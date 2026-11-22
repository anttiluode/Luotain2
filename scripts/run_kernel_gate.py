"""Gate IV-007 driver: blind discovery under spatially correlated sensor noise.

Blind kernel loop (see blind_kernel): uncorrected selection, a start kernel from a fixed
over-complete probe law, then alternate Wick-corrected selection with the echo kernel of
the selected law (radius 2, symmetric). At most 4 selections. Truth is read only by the scorer.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from luotain2.worlds import make_world
from luotain2.stream import Stream
from luotain2.grammar import shifted
from luotain2.noise_kernel import (BlindKernelDiscoverer, echo_kernel, probe_kernel,
                                   probe_terms, kernel_is_valid, scale_kernel)
from run_sensor_gate import sensor, score


def camera(paths, ratio, rng):
    """Signal-dependent AND spatially correlated: Poisson-like level x blurred white noise."""
    sd = ratio * float(np.mean([p.std() for p in paths]))
    out = []
    for p in paths:
        inten = p - p.min() + 0.25 * p.std()
        w = rng.standard_normal(p.shape)
        b = sum(shifted(w, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 3.0
        out.append(p + np.sqrt(inten / inten.mean()) * sd * b)
    return out, sd * sd


def data(world, seed, ratio, model):
    paths, law = make_world(world, seed=seed, n=64, T=40, noise=0.02, kind="process")
    rng = np.random.default_rng(seed + 99001)
    obs, R = camera(paths, ratio, rng) if model == "camera" else sensor(paths, model, ratio, rng)
    return obs, law, R


def _select(tr, va, seed, kern):
    d = BlindKernelDiscoverer(max_lag=3, margin=31, k_props=12, max_terms=12, delta=0.003,
                              pixel_stride=2, max_condition=1e5, seed=seed, estimator="iv",
                              k_products=4)
    return d.run(tr, va, kernel=kern)


def _echo(obs, f, kern, radius):
    ks = [echo_kernel(o, f.terms, f.coef, radius=radius, kernel0=kern) for o in obs[:2]]
    ks = [k for k in ks if k is not None]
    return {D: float(np.mean([k[D] for k in ks])) for D in ks[0]} if ks else None


def blind_kernel(obs, seed, radius=2, max_rounds=4, haircut=0.85):
    """Round 1: uncorrected selection. Start kernel: echo of a fixed over-complete probe law
    (local stencil, u^2, u^3, nearest products at lag 1, plus round 1's non-local shifts),
    checked for positive semidefiniteness and multiplied by `haircut`; if the probe is
    refused or invalid, the echo of the round-1 law is used instead. Later rounds: echo of
    the law just selected, used as is. Stop when C(0) moves < 5%."""
    tr, va = [Stream(p, mode="dynamics", max_lag=3) for p in obs[:2]]
    f = _select(tr, va, seed, {})
    rounds = [{"C0": 0.0, "n_terms": len(f.terms), "source": "none"}]
    if not f.terms:
        return f, {}, rounds
    extra = [t for t in f.terms if t[0] == "shift" and max(abs(t[3]), abs(t[4])) > 1]
    kern, _ = probe_kernel(obs[:2], probe_terms(extra), radius=radius)
    source = "probe"
    if kern is None or not kernel_is_valid(kern):
        kern, source = _echo(obs, f, {}, radius), "round-1 echo"
    if kern is None:
        return f, {}, rounds
    kern = scale_kernel(kern, haircut)
    for _ in range(max_rounds - 1):
        f = _select(tr, va, seed, kern)
        rounds.append({"C0": kern[(0, 0)], "n_terms": len(f.terms), "source": source})
        new = _echo(obs, f, kern, radius) if f.terms else None
        if new is None:
            break
        old0 = kern[(0, 0)]
        kern, source = new, "echo"
        if abs(new[(0, 0)] - old0) / old0 < 0.05:
            break
    return f, kern, rounds


def one(world, seed, ratio, model):
    obs, law, R = data(world, seed, ratio, model)
    f, kern, rounds = blind_kernel(obs, seed)
    k = {f"{D[0]},{D[1]}": v / R for D, v in kern.items()} if kern else {}
    return {"world": world, "seed": seed, "sensor_ratio": ratio, "noise_model": model,
            "kernel_over_R": k, "rounds": rounds, **score(f.terms, law, f.describe())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[901, 902, 903])
    p.add_argument("--conditions", nargs="+", default=["blurred:0.04", "camera:0.04", "gauss:0.04"])
    p.add_argument("--worlds", nargs="+", default=["allen_cahn", "burgers", "wormhole", "null"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for cond in a.conditions:
        model, ratio = cond.split(":")
        for w in a.worlds:
            for s in a.seeds:
                ti = time.time(); row = one(w, s, float(ratio), model); row["seconds"] = round(time.time() - ti, 1)
                rows.append(row)
                print(cond, w, s, row["status"], round(row["kernel_over_R"].get("0,0", 0), 2),
                      round(row["kernel_over_R"].get("0,1", 0), 2), row["extra"], row["missing"], row["seconds"], flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps({"seeds": a.seeds, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()

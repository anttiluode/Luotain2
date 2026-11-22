"""Gate IV-007 driver: blind discovery under spatially correlated sensor noise.

Blind kernel loop: start with no correction, select a law, estimate the noise kernel
(radius 2, symmetric) from that law's echo on the train and validation trajectories,
select again with Wick-corrected fits and validation risk; stop when C(0) moves < 5%
(at most 4 selections). Truth is read only by the scorer.
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
from luotain2.noise_kernel import BlindKernelDiscoverer, echo_kernel
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


def blind_kernel(obs, seed, radius=2, max_rounds=4):
    tr, va = [Stream(p, mode="dynamics", max_lag=3) for p in obs[:2]]
    kern, rounds, f = {}, [], None
    for _ in range(max_rounds):
        d = BlindKernelDiscoverer(max_lag=3, margin=31, k_props=12, max_terms=12, delta=0.003,
                                  pixel_stride=2, max_condition=1e5, seed=seed, estimator="iv",
                                  k_products=4)
        f = d.run(tr, va, kernel=kern)
        ks = [echo_kernel(o, f.terms, f.coef, radius=radius, kernel0=kern) for o in obs[:2]]
        ks = [k for k in ks if k is not None]
        rounds.append({"C0": kern.get((0, 0), 0.0), "n_terms": len(f.terms)})
        if not ks:
            break
        new = {D: float(np.mean([k[D] for k in ks])) for D in ks[0]}
        old0 = kern.get((0, 0), 0.0)
        kern = new
        if old0 > 0 and abs(new[(0, 0)] - old0) / old0 < 0.05:
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
    p.add_argument("--conditions", nargs="+", default=["blurred:0.04", "blurred:0.08", "camera:0.04", "gauss:0.04"])
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

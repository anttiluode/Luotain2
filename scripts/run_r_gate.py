"""Gate IV-005 driver: unknown sensor variance and weak excitation for polynomial laws.

part A  miscalibrated R: the selector is told R_hat = factor x R.
part B  blind R: start from R_hat = 0, then alternate (select law -> R_hat from the
        noise echo of that law on the train+validation trajectories -> select again)
        until R_hat moves less than 5% (at most 4 selections).
part C  weak excitation: process noise 0, 0.5%, 1%; known R; IV vs OLS.
Truth is read only by the scorer; test trajectory only after selection.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from luotain2.worlds import make_world
from luotain2.stream import Stream
from luotain2.blind_poly import BlindPolyDiscoverer
from luotain2.noise_echo import echo_variance


def data(world, seed, sensor_ratio, process_ratio, n=64, T=40):
    paths, law = make_world(world, seed=seed, n=n, T=T, noise=process_ratio, kind="process")
    rng = np.random.default_rng(seed + 99001)
    sd = sensor_ratio * float(np.mean([p.std() for p in paths]))
    obs = [p + sd * rng.standard_normal(p.shape) for p in paths]
    return obs, law, sd * sd


def select(obs, R_hat, seed, estimator="iv"):
    tr, va = [Stream(p, mode="dynamics", max_lag=3) for p in obs[:2]]
    d = BlindPolyDiscoverer(max_lag=3, margin=31, k_props=12, max_terms=12, delta=0.003,
                            pixel_stride=2, max_condition=1e5, seed=seed, estimator=estimator,
                            correction="exact", k_products=4)
    return d.run(tr, va, sensor_variance=R_hat)


def score(found, law):
    t, g = set(law), set(found.terms)
    return {"status": "exact" if t == g else ("null-false-positive" if not t else "miss"),
            "n_found": len(g), "n_truth": len(t),
            "missing": sorted(repr(x) for x in t - g), "extra": sorted(repr(x) for x in g - t),
            "law": found.describe()}


def part_a(world, seed, s, factor):
    obs, law, R = data(world, seed, s, 0.02)
    f = select(obs, factor * R, seed)
    return {"part": "A", "world": world, "seed": seed, "sensor_ratio": s, "R_factor": factor, **score(f, law)}


def part_b(world, seed, s, max_rounds=4):
    obs, law, R = data(world, seed, s, 0.02)
    R_hat, rounds = 0.0, []
    for k in range(max_rounds):
        f = select(obs, R_hat, seed)
        est = [echo_variance(o, f.terms, f.coef, R0=R_hat) for o in obs[:2]]
        est = [e for e in est if e is not None]
        new = float(np.mean(est)) if est else R_hat
        rounds.append({"R_hat_over_R": R_hat / R, "n_terms": len(f.terms),
                       "exact": set(f.terms) == set(law)})
        done = R_hat > 0 and abs(new - R_hat) / R_hat < 0.05
        if done or not est:
            break
        R_hat = new
    return {"part": "B", "world": world, "seed": seed, "sensor_ratio": s,
            "final_R_hat_over_R": R_hat / R, "rounds": rounds, **score(f, law)}


def part_c(world, seed, process, estimator):
    obs, law, R = data(world, seed, 0.04, process)
    f = select(obs, R, seed, estimator)
    return {"part": "C", "world": world, "seed": seed, "sensor_ratio": 0.04, "process_ratio": process,
            "estimator": estimator, **score(f, law)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["A", "B", "C"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[701, 702, 703])
    p.add_argument("--worlds", nargs="+", default=None)
    p.add_argument("--sensor-ratios", type=float, nargs="+", default=[0.04, 0.08])
    p.add_argument("--factors", type=float, nargs="+", default=[0.75, 0.9, 1.0, 1.1, 1.25])
    p.add_argument("--processes", type=float, nargs="+", default=[0.0, 0.005, 0.01])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    t0 = time.time()
    if a.part == "C":
        worlds = a.worlds or ["allen_cahn", "burgers"]
        jobs = [(w, s, pr, e) for pr in a.processes for w in worlds for s in a.seeds for e in ("iv", "ols")]
        fn = part_c
    else:
        worlds = a.worlds or ["allen_cahn", "burgers", "wormhole", "null"]
        if a.part == "A":
            jobs = [(w, s, r, f) for r in a.sensor_ratios for f in a.factors for w in worlds for s in a.seeds]
            fn = part_a
        else:
            jobs = [(w, s, r) for r in a.sensor_ratios for w in worlds for s in a.seeds]
            fn = part_b
    for j in jobs:
        ti = time.time(); row = fn(*j); row["seconds"] = round(time.time() - ti, 1); rows.append(row)
        print(a.part, j, row["status"], row.get("final_R_hat_over_R", ""), row["extra"], row["missing"], row["seconds"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"part": a.part, "seeds": a.seeds, "rows": rows,
                                       "seconds": round(time.time() - t0)}, indent=1) + "\n")


if __name__ == "__main__":
    main()

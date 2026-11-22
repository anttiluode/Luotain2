"""Gate IV-006 driver: toward real sensors.

part A  estimator choice: run IV and OLS selections with the same (known) R and keep the
        law with the lower corrected validation risk (+ the search's complexity penalty).
part B  conservative R: the blind echo loop of IV-005 with a haircut h on every R_hat.
part C  sensor noise that breaks the Gaussian i.i.d. assumption, blind echo loop, IV:
          gauss      i.i.d. Gaussian (reference)
          quantized  uniform rounding to a grid (non-Gaussian, a deterministic function of X)
          poisson    signal-dependent variance (Poisson-Gaussian, variance ~ intensity)
          blurred    spatially correlated (3x3 box-blurred white noise), white in time
          heavy      Student-t, 4 degrees of freedom
        every model is scaled to the same average per-pixel standard deviation.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from luotain2.worlds import make_world
from luotain2.stream import Stream
from luotain2.grammar import shifted
from luotain2.blind_poly import BlindPolyDiscoverer
from luotain2.noise_echo import echo_variance

PENALTY = 2e-6


def sensor(paths, model, ratio, rng):
    sd = ratio * float(np.mean([p.std() for p in paths]))
    out = []
    for p in paths:
        if model == "gauss":
            e = sd * rng.standard_normal(p.shape)
        elif model == "quantized":
            step = sd * np.sqrt(12.0)
            off = rng.uniform(0, step)              # one fixed random grid offset per trajectory
            e = step * np.round((p - off) / step) + off - p
        elif model == "poisson":
            inten = p - p.min() + 0.25 * p.std()    # positive "light level"
            e = np.sqrt(inten / inten.mean()) * sd * rng.standard_normal(p.shape)
        elif model == "blurred":
            w = rng.standard_normal(p.shape)
            b = sum(shifted(w, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 3.0
            e = sd * b
        elif model == "heavy":
            e = sd * rng.standard_t(4, size=p.shape) / np.sqrt(2.0)   # t4 variance is 2
        else:
            raise KeyError(model)
        out.append(p + e)
    return out, sd * sd


def data(world, seed, ratio, process, model="gauss"):
    paths, law = make_world(world, seed=seed, n=64, T=40, noise=process, kind="process")
    obs, R = sensor(paths, model, ratio, np.random.default_rng(seed + 99001))
    return obs, law, R


def selector(seed, estimator):
    return BlindPolyDiscoverer(max_lag=3, margin=31, k_props=12, max_terms=12, delta=0.003,
                               pixel_stride=2, max_condition=1e5, seed=seed, estimator=estimator,
                               correction="exact", k_products=4)


def select(obs, R_hat, seed, estimator="iv"):
    tr, va = [Stream(p, mode="dynamics", max_lag=3) for p in obs[:2]]
    d = selector(seed, estimator)
    f = d.run(tr, va, sensor_variance=R_hat)
    return f, d, va


def score(terms, law, describe=""):
    t, g = set(law), set(terms)
    return {"status": "exact" if t == g else ("null-false-positive" if not t else "miss"),
            "n_found": len(g), "n_truth": len(t),
            "missing": sorted(repr(x) for x in t - g), "extra": sorted(repr(x) for x in g - t), "law": describe}


def blind(obs, seed, haircut=1.0, max_rounds=4):
    R_hat, rounds, f = 0.0, [], None
    for _ in range(max_rounds):
        f, _, _ = select(obs, haircut * R_hat, seed)
        est = [echo_variance(o, f.terms, f.coef, R0=haircut * R_hat) for o in obs[:2]]
        est = [e for e in est if e is not None]
        new = float(np.mean(est)) if est else R_hat
        rounds.append({"R_hat": R_hat, "used": haircut * R_hat, "n_terms": len(f.terms)})
        done = R_hat > 0 and abs(new - R_hat) / R_hat < 0.05
        if done or not est:
            break
        R_hat = new
    return f, R_hat, rounds


def part_a(world, seed, ratio, process):
    obs, law, R = data(world, seed, ratio, process)
    res = {}
    for est in ("iv", "ols"):
        f, d, va = select(obs, R, seed, est)
        risk = d.corrected_risk(va, f.terms, f.coef) + PENALTY * len(f.terms)
        res[est] = (f, risk)
    pick = min(res, key=lambda k: res[k][1])
    row = {"part": "A", "world": world, "seed": seed, "sensor_ratio": ratio, "process_ratio": process,
           "picked": pick, **score(res[pick][0].terms, law, res[pick][0].describe())}
    for k, (f, r) in res.items():
        row[f"{k}_status"] = score(f.terms, law)["status"]
        row[f"{k}_val_risk"] = r
    return row


def part_b(world, seed, ratio, haircut):
    obs, law, R = data(world, seed, ratio, 0.02)
    f, R_hat, rounds = blind(obs, seed, haircut)
    return {"part": "B", "world": world, "seed": seed, "sensor_ratio": ratio, "haircut": haircut,
            "R_hat_over_R": R_hat / R, "rounds": rounds, **score(f.terms, law, f.describe())}


def part_c(world, seed, model):
    obs, law, R = data(world, seed, 0.04, 0.02, model)
    f, R_hat, rounds = blind(obs, seed)
    return {"part": "C", "world": world, "seed": seed, "noise_model": model, "sensor_ratio": 0.04,
            "R_hat_over_R": R_hat / R, "rounds": rounds, **score(f.terms, law, f.describe())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["A", "B", "C"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[801, 802, 803])
    p.add_argument("--models", nargs="+", default=["gauss", "quantized", "poisson", "blurred", "heavy"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.part == "A":
        jobs = [(w, s, 0.04, pr) for pr in (0.0, 0.005, 0.02) for w in ("allen_cahn", "burgers") for s in a.seeds]
        jobs += [("allen_cahn", s, 0.08, 0.02) for s in a.seeds]
        fn = part_a
    elif a.part == "B":
        jobs = [(w, s, r, 0.9) for r in (0.04, 0.08) for w in ("allen_cahn", "burgers", "wormhole", "null") for s in a.seeds]
        fn = part_b
    else:
        jobs = [(w, s, m) for m in a.models for w in ("allen_cahn", "burgers", "wormhole", "null") for s in a.seeds]
        fn = part_c
    rows, t0 = [], time.time()
    for j in jobs:
        ti = time.time(); row = fn(*j); row["seconds"] = round(time.time() - ti, 1); rows.append(row)
        print(a.part, j, row["status"], {k: v for k, v in row.items() if k in ("picked", "iv_status", "ols_status", "R_hat_over_R")},
              row["extra"], row["missing"], row["seconds"], flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"part": a.part, "seeds": a.seeds, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()

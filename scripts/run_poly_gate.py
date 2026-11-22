"""Gate IV-004 driver: blind polynomial-law discovery under sensor noise.

Same three independent trajectories as IV-002/003 (train proposes and fits,
validation selects, test is read only after selection). Latent paths come from
the original simulator with process noise; independent Gaussian sensor noise of
KNOWN standard deviation (sensor_ratio x latent std) is added to what is observed.
Paths are NOT rescaled: rescaling changes the coefficients of u^3 and u*u laws.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from luotain2.worlds import make_world
from luotain2.stream import Stream
from luotain2.blind_poly import BlindPolyDiscoverer, corrected_poly_risk_raw
from luotain2.discover import nmse

WORLDS = ["allen_cahn", "burgers", "heat", "wormhole", "null"]


def one(world, seed, estimator, correction, sensor_ratio, process_ratio, n=64, T=40,
        k_props=12, max_terms=12, k_products=4):
    ti = time.time()
    paths, law = make_world(world, seed=seed, n=n, T=T, noise=process_ratio, kind="process")
    rng = np.random.default_rng(seed + 99001)
    sd = sensor_ratio * float(np.mean([p.std() for p in paths]))
    obs = [p + sd * rng.standard_normal(p.shape) for p in paths]
    R = sd * sd
    train, val, test = [Stream(p, mode="dynamics", max_lag=3) for p in obs]
    d = BlindPolyDiscoverer(max_lag=3, margin=min(32, n // 2 - 1), k_props=k_props,
                            max_terms=max_terms, delta=0.003, pixel_stride=2,
                            max_condition=1e5, seed=seed, estimator=estimator,
                            correction=correction, k_products=k_products)
    found = d.run(train, val, sensor_variance=R)
    truth, got = set(law), set(found.terms)
    verdict = "exact" if truth == got else ("null-false-positive" if not truth else "miss")
    clean_test = Stream(paths[2], mode="dynamics", max_lag=3)
    row = {"world": world, "seed": seed, "estimator": estimator, "correction": correction,
           "sensor_ratio": sensor_ratio, "process_ratio": process_ratio, "sensor_var": R,
           "status": verdict, "n_found": len(got), "n_truth": len(truth),
           "missing": sorted(repr(t) for t in truth - got), "extra": sorted(repr(t) for t in got - truth),
           "law": found.describe(),
           "corrected_test_risk": d.corrected_risk(test, found.terms, found.coef),
           "clean_test_nmse": nmse(clean_test.y, clean_test.design(found.terms) @ found.coef),
           "iv_refusals": found.refusals, "fits": found.fit_calls, "seconds": round(time.time() - ti, 1)}
    if truth == got:
        tv = np.array([law[t] for t in found.terms])
        row["coef_rel_error"] = float(np.linalg.norm(found.coef[1:] - tv) / np.linalg.norm(tv))
    print(f"{estimator:3s} {correction:6s} {world:10s} {seed} s={sensor_ratio}: {verdict:20s} "
          f"{len(got)}/{len(truth)} missing={row['missing']} extra={row['extra']} {row['seconds']}s", flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", nargs="+", default=WORLDS)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    p.add_argument("--estimators", nargs="+", default=["iv", "ols"])
    p.add_argument("--corrections", nargs="+", default=["exact", "linear"])
    p.add_argument("--sensor-ratios", type=float, nargs="+", default=[0.01])
    p.add_argument("--process-ratio", type=float, default=0.02)
    p.add_argument("--out", default="results/poly_gate.json")
    a = p.parse_args()
    rows = []
    for s in a.sensor_ratios:
        for w in a.worlds:
            for seed in a.seeds:
                for e in a.estimators:
                    for c in a.corrections:
                        rows.append(one(w, seed, e, c, s, a.process_ratio))
    summ = {}
    for r in rows:
        k = f"{r['estimator']}|{r['correction']}|sensor {r['sensor_ratio']}"
        s = summ.setdefault(k, {"cases": 0, "exact": 0, "by_world": {}})
        s["cases"] += 1; s["exact"] += r["status"] == "exact"
        bw = s["by_world"].setdefault(r["world"], [0, 0]); bw[0] += r["status"] == "exact"; bw[1] += 1
    report = {"summary": summ, "process_ratio": a.process_ratio, "seeds": a.seeds, "rows": rows,
              "support_given_to_search": False, "sensor_variance_known_to_selector": True}
    print("RESULT", json.dumps(summ), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Exploratory Gate IV-001: known-support coefficient recovery under sensor noise.

The original Luotain2 discovery benchmark fails to identify non-null exact laws
under 1% observation noise. Before changing the open-grammar search, this
isolated gate tests whether disjoint-time IV can fit the TRUE, supplied terms.
A successful oracle-support fit is NOT discovery and must not replace the
original result ledger. WAIVES (Ruzayqat, Bortz, Dukic, 2026) is a continuous
Itô SDE estimator; here we test only an analogous discrete-time IV principle.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from luotain2.iv import estimate, coefficient_error
from luotain2.worlds import make_world, WORLDS, STRESS
from luotain2.stream import Stream
from luotain2.discover import nmse

DEFAULT_WORLDS = [x for x in WORLDS + STRESS if x != "null"]


def score_clean(terms, coefficients, clean):
    max_lag = max(t[2] if t[0] == "shift" else t[1] for t in terms)
    stream = Stream(clean, mode="dynamics", max_lag=max_lag)
    return nmse(stream.y, stream.design(terms) @ coefficients)


def run(worlds=DEFAULT_WORLDS, seeds=(101, 102, 103, 104, 105),
        window=1, gap=0, max_condition=1e9, n=64, T=40,
        measurement_ratio=.01, kind="measurement", process_ratio=.02):
    results = []
    for world in worlds:
        for seed in seeds:
            if kind == "combined":
                raw, truth = make_world(world, seed=seed, n=n, T=T,
                                        noise=process_ratio, kind="process")
                # Independent sensor-only corruption is now added *after*
                # the process-noise-driven state trajectory is complete.
                latent = raw[0]
                noise_rng = np.random.default_rng(seed + 700000)
                observed = latent + (measurement_ratio * latent.std()) * noise_rng.standard_normal(latent.shape)
            else:
                raw, truth = make_world(world, seed=seed, n=n, T=T,
                                        noise=measurement_ratio if kind == "measurement" else process_ratio,
                                        kind=kind)
                observed = raw[0]
            terms = list(truth)
            if not terms:
                results.append({"world": world, "seed": seed,
                                "status": "null: no oracle nonzero law"})
                continue
            # KNOWN corruption-level calibration from the benchmark design.
            # Do not pretend this is estimated from a continuous-SDE variogram.
            sensor_variance = float((measurement_ratio * observed.std())**2) if kind in ("measurement", "combined") else 0.0
            fitted = estimate(observed, terms, sensor_var=sensor_variance,
                              window=window, gap=gap, max_condition=max_condition)
            quality = coefficient_error(fitted, terms, truth)
            # Independently simulated clean test trajectory, never used in the fit.
            clean = make_world(world, seed=seed + 10000, n=n, T=T,
                               noise=process_ratio if kind == "combined" else 0.0,
                               kind="process" if kind == "combined" else "measurement")[0][2]
            quality["world"] = world
            quality["seed"] = seed
            quality["oracle_support"] = True
            quality["sensor_variance_calibrated"] = sensor_variance
            quality["clean_test_nmse_ols"] = score_clean(terms, fitted.ols_coef, clean)
            quality["clean_test_nmse_iv"] = (
                score_clean(terms, fitted.coef, clean) if fitted.identified else None)
            quality["clean_test_nmse_truth"] = score_clean(
                terms, np.array([0.0] + [truth[t] for t in terms]), clean)
            results.append(quality)
            flag = ("REFUSE" if not fitted.identified else
                    "IV better" if quality["iv_l2"] < quality["ols_l2"] else "OLS better")
            print(f"{world:11s} {seed} {flag:11s} "
                  f"coef L2 OLS={quality['ols_l2']:.5g} "
                  f"IV={quality['iv_l2'] if fitted.identified else 'n.i.'} "
                  f"condition={fitted.condition:.3g}", flush=True)
    good = [r for r in results if r.get("identified")]
    attempted = [r for r in results if "identified" in r]
    summary = {
        "status": "oracle-support diagnostic, not blind discovery",
        "model": "discrete-time spatial autoregression, not WAIVES continuous SDE",
        "kind": kind,
        "measurement_ratio": measurement_ratio,
        "process_ratio": (process_ratio if kind == "combined" else 0.0),
        "window": window,
        "gap": gap,
        "conditions": {"n": n, "T": T, "noise_variance": "known 1% ratio times observed trajectory std^2" if kind in ("measurement", "combined") else "none"},
        "attempted": len(attempted),
        "identified": len(good),
        "refused": len(attempted)-len(good),
        "iv_better_coefficient_l2": sum(r["iv_l2"] < r["ols_l2"] for r in good),
        "iv_better_clean_test_nmse": sum(r["clean_test_nmse_iv"] < r["clean_test_nmse_ols"] for r in good),
        "mean_coefficient_l2_ols_when_identified": (float(np.mean([r["ols_l2"] for r in good])) if good else None),
        "mean_coefficient_l2_iv_when_identified": (float(np.mean([r["iv_l2"] for r in good])) if good else None),
        "warnings": [
            "An exact true-term support is supplied; this benchmark does not recover an unknown equation.",
            "Only 1% i.i.d. observation noise and a known calibration ratio are assumed.",
            "No continuous-time SDE or SPDE consistency theorem is claimed.",
            "Spatial samples are correlated; the condition statistic is a numerical gate, not an F test.",
            "A good IV estimate may still be noisier on a single finite trajectory."
        ]
    }
    return {"summary": summary, "rows": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=DEFAULT_WORLDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--max-condition", type=float, default=1e9)
    ap.add_argument("--kind", choices=("measurement", "process", "combined"), default="measurement")
    ap.add_argument("--measurement-ratio", type=float, default=.01)
    ap.add_argument("--process-ratio", type=float, default=.02)
    ap.add_argument("--out", default="results/iv_gate.json")
    args = ap.parse_args()
    out = run(args.worlds, args.seeds, args.window, args.gap, args.max_condition,
              args.n, args.T, args.measurement_ratio, args.kind, args.process_ratio)
    print(json.dumps(out["summary"], indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()

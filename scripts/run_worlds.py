"""Benchmark: can the law of each mystery world be recovered, and by which method?

Seeds 1-5 were used while developing the method. This script reports fresh seeds (101+).
Writes results/worlds.json and prints a table.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from luotain2.worlds import make_world, WORLDS, STRESS
from luotain2.stream import Stream
from luotain2.discover import Discoverer, lstsq, nmse
from luotain2.baselines import default_library, stlsq
from luotain2.grammar import term_name

CONDITIONS = {
    "process 2%": dict(noise=0.02, kind="process"),
    "process 10%": dict(noise=0.10, kind="process"),
    "measurement 1%": dict(noise=0.01, kind="measurement"),
}


def classify(found, law, tr, te):
    truth = list(law)
    if set(found.terms) == set(truth):
        verdict = "exact"
    else:
        verdict = None
    c_or = lstsq(tr.design(truth), tr.y)
    oracle = nmse(te.y, te.design(truth) @ c_or)
    test = nmse(te.y, te.design(found.terms) @ found.coef)
    if verdict is None:
        verdict = "equivalent" if test <= oracle * 1.01 else "miss"
    return verdict, test, oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--worlds", nargs="+", default=WORLDS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--methods", nargs="+", default=["residue", "random", "stlsq"])
    ap.add_argument("--out", default="results/worlds.json")
    a = ap.parse_args()
    rows = []
    if os.path.exists(a.out):
        rows = json.load(open(a.out))
    done = {(r["world"], r["seed"], r["condition"], r["method"]) for r in rows}
    lib = default_library()
    for cond in a.conditions:
        for w in a.worlds:
            for seed in a.seeds:
                trajs, law = make_world(w, seed=seed, **CONDITIONS[cond])
                tr, va, te = [Stream(x, mode="dynamics", max_lag=3) for x in trajs]
                hold = nmse(te.y, te.frames[te.targets - 1][:, te.mask].ravel())
                for m in a.methods:
                    if (w, seed, cond, m) in done:
                        continue
                    t0 = time.time()
                    if m == "stlsq":
                        L = stlsq(tr, va, lib)
                    else:
                        d = Discoverer(mode="dynamics", max_lag=3, margin=32, proposer=m, seed=seed)
                        L = d.run(tr, va, record_every=False)
                    verdict, test, oracle = classify(L, law, tr, te)
                    rows.append(dict(world=w, seed=seed, condition=cond, method=m, verdict=verdict,
                                     test_nmse=test, oracle_nmse=oracle, hold_last_nmse=hold,
                                     n_terms=len(L.terms), n_true=len(law), probes=L.probes,
                                     killed=L.killed, full_fits=L.full_fits, seconds=time.time() - t0,
                                     law=L.describe(4), truth=" ".join(f"{c:+.4f}*{term_name(t)}" for t, c in law.items())))
                    print(f"{cond:15s} {w:11s} {seed} {m:8s} {verdict:10s} test/oracle {test/oracle:6.3f} terms {len(L.terms):2d}/{len(law)} {time.time()-t0:5.1f}s", flush=True)
                    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

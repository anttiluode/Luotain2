"""Blind IV-002: exact structural recovery, no true support passed to search.

Uses original synthetic world generator and three independent trajectories:
train proposes/IV-fits, validation selects using an observable noise-corrected
risk, and test is not read until after final selection. This gate only searches
linear shifted fields (heat, diagonal, drift, wave, wormhole, echo, null).
Nonlinear examples are explicitly NOT evaluated as positive recoveries.
"""
from __future__ import annotations
import argparse,json,sys,os,time
from pathlib import Path
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from luotain2.worlds import make_world
from luotain2.stream import Stream
from luotain2.blind_iv import BlindIVDiscoverer
from luotain2.discover import nmse

WORLDS=["heat","diagonal","drift","wave","wormhole","echo","null"]


def run(worlds=WORLDS,seeds=(101,102),noise=.01,kind="measurement",
        n=64,T=40,delta=.003,k_props=20,max_terms=18,
        pixel_stride=2,max_condition=1e5,process_ratio=.02,
        estimator="iv",sensor_std=None,match_latent_std=None):
    records=[]
    for world in worlds:
        for seed in seeds:
            ti=time.time()
            if kind=="combined":
                # Draw process-driven paths, then independently contaminate
                # only observations. Same synthetic ground truth, no oracle
                # access during search, all three trajectories independent.
                paths,law=make_world(world,seed=seed,n=n,T=T,
                                     noise=process_ratio,kind="process")
                latent_pre_std=[float(p.std()) for p in paths]
                if match_latent_std is not None:
                    if match_latent_std <= 0:
                        raise ValueError("match_latent_std must be positive")
                    paths=[p*(match_latent_std/max(p.std(),1e-12)) for p in paths]
                latent_std=[float(p.std()) for p in paths]
                rng=np.random.default_rng(seed+99001)
                paths=[p+(sensor_std if sensor_std is not None else noise*p.std())
                         *rng.standard_normal(p.shape) for p in paths]
            else:
                paths,law=make_world(world,seed=seed,n=n,T=T,
                                     noise=noise,kind="measurement")
            if estimator not in ("iv","ols"):
                raise ValueError("estimator must be iv or ols")
            if sensor_std is not None and (not np.isfinite(sensor_std) or sensor_std < 0):
                raise ValueError("sensor_std must be nonnegative and finite")
            # Same absolute measurement-error variance, including in held-out
            # corrected validation. No clean target is consulted by search.
            fixed_sensor_var=(sensor_std**2 if sensor_std is not None else None)
            train,val,test=[Stream(p,mode="dynamics",max_lag=3) for p in paths]
            d=BlindIVDiscoverer(max_lag=3,margin=min(32,n//2-1),
                  k_props=k_props,max_terms=max_terms,delta=delta,
                  pixel_stride=pixel_stride,max_condition=max_condition,seed=seed,
                  estimator=estimator)
            found=d.run(train,val,sensor_ratio=noise,
                        sensor_variance=fixed_sensor_var,verbose=False)
            true=set(law)
            got=set(found.terms)
            # The truth and held-out test data are ONLY used here, not by the search.
            verdict=("exact" if true==got else
                     "null-false-positive" if not true else "miss")
            observed_test=nmse(test.y,test.design(found.terms)@found.coef)
            corrected_test=d.corrected_risk(test,found.terms,found.coef,noise,
                                             sensor_variance=fixed_sensor_var)
            row={"world":world,"seed":seed,"kind":kind,
                "status":verdict,"found":[repr(t) for t in found.terms],
                "truth":[repr(t) for t in law],
                "n_found":len(found.terms),"n_truth":len(law),
                "corrected_validation_risk":found.val_risk,
                "corrected_test_risk":corrected_test,
                "noisy_test_nmse":observed_test,
                "iv_refusals":found.refusals,"iv_fits":found.fit_calls,
                "n_steps":len(found.steps),"steps":found.steps,
                "seconds":time.time()-ti,"law":found.describe(),
                "estimator":estimator,"sensor_std":sensor_std,
                "match_latent_std":match_latent_std,
                "latent_std_before_matching":(latent_pre_std if kind=="combined" else None),
                "latent_std_after_matching":(latent_std if kind=="combined" else None),
                "observed_std":[float(p.std()) for p in paths]}
            records.append(row)
            print(f"{estimator:3s} {kind:11} {world:9} {seed}: {verdict:18} "
                  f"{len(found.terms)}/{len(law)} terms "
                  f"score {corrected_test:.5g} fits {found.fit_calls} "
                  f"refused {found.refusals} time {row['seconds']:.1f}s",flush=True)
    summary={
        "claim":"blind linear-shift term discovery under i.i.d. sensor noise",
        "support_given_to_search":False,
        "noise_known_to_selector":True,
        "nonlinear_term_grammar_enabled":False,
        "independent_train_validation_test_trajectories":True,
        "worlds":list(worlds),"seeds":list(seeds),"kind":kind,
        "estimator":estimator,"sensor_std":sensor_std,
        "match_latent_std":match_latent_std,
        "n":n,"T":T,"noise":noise,"process_noise":process_ratio if kind=="combined" else 0,
        "cases":len(records),"exact":sum(x["status"]=="exact" for x in records),
        "miss":sum(x["status"]=="miss" for x in records),
        "null_false_positive":sum(x["status"]=="null-false-positive" for x in records),
        "note":"A passed original-simulator benchmark is not new science or a WAIVES theorem."
    }
    return {"summary":summary,"rows":records}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worlds",nargs="+",default=WORLDS)
    p.add_argument("--seeds",type=int,nargs="+",default=[101,102])
    p.add_argument("--n",type=int,default=64)
    p.add_argument("--T",type=int,default=40)
    p.add_argument("--noise",type=float,default=.01)
    p.add_argument("--process-ratio",type=float,default=.02)
    p.add_argument("--estimator",choices=["iv","ols"],default="iv")
    p.add_argument("--sensor-std",type=float,default=None)
    p.add_argument("--match-latent-std",type=float,default=None)
    p.add_argument("--kind",choices=["measurement","combined"],default="measurement")
    p.add_argument("--delta",type=float,default=.003)
    p.add_argument("--k-props",type=int,default=20)
    p.add_argument("--max-terms",type=int,default=18)
    p.add_argument("--pixel-stride",type=int,default=2)
    p.add_argument("--max-condition",type=float,default=1e5)
    p.add_argument("--out",default="results/blind_iv_gate.json")
    a=p.parse_args()
    report=run(a.worlds,a.seeds,a.noise,a.kind,a.n,a.T,
               a.delta,a.k_props,a.max_terms,a.pixel_stride,a.max_condition,
               a.process_ratio,a.estimator,a.sensor_std,a.match_latent_std)
    print("RESULT",json.dumps(report["summary"],sort_keys=True),flush=True)
    if a.out:
        dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2)+"\n")
        print("WROTE",str(dest),flush=True)


if __name__=="__main__":
    main()

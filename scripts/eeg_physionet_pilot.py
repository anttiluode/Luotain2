"""First *real EEG* observability pilot; not an anatomical connectivity estimator.

Uses PhysioNet EEG Motor Movement/Imagery subject 1, baseline runs 1 (eyes open)
and 2 (eyes closed), 64 channels at 160 Hz. The model is an illustrative
8-channel scalp-sensor autoregression, not the 2-D toroidal Luotain2
spatial grammar. Cross-channel weights are NOT brain connections.

Data source: https://physionet.org/content/eegmmidb/1.0.0/
Method: posterior alpha-band energy, held-out temporal predictability, matched
OLS-vs-disjoint-IV, and a synthetic *independent sources + volume mixing* null.
"""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np

PICKS=("Fp1","Fp2","C3","C4","P3","P4","O1","O2")
POSTERIOR=("P3","P4","O1","O2")


def canonical(name):
    return "".join(ch for ch in name.upper() if ch.isalnum())


def choose_channels(names, wanted=PICKS):
    lookup={canonical(s):i for i,s in enumerate(names)}
    chosen=[(name,lookup[canonical(name)]) for name in wanted
            if canonical(name) in lookup]
    if len(chosen)<6:
        raise ValueError(f"Fewer than 6 selected electrodes: {chosen}; names={names}")
    return chosen


def bands(eeg, sfreq, names):
    """Welch power spectra; compare alpha 8-12 Hz to 4-30 Hz on posterior leads."""
    from scipy.signal import welch
    spectra=[]
    for row in eeg:
        hz,power=welch(row,sfreq,nperseg=min(int(sfreq*4),len(row)),
                       noverlap=min(int(sfreq*2),len(row)//2))
        alpha=np.trapezoid(power[(hz>=8)&(hz<12)],hz[(hz>=8)&(hz<12)])
        broad=np.trapezoid(power[(hz>=4)&(hz<30)],hz[(hz>=4)&(hz<30)])
        spectra.append({"channel":row is None,"alpha":float(alpha),
                        "broad":float(broad),"ratio":float(alpha/(broad+1e-20))})
    for k,n in enumerate(names):spectra[k]["channel"]=n
    post=[v["ratio"] for v in spectra if v["channel"] in POSTERIOR]
    return {"per_channel":spectra,"posterior_alpha_fraction_mean":(
            float(np.mean(post)) if post else None)}


def regression(eeg, sfreq, ridge=.01):
    """Baseline and linear OLS/IV on EXACTLY matching time rows."""
    if eeg.ndim!=2 or eeg.shape[0]<6:
        raise ValueError("eeg shape must be channels x time, >=6 channels")
    x=np.asarray(eeg,dtype=float).T
    split=int(len(x)*.65)
    if len(x)-split<int(sfreq*4):raise ValueError("too short")
    # No data from test is used to choose the normalizer.
    mu=x[:split].mean(axis=0)
    sd=np.maximum(x[:split].std(axis=0),1e-9)
    x=(x-mu)/sd
    target=np.arange(2,len(x))
    train=target<split
    test=target>=split+int(sfreq)  # 1 s guard, no overlap with fitting
    y=x[target]
    A=x[target-1]
    Z=x[target-2]
    at,zt,bt=A[train],Z[train],y[train]
    ae,be=A[test],y[test]
    channels=x.shape[1]
    cov=at.T@at/len(at)
    rhs=at.T@bt/len(at)
    W=np.linalg.solve(cov+ridge*np.eye(channels),rhs)
    moment=zt.T@at/len(at)
    ivrhs=zt.T@bt/len(at)
    singular=np.linalg.svd(moment,compute_uv=False)
    condition=float(singular[0]/max(singular[-1],1e-20))
    # Not a first-stage F. Correlated EEG channels inflate effective
    # dimensionality and the sample count isn't independent.
    iv_ok=condition<1e4 and singular[-1]>1e-8
    V=np.linalg.solve(moment+ridge*np.eye(channels),ivrhs) if iv_ok else None
    diagonal=np.sum(at*bt,axis=0)/np.maximum(np.sum(at*at,axis=0),1e-12)
    denom=max(float(np.mean(be**2)),1e-12)
    mse=lambda pred:float(np.mean((be-pred)**2)/denom)
    off=W-np.diag(np.diag(W))
    return {
      "train_time_samples":int(train.sum()),"test_time_samples":int(test.sum()),
      "heldout_persistence_nmse":mse(ae),
      "heldout_channelwise_ar1_nmse":mse(ae*diagonal),
      "heldout_multichannel_ols_nmse":mse(ae@W),
      "heldout_multichannel_iv_nmse":mse(ae@V) if iv_ok else None,
      "iv_identified_numerically":bool(iv_ok),
      "iv_crossmoment_condition":condition,
      "iv_strength_ratio":float(singular[-1]/max(singular[0],1e-20)),
      "ols_offdiagonal_weight_fraction":float(
         np.linalg.norm(off)/max(np.linalg.norm(W),1e-20)),
      "warning":"EEG sensor-space coefficients are not anatomical neural connections"
    }


def independent_sources_mixed(seed=1,n=20000):
    """Negative demonstration: mixed independent AR sources create dense sensor AR."""
    rng=np.random.default_rng(seed)
    poles=np.linspace(.2,.94,8)
    S=np.zeros((n,8))
    for t in range(1,n):
        S[t]=poles*S[t-1]+rng.standard_normal(8)*np.sqrt(1-poles**2)
    M=np.eye(8)+.5*rng.standard_normal((8,8))
    Y=S@M.T
    truth=M@np.diag(poles)@np.linalg.inv(M)
    estimated=np.linalg.lstsq(Y[:-1],Y[1:],rcond=None)[0].T
    off=lambda W:float(np.linalg.norm(W-np.diag(np.diag(W)))/np.linalg.norm(W))
    return {"source_true_cross_couplings":0,
            "mixed_sensor_ar_offdiagonal_fraction":off(estimated),
            "mixed_sensor_theoretical_fraction":off(truth),
            "relative_fit_error":float(
                np.linalg.norm(estimated-truth)/np.linalg.norm(truth)),
            "conclusion":"mixing alone creates apparent cross-channel coupling"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subject",type=int,default=1)
    ap.add_argument("--out",default="results/eeg_physionet_001.json")
    args=ap.parse_args()
    import mne
    from mne.datasets import eegbci
    files=eegbci.load_data(args.subject,[1,2],verbose=False)
    records={}
    for condition,filename in zip(["eyes_open","eyes_closed"],files):
        raw=mne.io.read_raw_edf(filename,preload=True,verbose=False)
        eegbci.standardize(raw)
        selected=choose_channels(raw.ch_names)
        data=raw.get_data(picks=[ix for _,ix in selected])
        names=[name for name,_ in selected]
        fs=float(raw.info["sfreq"])
        records[condition]={
            "duration_s":float(data.shape[1]/fs),
            "channels":names,"sfreq":fs,
            "bands":bands(data,fs,names),
            "prediction":regression(data,fs)
        }
        print(condition,"posterior alpha fraction",
              records[condition]["bands"]["posterior_alpha_fraction_mean"],
              "OLS nmse",records[condition]["prediction"]["heldout_multichannel_ols_nmse"],
              "IV nmse",records[condition]["prediction"]["heldout_multichannel_iv_nmse"],
              flush=True)
    o=records["eyes_open"]["bands"]["posterior_alpha_fraction_mean"]
    c=records["eyes_closed"]["bands"]["posterior_alpha_fraction_mean"]
    result={
        "dataset":"PhysioNet EEG Motor Movement/Imagery Dataset",
        "source":"https://physionet.org/content/eegmmidb/1.0.0/",
        "subject":args.subject,
        "runs":{"eyes_open":1,"eyes_closed":2},
        "model":"8-electrode scalp sensor-space linear AR1, not 2D field dynamics",
        "training":"first 65% of each run, no cross-run learning",
        "heldout":"last 35% after 1-second gap",
        "recordings":records,
        "posterior_closed_over_open_alpha_fraction":(
            float(c/o) if c is not None and o is not None and o>0 else None),
        "mixing_null":independent_sources_mixed(),
        "limitations":[
         "No neural wiring identified; scalp EEG is an instantaneous volume-conducted mixture",
         "Only 8 of 64 electrodes and one public volunteer; this is feasibility, not generalization",
         "Eyes open and closed runs are different times, not randomized interventions",
         "Common reference, blink, muscle and line-noise artifacts can mimic connectivity",
         "Temporal predictability can be trivial autocorrelation at 160 Hz",
         "Strictly earlier IV is not automatically a valid causal instrument",
         "No medical interpretation of the participant or user"
        ]
    }
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print("WROTE",path,flush=True)


if __name__=="__main__":
    main()

"""Blind support discovery with IV refits and sensor-noise-corrected validation.

Experimental Gate IV-002: term support is NEVER passed to the search.
The old Discoverer remains untouched. This deliberately begins with linear
shift terms only; polynomial and product selection need a richer corrected
validation risk and are NOT claimed here.

Proposal mechanism: Luotain2's original full-offset FFT residual correlations.
Coefficient fitting: earlier disjoint-time instruments (iv.py).
Term selection: independent-trajectory *latent-risk-corrected* validation
for observed i.i.d. additive noise of KNOWN variance.

For a linear discrete law with distinct input time/pixel coordinates,
E[(Y_t - b0 - sum beta_j Y_{t-lag_j}(x+offset_j))^2]
 = latent innovation risk + R_t + sum beta_j^2 R_input_j.
We subtract this known sensor contribution, avoiding selection of noisy
multi-frame denoisers for better observed-pixel NMSE. Multiple terms that
read the same (lag,dy,dx) are not allowed. This formula is NOT valid
for general nonlinear terms or correlated sensor noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .discover import Discoverer, lstsq
from .iv import estimate
from .grammar import term_name


@dataclass
class BlindIVLaw:
    terms: list
    coef: np.ndarray
    val_risk: float
    train_risk: float
    steps: list = field(default_factory=list)
    fit_calls: int = 0
    refusals: int = 0
    status: str = "stalled"

    def describe(self):
        lines = [f"{self.coef[0]:+.6f}"]
        lines.extend(f"{v:+.6f}*{term_name(t)}" for t,v in zip(self.terms, self.coef[1:]))
        return " ".join(lines)


class BlindIVDiscoverer:
    def __init__(self, max_lag=3, margin=32, k_props=20,
                 max_terms=18, delta=0.003, abs_floor=1e-6,
                 pixel_stride=2, max_condition=1e5, seed=0, complexity_penalty=2e-6,
                 estimator="iv"):
        self.max_lag = max_lag
        self.max_terms = max_terms
        self.delta = delta
        self.abs_floor = abs_floor
        self.pixel_stride = pixel_stride
        self.max_condition = max_condition
        self.complexity_penalty = complexity_penalty
        if estimator not in ("iv", "ols"):
            raise ValueError("estimator must be iv or ols")
        self.estimator = estimator
        self.original = Discoverer(
            mode="dynamics", max_lag=max_lag, margin=margin,
            families=("id",), k_props=k_props, k_products=0,
            max_terms=max_terms, snap=False, seed=seed)
        self.fit_calls = 0
        self.refusals = 0

    def fit(self, train, terms):
        self.fit_calls += 1
        if not terms:
            return np.array([train.y.mean()])
        z = estimate(train.frames, terms, window=1, gap=0,
                     pixel_stride=self.pixel_stride,
                     max_condition=self.max_condition)
        if self.estimator == "ols":
            return z.ols_coef if np.all(np.isfinite(z.ols_coef)) else None
        if not z.identified or not np.all(np.isfinite(z.coef)):
            self.refusals += 1
            return None
        return z.coef

    @staticmethod
    def corrected_risk(stream, terms, coef, sensor_ratio=0.01,
                       sensor_variance=None):
        if coef is None:
            return float("inf")
        if any(t[0] != "shift" or t[1] != "id" for t in terms):
            raise ValueError("Gate IV-002 only validates linear shift terms")
        # No oracle coefficients/clean frames. Noise ratio was supplied by
        # benchmark experiment, and estimated from observable variance.
        sensor_var = ((sensor_ratio * float(stream.frames.std())) ** 2
                      if sensor_variance is None else float(sensor_variance))
        if not np.isfinite(sensor_var) or sensor_var < 0:
            raise ValueError("sensor_variance must be finite and nonnegative")
        pred = stream.design(terms) @ coef
        raw = float(np.mean((stream.y - pred) ** 2))
        # Unique shift terms read distinct observation noise samples, so
        # independent additive sensor contributions sum in variance.
        correction = sensor_var * (1.0 + float(np.dot(coef[1:], coef[1:])))
        return (raw - correction) / max(float(np.var(stream.y)) - sensor_var, 1e-12)

    def extra_groups(self, terms):
        """Explicit local-stencil control, separate from open FFT discovery.

        This is a prior/control, not a newly invented open-grammar proposal.
        We deliberately include whole four-neighbor stencils to let a local
        physical law compete against delayed copies. The arbitrary-offset
        FFT proposals are retained for wormholes and directed advection.
        """
        present=set(terms)
        groups=[]
        seen=set()

        def append(ts):
            group=tuple(t for t in ts if t not in present)
            if not group or len(group)+len(terms)>self.max_terms:
                return
            key=tuple(sorted(group))
            if key not in seen:
                seen.add(key)
                groups.append(group)

        for lag in range(1,self.max_lag+1):
            make=lambda dy,dx:("shift","id",lag,dy,dx)
            ns=[make(1,0),make(-1,0),make(0,1),make(0,-1)]
            ds=[make(1,1),make(1,-1),make(-1,1),make(-1,-1)]
            append((make(0,0),))
            for t in ns+ds:
                append((t,))
            append(tuple(ns))
            append((ns[0],ns[1]))
            append((ns[2],ns[3]))
            append(tuple(ds))
            append((ds[0],ds[3]))
            append((ds[1],ds[2]))
        return groups

    def run(self, train, val, *, sensor_ratio=0.01, sensor_variance=None,
            verbose=False):
        # Both IV and OLS use exactly the same validation correction.
        def risk(stream, terms, coef):
            return self.corrected_risk(stream, terms, coef, sensor_ratio,
                                       sensor_variance=sensor_variance)
        terms=[]
        coef=self.fit(train,terms)
        score=risk(val,terms,coef)
        steps=[]
        for iteration in range(self.max_terms):
            r=train.y-train.design(terms)@coef
            candidates,_=self.original.propose_residue(
                train,train.to_fields(r),set(terms),terms)
            groups=self.original.with_symmetry(candidates,terms)
            groups+=self.extra_groups(terms)
            # Deduplicate groups, including symmetric and manually anchored.
            uniq={}
            for group in groups:
                if not any(t in terms for t in group) and len(group)+len(terms)<=self.max_terms:
                    uniq[tuple(sorted(group))]=group
            groups=list(uniq.values())
            base_obj=score+self.complexity_penalty*len(terms)
            proposals=[]
            for group in groups:
                trial_terms=terms+list(group)
                c=self.fit(train,trial_terms)
                raw=risk(val,trial_terms,c)
                if np.isfinite(raw):
                    obj=raw+self.complexity_penalty*len(trial_terms)
                    proposals.append((obj,raw,group,c,None))
            # One-for-one/group exchange lets a real local law displace a
            # proxy already selected, rather than merely piling on more
            # correlated copies. This is expensive but traceable.
            if terms:
                for old in terms:
                    rest=[t for t in terms if t!=old]
                    for group in groups[:30]:
                        if any(t in rest for t in group) or len(rest)+len(group)>self.max_terms:
                            continue
                        trial_terms=rest+list(group)
                        c=self.fit(train,trial_terms)
                        raw=risk(val,trial_terms,c)
                        if np.isfinite(raw):
                            obj=raw+self.complexity_penalty*len(trial_terms)
                            proposals.append((obj,raw,group,c,old))
            proposals.sort(key=lambda row:row[0])
            need=max(self.abs_floor,abs(base_obj)*self.delta)
            chosen=next((p for p in proposals if base_obj-p[0]>need),None)
            if chosen is None:
                break
            obj,score,chosen_group,coef,old=chosen
            if old is not None:
                terms.remove(old)
            terms+=list(chosen_group)
            dropped=[]
            while len(terms)>1:
                options=[]
                for t in terms:
                    reduced=[v for v in terms if v!=t]
                    c=self.fit(train,reduced)
                    raw=risk(val,reduced,c)
                    old_obj=score+self.complexity_penalty*len(terms)
                    new_obj=raw+self.complexity_penalty*len(reduced)
                    if np.isfinite(raw) and new_obj-old_obj<=max(
                            self.abs_floor,abs(old_obj)*self.delta/2):
                        options.append((new_obj,raw,t,reduced,c))
                if not options:
                    break
                _,score,t,terms,coef=min(options,key=lambda x:x[0])
                dropped.append(term_name(t))
            steps.append({"iteration":iteration+1,
                          "added":[term_name(x) for x in chosen_group],
                          "exchange_removed":term_name(old) if old else None,
                          "dropped":dropped,"corrected_val_risk":score,
                          "fit_calls":self.fit_calls,"iv_refusals":self.refusals})
            if verbose:
                print(f"  {iteration+1:2d} risk={score:.6g} "
                      f"added={steps[-1]['added']} "
                      f"exchange={steps[-1]['exchange_removed']} "
                      f"dropped={dropped}",flush=True)
        trrisk=risk(train,terms,coef)
        return BlindIVLaw(terms,coef,score,trrisk,steps,
                          self.fit_calls,self.refusals,
                          "found" if terms else "empty")

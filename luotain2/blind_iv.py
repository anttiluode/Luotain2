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
                 pixel_stride=2, max_condition=1e5, seed=0):
        self.max_lag = max_lag
        self.max_terms = max_terms
        self.delta = delta
        self.abs_floor = abs_floor
        self.pixel_stride = pixel_stride
        self.max_condition = max_condition
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
        if not z.identified or not np.all(np.isfinite(z.coef)):
            self.refusals += 1
            return None
        return z.coef

    @staticmethod
    def corrected_risk(stream, terms, coef, sensor_ratio=0.01):
        if coef is None:
            return float("inf")
        if any(t[0] != "shift" or t[1] != "id" for t in terms):
            raise ValueError("Gate IV-002 only validates linear shift terms")
        # No oracle coefficients/clean frames. Noise ratio was supplied by
        # benchmark experiment, and estimated from observable variance.
        sensor_var = (sensor_ratio * float(stream.frames.std())) ** 2
        pred = stream.design(terms) @ coef
        raw = float(np.mean((stream.y - pred) ** 2))
        # Unique shift terms read distinct observation noise samples, so
        # independent additive sensor contributions sum in variance.
        correction = sensor_var * (1.0 + float(np.dot(coef[1:], coef[1:])))
        return (raw - correction) / max(float(np.var(stream.y)) - sensor_var, 1e-12)

    def run(self, train, val, *, sensor_ratio=0.01, verbose=False):
        terms=[]
        coef=self.fit(train,terms)
        score=self.corrected_risk(val,terms,coef,sensor_ratio)
        first=score
        steps=[]
        for iteration in range(self.max_terms):
            r=train.y-train.design(terms)@coef
            candidates, maps=self.original.propose_residue(
                train,train.to_fields(r),set(terms),terms)
            groups=self.original.with_symmetry(candidates,terms)
            proposals=[]
            for group in groups:
                if any(t in terms for t in group):
                    continue
                trial_terms=terms+list(group)
                if len(trial_terms)>self.max_terms:
                    continue
                c=self.fit(train,trial_terms)
                s=self.corrected_risk(val,trial_terms,c,sensor_ratio)
                if np.isfinite(s):
                    proposals.append((s,group,c))
            proposals.sort(key=lambda row:row[0])
            need=max(self.abs_floor,abs(score)*self.delta)
            chosen=next(((s,g,c) for s,g,c in proposals if score-s>need),None)
            if chosen is None:
                break
            next_score,chosen_group,chosen_coef=chosen
            terms+=list(chosen_group)
            coef=chosen_coef
            score=next_score
            dropped=[]
            while len(terms)>1:
                options=[]
                for t in terms:
                    reduced=[v for v in terms if v!=t]
                    c=self.fit(train,reduced)
                    s=self.corrected_risk(val,reduced,c,sensor_ratio)
                    if np.isfinite(s) and s-score<=max(self.abs_floor,abs(score)*self.delta/2):
                        options.append((s,t,reduced,c))
                if not options:
                    break
                s,t,terms,coef=min(options,key=lambda x:x[0])
                score=s
                dropped.append(term_name(t))
            step={"iteration":iteration+1,"added":[term_name(x) for x in chosen_group],
                  "dropped":dropped,"corrected_val_risk":score,
                  "raw_validation_mse":float(np.mean(
                      (val.y-val.design(terms)@coef)**2)),
                  "fit_calls":self.fit_calls,"iv_refusals":self.refusals}
            steps.append(step)
            if verbose:
                print(f"  {iteration+1:2d} risk={score:.6g} +"
                      f"{' & '.join(step['added'])} -{dropped}",flush=True)
        trrisk=self.corrected_risk(train,terms,coef,sensor_ratio)
        return BlindIVLaw(terms,coef,score,trrisk,steps,
                          self.fit_calls,self.refusals,
                          "found" if terms else "empty")

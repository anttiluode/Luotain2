"""Gate IV-004: blind discovery of POLYNOMIAL field laws under sensor noise.

Blind IV-002/003 search linear shifted copies only. Their validation score
subtracts R (1 + sum beta^2), which is exact for distinct linear inputs and
wrong for u^2, u^3 or products. This module extends the same search to the
full grammar (id / sq / cube shifts and nearby products) with two pieces:

1. Debiased regressors and instruments for the IV fit (already in iv.py):
   Y^2 - R, Y^3 - 3RY, same-pixel products minus R. (The WAIVES corrected
   score idea, applied to a discrete field law; this is not WAIVES itself.)

2. An exact corrected validation risk for any polynomial law. Under
   Y = X + eta with eta i.i.d. N(0, R) over pixels and times, the latent
   squared error

       L(X) = (X_t - c0 - sum_j beta_j g_j(X_{earlier}))^2

   is a polynomial in the latent pixel values it touches. Every monomial
   prod_v X_v^{k_v} (distinct pixel-times v) has the unbiased estimator
   prod_v He_{k_v}(Y_v; R), with the scaled Hermite polynomials
   He_0 = 1, He_1 = y, He_{k+1} = y He_k - k R He_{k-1}.
   Summing gives an unbiased estimate of the latent risk from observed data
   only. No clean frames, no true support, no true coefficients.

Assumptions (stated, not tested here): Gaussian, i.i.d., known R; noise
independent of the latent field; correct-grammar polynomial terms. Wrong R,
correlated sensor noise or heavy tails make the correction biased.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .blind_iv import BlindIVDiscoverer
from .discover import Discoverer
from .grammar import shifted
from .iv import estimate


# ----------------------------------------------------------- polynomial algebra
def _term_poly(term):
    """A grammar term as {monomial: coef}; monomial = tuple of ((lag,dy,dx), power)."""
    if term[0] == "shift":
        _, f, lag, dy, dx = term
        k = {"id": 1, "sq": 2, "cube": 3}[f]
        return {(((lag, dy, dx), k),): 1.0}
    if term[0] == "prod":
        _, lag, dy1, dx1, dy2, dx2 = term
        a, b = (lag, dy1, dx1), (lag, dy2, dx2)
        if a == b:
            return {((a, 2),): 1.0}
        return {tuple(sorted(((a, 1), (b, 1)))): 1.0}
    raise ValueError(term)


def _mono_mul(m1, m2):
    d = defaultdict(int)
    for v, k in m1 + m2:
        d[v] += k
    return tuple(sorted(d.items()))


def _poly_mul(p, q):
    out = defaultdict(float)
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            out[_mono_mul(m1, m2)] += c1 * c2
    return dict(out)


def residual_poly(terms, coef):
    """q = X_target - c0 - sum beta_j g_j. Target variable is (0,0,0)."""
    q = defaultdict(float)
    q[(((0, 0, 0), 1),)] += 1.0
    q[()] -= float(coef[0])
    for t, b in zip(terms, coef[1:]):
        for m, c in _term_poly(t).items():
            q[m] -= float(b) * c
    return dict(q)


def hermite(y, k, R):
    """Unbiased estimator of X^k from Y = X + N(0,R): scaled probabilists' Hermite."""
    if k == 0:
        return np.ones_like(y)
    h0, h1 = np.ones_like(y), y
    for n in range(1, k):
        h0, h1 = h1, y * h1 - n * R * h0
    return h1


class _Vars:
    """Observed pixel-time variables of a Stream, cached per (lag,dy,dx)."""

    def __init__(self, stream):
        self.s = stream
        self.cache = {}

    def get(self, v):
        if v not in self.cache:
            lag, dy, dx = v
            if lag == 0:
                self.cache[v] = self.s.y
            else:
                self.cache[v] = shifted(self.s.base("id", lag), dy, dx)[:, self.s.mask].ravel()
        return self.cache[v]


def corrected_poly_risk_raw(stream, terms, coef, R, vars_=None):
    """Unbiased estimate of the latent mean squared one-step error."""
    vars_ = vars_ or _Vars(stream)
    q = residual_poly(terms, coef)
    q2 = _poly_mul(q, q)
    total = 0.0
    herm = {}
    for mono, c in q2.items():
        if abs(c) == 0.0:
            continue
        val = None
        for v, k in mono:
            key = (v, k)
            if key not in herm:
                herm[key] = hermite(vars_.get(v), k, R)
            val = herm[key] if val is None else val * herm[key]
        total += c * (1.0 if val is None else float(np.mean(val)))
    return total


# ----------------------------------------------------------- the discoverer
class BlindPolyDiscoverer(BlindIVDiscoverer):
    """BlindIVDiscoverer with the full grammar and a choice of noise handling.

    correction="exact": debiased IV/OLS features + exact Hermite validation risk.
    correction="linear": the Gate IV-002 recipe applied unchanged to every term
                         (no feature debiasing, R(1+sum beta^2) subtracted):
                         the naive port, kept as the ablation.
    """

    def __init__(self, *args, correction="exact", families=("id", "sq", "cube"),
                 k_products=4, product_radius=1, **kw):
        super().__init__(*args, **kw)
        if correction not in ("exact", "linear"):
            raise ValueError("correction must be exact or linear")
        self.correction = correction
        self.R = None
        o = self.original
        self.original = Discoverer(
            mode="dynamics", max_lag=self.max_lag, margin=o.margin,
            families=families, k_props=o.k_props, k_products=k_products,
            product_radius=product_radius, max_terms=self.max_terms, snap=False,
            seed=0)
        self._vars = {}

    def fit(self, train, terms):
        self.fit_calls += 1
        if not terms:
            return np.array([train.y.mean()])
        sv = self.R if self.correction == "exact" else 0.0
        z = estimate(train.frames, terms, sensor_var=sv, window=1, gap=0,
                     pixel_stride=self.pixel_stride,
                     max_condition=self.max_condition)
        if self.estimator == "ols":
            return z.ols_coef if np.all(np.isfinite(z.ols_coef)) else None
        if not z.identified or not np.all(np.isfinite(z.coef)):
            self.refusals += 1
            return None
        return z.coef

    def corrected_risk(self, stream, terms, coef, sensor_ratio=0.01, sensor_variance=None):
        if coef is None:
            return float("inf")
        R = self.R if sensor_variance is None else float(sensor_variance)
        denom = max(float(np.var(stream.y)) - R, 1e-12)
        if self.correction == "linear":
            pred = stream.design(terms) @ coef
            raw = float(np.mean((stream.y - pred) ** 2))
            return (raw - R * (1.0 + float(np.dot(coef[1:], coef[1:])))) / denom
        key = id(stream)
        if key not in self._vars:
            self._vars[key] = _Vars(stream)
        return corrected_poly_risk_raw(stream, terms, coef, R, self._vars[key]) / denom

    def run(self, train, val, *, sensor_variance, verbose=False):
        if sensor_variance is None or not np.isfinite(sensor_variance) or sensor_variance < 0:
            raise ValueError("Gate IV-004 needs a known, finite sensor variance")
        self.R = float(sensor_variance)
        return super().run(train, val, sensor_variance=self.R, verbose=verbose)

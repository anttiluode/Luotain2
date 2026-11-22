"""Spatially correlated sensor noise: a noise KERNEL instead of a noise variance.

Real cameras blur their own noise (demosaicing, compression, optics), so the
sensor error of neighbouring pixels in the same frame is correlated:

    cov( eta_s(x), eta_s(x + D) ) = C(D),      white in time, C(D) = C(-D).

The i.i.d. case is C(0) = R, C(D != 0) = 0.

1. Wick-corrected scores. For Gaussian noise with covariance C, the unbiased
   estimator of a latent monomial prod X_v is its Wick product :prod Y_v:

       sum over partial pairings M of the factors
           (-1)^|M|  prod_{(a,b) in M} C(a,b)  prod_{unpaired c} Y_c

   (Hermite polynomials are the single-pixel case.) Factors at different
   times never pair. This corrects both the IV/OLS features (degree <= 3) and
   the validation risk (the squared residual polynomial, degree <= 6).

2. Kernel echo. Stein's lemma in several variables, E[eta_a f] = sum_b C_ab E[df/dY_b],
   turns the IV-005 echo into a deconvolution. For the TRUE law,

       cov( r_s(x + o), r_{s+L}(x) ) = - sum_o' d_L(o') C(o - o'),

   where d_L(o') is the mean gradient of the (Wick-debiased) prediction with
   respect to Y_s(x + o'). Measured for many o, this is a linear system for
   the kernel values on a small symmetric support; it is solved by least
   squares and iterated (the debiased gradients depend on C).
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from .blind_poly import residual_poly, _poly_mul, _term_poly, _Vars
from .grammar import shifted


# ------------------------------------------------------------------ kernels
def kernel_support(radius):
    """Half-plane representatives of offsets with |dy|,|dx| <= radius (D and -D share a value)."""
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dy, dx) == (0, 0) or (dy, dx) > (0, 0):
                out.append((dy, dx))
    return out


def kernel_value(kernel, a, b):
    """Noise covariance between observed pixel-times a=(lag,dy,dx) and b."""
    if a[0] != b[0]:
        return 0.0
    d = (b[1] - a[1], b[2] - a[2])
    if d < (0, 0):
        d = (-d[0], -d[1])
    return kernel.get(d, 0.0)


def iid_kernel(R):
    return {(0, 0): float(R)}


# ------------------------------------------------------------------ Wick products
@lru_cache(maxsize=None)
def _pairings(n):
    """All partial pairings of n labelled factors: tuple of (pairs, unpaired)."""
    if n == 0:
        return (((), ()),)
    out = []
    for pairs, single in _pairings(n - 1):
        new = n - 1
        out.append((pairs, single + (new,)))           # the new factor stays unpaired
        for i, s in enumerate(single):                   # or pairs with an unpaired one
            out.append((pairs + ((s, new),), single[:i] + single[i + 1:]))
    return tuple(out)


def wick(factors, get, kernel):
    """Unbiased estimate (array) of prod_v X_v for a list of pixel-time factors."""
    n = len(factors)
    if n == 0:
        return 1.0
    total = 0.0
    for pairs, single in _pairings(n):
        coef = 1.0
        for i, j in pairs:
            c = kernel_value(kernel, factors[i], factors[j])
            if c == 0.0:
                coef = 0.0
                break
            coef *= -c
        if coef == 0.0:
            continue
        val = coef
        for k in single:
            val = val * get(factors[k])
        total = total + val
    return total


def _factors(mono):
    out = []
    for v, k in mono:
        out.extend([v] * k)
    return out


def corrected_risk_raw_kernel(stream, terms, coef, kernel, vars_=None):
    """Unbiased latent mean squared one-step error under kernel-correlated noise."""
    vars_ = vars_ or _Vars(stream)
    q = residual_poly(terms, coef)
    q2 = _poly_mul(q, q)
    total = 0.0
    for mono, c in q2.items():
        if c == 0.0:
            continue
        w = wick(_factors(mono), vars_.get, kernel)
        total += c * float(np.mean(w))
    return total


def feature_value(frames, t, term, kernel):
    """Wick-debiased grammar term evaluated on observed frames (for IV/OLS fits)."""
    (mono, c), = _term_poly(term).items()
    get = lambda v: shifted(frames[t - v[0]], v[1], v[2])
    return c * wick(_factors(mono), get, kernel)


# ------------------------------------------------------------------ kernel echo
def _lag(term):
    return int(term[2] if term[0] == "shift" else term[1])


def _pred_and_grads(frames, t, terms, coef, kernel):
    get = lambda v: shifted(frames[t - v[0]], v[1], v[2])
    p = np.full(frames.shape[1:], float(coef[0]))
    grads = {}
    for term, b in zip(terms, coef[1:]):
        for mono, c in _term_poly(term).items():
            fac = _factors(mono)
            p = p + b * c * wick(fac, get, kernel)
            for v in set(fac):
                k = fac.count(v)
                rest = list(fac)
                rest.remove(v)
                g = b * c * k * wick(rest, get, kernel)
                grads[v] = grads.get(v, 0.0) + g
    return p, grads


def echo_kernel(frames, terms, coef, radius=1, kernel0=None, iters=3, window=None):
    """Estimate the noise kernel (dict offset -> covariance) from a law's echo."""
    frames = np.asarray(frames, dtype=np.float64)
    if not terms:
        return None
    L = max(_lag(t) for t in terms)
    T = len(frames)
    kern = dict(kernel0 or {})
    sup = kernel_support(radius)
    for _ in range(iters):
        res, gs = {}, {}
        for t in range(L, T):
            p, g = _pred_and_grads(frames, t, terms, coef, kern)
            r = frames[t] - p
            res[t] = r - r.mean()
            for v, gf in g.items():
                gs.setdefault(v, []).append(float(np.mean(gf)))
        d = {v: float(np.mean(x)) for v, x in gs.items()}
        lags = sorted({v[0] for v in d})
        reach = max(max(abs(v[1]), abs(v[2])) for v in d) + radius
        W = window if window is not None else min(reach, 6)
        rows, rhs = [], []
        for lag in lags:
            for oy in range(-W, W + 1):
                for ox in range(-W, W + 1):
                    cs = [float(np.mean(shifted(res[s], oy, ox) * res[s + lag]))
                          for s in range(L, T - lag)]
                    if not cs:
                        continue
                    row = np.zeros(len(sup))
                    for (vl, vy, vx), dv in d.items():
                        if vl != lag:
                            continue
                        D = (oy - vy, ox - vx)
                        if D < (0, 0):
                            D = (-D[0], -D[1])
                        if D in sup:
                            row[sup.index(D)] += -dv
                    if np.any(row):
                        rows.append(row)
                        rhs.append(float(np.mean(cs)))
        if not rows:
            return None
        sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        kern = {D: float(v) for D, v in zip(sup, sol)}
        kern[(0, 0)] = max(kern[(0, 0)], 0.0)
    return kern


# ------------------------------------------------------------------ discoverer
from .blind_poly import BlindPolyDiscoverer  # noqa: E402
from .blind_iv import BlindIVDiscoverer  # noqa: E402


class BlindKernelDiscoverer(BlindPolyDiscoverer):
    """BlindPolyDiscoverer whose fits and validation risk use a noise kernel."""

    def __init__(self, *args, **kw):
        super().__init__(*args, correction="exact", **kw)
        self.kernel = None

    def fit(self, train, terms):
        self.fit_calls += 1
        if not terms:
            return np.array([train.y.mean()])
        from .iv import estimate
        z = estimate(train.frames, terms, sensor_var=self.kernel, window=1, gap=0,
                     pixel_stride=self.pixel_stride, max_condition=self.max_condition)
        if self.estimator == "ols":
            return z.ols_coef if np.all(np.isfinite(z.ols_coef)) else None
        if not z.identified or not np.all(np.isfinite(z.coef)):
            self.refusals += 1
            return None
        return z.coef

    def corrected_risk(self, stream, terms, coef, sensor_ratio=0.01, sensor_variance=None):
        if coef is None:
            return float("inf")
        key = id(stream)
        if key not in self._vars:
            self._vars[key] = _Vars(stream)
        denom = max(float(np.var(stream.y)) - self.kernel.get((0, 0), 0.0), 1e-12)
        return corrected_risk_raw_kernel(stream, terms, coef, self.kernel, self._vars[key]) / denom

    def run(self, train, val, *, kernel, verbose=False):
        self.kernel = dict(kernel)
        self.R = self.kernel.get((0, 0), 0.0)
        return BlindIVDiscoverer.run(self, train, val, sensor_variance=self.R, verbose=verbose)

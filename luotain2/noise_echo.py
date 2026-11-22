"""Blind sensor-variance estimate from the noise *echo* of a candidate law.

Observed Y_s = X_s + eta_s, eta i.i.d. Gaussian with unknown variance R.
Residual of a law p (Hermite-debiased, see blind_poly.py):

    r_t(x) = Y_t(x) - p~(Y_{t-1}, Y_{t-2}, ...)(x)

The sensor noise of frame s shows up twice: with weight +1 in r_s(x'), and
inside every later residual r_{s+L}(x) whose law reads pixel x' at lag L. By
Stein's lemma, E[eta f(X + eta)] = R E[f'(X + eta)], so for the TRUE law

    cov( r_s(x + o), r_{s+L}(x) ) = - R * E[ d p~ / d Y_{s}(x + o) ]      (lag L)

and nothing else contributes: process innovations and the law's own
prediction error are uncorrelated with the noise. R is then the least-squares
slope of the measured echo covariances against the measured mean gradients,
over every (lag, offset) the law reads. Gradients of Hermite-debiased terms
are themselves debiased, so the estimate needs R; it is iterated to a fixed
point (two or three passes are enough).

For a wrong law, the law's own prediction error is usually correlated over
time and biases R. In blind search the estimate is therefore alternated with
selection: estimate R from the law found so far, select again, re-estimate.
Assumptions: Gaussian, i.i.d. over pixels and frames, independent of X.
"""
from __future__ import annotations

import numpy as np

from .blind_poly import hermite, _term_poly
from .grammar import shifted


def _lag(term):
    return int(term[2] if term[0] == "shift" else term[1])


def _pred_and_grads(frames, t, terms, coef, R):
    p = np.full(frames.shape[1:], float(coef[0]))
    grads = {}
    for term, b in zip(terms, coef[1:]):
        for mono, c in _term_poly(term).items():
            vals = {v: shifted(frames[t - v[0]], v[1], v[2]) for v, _ in mono}
            hv = {v: hermite(vals[v], k, R) for v, k in mono}
            val = np.ones_like(p)
            for v, _ in mono:
                val = val * hv[v]
            p = p + b * c * val
            for v, k in mono:
                g = np.full_like(p, b * c * k)
                for w, kw in mono:
                    g = g * (hermite(vals[w], kw - 1, R) if w == v else hv[w])
                grads[v] = grads.get(v, 0.0) + g
    return p, grads


def echo_variance(frames, terms, coef, R0=0.0, iters=3, return_detail=False):
    """Estimate sensor variance R from the noise echo of the given law.

    Returns None if the law reads no earlier pixel (no echo to measure)."""
    frames = np.asarray(frames, dtype=np.float64)
    if not terms:
        return None
    L = max(_lag(t) for t in terms)
    T = len(frames)
    R = float(R0)
    detail = None
    for _ in range(iters):
        res, gsum = {}, {}
        for t in range(L, T):
            p, g = _pred_and_grads(frames, t, terms, coef, R)
            r = frames[t] - p
            res[t] = r - r.mean()
            for v, gf in g.items():
                gsum.setdefault(v, []).append(float(gf.mean()))
        num = den = 0.0
        detail = []
        for (lag, dy, dx), ds in gsum.items():
            d = float(np.mean(ds))
            cs = [float(np.mean(shifted(res[s], dy, dx) * res[s + lag]))
                  for s in range(L, T - lag)]
            if not cs:
                continue
            c = float(np.mean(cs))
            num += -c * d
            den += d * d
            detail.append({"lag": lag, "offset": [dy, dx], "mean_grad": d, "echo_cov": c})
        if den <= 0:
            return None
        R = max(num / den, 0.0)
    return (R, detail) if return_detail else R

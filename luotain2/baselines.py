"""Controls the method has to beat.

1. Random proposals: the same loop, probes, validation and symmetry moves, but proposals are
   drawn at random from the open grammar instead of being read off the residue.
2. Fixed library + sequential thresholded least squares (the PDE-FIND / SINDy recipe):
   every term within a radius-2 stencil, three lags, local nonlinearities and products, one big
   regression, small coefficients thresholded away. The threshold is chosen on validation, and
   among thresholds within (1 + delta) of the best, the sparsest law wins.
"""
from __future__ import annotations

import numpy as np

from .discover import Law, lstsq, nmse
from .grammar import canonical


def default_library(max_lag=3, radius=2, nl_radius=1, mode="dynamics"):
    lib = []
    lags = range(1, max_lag + 1) if mode == "dynamics" else [0]
    for lag in lags:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if mode != "dynamics" and (dy, dx) == (0, 0):
                    continue
                if mode == "static-causal" and not (dy < 0 or (dy == 0 and dx < 0)):
                    continue
                lib.append(("shift", "id", lag, dy, dx))
    if mode == "dynamics":
        for f in ("sq", "cube"):
            for dy in range(-nl_radius, nl_radius + 1):
                for dx in range(-nl_radius, nl_radius + 1):
                    lib.append(("shift", f, 1, dy, dx))
        for dy in range(-nl_radius, nl_radius + 1):
            for dx in range(-nl_radius, nl_radius + 1):
                if (dy, dx) != (0, 0):
                    lib.append(canonical(("prod", 1, 0, 0, dy, dx)))
    return lib


def stlsq(train, val, library, delta=0.0025, grid=None, iters=10):
    X = train.design(library)
    Xv = val.design(library)
    sd = X[:, 1:].std(axis=0)
    sd[sd == 0] = 1
    scale = np.concatenate([[1.0], sd])
    ys = train.y.std() or 1
    Xn = X / scale
    G = Xn.T @ Xn                      # normal equations once; every refit is a small solve
    b = Xn.T @ (train.y / ys)
    ridge = 1e-10 * np.trace(G) / len(G)

    def solve(keep):
        idx = np.flatnonzero(keep)
        xi = np.zeros(len(G))
        xi[idx] = np.linalg.solve(G[np.ix_(idx, idx)] + ridge * np.eye(len(idx)), b[idx])
        return xi

    grid = grid if grid is not None else np.concatenate([[0], np.logspace(-4, 0, 33)])
    results = []
    for lam in grid:
        keep = np.ones(len(G), bool)
        xi = solve(keep)
        for _ in range(iters):
            new = np.abs(xi) >= lam
            new[0] = True
            if np.array_equal(new, keep) and _ > 0:
                break
            keep = new
            xi = solve(keep)
        coef = xi * ys / scale
        e = nmse(val.y, Xv @ coef)
        terms = [library[i - 1] for i in range(1, len(G)) if keep[i]]
        results.append((e, len(terms), lam, terms, coef))
    best = min(r[0] for r in results)
    ok = [r for r in results if r[0] <= best * (1 + delta)]
    e, k, lam, terms, coef = min(ok, key=lambda r: (r[1], r[0]))
    c = lstsq(train.design(terms), train.y)
    return Law(terms=terms, coef=c, val_nmse=nmse(val.y, val.design(terms) @ c),
               probes=len(grid), full_fits=len(grid))

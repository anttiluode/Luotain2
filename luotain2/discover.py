"""Residue-guided law discovery.

The loop, in words:

1. Fit the current law on the training split. What it fails to predict is the residue.
2. Ask the residue which term it wants. One FFT cross-correlation per family of base fields
   (u, u^2, u^3 at each lag) scores *every* shift of that family at once; products of nearby
   values are scored directly. The highest peaks are the proposals.
3. Probe: refit on a small random subsample with each proposal added. Proposals that do not
   help on a validation subsample are killed without a full fit.
4. The survivors get a full fit. The best one joins the law only if it lowers the error on the
   validation split (data the coefficients never saw) by more than `delta`.
5. Prune: any term whose removal costs less than delta/2 on validation is dropped.
6. Repeat until nothing passes. Finally, snap coefficients to simple fractions when that costs
   (almost) nothing on validation, so the law can be read.

Step 2 is orthogonal matching pursuit with an implicit dictionary: the FFT lets the residue
point at any offset, so the grammar does not have to be enumerated in advance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .grammar import canonical, causal_ok, shifted, term_name


@dataclass
class Law:
    terms: list
    coef: np.ndarray          # coef[0] is the intercept
    val_nmse: float
    history: list = field(default_factory=list)
    probes: int = 0
    killed: int = 0
    full_fits: int = 0
    snapped: dict = field(default_factory=dict)

    def describe(self, digits=4):
        parts = []
        c0 = self.coef[0]
        if abs(c0) > 10 ** -digits:
            parts.append(f"{c0:+.{digits}f}")
        for t, c in sorted(zip(self.terms, self.coef[1:]), key=lambda z: -abs(z[1])):
            s = self.snapped.get(t)
            cs = f"{'+' if s >= 0 else '-'}{abs(s)}" if s is not None else f"{c:+.{digits}f}"
            parts.append(f"{cs}*{term_name(t)}")
        return " ".join(parts) if parts else "(nothing)"


def nmse(y, yhat):
    v = np.var(y)
    return float(np.mean((y - yhat) ** 2) / (v if v > 0 else 1.0))


def lstsq(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


class Discoverer:
    def __init__(self, mode="dynamics", max_lag=3, margin=32, families=("id", "sq", "cube"),
                 k_props=12, k_products=4, product_radius=2, delta=0.0025, max_terms=24,
                 probe_rows=20000, snap=True, proposer="residue", seed=0):
        self.mode = mode
        self.lags = list(range(1, max_lag + 1)) if mode == "dynamics" else [0]
        self.margin = margin
        self.families = families if mode == "dynamics" else ("id",)
        self.k_props = k_props
        self.k_products = k_products if mode == "dynamics" else 0
        self.product_radius = product_radius
        self.delta = delta
        self.max_terms = max_terms
        self.probe_rows = probe_rows
        self.snap = snap
        self.proposer = proposer          # "residue" (the method) or "random" (control)
        self.rng = np.random.default_rng(seed)

    # ---------- legality ----------
    def legal(self, t):
        if t[0] == "shift":
            _, f, lag, dy, dx = t
            if max(abs(dy), abs(dx)) > self.margin:
                return False
            if self.mode != "dynamics":
                if (dy, dx) == (0, 0):
                    return False
                if self.mode == "static-causal" and not causal_ok(dy, dx):
                    return False
            return True
        if t[0] == "prod":
            return self.mode == "dynamics"
        return False

    # ---------- proposals ----------
    def offset_mask(self, H, W):
        dy = np.arange(H)
        dy = np.where(dy > H // 2, dy - H, dy)
        dx = np.arange(W)
        dx = np.where(dx > W // 2, dx - W, dx)
        DY, DX = np.meshgrid(dy, dx, indexing="ij")
        ok = (np.abs(DY) <= self.margin) & (np.abs(DX) <= self.margin)
        if self.mode != "dynamics":
            ok &= ~((DY == 0) & (DX == 0))
            if self.mode == "static-causal":
                ok &= (DY < 0) | ((DY == 0) & (DX < 0))
        return ok, DY, DX

    def propose_residue(self, train, r_fields, present, terms):
        """Score every offset of every base family by how much it would lower the training error
        *after* the terms already in the law have taken their share (order-recursive matching
        pursuit). All inner products against shifted copies are FFT cross-correlations."""
        H, W = train.H, train.W
        ok, DY, DX = self.offset_mask(H, W)
        corr = lambda Af, B: np.real(np.fft.ifft2(Af * np.fft.fft2(B))).sum(axis=0)
        Rf = np.conj(np.fft.fft2(r_fields))
        m = np.broadcast_to(train.mask, r_fields.shape).astype(float)
        Mf = np.conj(np.fft.fft2(m))
        X = [m] + [train.field(t) * m for t in terms]          # current design as fields
        Xf = [np.conj(np.fft.fft2(x)) for x in X]
        Xd = train.design(terms)
        Ginv = np.linalg.pinv(Xd.T @ Xd)
        scored, maps = [], {}
        for f in self.families:
            for lag in self.lags:
                B = train.base(f, lag)
                C = corr(Rf, B)
                E = corr(Mf, B * B)
                g = np.stack([corr(xf, B) for xf in Xf])          # (k+1, H, W)
                proj = np.einsum("ihw,ij,jhw->hw", g, Ginv, g)
                den = np.maximum(E - proj, 1e-9 * np.maximum(E, 1e-30))
                S = np.where(ok, C * C / den, -np.inf)
                for t in present:
                    if t[0] == "shift" and t[1] == f and t[2] == lag:
                        S[t[3] % H, t[4] % W] = -np.inf
                maps[(f, lag)] = S
                flat = np.argpartition(S.ravel(), -self.k_props)[-self.k_props:]
                for i in flat:
                    if not np.isfinite(S.flat[i]):
                        continue
                    t = ("shift", f, lag, int(DY.flat[i]), int(DX.flat[i]))
                    if t not in present:
                        scored.append((float(S.flat[i]), t))
        scored.sort(key=lambda z: -z[0])
        cands = [t for _, t in scored[: self.k_props]]
        if self.k_products:
            rp, rv = [], r_fields[:, train.mask].ravel()
            R = self.product_radius
            for lag in self.lags:
                for dy in range(-R, R + 1):
                    for dx in range(-R, R + 1):
                        t = canonical(("prod", lag, 0, 0, dy, dx))
                        if (dy, dx) == (0, 0) or t in present:
                            continue
                        p = train.feature(t)
                        v = Xd.T @ p
                        e = max(float(p @ p - v @ Ginv @ v), 1e-12)
                        rp.append(((rv @ p) ** 2 / e, t))
            rp.sort(key=lambda z: -z[0])
            cands += [t for _, t in rp[: self.k_products]]
        return cands, maps

    def propose_random(self, present):
        cands = []
        while len(cands) < self.k_props + self.k_products:
            if self.k_products and self.rng.random() < self.k_products / (self.k_props + self.k_products):
                R = self.product_radius
                t = canonical(("prod", int(self.rng.choice(self.lags)), 0, 0,
                               int(self.rng.integers(-R, R + 1)), int(self.rng.integers(-R, R + 1))))
                if t[4:] == (0, 0):
                    continue
            else:
                t = ("shift", str(self.rng.choice(self.families)), int(self.rng.choice(self.lags)),
                     int(self.rng.integers(-self.margin, self.margin + 1)),
                     int(self.rng.integers(-self.margin, self.margin + 1)))
            if self.legal(t) and t not in present and t not in cands:
                cands.append(t)
        return cands

    def exchange(self, terms, groups, train, val, cur, tr_rows, va_rows, law, max_groups=24):
        """Growth has stalled: try replacing one term of the law by a proposal (stepwise exchange)."""
        if not terms:
            return None
        trial = []
        for old in terms:
            rest = [x for x in terms if x != old]
            Xt0, Xv0 = train.design(rest, tr_rows), val.design(rest, va_rows)
            for g in groups[:max_groups]:
                if old in g:
                    continue
                law.probes += 1
                Xt = np.column_stack([Xt0] + [train.feature(t)[tr_rows] for t in g])
                c = lstsq(Xt, train.y[tr_rows])
                Xv = np.column_stack([Xv0] + [val.feature(t)[va_rows] for t in g])
                trial.append((nmse(val.y[va_rows], Xv @ c), old, g))
        trial.sort(key=lambda z: z[0])
        best = None
        for _, old, g in trial[:6]:
            law.full_fits += 1
            new = [x for x in terms if x != old] + list(g)
            c = self.fit(new, train)
            e = self.score(new, c, val)
            need = cur * (1 - self.delta * max(len(g) - 1, 0.5))
            if e <= need and (best is None or e < best[0]):
                best = (e, g, c, old)
        return best

    def mirror(self, t):
        if t[0] == "shift":
            return ("shift", t[1], t[2], -t[3], -t[4])
        if t[0] == "prod":
            return canonical(("prod", t[1], -t[2], -t[3], -t[4], -t[5]))
        return t

    def with_symmetry(self, cands, terms):
        """Symmetry moves: laws of nature are often mirror-symmetric, and a mirror pair can be
        worth adding even when each half alone is too weak to pass. Candidates become groups."""
        present = set(terms)
        groups, seen = [], set()

        def add(g):
            key = tuple(sorted(g))
            if key not in seen and all(self.legal(x) and x not in present for x in g):
                seen.add(key)
                groups.append(tuple(g))
        for t in cands:
            add((t,))
            m = self.mirror(t)
            if m != t:
                add((t, m))
        missing = []
        for t in terms:
            m = self.mirror(t)
            if m != t and m not in present and m not in missing and self.legal(m):
                add((m,))
                missing.append(m)
        if len(missing) > 1:
            add(tuple(missing))      # complete every mirror at once
        return groups

    # ---------- fitting ----------
    def fit(self, terms, train):
        return lstsq(train.design(terms), train.y)

    def score(self, terms, coef, stream):
        return nmse(stream.y, stream.design(terms) @ coef)

    # ---------- main loop ----------
    def run(self, train, val, verbose=False, record_every=True):
        terms, coef = [], self.fit([], train)
        cur = self.score(terms, coef, val)
        law = Law(terms=[], coef=coef, val_nmse=cur)
        tr_rows = self.rng.choice(train.n, size=min(self.probe_rows, train.n), replace=False)
        va_rows = self.rng.choice(val.n, size=min(self.probe_rows, val.n), replace=False)
        while len(terms) < self.max_terms:
            resid = train.y - train.design(terms) @ coef
            r_fields = train.to_fields(resid)
            if self.proposer == "residue":
                cands, maps = self.propose_residue(train, r_fields, set(terms), terms)
            else:
                cands, maps = self.propose_random(set(terms)), {}
            groups = self.with_symmetry(cands, terms)
            # probe stage on subsamples
            Xv0 = val.design(terms, va_rows)
            Xt0 = train.design(terms, tr_rows)
            c_sub = lstsq(Xt0, train.y[tr_rows])
            base_sub = nmse(val.y[va_rows], Xv0 @ c_sub)
            survivors = []
            for g in groups:
                law.probes += 1
                Xt = np.column_stack([Xt0] + [train.feature(t)[tr_rows] for t in g])
                c = lstsq(Xt, train.y[tr_rows])
                Xv = np.column_stack([Xv0] + [val.feature(t)[va_rows] for t in g])
                e = nmse(val.y[va_rows], Xv @ c)
                if e < base_sub:
                    survivors.append((e, g))
                else:
                    law.killed += 1
            survivors.sort(key=lambda z: z[0])
            best = None
            for _, g in survivors[:4]:
                law.full_fits += 1
                c = self.fit(terms + list(g), train)
                e = self.score(terms + list(g), c, val)
                # a pair must earn more than a single term would
                need = cur * (1 - self.delta * len(g))
                if e <= need and (best is None or e < best[0]):
                    best = (e, g, c)
            if best is None:
                swap = self.exchange(terms, groups, train, val, cur, tr_rows, va_rows, law)
                if swap is None:
                    break
                e, g, c, old = swap
                terms = [x for x in terms if x != old]
                law.history.append({"added": list(g), "name": " & ".join(term_name(x) for x in g),
                                    "removed": term_name(old), "move": "exchange", "val_before": cur,
                                    "val_after": e, "rank": -1, "proposals": [], "pruned": []})
                if verbose:
                    print(f"  ~ swap {term_name(old)} -> {' & '.join(term_name(x) for x in g)}  val NMSE {e:.3e}")
                terms, coef, cur = terms + list(g), c, e
                terms, coef, cur, _ = self.prune(terms, train, val, cur)
                continue
            e, g, c = best
            t = g[0]
            step = {"added": list(g), "name": " & ".join(term_name(x) for x in g), "val_before": cur,
                    "val_after": e, "proposals": [term_name(x) for x in cands],
                    "rank": cands.index(t) if t in cands else -1,
                    "move": "residue peak" if (len(g) == 1 and t in cands) else ("mirror pair" if (len(g) == 2 and t in cands) else "mirror completion")}
            if record_every and t[0] == "shift" and maps and (t[1], t[2]) in maps:
                step["resid_frame"] = r_fields[len(r_fields) // 2].astype(np.float32)
                step["corr_map"] = np.where(np.isfinite(maps[(t[1], t[2])]), maps[(t[1], t[2])], 0).astype(np.float32)
                step["peak"] = (t[3], t[4])
            terms, coef, cur = terms + list(g), c, e
            terms, coef, cur, dropped = self.prune(terms, train, val, cur)
            step["pruned"] = [term_name(x) for x in dropped]
            law.history.append(step)
            if verbose:
                print(f"  + {step['name']:>34s}  val NMSE {cur:.3e}  ({step['move']}, rank {step['rank']}, pruned {step['pruned']})")
        law.terms, law.coef, law.val_nmse = terms, coef, cur
        if self.snap and terms:
            self.snap_law(law, train, val)
        return law

    def prune(self, terms, train, val, cur):
        dropped, changed = [], True
        while changed and len(terms) > 1:
            changed = False
            for t in list(terms):
                rest = [x for x in terms if x != t]
                c = self.fit(rest, train)
                e = self.score(rest, c, val)
                if e <= cur * (1 + self.delta / 2):
                    terms, cur, changed = rest, e, True
                    dropped.append(t)
                    break
        coef = self.fit(terms, train)
        return terms, coef, self.score(terms, coef, val), dropped

    def snap_law(self, law, train, val, max_den=20, tol=0.002):
        """Replace coefficients by simple fractions one at a time, keeping the rest refitted."""
        fixed = {}
        order = sorted(range(len(law.terms)), key=lambda i: -abs(law.coef[i + 1]))
        base = law.val_nmse
        for i in order:
            t = law.terms[i]
            c = law.coef[i + 1]
            fr = Fraction(c).limit_denominator(max_den)
            if abs(float(fr) - c) > 0.03 * max(abs(c), 0.05):
                continue
            trial = dict(fixed)
            trial[t] = fr
            coef = self._fit_fixed(law.terms, trial, train)
            e = self.score(law.terms, coef, val)
            if e <= base * (1 + tol):
                fixed = trial
                law.coef = coef
        law.snapped = fixed
        law.val_nmse = self.score(law.terms, law.coef, val)

    def _fit_fixed(self, terms, fixed, train):
        free = [t for t in terms if t not in fixed]
        y = train.y.copy()
        for t, fr in fixed.items():
            y -= float(fr) * train.feature(t)
        cf = lstsq(train.design(free), y)
        coef = np.zeros(len(terms) + 1)
        coef[0] = cf[0]
        j = 1
        for i, t in enumerate(terms):
            if t in fixed:
                coef[i + 1] = float(fixed[t])
            else:
                coef[i + 1] = cf[j]
                j += 1
        return coef

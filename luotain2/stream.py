"""A stream is the data a law is fitted to: frames plus which pixels count."""
from __future__ import annotations

import numpy as np

from .grammar import FUNCS, shifted


class Stream:
    """Frames (T, H, W) seen through a mask.

    mode "dynamics": targets are frames t >= max_lag, predicted from earlier frames.
    mode "static" / "static-causal": one target frame (lag 0), predicted from its own neighbours.

    margin: pixels within `margin` of the border are never targets when the data is not
    periodic, so every term with reach <= margin reads real pixels (no wrap-around).
    region: optional (H, W) bool mask choosing which pixels belong to this split.
    """

    def __init__(self, frames, mode="dynamics", max_lag=3, margin=0, periodic=True, region=None):
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim == 2:
            frames = frames[None]
        self.frames = frames
        self.mode = mode
        self.max_lag = max_lag if mode == "dynamics" else 0
        T, H, W = frames.shape
        self.H, self.W = H, W
        self.targets = np.arange(self.max_lag, T) if mode == "dynamics" else np.array([0])
        mask = np.ones((H, W), bool)
        if not periodic and margin > 0:
            mask[:margin] = False
            mask[-margin:] = False
            mask[:, :margin] = False
            mask[:, -margin:] = False
        if region is not None:
            mask &= region
        self.mask = mask
        self.periodic = periodic
        self.margin = margin
        self._base = {}
        self._feat = {}
        self.y = frames[self.targets][:, mask].ravel()
        self.n = self.y.size

    def base(self, f, lag):
        key = (f, lag)
        if key not in self._base:
            self._base[key] = FUNCS[f](self.frames[self.targets - lag])
        return self._base[key]

    def field(self, t):
        """The term evaluated as full fields (nT, H, W)."""
        if t[0] == "shift":
            _, f, lag, dy, dx = t
            return shifted(self.base(f, lag), dy, dx)
        if t[0] == "prod":
            _, lag, dy1, dx1, dy2, dx2 = t
            b = self.base("id", lag)
            return shifted(b, dy1, dx1) * shifted(b, dy2, dx2)
        raise ValueError(t)

    def feature(self, t):
        if t not in self._feat:
            self._feat[t] = self.field(t)[:, self.mask].ravel()
        return self._feat[t]

    def to_fields(self, vec):
        out = np.zeros((len(self.targets), self.H, self.W))
        out[:, self.mask] = vec.reshape(len(self.targets), -1)
        return out

    def design(self, terms, rows=None):
        cols = [np.ones(self.n)] + [self.feature(t) for t in terms]
        X = np.stack(cols, axis=1)
        return X if rows is None else X[rows]

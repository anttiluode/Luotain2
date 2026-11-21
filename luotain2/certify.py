"""Check a law without trusting the code that found it.

A law is saved as plain JSON: {"intercept": c0, "terms": [[kind, ...], ...], "coef": [...]}
This file re-implements prediction from scratch with numpy only (no imports from the
discovery engine), so anyone can recompute the error a law claims on any data.

    python -m luotain2.certify law.json frames.npy --mode dynamics
"""
from __future__ import annotations

import json
import sys

import numpy as np


def _value(frames, t, term, mode):
    kind = term[0]
    if kind == "shift":
        _, f, lag, dy, dx = term
        a = np.roll(frames[t - lag], shift=(-dy, -dx), axis=(0, 1))
        return a if f == "id" else a ** 2 if f == "sq" else a ** 3
    if kind == "prod":
        _, lag, dy1, dx1, dy2, dx2 = term
        b = frames[t - lag]
        return np.roll(b, (-dy1, -dx1), (0, 1)) * np.roll(b, (-dy2, -dx2), (0, 1))
    raise ValueError(term)


def predict(law, frames, t, mode="dynamics"):
    out = np.full(frames.shape[1:], float(law["intercept"]))
    for term, c in zip(law["terms"], law["coef"]):
        out += c * _value(frames, t, term, mode)
    return out


def score(law, frames, mode="dynamics", margin=0, max_lag=None):
    frames = np.asarray(frames, float)
    if frames.ndim == 2:
        frames = frames[None]
    lags = [t[2] if t[0] == "shift" else t[1] for t in law["terms"]] or [0]
    start = (max_lag if max_lag is not None else max(lags)) if mode == "dynamics" else 0
    ys, ps = [], []
    sl = (slice(margin, -margin or None), slice(margin, -margin or None))
    for t in range(start, len(frames)):
        ys.append(frames[t][sl].ravel())
        ps.append(predict(law, frames, t, mode)[sl].ravel())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float(np.mean((y - p) ** 2) / np.var(y))


def to_json(law_obj):
    """Convert an engine Law into the plain certificate format."""
    return {"intercept": float(law_obj.coef[0]),
            "terms": [list(t) for t in law_obj.terms],
            "coef": [float(c) for c in law_obj.coef[1:]]}


if __name__ == "__main__":
    law = json.load(open(sys.argv[1]))
    frames = np.load(sys.argv[2])
    mode = "static" if "--static" in sys.argv else "dynamics"
    print(f"NMSE {score(law, frames, mode):.6e}")

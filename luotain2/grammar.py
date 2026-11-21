"""The grammar of laws.

A law predicts a field value from other values of the same field:

    u_t(x)  ~=  c0  +  sum_k  a_k * term_k(x)

Each term is one of

    ("shift", f, lag, dy, dx)            f(u_{t-lag})(y+dy, x+dx),  f in {id, sq, cube}
    ("prod",  lag, dy1, dx1, dy2, dx2)   u_{t-lag}(y+dy1, x+dx1) * u_{t-lag}(y+dy2, x+dx2)

For a moving field (mode "dynamics") lag >= 1: the law says how the next frame is made from
earlier frames. For a still image (mode "static" or "static-causal") lag = 0 and the term
may not read the pixel it predicts: the law says how each pixel is made from its neighbours.
In "static-causal" mode a term may only read pixels that come earlier in raster order, so the
law is a predictor a lossless coder can use.

The grammar is open: offsets are not limited to a small stencil. Any (dy, dx) inside the
search margin is a legal term, so a law may couple a pixel to one 17 pixels away.
"""
from __future__ import annotations

import numpy as np

FUNCS = {
    "id": lambda a: a,
    "sq": lambda a: a * a,
    "cube": lambda a: a * a * a,
}


def term_name(t) -> str:
    if t[0] == "shift":
        _, f, lag, dy, dx = t
        core = "u" if lag == 0 else f"u[t-{lag}]"
        where = "" if (dy, dx) == (0, 0) else f"({dy:+d},{dx:+d})"
        base = f"{core}{where}"
        return {"id": base, "sq": f"{base}^2", "cube": f"{base}^3"}[f]
    if t[0] == "prod":
        _, lag, dy1, dx1, dy2, dx2 = t
        core = "u" if lag == 0 else f"u[t-{lag}]"
        a = core if (dy1, dx1) == (0, 0) else f"{core}({dy1:+d},{dx1:+d})"
        b = core if (dy2, dx2) == (0, 0) else f"{core}({dy2:+d},{dx2:+d})"
        return f"{a}*{b}"
    raise ValueError(t)


def canonical(t):
    """Products are symmetric: store the two factors in sorted order."""
    if t[0] == "prod":
        _, lag, a, b, c, d = t
        (a, b), (c, d) = sorted([(a, b), (c, d)])
        return ("prod", lag, a, b, c, d)
    return t


def term_reach(t) -> int:
    if t[0] == "shift":
        return max(abs(t[3]), abs(t[4]))
    if t[0] == "prod":
        return max(abs(t[2]), abs(t[3]), abs(t[4]), abs(t[5]))
    return 0


def causal_ok(dy: int, dx: int) -> bool:
    """True when pixel (y+dy, x+dx) is decoded before (y, x) in raster order."""
    return dy < 0 or (dy == 0 and dx < 0)


def shifted(field: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """field(y+dy, x+dx) on the last two axes, periodic."""
    if dy == 0 and dx == 0:
        return field
    return np.roll(field, shift=(-dy, -dx), axis=(-2, -1))

"""Mystery worlds: fields made by a known law, so a discovered law can be checked.

Every world returns three independent trajectories (train / validation / test), each started
from a different random smooth field, with a little measurement noise added after simulation.
The true law is returned as {term: coefficient} in the grammar's own terms.
"""
from __future__ import annotations

import numpy as np

from .grammar import shifted

N4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def smooth_field(rng, n, scale=6.0):
    k = np.fft.fftfreq(n) * n
    KY, KX = np.meshgrid(k, k, indexing="ij")
    amp = np.exp(-(KY ** 2 + KX ** 2) / (2 * scale ** 2))
    z = np.fft.ifft2(amp * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))).real
    return (z - z.mean()) / (z.std() + 1e-12)


def lap(u):
    return sum(shifted(u, dy, dx) for dy, dx in N4) - 4 * u


def S(f, lag, dy, dx):
    return ("shift", f, lag, dy, dx)


def heat_law(D):
    law = {S("id", 1, 0, 0): 1 - 4 * D}
    for dy, dx in N4:
        law[S("id", 1, dy, dx)] = D
    return law


def _simulate(step, rng, n, T, lags, init_scale, noise, amp=1.0, kind="process", clip=None):
    """kind="process": noise is injected into every new frame and becomes part of the state, so
    the true law is the best possible predictor. kind="measurement": the state evolves cleanly and
    noise is added only to what is observed (then averaging old frames helps, and the best
    predictor is no longer the sparse true law: the errors-in-variables problem)."""
    hist = [amp * smooth_field(rng, n, init_scale) for _ in range(lags)]
    if clip is not None:
        hist = [np.clip(h, -clip, clip) for h in hist]
    if lags == 2:  # wave: start at rest
        hist = [hist[0], hist[0].copy()]
    frames = list(hist)
    while len(frames) < T:
        nxt = step(frames)
        if kind == "process":
            nxt = nxt + noise * amp * rng.standard_normal(nxt.shape)
        frames.append(nxt)
    F = np.array(frames)
    if kind == "measurement":
        F = F + noise * F.std() * rng.standard_normal(F.shape)
    return F


def make_world(name, seed=0, n=64, T=40, noise=0.02, kind="process"):
    rng = np.random.default_rng(seed)
    if name == "heat":
        D = 0.2
        law = heat_law(D)
        step = lambda h: h[-1] + D * lap(h[-1])
        spec = dict(lags=1, init_scale=8)
    elif name == "diagonal":   # diffusion only along the main diagonal, plus a weak isotropic part
        a, b = 0.2, 0.05
        law = {S("id", 1, 0, 0): 1 - 2 * a - 4 * b, S("id", 1, 1, 1): a, S("id", 1, -1, -1): a}
        for dy, dx in N4:
            law[S("id", 1, dy, dx)] = b
        step = lambda h: h[-1] + a * (shifted(h[-1], 1, 1) + shifted(h[-1], -1, -1) - 2 * h[-1]) + b * lap(h[-1])
        spec = dict(lags=1, init_scale=8)
    elif name == "drift":      # the field slides by (+1, +2) per frame while slowly blending in place
        law = {S("id", 1, -1, -2): 0.75, S("id", 1, 0, 0): 0.25}
        step = lambda h: 0.75 * shifted(h[-1], -1, -2) + 0.25 * h[-1]
        spec = dict(lags=1, init_scale=6)
    elif name == "wave":
        c2 = 0.2
        law = {S("id", 1, 0, 0): 2 - 4 * c2, S("id", 2, 0, 0): -1.0}
        for dy, dx in N4:
            law[S("id", 1, dy, dx)] = c2
        step = lambda h: 2 * h[-1] - h[-2] + c2 * lap(h[-1])
        spec = dict(lags=2, init_scale=5)
    elif name == "allen_cahn":  # u' = u + D lap u + r (u - u^3)
        D, r = 0.15, 0.2
        law = heat_law(D)
        law[S("id", 1, 0, 0)] += r
        law[S("cube", 1, 0, 0)] = -r
        step = lambda h: h[-1] + D * lap(h[-1]) + r * (h[-1] - h[-1] ** 3)
        spec = dict(lags=1, init_scale=6, amp=0.8, clip=1.4)
    elif name == "wormhole":   # local averaging plus a link to a pixel 17 rows down and 9 columns left
        law = {S("id", 1, 0, 0): 0.5, S("id", 1, 17, -9): 0.1}
        for dy, dx in N4:
            law[S("id", 1, dy, dx)] = 0.1
        step = lambda h: 0.5 * h[-1] + 0.1 * sum(shifted(h[-1], dy, dx) for dy, dx in N4) + 0.1 * shifted(h[-1], 17, -9)
        spec = dict(lags=1, init_scale=4)
    elif name == "burgers":    # u' = u - c u (u(x+1) - u(x-1))/2 + nu lap u
        c, nu = 0.25, 0.2
        law = heat_law(nu)
        law[("prod", 1, 0, 0, 0, 1)] = -c / 2
        law[("prod", 1, 0, -1, 0, 0)] = c / 2
        step = lambda h: h[-1] - c * h[-1] * (shifted(h[-1], 0, 1) - shifted(h[-1], 0, -1)) / 2 + nu * lap(h[-1])
        spec = dict(lags=1, init_scale=6, amp=0.5)
    elif name == "echo":       # local averaging plus a delayed echo: three frames back, 11 rows up, 23 columns right
        law = {S("id", 1, 0, 0): 0.5, S("id", 3, -11, 23): 0.1}
        for dy, dx in N4:
            law[S("id", 1, dy, dx)] = 0.1
        step = lambda h: 0.5 * h[-1] + 0.1 * sum(shifted(h[-1], dy, dx) for dy, dx in N4) + 0.1 * shifted(h[-3], -11, 23)
        spec = dict(lags=3, init_scale=4)
    elif name == "null":       # every frame is a fresh random field: there is no law to find
        law = {}
        step = lambda h: smooth_field(rng, n, 6)
        spec = dict(lags=1, init_scale=6)
    else:
        raise KeyError(name)
    trajs = [_simulate(step, rng, n, T, spec["lags"], spec["init_scale"], noise, spec.get("amp", 1.0), kind, spec.get("clip")) for _ in range(3)]
    for tr in trajs:
        assert np.all(np.isfinite(tr)), f"{name} blew up"
    return trajs, law


WORLDS = ["heat", "diagonal", "drift", "wave", "allen_cahn", "wormhole", "burgers", "null"]
STRESS = ["echo"]


def causal_ar_texture(seed=0, n=192, coefs=None, noise=1.0):
    """A still texture made pixel by pixel in raster order from earlier pixels:
    x(y, x) = sum a_d x(y+dy, x+dx) + e, every (dy, dx) causal. Returns image and true law."""
    rng = np.random.default_rng(seed)
    coefs = coefs or {(0, -1): 0.45, (-1, 0): 0.35, (-6, -11): 0.15}
    pad = 16
    img = np.zeros((n + pad, n + 2 * pad))
    e = rng.standard_normal(img.shape) * noise
    for y in range(pad, n + pad):
        for x in range(pad, n + pad):
            v = e[y, x]
            for (dy, dx), a in coefs.items():
                v += a * img[y + dy, x + dx]
            img[y, x] = v
    img = img[pad:, pad:-pad]
    law = {S("id", 0, dy, dx): a for (dy, dx), a in coefs.items()}
    return img, law

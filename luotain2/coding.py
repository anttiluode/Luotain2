"""Lossless-coding yardsticks for still-image laws.

A causal law predicts each pixel from pixels already decoded. A lossless coder only has to
store the prediction residuals, so the empirical entropy of the integer residual (bits per
pixel) measures how much of the image the law explains. These are predictor-only numbers:
real coders (JPEG-LS, CALIC) add context modelling on top and do better than any bare predictor.
"""
from __future__ import annotations

import numpy as np


def entropy_bits(residual_int):
    v, c = np.unique(residual_int, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def med_predict(img):
    """LOCO-I / JPEG-LS median edge detector, from W, N, NW."""
    a = np.roll(img, 1, axis=1)      # W
    b = np.roll(img, 1, axis=0)      # N
    c = np.roll(np.roll(img, 1, axis=0), 1, axis=1)  # NW
    mx, mn = np.maximum(a, b), np.minimum(a, b)
    return np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))


def causal_stencil(radius):
    out = []
    for dy in range(-radius, 1):
        for dx in range(-radius, radius + 1):
            if dy < 0 or dx < 0:
                out.append(("shift", "id", 0, dy, dx))
    return out


def bits_per_pixel(stream, terms, coef, img_u8):
    """Entropy of the rounded residual of the law on the stream's own pixels."""
    pred = stream.design(terms) @ coef
    pred = np.clip(np.round(pred), 0, 255)
    truth = img_u8[stream.mask].astype(np.int64)
    return entropy_bits(truth - pred.astype(np.int64))

"""Disjoint-time instrumental variables for Luotain2's discrete field laws.

This is an IV-inspired experiment, NOT an implementation of the WAIVES SDE weak
form or its asymptotic theory. Separate time samples protect instruments from
independent observation noise. This only estimates coefficients conditional on
a supplied term support: it does not discover the terms.

Y_t(x) = X_t(x) + eta_t(x), with zero-mean i.i.d. sensor noise.
Latent discrete law: X_t = b0 + sum b_j g_j(X_{t-lag}) + process noise.
Polynomial corrected scores remove Gaussian measurement-noise bias.
No claims for temporally correlated sensors, hidden variables or SPDEs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grammar import shifted


@dataclass
class IVEstimate:
    coef: np.ndarray | None
    ols_coef: np.ndarray
    condition: float
    instrument_strength: float
    n_moments: int
    n_windows: int
    identified: bool
    reason: str
    lag_shift: int
    moment_residual: float | None


def _value(frames, t, term, sensor_var):
    """Evaluate one existing Luotain2 grammar term with Gaussian debiasing."""
    kind = term[0]
    if kind == "shift":
        _, f, lag, dy, dx = term
        y = shifted(frames[t - lag], dy, dx)
        if f == "id":
            return y
        if f == "sq":
            return y * y - sensor_var
        if f == "cube":
            return y * y * y - 3.0 * sensor_var * y
        raise ValueError(f"unsupported function: {f}")
    if kind == "prod":
        _, lag, dy1, dx1, dy2, dx2 = term
        a = shifted(frames[t - lag], dy1, dx1)
        b = shifted(frames[t - lag], dy2, dx2)
        # Independent sensor noise correlates only at the same pixel.
        correction = sensor_var if (dy1, dx1) == (dy2, dx2) else 0.0
        return a * b - correction
    raise ValueError(term)


def _lag(term):
    """Grammar: shift uses index 2; product uses index 1."""
    return int(term[2] if term[0] == "shift" else term[1])


def moment_matrices(frames, terms, *, sensor_var=0.0, window=1,
                    gap=0, pixel_stride=1, step=None):
    """Construct A, Z, b from distinct time samples.

    Each target window averages consecutive discrete equations. Instrument
    columns evaluate the same functions on strictly earlier observed frames.
    Sensor variance is a calibration INPUT, not the continuous-SDE variogram
    estimate from WAIVES. Spatial pixels are correlated, so do not count them
    as independent replicates when assessing strength.
    """
    f = np.asarray(frames, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError("frames must have shape (time,height,width)")
    if not terms:
        raise ValueError("pass a nonempty, known candidate support")
    if any(t[0] not in ("shift", "prod") or _lag(t) < 1 for t in terms):
        raise ValueError("dynamics terms must use time lag >= 1")
    if not (window >= 1 and gap >= 0 and pixel_stride >= 1):
        raise ValueError("window and pixel_stride >=1, gap >=0")
    if sensor_var < 0 or not np.isfinite(sensor_var):
        raise ValueError("sensor_var must be finite and nonnegative")
    lag_max = max(_lag(t) for t in terms)
    # For W equations the regressor observation times run from t-lag_max
    # to t+W-2. The latest instrument observation is t+W-2-shift;
    # shift=max_lag+W-1+gap makes it strictly earlier than t-lag_max.
    shift = lag_max + window - 1 + gap
    first = lag_max + shift
    stride = window if step is None else int(step)
    if stride < 1:
        raise ValueError("step must be positive")
    starts = np.arange(first, len(f) - window + 1, stride, dtype=int)
    if starts.size == 0:
        raise ValueError("too few frames for disjoint-time instruments")
    n_pixels = f[0, ::pixel_stride, ::pixel_stride].size
    A_rows, Z_rows, b_rows = [], [], []
    for t0 in starts:
        av = [np.ones(n_pixels)]
        zv = [np.ones(n_pixels)]
        for term in terms:
            xa = sum((_value(f, t0 + k, term, sensor_var)
                      for k in range(window))) / window
            zi = sum((_value(f, t0 + k - shift, term, sensor_var)
                      for k in range(window))) / window
            av.append(xa[::pixel_stride, ::pixel_stride].ravel())
            zv.append(zi[::pixel_stride, ::pixel_stride].ravel())
        b = np.mean(f[t0:t0 + window], axis=0)
        A_rows.append(np.column_stack(av))
        Z_rows.append(np.column_stack(zv))
        b_rows.append(b[::pixel_stride, ::pixel_stride].ravel())
    return (np.concatenate(A_rows),
            np.concatenate(Z_rows),
            np.concatenate(b_rows),
            {"n_windows": len(starts), "lag_shift": shift,
             "earliest_target_time": int(starts[0]),
             "n_spatial_locations": n_pixels})


def estimate(frames, terms, *, sensor_var=0.0, window=1,
             gap=0, pixel_stride=1, step=None, max_condition=1e9):
    """Return aligned OLS and IV coefficients or an IV refusal.

    Refusal uses numerical rank/condition, not a Cragg-Donald F statistic.
    The massive number of spatial samples should not create false statistical
    confidence when they are correlated.
    """
    A, Z, b, meta = moment_matrices(
        frames, terms, sensor_var=sensor_var, window=window,
        gap=gap, pixel_stride=pixel_stride, step=step
    )
    ols = np.linalg.lstsq(A, b, rcond=None)[0]
    M = (Z.T @ A) / len(b)
    rhs = (Z.T @ b) / len(b)
    sing = np.linalg.svd(M, compute_uv=False)
    ratio = float(sing[-1] / sing[0]) if sing[0] > 0 else 0.0
    cond = float(sing[0] / sing[-1]) if sing[-1] > 0 else float("inf")
    identified = bool(np.isfinite(cond) and cond <= max_condition
                      and sing[-1] > 1e-13)
    if not identified:
        return IVEstimate(None, ols, cond, ratio, len(b), meta["n_windows"],
                          False, "weak or rank-deficient instruments",
                          meta["lag_shift"], None)
    coef = np.linalg.solve(M, rhs)
    residual = b - A @ coef
    mres = float(np.linalg.norm(Z.T @ residual / len(b)))
    return IVEstimate(coef, ols, cond, ratio, len(b), meta["n_windows"],
                      True, "numerically identified for the supplied model",
                      meta["lag_shift"], mres)


def true_coefficient_vector(terms, law):
    """Oracle-support synthetic diagnostic only, not blind discovery."""
    return np.array([0.0] + [law[t] for t in terms], dtype=float)


def coefficient_error(est, terms, law):
    """L2 error against known synthetic generating coefficients."""
    truth = true_coefficient_vector(terms, law)
    return {
        "ols_l2": float(np.linalg.norm(est.ols_coef - truth)),
        "iv_l2": (None if est.coef is None
                  else float(np.linalg.norm(est.coef - truth))),
        "identified": est.identified,
        "condition": est.condition,
        "strength_ratio": est.instrument_strength,
        "n_windows": est.n_windows,
        "reason": est.reason,
    }

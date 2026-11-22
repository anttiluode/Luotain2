"""Tests of IV's identification boundary; not a test of open-grammar discovery."""
import numpy as np
import pytest

from luotain2.iv import _value, estimate, moment_matrices
from luotain2.worlds import make_world


TERM = ("shift", "id", 1, 0, 0)


def test_instruments_are_strictly_in_the_past():
    frames = np.arange(14, dtype=float)[:, None, None] * np.ones((1, 3, 3))
    A, Z, b, meta = moment_matrices(frames, [TERM], window=1, gap=0, step=100)
    assert meta["earliest_target_time"] == 2
    assert meta["lag_shift"] == 1
    assert np.all(A[:, 1] == 1)  # regressor reads t-1
    assert np.all(Z[:, 1] == 0)  # instrument reads t-2
    assert np.all(b == 2)

    A, Z, b, meta = moment_matrices(frames, [TERM], window=2, gap=2, step=100)
    assert meta["lag_shift"] == 4
    assert meta["earliest_target_time"] == 5
    assert np.allclose(A[:, 1], (4 + 5) / 2)
    assert np.allclose(Z[:, 1], (0 + 1) / 2)
    assert np.allclose(b, (5 + 6) / 2)


def test_polynomial_corrected_scores_are_unbiased_in_expectation():
    rng = np.random.default_rng(1001)
    x = 0.65
    variance = 0.16
    data = x + rng.normal(0, np.sqrt(variance), size=(1, 1, 120000))
    sq = _value(data, 1, ("shift", "sq", 1, 0, 0), variance)
    cube = _value(data, 1, ("shift", "cube", 1, 0, 0), variance)
    same = _value(data, 1, ("prod", 1, 0, 0, 0, 0), variance)
    assert abs(sq.mean() - x * x) < 0.01
    assert abs(cube.mean() - x ** 3) < 0.01
    assert abs(same.mean() - x * x) < 0.01


def _ar_data(seed=0, n=16, T=260, a=.75, process=.25, sensor=.4):
    rng = np.random.default_rng(seed)
    x = np.empty((T, n, n))
    x[0] = rng.normal(0, process / np.sqrt(1 - a*a), (n, n))
    for t in range(1, T):
        x[t] = a * x[t-1] + process * rng.standard_normal((n, n))
    return x + sensor*rng.standard_normal(x.shape)


@pytest.mark.parametrize("seed", [13, 17, 29])
def test_iv_repairs_classical_independent_sensor_attenuation(seed):
    y = _ar_data(seed=seed)
    r = estimate(y, [TERM], sensor_var=.16, window=1)
    assert r.identified
    assert abs(r.coef[1] - .75) < .055
    assert abs(r.coef[1] - .75) < abs(r.ols_coef[1] - .75)*.35
    assert r.instrument_strength > 0


def test_rank_deficiency_causes_refusal_not_a_fake_answer():
    y = np.ones((16, 8, 8))
    r = estimate(y, [TERM], sensor_var=0.01)
    assert not r.identified
    assert r.coef is None
    assert np.isinf(r.condition) or r.condition > 1e9


def test_original_discrete_world_is_supported_without_promising_recovery():
    trajectories, law = make_world("wormhole", seed=102, n=32, T=27,
                                  noise=.01, kind="measurement")
    # The oracle-support diagnostic is separate from blind support discovery.
    terms = list(law)
    r = estimate(trajectories[0], terms, sensor_var=0.0, window=1)
    assert np.all(np.isfinite(r.ols_coef))
    assert r.n_windows > 0 and r.n_moments > 0
    assert np.isfinite(r.instrument_strength)
    # This system may have weak instruments: numeric abstention is allowed.
    assert r.coef is None or np.all(np.isfinite(r.coef))


def test_invalid_time_geometry_refuses():
    with pytest.raises(ValueError):
        moment_matrices(np.ones((2, 4, 4)), [TERM], window=2, gap=5)
    with pytest.raises(ValueError):
        moment_matrices(np.ones((12, 4, 4)), [TERM], sensor_var=-1)

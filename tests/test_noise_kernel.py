import numpy as np
from luotain2.noise_kernel import (wick, _pairings, iid_kernel, corrected_risk_raw_kernel,
                                   echo_kernel, kernel_value)
from luotain2.blind_poly import corrected_poly_risk_raw
from luotain2.stream import Stream
from luotain2.worlds import make_world
from luotain2.grammar import shifted


def test_pairings_count():
    # partial matchings of n items: 1, 1, 2, 4, 10, 26, 76 (telephone numbers)
    assert [len(_pairings(n)) for n in range(7)] == [1, 1, 2, 4, 10, 26, 76]


def test_wick_is_unbiased_for_correlated_pair_and_cube():
    rng = np.random.default_rng(0)
    n = 400_000
    C = np.array([[0.04, 0.025], [0.025, 0.04]])
    eta = rng.multivariate_normal([0, 0], C, size=n)
    xa, xb = 0.8, -0.5
    Y = {(1, 0, 0): xa + eta[:, 0], (1, 0, 1): xb + eta[:, 1]}
    kern = {(0, 0): 0.04, (0, 1): 0.025}
    get = Y.__getitem__
    a, b = (1, 0, 0), (1, 0, 1)
    assert abs(np.mean(wick([a, b], get, kern)) - xa * xb) < 2e-3
    assert abs(np.mean(wick([a, a, a], get, kern)) - xa ** 3) < 5e-3
    assert abs(np.mean(wick([a, a, b, b], get, kern)) - xa ** 2 * xb ** 2) < 5e-3


def test_iid_kernel_equals_hermite_risk():
    paths, law = make_world("allen_cahn", seed=3, kind="process", noise=0.02)
    s = Stream(paths[0] + 0.03 * np.random.default_rng(9).standard_normal(paths[0].shape), max_lag=3)
    terms = list(law); coef = np.array([0.0] + [law[t] for t in terms])
    assert np.isclose(corrected_risk_raw_kernel(s, terms, coef, iid_kernel(9e-4)),
                      corrected_poly_risk_raw(s, terms, coef, 9e-4), rtol=1e-10)


def test_kernel_echo_recovers_blurred_noise():
    paths, law = make_world("wormhole", seed=4, kind="process", noise=0.02)
    rng = np.random.default_rng(1234)
    w = rng.standard_normal(paths[0].shape)
    sd = 0.05 * paths[0].std()
    e = sd * sum(shifted(w, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 3.0
    terms = list(law); coef = np.array([0.0] + [law[t] for t in terms])
    k = echo_kernel(paths[0] + e, terms, coef, radius=2)
    R = sd * sd
    for D, truth in {(0, 0): 1, (0, 1): 6 / 9, (1, 1): 4 / 9, (0, 2): 3 / 9, (2, 2): 1 / 9}.items():
        assert abs(k[D] / R - truth) < 0.08, (D, k[D] / R)


def test_kernel_value_symmetry_and_time():
    k = {(0, 0): 1.0, (0, 1): 0.5}
    assert kernel_value(k, (1, 3, 3), (1, 3, 4)) == 0.5 == kernel_value(k, (1, 3, 4), (1, 3, 3))
    assert kernel_value(k, (1, 3, 3), (2, 3, 4)) == 0.0


def test_echo_kernel_two_lag_law_needs_cross_lag_term():
    """Two-lag linear law: the law's own lag-1 and lag-2 terms meet across time.
    With the full echo formula the box-blur kernel is recovered from the true law."""
    from luotain2.grammar import shifted
    from luotain2.noise_kernel import echo_kernel
    rng = np.random.default_rng(5)
    n, T = 64, 120
    X = [rng.standard_normal((n, n)), rng.standard_normal((n, n))]
    for _ in range(T - 2):
        X.append(0.55 * X[-1] + 0.4 * shifted(X[-2], 1, 0) + rng.standard_normal((n, n)))
    X = np.array(X)
    nrng = np.random.default_rng(777)
    sd = 0.5
    Y = []
    for x in X:
        w = nrng.standard_normal((n, n))
        b = sum(shifted(w, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 3.0
        Y.append(x + sd * b)
    Y = np.array(Y)
    terms = [("shift", "id", 1, 0, 0), ("shift", "id", 2, 1, 0)]
    coef = np.array([0.0, 0.55, 0.4])
    k = echo_kernel(Y, terms, coef, radius=2, iters=1)
    R = sd * sd
    for D, want in [((0, 0), 1.0), ((0, 1), 6 / 9), ((1, 1), 4 / 9), ((0, 2), 3 / 9)]:
        assert abs(k[D] / R - want) < 0.07, (D, k[D] / R, want)

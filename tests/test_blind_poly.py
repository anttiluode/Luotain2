import numpy as np
from luotain2.blind_poly import hermite, residual_poly, corrected_poly_risk_raw, BlindPolyDiscoverer
from luotain2.blind_iv import BlindIVDiscoverer
from luotain2.stream import Stream
from luotain2.worlds import make_world


def test_hermite_is_unbiased_for_powers():
    rng = np.random.default_rng(0)
    x, R = 0.7, 0.09
    y = x + np.sqrt(R) * rng.standard_normal(2_000_000)
    for k in range(7):
        assert abs(hermite(y, k, R).mean() - x ** k) < 0.01 * max(1, x ** k) + 0.004


def test_exact_risk_equals_linear_formula_for_linear_laws():
    paths, law = make_world("wormhole", seed=1, kind="process", noise=0.02)
    s = Stream(paths[0] + 0.05 * np.random.default_rng(1).standard_normal(paths[0].shape), max_lag=3)
    terms = list(law); coef = np.array([0.01] + [law[t] for t in terms])
    R = 0.0025
    exact = corrected_poly_risk_raw(s, terms, coef, R)
    raw = np.mean((s.y - s.design(terms) @ coef) ** 2)
    assert np.isclose(exact, raw - R * (1 + coef[1:] @ coef[1:]), rtol=1e-10, atol=1e-14)


def test_zero_noise_gives_plain_mse():
    paths, law = make_world("burgers", seed=2, kind="process", noise=0.02)
    s = Stream(paths[0], max_lag=3)
    terms = list(law); coef = np.array([0.0] + [law[t] for t in terms])
    assert np.isclose(corrected_poly_risk_raw(s, terms, coef, 0.0),
                      np.mean((s.y - s.design(terms) @ coef) ** 2), rtol=1e-10)


def test_corrected_risk_unbiased_for_nonlinear_laws():
    rng = np.random.default_rng(3)
    for world in ("allen_cahn", "burgers"):
        paths, law = make_world(world, seed=4, kind="process", noise=0.02)
        X = paths[0]; terms = list(law); coef = np.array([0.0] + [law[t] for t in terms])
        latent = corrected_poly_risk_raw(Stream(X, max_lag=3), terms, coef, 0.0)
        R = (0.04 * X.std()) ** 2
        est = [corrected_poly_risk_raw(Stream(X + np.sqrt(R) * rng.standard_normal(X.shape), max_lag=3),
                                       terms, coef, R) for _ in range(12)]
        raw_bias = R  # at least the target's own noise
        assert abs(np.mean(est) - latent) < 0.1 * raw_bias


def test_residual_poly_shares_pixels():
    q = residual_poly([("shift", "id", 1, 0, 0), ("shift", "cube", 1, 0, 0), ("prod", 1, 0, 0, 0, 1)],
                      np.array([0.1, 2.0, -3.0, 0.5]))
    assert q[(((1, 0, 0), 1),)] == -2.0 and q[(((1, 0, 0), 3),)] == 3.0
    assert q[tuple(sorted((((1, 0, 0), 1), ((1, 0, 1), 1))))] == -0.5 and q[()] == -0.1


def test_requires_known_variance():
    paths, _ = make_world("heat", seed=1, kind="process", noise=0.02)
    s = Stream(paths[0], max_lag=3)
    try:
        BlindPolyDiscoverer(max_lag=3, margin=31).run(s, s, sensor_variance=None)
        assert False
    except ValueError:
        pass

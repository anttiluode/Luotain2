import numpy as np
from luotain2.noise_echo import echo_variance
from luotain2.worlds import make_world


def _noisy(world, ratio, seed=5):
    paths, law = make_world(world, seed=seed, kind="process", noise=0.02)
    sd = ratio * paths[0].std()
    y = paths[0] + sd * np.random.default_rng(seed + 12345).standard_normal(paths[0].shape)
    return y, law, sd * sd


def test_true_law_echo_recovers_R():
    for world in ("allen_cahn", "burgers", "wormhole", "echo"):
        y, law, R = _noisy(world, 0.05)
        terms = list(law); coef = np.array([0.0] + [law[t] for t in terms])
        assert abs(echo_variance(y, terms, coef) / R - 1) < 0.06, world


def test_no_terms_no_estimate():
    y, _, _ = _noisy("heat", 0.05)
    assert echo_variance(y, [], np.array([0.0])) is None


def test_noise_free_gives_near_zero():
    paths, law = make_world("wormhole", seed=6, kind="process", noise=0.02)
    terms = list(law); coef = np.array([0.0] + [law[t] for t in terms])
    R = echo_variance(paths[0], terms, coef)
    assert R < 1e-3 * paths[0].var()

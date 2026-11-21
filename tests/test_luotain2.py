import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from luotain2.stream import Stream
from luotain2.discover import Discoverer
from luotain2.worlds import make_world, causal_ar_texture
from luotain2 import certify


def test_term_semantics():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((5, 16, 16))
    s = Stream(F, mode="dynamics", max_lag=2)
    f = s.field(("shift", "id", 1, 3, -2))
    t, y, x = 1, 4, 7          # target index 1 -> frame 3, reads frame 2
    assert np.isclose(f[t, y, x], F[s.targets[t] - 1, y + 3, x - 2])


def test_fft_score_matches_direct_correlation():
    rng = np.random.default_rng(1)
    F = rng.standard_normal((6, 32, 32))
    tr = Stream(F, mode="dynamics", max_lag=1)
    r = rng.standard_normal((len(tr.targets), 32, 32))
    C = np.real(np.fft.ifft2(np.conj(np.fft.fft2(r)) * np.fft.fft2(tr.base("id", 1)))).sum(0)
    dy, dx = 5, -7
    direct = np.sum(r * tr.field(("shift", "id", 1, dy, dx)))
    assert np.isclose(C[dy % 32, dx % 32], direct)


def test_heat_law_recovered_exactly():
    trajs, law = make_world("heat", seed=1)
    tr, va, te = [Stream(x, mode="dynamics", max_lag=3) for x in trajs]
    L = Discoverer(mode="dynamics", max_lag=3, margin=32).run(tr, va)
    assert set(L.terms) == set(law)


def test_long_range_link_recovered():
    trajs, law = make_world("wormhole", seed=2)
    tr, va, te = [Stream(x, mode="dynamics", max_lag=2) for x in trajs]
    L = Discoverer(mode="dynamics", max_lag=2, margin=32).run(tr, va)
    assert ("shift", "id", 1, 17, -9) in L.terms


def test_null_world_finds_nothing():
    trajs, law = make_world("null", seed=3)
    tr, va, te = [Stream(x, mode="dynamics", max_lag=3) for x in trajs]
    L = Discoverer(mode="dynamics", max_lag=3, margin=32).run(tr, va)
    assert L.terms == []


def test_causal_texture_law():
    img, law = causal_ar_texture(seed=7, n=128)
    H, W = img.shape
    rows = np.arange(H)[:, None] * np.ones((1, W), bool)
    reg = lambda a, b: (rows >= a) & (rows < b)
    S = lambda r: Stream(img, mode="static-causal", margin=16, periodic=False, region=r)
    tr, va = S(reg(16, 70)), S(reg(70, 112))
    L = Discoverer(mode="static-causal", margin=16).run(tr, va)
    assert set(L.terms) == set(law)


def test_certificate_matches_engine():
    trajs, law = make_world("allen_cahn", seed=1)
    tr, va, te = [Stream(x, mode="dynamics", max_lag=3) for x in trajs]
    d = Discoverer(mode="dynamics", max_lag=3, margin=32)
    L = d.run(tr, va)
    cert = certify.to_json(L)
    assert np.isclose(certify.score(cert, trajs[2], max_lag=3), d.score(L.terms, L.coef, te), rtol=1e-9)

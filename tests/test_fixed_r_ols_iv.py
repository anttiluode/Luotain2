"""Gate IV-003 matched OLS/IV and fixed calibrated sensor-noise controls."""
import numpy as np
from luotain2.blind_iv import BlindIVDiscoverer
from luotain2.iv import estimate
from luotain2.stream import Stream
from luotain2.worlds import make_world
from scripts.run_blind_iv_gate import run


def test_corrected_risk_known_variance_matches_definition():
    rng=np.random.default_rng(17)
    frames=rng.standard_normal((10,8,8))
    s=Stream(frames,max_lag=2)
    coef=np.array([.1,.7])
    term=[("shift","id",1,0,1)]
    r=.04
    predicted=s.design(term) @ coef
    manual=(np.mean((s.y-predicted)**2)-r*(1+.7**2))/(np.var(s.y)-r)
    got=BlindIVDiscoverer.corrected_risk(
        s,term,coef,sensor_ratio=0,
        sensor_variance=r)
    assert np.isclose(got,manual,atol=1e-12)


def test_exactly_matched_ols_and_iv_fits_from_same_time_rows():
    y,_=make_world("drift",seed=34,n=24,T=23,
                    noise=.01,kind="measurement")
    s=Stream(y[0],mode="dynamics",max_lag=3)
    terms=[("shift","id",1,-1,-2),("shift","id",1,0,0)]
    kwargs=dict(max_lag=3,margin=10,k_props=6,pixel_stride=2)
    a=BlindIVDiscoverer(estimator="iv",**kwargs)
    b=BlindIVDiscoverer(estimator="ols",**kwargs)
    estimate0=estimate(s.frames,terms,window=1,pixel_stride=2)
    assert np.allclose(b.fit(s,terms),estimate0.ols_coef)
    assert np.allclose(a.fit(s,terms),estimate0.coef)


def test_fixed_noise_matched_latent_power_pipeline_is_deterministic():
    options=dict(worlds=["drift"],seeds=[440],kind="combined",
                process_ratio=.01,sensor_std=.02,
                match_latent_std=.25,n=24,T=16,max_terms=2,
                k_props=4,pixel_stride=3)
    r=run(estimator="ols",**options)
    r2=run(estimator="ols",**options)
    p=r["rows"][0]
    assert r["summary"]["estimator"]=="ols"
    assert r["summary"]["sensor_std"]==.02
    assert all(np.isclose(v,.25,atol=1e-10) for v in p["latent_std_after_matching"])
    assert p["status"]==r2["rows"][0]["status"]
    assert p["found"]==r2["rows"][0]["found"]

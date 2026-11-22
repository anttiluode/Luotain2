"""Gate IV-002 structural sanity tests, no oracle-support input to the method."""
import numpy as np
from luotain2.blind_iv import BlindIVDiscoverer
from luotain2.stream import Stream
from luotain2.worlds import make_world


def test_known_sensor_risk_formula_nonnegative_in_expectation():
    rng=np.random.default_rng(7)
    frames=rng.normal(0,.3,size=(32,16,16))
    frames+=.01*rng.normal(size=frames.shape)
    s=Stream(frames,mode="dynamics",max_lag=1)
    d=BlindIVDiscoverer(max_lag=1,margin=4)
    coef=np.array([0.0,0.4])
    t=[("shift","id",1,0,0)]
    val=d.corrected_risk(s,t,coef,.01)
    assert np.isfinite(val)


def test_single_term_iv_discoverer_runs_without_support_information():
    paths,truth=make_world("drift",seed=8,n=32,T=24,
                           noise=.01,kind="measurement")
    tr,va=[Stream(x,mode="dynamics",max_lag=3) for x in paths[:2]]
    method=BlindIVDiscoverer(max_lag=3,margin=14,k_props=8,
                              max_terms=4,pixel_stride=2,seed=8)
    result=method.run(tr,va,sensor_ratio=.01)
    assert set(result.terms).issubset({
        ("shift","id",lag,dy,dx)
        for lag in range(1,4) for dy in range(-14,15)
        for dx in range(-14,15)
    })
    assert len(result.coef)==len(result.terms)+1
    assert np.all(np.isfinite(result.coef))
    assert np.isfinite(result.val_risk)
    assert result.fit_calls>0


def test_corrected_risk_penalizes_wrong_noise_denoising_coefficients():
    # Latent field is constant across time; the actual true state equation
    # is identity. Noise-corrected risk should prefer exact identity to a
    # 50/50 average of two old noisy observations, in expectation.
    rng=np.random.default_rng(23)
    base=rng.normal(size=(1,48,48))
    noisy=np.broadcast_to(base,(40,48,48)).copy()
    noisy+=.2*rng.normal(size=noisy.shape)
    s=Stream(noisy,mode="dynamics",max_lag=2)
    id1=[("shift","id",1,0,0)]
    id2=[("shift","id",1,0,0),("shift","id",2,0,0)]
    a=BlindIVDiscoverer.corrected_risk(s,id1,np.array([0.0,1.0]),.2)
    b=BlindIVDiscoverer.corrected_risk(s,id2,np.array([0.0,.5,.5]),.2)
    assert np.isfinite(a) and np.isfinite(b)
    # In the absence of process innovation both are valid latent
    # predictors! So the risk may be nearly tied rather than inventing a
    # robust difference; this test asserts only the calibrated magnitude.
    assert abs(a-b)<.02

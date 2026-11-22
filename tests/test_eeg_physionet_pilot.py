"""EEG pilot null and numerical regression tests; never access personal EEG."""
import numpy as np
from scripts.eeg_physionet_pilot import (
    canonical, choose_channels, regression, independent_sources_mixed)


def test_montage_normalization():
    names=["EEG Fp1.","EEG Fp2.","EEG C3.","EEG C4.",
           "EEG P3.","EEG P4.","EEG O1.","EEG O2."]
    # MNE's eegbci.standardize produces simplified lead names.
    simplified=[n.removeprefix("EEG ").rstrip(".") for n in names]
    got=choose_channels(simplified)
    assert [name for name,_ in got]==["Fp1","Fp2","C3","C4","P3","P4","O1","O2"]


def test_synthetic_independent_sources_can_fake_sensor_connectivity():
    null=independent_sources_mixed(seed=99,n=10000)
    assert null["source_true_cross_couplings"]==0
    assert null["mixed_sensor_ar_offdiagonal_fraction"]>.1
    assert null["relative_fit_error"]<.1


def test_temporal_regressor_and_iv_have_finite_heldout_metrics():
    rng=np.random.default_rng(2)
    eeg=np.zeros((8,3000))
    for t in range(1,eeg.shape[1]):
        eeg[:,t]=.7*eeg[:,t-1]+rng.normal(0,.5,size=8)
    p=regression(eeg,160)
    assert p["train_time_samples"]>1000
    assert p["test_time_samples"]>500
    assert p["heldout_multichannel_ols_nmse"]<.8
    assert np.isfinite(p["iv_crossmoment_condition"])

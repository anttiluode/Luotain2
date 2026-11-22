# EEG Gate 000 — Do sensor-space rhythms reveal a real dynamical law?

This first real-world study deliberately **does not run Luotain2's 2-D convolution/FFT spatial grammar on EEG electrodes**. Scalp EEG electrodes do not lie on a periodic rectangular lattice and do not record independent local neural sources. Volume conduction, reference choice, common sources and sensor artifacts may create apparently directed cross-electrode coupling without any neural interaction.

## Public recording

PhysioNet EEG Motor Movement/Imagery dataset, subject S001, 64 EEG channels (10-10 montage), sampling 160 Hz. This is a public demonstration dataset, not the user's EEG and not a clinical interpretation.

- Run 1: one-minute eyes-open baseline
- Run 2: one-minute eyes-closed baseline

Reference: https://physionet.org/content/eegmmidb/1.0.0/

Fetch through MNE-Python's eegbci loader; the initial pilot is GitHub Actions .github/workflows/eeg_physionet_pilot.yml, executable locally with:

    pip install numpy scipy mne pytest
    python scripts/eeg_physionet_pilot.py --subject 1 --out results/eeg_physionet_001.json

## What is actually measured

1. **Spectral decomposition:** Welch estimates 8-12 Hz alpha energy as a fraction of 4-30 Hz power at four posterior electrodes (P3/P4/O1/O2). This is a known physiological contrast, *not a new signal law*. The two baselines differ in behavioral state and acquisition time.
2. **Time-series prediction:** The first 65% of each run trains models; the final 35% (after a one-second gap) tests them. Baselines: persistence (use previous observation) and channelwise AR(1). Competing sensor-space models: eight-channel ridge OLS AR(1), and an illustrative disjoint-lag IV AR(1) using lag-two observations as instruments. These models do not search the spatial grammar and do not identify brain connections.
3. **Blindness diagnostic:** the normalized lagged cross-moment condition number is printed. If too ill-conditioned, the illustrative IV model refuses to score. This is *not* a Cragg-Donald F test; scalp channels are spatially correlated and temporal samples are autocorrelated.
4. **False-discovery null:** simulate eight independent AR sources with different autoregressive poles, mix them through a fixed dense observation matrix, and measure the nonzero off-diagonal AR weights that appear in the observed channels. There are no source-to-source connections in this null. If the observed model reports them, that is a sensor mixture, not a neurological discovery.

## Why this is deliberately not a brain-network claim

The sensor-space regression y[t]≈W*y[t−1] is useful for prediction. But even if the *true* underlying source dynamics are diagonal, an instantaneous sensor mixing y=M*s creates an apparently dense observed operator W=M*A*M⁻¹. **Off-diagonal weights can therefore arise without a single biological connection between sources.** EEG volume conduction is a real physical example of this ambiguity.

Additional failure modes include ocular/muscle artifacts, common-reference contamination, sample filtering and leakage, unknown noise covariance, interelectrode volume conduction and nonstationarity. Those make neither ordinary OLS nor time-disjoint IV automatically a valid neural causal estimator.

The tiny synthetic process-noise ratios (1–4%) in the earlier wormhole simulator are numerical settings, not estimates of physiological EEG process noise. It is incorrect to claim that a spontaneous scalp EEG trace lives in the exact "2% process+1% sensor" regime without measuring its observation model and excitation structure.

## Next scientific gate, only if this pilot completes

Use independent participants and recording sessions. Compare posterior alpha and generalization across subject/test splits. Evaluate whether any improvement remains after alternative referencing, non-zero-lag controls, independent artifact rejection and spectral-shape-matched surrogate signals. Without robust negative controls, mark the output as **sensor dynamics only**.

A more plausible *physical system ID* experiment uses known stimulus timing and well-characterized channels (steady-state evoked potentials, isolated electrode/MEA recordings or controlled electronics), but even stimulus locking alone does not uniquely identify neural connections.

**Success criterion:** reproducible EEG frequency/mode decomposition and calibrated prediction with a clear measurement-bias alarm, rather than an unsupported reconstructed brain wiring diagram.

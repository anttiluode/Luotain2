# EEG Gate 000 — A real scalp EEG, measured without inventing connections

**Status: completed public-data feasibility pilot, November 22, 2026.**
[Successful GitHub Actions run](https://github.com/anttiluode/Luotain2/actions/runs/45022967372) · [Downloadable complete JSON results](https://github.com/anttiluode/Luotain2/actions/runs/45022967372/artifacts/14754154662) · [Source code](scripts/eeg_physionet_pilot.py).

PhysioNet EEG Motor Movement/Imagery subject **S001**; runs **1: eyes open** and **2: eyes closed**. Both baseline runs are one minute at **160 Hz**, recorded as part of a **64-channel EEG** dataset. For the pilot, eight scalp electrodes were used: Fp1, Fp2, C3, C4, P3, P4, O1 and O2. All recorded EEG traces were real public data; the false-connectivity null was synthetic.

Source: https://physionet.org/content/eegmmidb/1.0.0/

## 1. The expected alpha difference is visible

Welch spectral estimates, averaged across posterior P3/P4/O1/O2 electrodes, using **8–12 Hz power as a fraction of 4–30 Hz power**:

| Recording | Posterior alpha fraction |
|---|---:|
| Eyes open | **0.222644** |
| Eyes closed | **0.702464** |
| Closed/open ratio | **3.15509** |

This is a well-known EEG eyes-closed alpha-power effect, not a new biological mechanism. The sessions are sequential runs, not blinded randomized conditions.

## 2. Held-out signal prediction: OLS beats our IV

Training first **65%** of each run; held-out last **35% after a 1-second buffer**. All values are mean-squared error normalized by the held-out target energy, so smaller is better.

| Model | Eyes open | Eyes closed |
|---|---:|---:|
| Persistence (previous observation) | 0.106847 | 0.143299 |
| Independent per-electrode AR(1) | 0.103669 | 0.138010 |
| Multichannel ridge OLS AR(1) | **0.091577** | **0.123933** |
| Earlier-lag ridge IV AR(1) | 0.101964 | 0.137344 |

Both IV fits were numerically accepted by the declared cross-moment cutoff; condition numbers were **396.98** eyes open and **153.92** eyes closed. This does *not* establish valid causal instrumentation. The eight-channel ridge regression is deliberately a simpler sensor-space observer, not a direct run of the 2-D Luotain2 FFT grammar.

## 3. Exact false-connection demonstration

A separate null world constructed **eight independent AR processes**, with zero true cross-source coupling. A fixed instantaneous mixing matrix created eight observed sensor traces. Regressing the mixed traces produced an observed AR operator with **0.7206 off-diagonal weight fraction**, versus **0.7193** in the exact theoretical mixed operator. Sources never interacted.

For comparison, the **real EEG sensor-space OLS** operator had off-diagonal weight fractions **0.3825** (eyes open) and **0.4158** (eyes closed). **Those percentages are NOT estimates of neural connections.** The synthetic control demonstrates how volume conduction or reference mixing can produce the same qualitative pattern without any anatomical links.

## What this does not show

- EEG scalp sensors measure mixtures of current sources; a sensor-space operator is not a recovered network of neurons.
- IV did **not** beat OLS on these held-out EEG recordings. It has no measured advantage here.
- The fixed-R process-excitation phenomenon found in discrete synthetic fields cannot be transferred as a numerical "1% threshold" to living EEG.
- EEG preprocessing, artifact rejection, referencing changes, source separation and participant-to-participant validation remain untested.
- Two recordings from one volunteer are not evidence of generalizable identification.
- No clinical diagnosis, seizure detection or health interpretation is attempted.

**Useful next rung:** extract multiscale alpha/other-band residues from each channel, inspect reproducibility across subjects and tasks, and test model confidence under channel mixing, reference transformations, ocular artifacts and phase-matched surrogates. The first goal is a reliable *instrument that knows what it can measure*, not a claim that it has decoded hidden synapses.

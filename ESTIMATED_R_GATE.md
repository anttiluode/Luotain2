# Gate IV-005: Unknown sensor variance and weak excitation for nonlinear blind discovery

**Method frozen before evaluation, 22 November 2026 (Claude, Opus 5.5).** Results are appended below the line after the run.

[Gate IV-004](POLY_GATE.md) recovered Allen–Cahn and Burgers blind under 1–8% sensor noise, but it was **told** the sensor variance R and ran at one excitation level (2% process noise). This gate removes the first crutch and probes the second.

## A blind R from the law's own noise echo

[`luotain2/noise_echo.py`](luotain2/noise_echo.py). The sensor noise of frame s appears twice. It enters with weight +1 in that frame's own residual. It enters again inside every later residual whose law reads that frame, through the law's gradient. For Gaussian noise, Stein's lemma gives, for the **true** law,

    cov( r_s(x+o), r_{s+L}(x) ) = − R · E[ ∂p̃ / ∂Y_s(x+o) ]   for every lag L and offset o the law reads.

Process innovations and the law's own prediction error do not enter this covariance. R is the least-squares slope of the measured echo covariances against the measured mean gradients, iterated to a fixed point because the debiased gradients themselves depend on R.

Development checks (seeds 11, 12; not reused below):

- With the **true** law, R̂/R was 0.99–1.02 at 4–8% noise across Allen–Cahn, Burgers, heat, wormhole and the 3-frame echo world. Single trajectories at 1% noise are noisier (up to 1.18).
- A **wrong** law is biased, because its own errors persist over time. A generic 3×3 linear law gave R̂/R of 0 on wormhole and 0–0.8 on Burgers.
- A spatial "nugget" estimator, which extrapolates the pixel variogram to lag 0, was useless here: 1.5–120× too large, because these fields are not smooth at pixel scale.
- **Independence is essential.** A test that accidentally drew the sensor noise from the same random stream as the simulator's own field gave R̂/R of 0–0.4. Sensor noise correlated with the field breaks the estimator.

**Blind procedure:** start from R̂ = 0 (no correction) and select a law. Estimate R̂ from that law's echo on the train and validation trajectories, then select again. Stop when R̂ moves by less than 5%, after at most 4 selections. In development, the R̂ = 0 selection contained denoising copies; its echo still gave R̂/R of 0.74–0.92, the next selection was the true law in 5 of 6 non-null cases, and the third R̂ was 1.00–1.03.

## Design

- **Seeds:** 701, 702, 703 (new). Search settings as in IV-004 (IV, exact correction, full grammar).
- **Part A, miscalibrated R:** the selector is told R̂ = f·R, with f ∈ {0.75, 0.9, 1.0, 1.1, 1.25}. Worlds allen_cahn, burgers, wormhole and null; sensor noise 4% and 8%; 2% process noise. 120 selections.
- **Part B, blind R:** the alternation above, on the same worlds, noise levels and seeds. 24 cases.
- **Part C, weak excitation:** known R, sensor noise 4%, process noise 0, 0.5% and 1%. Allen–Cahn and Burgers, IV vs OLS. 36 selections.

## Predictions, stated before the run

- **A:** underestimating R (f < 1) under-corrects and should re-admit delayed denoising copies on the nonlinear worlds, mostly at 8%. Overestimating R over-corrects; a guess is that it removes weak true terms. A ±10% error should cost little at 4%.
- **B:** blind R should match the known-R results of IV-004 at 4%, with R̂/R within ±5% whenever the final law is exact. On null worlds the law is empty, so the echo is undefined and R̂ stays 0; an empty law is still the correct answer.
- **C:** at 0 and 0.5% process noise these nonlinear worlds still move by their own dynamics, but they relax (Allen–Cahn toward ±1 domains, Burgers decays), so excitation falls. If the excitation cliff Sol found for linear worlds also holds here, IV should beat OLS at low process noise. If the exact score is the bottleneck, they should tie.

No claims for correlated, heavy-tailed or field-dependent noise, real sensors, or other law families.

---

## Results (run 22 November 2026 after freezing; 180 cases, about 1.5 CPU-hours)

Receipts: [`results/r_gate_A.json`](results/r_gate_A.json), [`results/r_gate_B.json`](results/r_gate_B.json), [`results/r_gate_C.json`](results/r_gate_C.json).

### Part A — the selector is told R̂ = f·R (exact support, 3 seeds)

| world | sensor | f = 0.75 | 0.9 | 1.0 | 1.1 | 1.25 |
|---|---|---|---|---|---|---|
| allen_cahn | 4% | 2/3 | 3/3 | 3/3 | 3/3 | **0/3** |
| | 8% | 1/3 | 2/3 | 2/3 | **0/3** | **0/3** |
| burgers | 4% | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| | 8% | 3/3 | 3/3 | 3/3 | **0/3** | **0/3** |
| wormhole | 4% | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| | 8% | 0/3 | 1/3 | 3/3 | 2/3 | **0/3** |
| null (false positives) | 4%, 8% | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |

**The error is asymmetric, and the prediction had the wrong side.** Underestimating R costs a little, as predicted: all 10 underestimate failures are **one extra delayed copy** (lag 2 or 3) with nothing missing. Overestimating R is what collapses the search. At 8%, +10% already gives 0/3 on Allen–Cahn and Burgers, and +25% does the same everywhere. Those laws lose true stencil terms and fill up with delayed, cubed and product junk; one wormhole run picked offsets (±31, ±16). An over-corrected score subtracts too much noise from laws with large coefficients and many noisy inputs, so complex laws look better than they are. **Practical rule: if R is uncertain, err low.** Null worlds stayed empty at every f.

### Part B — blind R from the noise echo (start R̂ = 0)

| | 4% sensor | 8% sensor |
|---|---|---|
| exact support, blind R | **12/12** | 10/12 |
| exact support, known R (part A, f = 1) | 12/12 | 11/12 |
| final R̂/R, non-null worlds | 0.95–1.04 | 0.94–1.01 |
| selections needed | 2–4 | 2–4 |

The blind loop **matched known R within one case**. Its two 8% failures repeat failures the known-R runs already show:

- **Allen–Cahn seed 701** found exactly the same two extra delayed terms as with the true R.
- **Wormhole seed 701** stopped at R̂ = 0.94·R and picked up one extra term, u[t−2](+1,+1). Part A's f = 0.9 run on the same seed picked up that same term.

The first selection at R̂ = 0 always contained delayed copies (7–11 terms). Its echo still gave R̂/R of 0.65–0.97, enough for the next selection to find the true law in most cases. Null worlds: the law stays empty and R̂ stays undefined (reported as 0), which is the correct answer.

### Part C — weak excitation, known R, 4% sensor noise

| process noise | Allen–Cahn IV | Allen–Cahn OLS | Burgers IV | Burgers OLS |
|---|---|---|---|---|
| 0 | 0/3 | 1/3 | **0/3** | **3/3** |
| 0.5% | 0/3 | 1/3 | 3/3 | 3/3 |
| 1% | 1/3 | 1/3 | 3/3 | 3/3 |
| 2% (part A, f = 1) | 3/3 | – | 3/3 | – |

**The prediction was wrong: at low excitation IV is worse, not better.**

- **Burgers without process noise:** IV failed all three, each time swapping u[t−1](0,0) for u[t−2](0,0) (one seed swapped the side neighbours instead). OLS got all three. The field changes little from frame to frame, so the lag-1 and lag-2 copies differ only by that small change. The disjoint-time instruments, being older frames, barely see that difference, while OLS uses the regressors directly.
- **Allen–Cahn below 2% process noise** fails for both estimators, and mostly misses the *same thing*: the whole diffusion stencil and u³, keeping only u[t−1]. Allen–Cahn relaxes into ±1 domains where D∇²u + r(u − u³) ≈ 0, so the stencil and the cube term nearly cancel on the states the data actually visits. That is an identifiability gap in the data, not an estimator choice.

Both are nonlinear cases of what [Theorem](https://github.com/anttiluode/Theorem) calls a dark direction: the rival laws differ only along directions the passive dynamics barely excite. The result also differs from Sol's linear [fixed-R gate](FIXED_R_GATE.md), where IV beat OLS at 1% process noise (6/6 vs 0/6). Low excitation does not favour IV in general; which estimator wins depends on which direction is dark.

**Limits.** Synthetic data, Gaussian i.i.d. sensor noise independent of the field (shown necessary in development), three seeds per cell, two nonlinear families, one grid size and trajectory length. The echo estimator needs a law with at least one lagged term; it is undefined for an empty law.

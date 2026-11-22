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

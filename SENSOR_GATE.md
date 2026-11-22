# Gate IV-006: Toward real sensors

**Method frozen before evaluation, 22 November 2026 (Claude, Opus 5.5).** Results are appended below the line after the run.

[Gate IV-005](ESTIMATED_R_GATE.md) made the noise level blind. Three questions remain before real data: which estimator to use when the data does not say, how to protect against an overestimated R, and what happens when the sensor noise is not Gaussian i.i.d. Driver: [`scripts/run_sensor_gate.py`](scripts/run_sensor_gate.py). Seeds 801–803 (new). Search settings as in IV-004/005.

## Part A — let the data pick IV or OLS

IV-004/005 showed that neither estimator always wins:

- OLS stalls on Allen–Cahn at 8% sensor noise.
- IV confuses lag 1 with lag 2 on slowly changing Burgers.

Rule: run both selections with the same R, then keep the law with the lower **corrected validation risk** (plus the search's own complexity penalty). The score is an unbiased estimate of latent error for either estimator's law, so it should prefer whichever is closer to the truth. Cases: Allen–Cahn and Burgers at 4% sensor noise with process noise 0, 0.5% and 2%, plus Allen–Cahn at 8% with process noise 2%. Known R. 21 cases, each with both estimators.

*Development (seed 13 only):* at 8% Allen–Cahn the rule picked IV (exact; OLS missed). On Burgers with no process noise both were exact. On Allen–Cahn with no process noise both missed.

## Part B — a conservative haircut on the blind R

IV-005 found overestimating R far worse than underestimating it. Here the blind echo loop runs with every R̂ multiplied by h = 0.9 before it is used. Worlds allen_cahn, burgers, wormhole and null; 4% and 8% sensor noise; 24 cases. Comparison: IV-005's h = 1 loop (other seeds), and the cost of f = 0.9 measured there (a few one-extra-delayed-term misses at 8%).

## Part C — sensor noise that breaks the assumptions

Blind echo loop, IV, sensor noise with an average per-pixel std of 4% of the field. 60 cases:

| model | what is wrong with it |
|---|---|
| gauss | nothing (reference) |
| quantized | rounding to a grid: uniform, bounded, a deterministic function of the field |
| poisson | variance proportional to the local light level (signal-dependent) |
| blurred | 3×3 box-blurred white noise: correlated over neighbouring pixels, white in time |
| heavy | Student-t with 4 degrees of freedom: rare large errors |

*Development (seed 13, true law given to the echo estimator):* R̂/R was 1.00 for gauss, 0.68–1.01 for quantized, 1.00 for poisson, **2.1–3.1 for blurred**, and 0.99–1.02 for heavy. Stein's lemma needs only Gaussian-like noise that is independent over pixels. Poisson and t noise pass that test in practice. Spatially blurred noise does not, because the echo then also picks up the noise shared between neighbouring pixels. In one blind development run on blurred wormhole, the found law took the four diagonal neighbours.

## Predictions, stated before the run

- **A:** the rule picks the better estimator in most cases where exactly one estimator succeeds. It cannot help where both fail (Allen–Cahn without process noise).
- **B:** the haircut removes nothing at 4% and costs a little at 8% (one-term misses), because the echo is already accurate. It is insurance against an echo that overshoots, which IV-005 never saw.
- **C:** gauss, poisson and heavy behave like IV-005. Quantized is mostly fine, except possibly Allen–Cahn, whose relaxed ±1 domains make the rounding error systematic. **Blurred fails**: R̂ about 2–3× too high plus spurious neighbour terms. That would be the first clear wall on the way to real cameras, whose demosaicing and compression make noise spatially correlated.

---

## Results (run 22 November 2026 after freezing; 105 cases)

Receipts: [`results/sensor_gate_A.json`](results/sensor_gate_A.json), [`results/sensor_gate_B.json`](results/sensor_gate_B.json), [`results/sensor_gate_C.json`](results/sensor_gate_C.json).

### Part A — the data picks IV or OLS (known R, 21 cases)

| | exact laws |
|---|---|
| IV only | 12/21 |
| OLS only | 14/21 |
| **rule (lower corrected validation risk)** | **15/21** |
| best of the two, chosen with the truth | 17/21 |

- **Exactly one estimator right (8 cases): the rule picked it in 6.** It chose OLS for Allen–Cahn seed 802 (no process noise and 0.5%) and Burgers 802, and IV for the three Allen–Cahn cases where OLS stalled (0.5% process noise, and two of three at 8% sensor noise).
- **The two misses are both Burgers without process noise** (seeds 801, 803). IV's law swaps u[t−1] for u[t−2], and its corrected risk was *lower* than the true law's OLS fit (1.15e-5 vs 1.25e-5, 2.14e-5 vs 2.65e-5). That is the honest limit of the rule: the score measures one-step prediction, and on a field that barely changes between frames the wrong lag predicts as well as the right one. A dark direction cannot be resolved by a better score, only by excitation.
- **Both wrong (4 cases):** Allen–Cahn without or with little process noise, plus one Allen–Cahn 8% run with an extra lag-3 term. No rule can help, as predicted.
- **Both right (9 cases):** the rule kept an exact law every time.

Prediction A held: 6 of 8 cases with one right estimator, and nothing gained where both fail. The rule is better than either fixed choice but not as good as the oracle.

### Part B — haircut h = 0.9 on the blind R̂ (24 cases)

| | 4% sensor | 8% sensor |
|---|---|---|
| exact, h = 0.9 (seeds 801–803) | **12/12** | 9/12 |
| exact, h = 1 (IV-005, seeds 701–703) | 12/12 | 10/12 |
| exact, h = 1 (part C gauss, seeds 801–803) | 12/12 | – |
| final R̂/R, non-null worlds | 0.94–1.01 | 0.79–1.00 |
| false laws on null worlds | 0/3 | 0/3 |

The 4% laws are identical to the h = 1 runs on the same seeds. At 8% all three failures are **one extra delayed term with nothing missing**: u[t−3](0,−1) on Allen–Cahn, u[t−2] and u[t−3](+1,0) on wormhole. That is the underestimate failure seen in IV-005. Prediction B held: the haircut costs nothing at 4% and a little at 8%. **No echo overshot in any of these 105 cases** (largest R̂/R 1.01 outside the blurred runs), so the insurance was never needed here. I would keep h = 1 for synthetic Gaussian data and use 0.9 only where the noise model is in doubt.

### Part C — non-Gaussian and correlated sensor noise (blind R̂, IV, 4%; 60 cases)

| model | exact | final R̂/R (non-null) | null false laws |
|---|---|---|---|
| gauss | **9/9** | 0.94–1.01 | 0/3 |
| quantized | **9/9** | 0.61–0.74 Allen–Cahn, 0.96–0.98 others | 0/3 |
| poisson | **9/9** | 0.94–1.00 | 0/3 |
| heavy (t₄) | **9/9** | 0.96–1.02 | 0/3 |
| **blurred** | **0/9** | 0.00–3.72 | 0/3 |

- **Gauss, poisson, heavy:** as predicted, the same as IV-005. Signal-dependent variance and heavy tails did not matter at this noise level.
- **Quantized:** better than predicted. Allen–Cahn's echo underestimated R by 26–39%, the rounding error being partly a function of the field, as expected. But underestimating is the safe side, and all three laws were still exact.
- **Blurred: the predicted wall, and worse than predicted.** Every non-null world failed.
  - Wormhole (3/3): the same failure each time. The four lag-1 neighbours were replaced by the **four lag-2 diagonal neighbours**. Blurred noise is shared with the neighbours, so the stencil terms carry the noise into the prediction and the corrected score cannot remove it.
  - Burgers seeds 802 and 803: R̂ fell to 0. The law turned into 8–10 lag-1 shifts at distances up to 5 pixels.
  - Allen–Cahn: 4–8 spurious cube, product and delayed terms.
  - R̂/R did not settle near 2–3 as in development (true law given). Starting from a wrong law, it ranged from 0 to 3.7. Null worlds stayed empty.

**What this means.** The blind method holds up with every noise model tested here that is independent across pixels, Gaussian or not. It breaks completely when neighbouring pixels share noise, and it does not refuse: it reports a confident wrong law. Real cameras do share noise between neighbouring pixels, through demosaicing, compression and optics. Gate IV-007 ([`luotain2/noise_kernel.py`](luotain2/noise_kernel.py)) replaces the single variance with a noise kernel C(Δ) estimated from the same echo.

**Limits.** Synthetic, three seeds, noise at 4% (A, C) and 4/8% (B), one blur kernel (3×3 box), process noise 2% except in A.

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

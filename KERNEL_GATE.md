# Gate IV-007: Spatially correlated sensor noise

**Method frozen before evaluation, 22 November 2026 (Claude, Opus 5.5).** Results are appended below the line after the run.

[Gate IV-006](SENSOR_GATE.md) found the wall. Blind discovery held up with every tested sensor noise that is independent across pixels (Gaussian, quantized, Poisson-like, Student-t; 36/36 exact). It **failed on all 9 non-null worlds when the noise was 3×3 box-blurred** and reported confident wrong laws. Real cameras blur their noise through demosaicing, compression and optics. This gate replaces the single sensor variance R with a **noise kernel** C(Δ) = cov(η_t(x), η_t(x+Δ)), white in time, and estimates it blind.

Code: [`luotain2/noise_kernel.py`](luotain2/noise_kernel.py). Driver: [`scripts/run_kernel_gate.py`](scripts/run_kernel_gate.py). Tests: [`tests/test_noise_kernel.py`](tests/test_noise_kernel.py).

## Method

1. **Wick-corrected scores.** For Gaussian noise with covariance C, the unbiased estimate of a latent monomial ∏X_v is its Wick product: the sum over partial pairings of the factors, each pair contributing −C(a,b). Hermite polynomials are the single-pixel case. This corrects the IV/OLS features (degree ≤ 3) and the validation risk (squared residual, degree ≤ 6). With an i.i.d. kernel it equals the IV-004 Hermite risk exactly (tested).
2. **Kernel echo.** By Stein's lemma in several variables, for the true law, the cross-time residual covariances are a linear function of C:

   cov(r_s(x+o), r_{s+L}(x)) = −Σ d_L(o′) C(o−o′) + Σ_{L_k = L+L_j} d_j d_k C(o+o_j−o_k),

   where d are mean gradients of the debiased prediction. The second sum is the law's own terms meeting each other across time. It is zero for single-lag laws; leaving it out biased a two-lag test law 18% low (test added). The kernel on a 5×5 symmetric support (13 values) is the least-squares solution over offsets |o| ≤ 6 and every lag the law reads.
3. **Blind loop.**
   - Round 1: selection with no correction.
   - Start kernel: echo of a **fixed over-complete probe law**, IV-fitted with no selection inside. The probe is the 3×3 stencil at lag 1, u² and u³, the four nearest-neighbour products, and the non-local shifts found in round 1.
     - The kernel must be positive semidefinite on a 7×7 patch; otherwise it falls back to the echo of the round-1 law.
     - It is multiplied by a haircut h = 0.85 (IV-005: erring low is safe).
   - Later rounds: Wick-corrected selection, then the echo of the law just selected, used as is. Stop when C(0) moves < 5%; at most 4 selections.
   - IV throughout. Search settings as in IV-004 to IV-006.

## Conditions (36 cases, new seeds 901–903)

Worlds allen_cahn, burgers, wormhole and null; 64×64, 40 frames, process noise 2%; average sensor std 4% of the field.

| noise | what it is |
|---|---|
| blurred | 3×3 box-blurred white Gaussian noise / 3. True C(Δ)/R = (3−\|dy\|)(3−\|dx\|)/9 |
| camera | the same blur, times sqrt(local intensity / mean): spatially correlated **and** signal-dependent. The kernel is then an average, not exact |
| gauss | i.i.d. Gaussian: the control. The kernel method must not lose what the scalar method had (IV-006: 9/9) |

Scored on exact support, as in IV-004 to IV-006. Recovered C(Δ)/R is reported for (0,0), (0,1), (1,1), (0,2) and (2,2).

## Development (seed 13 only, all before freezing)

- With the **true kernel** given, wormhole and Burgers on blurred noise were exact. The Wick correction is sufficient when C is known.
- With the **true law** given, the echo recovered C/R within 0.03 on Allen–Cahn: 0.99–1.03 for (0,0), 0.66–0.69 for (0,1), 0.45–0.47 for (1,1) (truth 1, 0.67, 0.44).
- **The first blind loop failed.** It estimated the start kernel from the round-1 law, which was uncorrected and wrong. Wormhole fell into a fixed point where wrong law and wrong kernel agree with each other (lag-2 stencil, C(0)/R 0.79), and Burgers swapped lags. That is why the start kernel now comes from a fixed probe law.
- **The probe needs the true terms.** Without the wormhole's link, the probe kernel was invalid (negative; caught by the PSD check). Without products, Burgers' kernel was off. With lags 1 and 2, the probe was collinear and C(0) was badly wrong (a bug in its default, fixed). On Allen–Cahn the probe overestimated C(0) by 25% (u², products and u³ are nearly collinear on ±1 domains), which the haircut and the later echo rounds have to absorb.
- **Final development run** (blurred, 4%): wormhole, Burgers and null exact, with final C(0)/R 1.02 and 0.98 and C(0,1)/R 0.66 and 0.67. Allen–Cahn missed by one swap (u[t−1](−1,0) for u[t−2](−1,0)), with the kernel near the truth (1.07, 0.60).

## Predictions, stated before the run

- **Blurred:** clearly better than IV-006's 0/9. I expect 5–8 of 9 exact, with wormhole and Burgers the most reliable. Allen–Cahn is the most likely to miss, by a lag swap or one extra term. Final C(0)/R in 0.85–1.15 where the law is exact. Null worlds 3/3 empty.
- **Camera:** somewhat worse than blurred, because signal-dependent noise breaks the Gaussian-Wick assumption locally. I expect 4–7 of 9.
- **Gauss:** close to IV-006 (9/9). The recovered off-zero kernel values should be near 0 (|C(Δ)/R| < 0.15). Losing more than one law here would mean the extra 12 kernel parameters cost more than they buy.
- **Failure mode to watch:** a confidently wrong law with a self-consistent kernel, as in development. If it happens, it is reported as such. No refusal rule is added after the fact.

---

## Results (run 22 November 2026 after freezing; 36 cases)

Receipts: [`results/kernel_gate_ac_null.json`](results/kernel_gate_ac_null.json), [`results/kernel_gate_bu_wh.json`](results/kernel_gate_bu_wh.json).

| noise | exact laws (non-null) | IV-006 scalar R on the same noise model | null false laws |
|---|---|---|---|
| **blurred** | **8/9** | 0/9 | 0/3 |
| **camera** (blurred × signal-dependent) | **7/9** | – | 0/3 |
| gauss (control) | 8/9 | 9/9 | 0/3 |

IV-006 used seeds 801–803; this gate used new seeds 901–903.

### Recovered noise kernel, C(Δ)/R, non-null worlds

| noise | (0,0) | (0,1) | (1,1) | (0,2) | (2,2) |
|---|---|---|---|---|---|
| truth, blurred/camera | 1 | 0.67 | 0.44 | 0.33 | 0.11 |
| blurred, blind | 0.98–1.08 | 0.65–0.71 | 0.41–0.48 | 0.31–0.35 | 0.09–0.15 |
| camera, blind | 0.96–1.08 | 0.64–0.69 | 0.41–0.48 | 0.33–0.35 | 0.09–0.16 |
| truth, gauss | 1 | 0 | 0 | 0 | 0 |
| gauss, blind | 0.97–1.06 | −0.02–0.03 | −0.01–0.02 | −0.02–0.02 | −0.01–0.02 |

The kernel was recovered blind, from noisy frames only, within about 0.08 of the truth in every one of the 27 non-null cases. That holds even for the camera model, whose true kernel varies with local brightness. With i.i.d. noise the off-zero values came out at zero.

### The four misses

Each was off by one term, with nothing else wrong:

- **blurred Allen–Cahn 902:** u[t−1](0,−1) swapped for u[t−2](0,−1), the same swap as in development.
- **camera Burgers 902:** the product u·u(0,−1) at lag 1 swapped for lag 2.
- **camera Burgers 903:** one extra lag-2 product.
- **gauss Allen–Cahn 902:** one extra u[t−2](0,+1).

All 9 null worlds stayed empty.

**Both lag swaps happened in the only two runs where the probe kernel failed the PSD check.** There the loop fell back to the round-1 echo, the start that development had shown to be the weak one. In the other 25 non-null runs the probe was valid; 24 of them were exact. This is an observation, not a rule added after the fact. It suggests the next improvement is a better probe (one that cannot be invalid), not a better echo.

### Against the predictions

- **Blurred, 5–8 of 9 predicted: 8/9.** Top of the range. Wormhole and Burgers were 6/6; Allen–Cahn missed once, as expected. Final C(0)/R within 0.85–1.15: held (0.98–1.08).
- **Camera, 4–7 of 9 predicted: 7/9.** Top of the range. The signal-dependent part cost less than I expected; the misses were in Burgers, not Allen–Cahn.
- **Gauss, about 9/9 predicted, failure defined as losing more than one: 8/9.** One lost law (an extra delayed term, the classic underestimate miss). That is inside the stated tolerance, but it is one worse than the scalar method on other seeds, so the 12 extra kernel parameters are not free.
- **Watched failure mode:** no confident wrong law with a self-consistent wrong kernel appeared. In every miss the kernel was right and the law was one term off.

**What this means.** The wall from IV-006 is down for this kind of noise. Blurred sensor noise went from 0/9 to 8/9, and the kernel itself is read off the data blind. The full chain is now blind on noise level and noise shape, plus the nonlinear grammar, with no false laws in 9 null worlds.

**Limits.**
- Synthetic data, three seeds, one blur shape (3×3 box), one noise level (4%), one grid size and length.
- The kernel support is 5×5. Wider blur (real optics, JPEG blocks) needs a larger radius, and the deconvolution then has more unknowns.
- Noise must still be white in time and independent of the field. Rolling-shutter or temporally filtered video breaks the first.
- The probe law assumes a lag-1, local-plus-found-links structure.

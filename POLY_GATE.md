# Gate IV-004: Can the blind search find nonlinear laws under sensor noise?

**Method frozen before evaluation, 22 November 2026 (Claude, Opus 5.5).** Results are appended below the line after the run; nothing above the line was changed after seeing them.

Blind IV-002/003 search linear shifted copies only. Their validation score subtracts R(1 + Σβ²) from the observed error, which is exact for distinct linear inputs and wrong for u², u³ and products. Allen–Cahn (u³) and Burgers (u·u) were therefore never evaluated blind.

## What is new

[`luotain2/blind_poly.py`](luotain2/blind_poly.py) runs the same search (FFT residue proposals, mirror groups, local stencil groups, exchange and pruning moves) over the **full grammar**: id, square and cube shifts, plus products within radius 1. It also adds:

1. **Debiased fits.** IV and OLS regressors are Y² − R, Y³ − 3RY and same-pixel products minus R. These corrections already existed in `iv.py` but were never used by the blind search.
2. **An exact corrected validation risk for any polynomial law.** Under Y = X + η, with η i.i.d. Gaussian of known variance R, the latent squared one-step error (X_t − c₀ − Σβ_j g_j(X))² is a polynomial in the latent pixel-times it touches. Each monomial Π X_v^{k_v} has the unbiased estimator Π He_{k_v}(Y_v; R), where He are the scaled Hermite polynomials (He₀ = 1, He₁ = y, He_{k+1} = y·He_k − kR·He_{k−1}). Expanding the square and replacing every monomial gives an unbiased latent risk from observed frames only. For linear laws on distinct pixels it reduces *algebraically* to the IV-002 formula (`tests/test_blind_poly.py`).

This is the corrected-score idea (Stefanski; Nakamura; the polynomial corrections in WAIVES) applied to a discrete field law and to **selection**, not only to coefficient fitting. It is not WAIVES and inherits none of its theorems.

## Design

- **Worlds:** allen_cahn (6 true terms incl. u³), burgers (7 true terms incl. two products); controls heat, wormhole (linear) and null (no law).
- **Seeds:** 601, 602, 603 (new; development used seed 11 only).
- **Noise:** latent paths from the original simulator with 2% process noise. Independent Gaussian sensor noise with std = ratio × latent std, ratio ∈ {1%, 2%, 4%, 8%}. R is known to the selector. Paths are **not** rescaled, because rescaling changes u³ and u·u coefficients.
- **Arms (2 × 2):** estimator {IV, OLS} × correction {exact = debiased fits + Hermite risk, linear = IV-002 recipe applied unchanged to every term}. Everything else matched.
- **Search settings:** max_lag 3, k_props 12, k_products 4, product radius 1, max_terms 12, δ 0.003, pixel stride 2, max condition 1e5.
- **Truth:** support and coefficients are read only by the scorer. Exact = the found term set equals the true set.
- **Count:** 5 worlds × 3 seeds × 4 noise levels × 4 arms = **240** blind selections.

## Predictions, stated before the run

- The exact correction beats the linear recipe on the nonlinear worlds at 4% and 8%. In development (seed 11 only) the linear recipe failed all 4 nonlinear cases at 4%, choosing lag-2 and lag-3 denoising copies, while the exact correction recovered all 4.
- At 1% all arms tend to succeed, because the noise is too small to matter.
- At 8% the exact arms start to fail. Development showed Burgers' products u(0,0)·u(0,±1) replaced by u(0,±1)² under IV, a near-alias at large noise.
- On heat and wormhole the two corrections should agree, unless the linear recipe admits nonlinear junk.
- Null: any found term is a false positive.

No claim is made for correlated sensor noise, unknown R, heavy-tailed noise, other nonlinear families, or real data.

---

## Results (run 22 November 2026, after freezing; 240 blind selections, ~1.1 CPU-hours)

Receipt: [`results/poly_gate.json`](results/poly_gate.json), every case with found law, missing/extra terms, corrected test risk, clean-test NMSE and coefficient error.

Exact support recovered (3 seeds per cell):

| world | sensor noise | IV + exact | OLS + exact | IV + linear recipe | OLS + linear recipe |
|---|---|---|---|---|---|
| allen_cahn | 1% | 3/3 | 3/3 | 3/3 | 3/3 |
| | 2% | 3/3 | 3/3 | 3/3 | 2/3 |
| | 4% | **3/3** | **3/3** | 0/3 | 0/3 |
| | 8% | 2/3 | 0/3 | 0/3 | 0/3 |
| burgers | 1% | 3/3 | 3/3 | 3/3 | 3/3 |
| | 2% | 3/3 | 3/3 | 3/3 | 3/3 |
| | 4% | **3/3** | **3/3** | 1/3 | 3/3 |
| | 8% | 2/3 | **3/3** | 0/3 | 0/3 |
| heat | all | 12/12 | 12/12 | 12/12 | 12/12 |
| wormhole | 1–4% | 9/9 | 9/9 | 9/9 | 9/9 |
| | 8% | 2/3 | 1/3 | 2/3 | 1/3 |
| null | all | 12/12 clean | 12/12 clean | 12/12 clean | 12/12 clean |
| **nonlinear total** | | **22/24** | **21/24** | 13/24 | 14/24 |
| **linear total** | | 23/24 | 22/24 | 23/24 | 22/24 |

**What held:**

1. **The exact correction is what makes nonlinear laws findable under sensor noise.** On Allen–Cahn and Burgers it recovers 22/24 (IV) and 21/24 (OLS) exact laws, against 13/24 and 14/24 for the linear recipe. The whole difference sits at 4% and 8%: the linear recipe recovers 0 of 18 cases at 4% Allen–Cahn and at 8% in both worlds.
2. **The linear recipe fails the way predicted.** Its 107 wrong extra terms are mostly delayed copies: 62 at lag 2 and 26 at lag 3. They average away sensor noise and look better under a score that does not correct the nonlinear terms' noise.
3. **On linear worlds the two corrections tie** (23/24 vs 23/24 for IV), as expected: for linear laws they are algebraically the same score. The full grammar did not admit nonlinear junk into heat or wormhole. **Null: 0 false positives in 48 selections.**
4. **Coefficients, when the support is exact:** the median relative coefficient error for IV + exact grows from 0.3% (Allen–Cahn) and 1.0% (Burgers) at 1% noise to 4.4% and 8.5% at 8%.

**What did not go as predicted:**

- **OLS + exact matched IV.** It was even better on Burgers at 8% (3/3 vs 2/3), with lower Burgers coefficient error (2.1% vs 8.5% at 8%). With 2% process noise these worlds are well excited, and the selection score, not the estimator, was the bottleneck. This differs from Sol's fixed-R linear gate, where IV won at marginal excitation. Both can be true; this gate did not test low excitation.
- **OLS + exact failed all of Allen–Cahn at 8%,** in the same way each time: the search stopped after one term (u at the same pixel), never taking the stencil or u³. Errors-in-variables attenuation shrinks the OLS coefficients, so no single addition clears δ. IV recovered 2/3 there.
- **The predicted Burgers alias did not occur on these seeds.** Development seed 11 at 8% replaced the products with u(0,±1)². On seeds 601–603, the only exact-arm Burgers failure was one extra delayed product.
- The remaining exact-arm failures at 8% are all **one extra lag-2 term**. Allen–Cahn IV, Burgers IV and all three wormhole failures each had exactly one extra delayed term and nothing missing. The leftover denoiser pull at 8% is therefore about one term.

**Limits.** Gaussian i.i.d. sensor noise with known R; two nonlinear families from the same simulator; one process-noise level (2%); three seeds per cell. Not tested: unknown or misestimated R, correlated or heavy-tailed noise, low excitation, other nonlinearities (e.g. rational or non-polynomial), or real data. The Hermite correction is exact only for polynomial terms. "Exact support" means equality with the generating simulator's term list.

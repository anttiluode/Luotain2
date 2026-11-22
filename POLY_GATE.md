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

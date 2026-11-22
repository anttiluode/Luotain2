# IV Gate 002 — Blind field-law discovery under measurement noise

**Status: exploratory and separately benchmarked. Original discover.py is unchanged.**

The original WAIVES paper (Ruzayqat, Bortz, Dukić, 2026, arXiv:2611.23951v2) motivates disjoint-time instrumental variables. This implementation applies a discrete-time IV principle to Luotain2 and does *not* reproduce the paper's continuous-SDE weak-form estimator or its consistency proofs.

## What it does

- True terms and coefficients are **not passed** into the search. The search is limited to linear shift terms, with lag 1–3 and arbitrary spatial offsets within its margin. Nonlinear Burgers and Allen–Cahn law discovery are explicitly excluded.
- Reuses the original FFT residue proposer, which can suggest any spatial offset.
- Competes those suggestions against **declared conventional local stencils**. Success on heat and drift can come from this library prior, not necessarily open-grammar discovery.
- Fits each proposed support with earlier, time-disjoint IV observations. Badly conditioned instruments can be refused.
- Selects using independently held-out noisy frames, adjusting the observed-pixel loss by the independent sensor-noise variance R(1+sum beta_j²), with a fixed sparsity penalty.
- This corrected loss is justified only for *linear distinct observations with independent additive sensor noise*, not for nonlinear candidates or temporally correlated cameras.
- A third trajectory and the hidden generating law are reserved solely for evaluation, never for fitting or proposal.

## Pilot and development tuning (not independent evidence)

First pilot on seed 101 heat/drift/wormhole/null with FFT-only IV: **1/4 exact** (only null). The solver selected delayed-copy surrogates and even dropped a real wormhole link. We added the local-stencil challenger, penalized model selection and swap/exchange moves. Same development seed: **3/4 exact** (heat, drift, null; wormhole still missed). These are exploratory improvements *after examining the pilot*.

## Frozen independent evaluation

Before inspecting the next outcome, the method was frozen and a separate GitHub Actions workflow launched:

- Fresh seeds 201, 202, 203, 204, 205; heat, diagonal, drift, wave, wormhole, echo, null: **35 cases = 30 non-null linear + 5 null**.
- Combined 2% process plus 1% measurement noise, seeds 201 and 202 on heat, wave, wormhole, echo, null: **10 cases**.
- Fixed 64x64 fields, 40 frames, max 12 terms, 12 FFT proposals per pass.

See the scripts/run_blind_iv_gate.py entrypoint and .github/workflows/blind_iv.yml; JSON run artifacts contain every fitted law, selected term list, exact-support verdict, validation/test scores and identifiability refusals.

**Never equate this 35-case denominator to the original 35 *non-null* measurement-noise cases.** The original includes nonlinear Allen–Cahn/Burgers while the new benchmark includes a lag-three echo and five null trials.

The meaningful distinguishing targets are long-range **wormhole (+17,-9)** and delayed **echo (lag 3, -11,+23)**. Exact local-stencil recovery alone does not establish new scientific methodology.

## Reproduce

    pip install numpy scipy pytest
    python -m pytest -q tests
    python scripts/run_blind_iv_gate.py --worlds heat drift wormhole null --seeds 101 --max-terms 12 --k-props 12

Do not retune on fresh-test seeds without declaring a new exploratory phase and freezing another separate holdout. Process-noise failures, false positives and missing nonlocal links remain in the denominator.

## Frozen fresh-seed results — November 22, 2026

**[Successful workflow and complete downloadable row-level JSON receipts](https://github.com/anttiluode/Luotain2/actions/runs/45019980713)** (artifact name: blind-iv-pilot).

This was the first evaluation of the revised selector on previously unseen
seeds 201–205, after all settings were tuned on the separate seed-101 pilot.

| World | Exact supports, 1% independent sensor noise |
|---|---:|
| Heat | **5/5** |
| Diagonal diffusion | **0/5** |
| Drift | **5/5** |
| Wave | **5/5** |
| Wormhole (+17,-9, lag 1) | **0/5** |
| Delayed echo (-11,+23, lag 3) | **5/5** |
| Null | **5/5** |

**Total 25/35** including null: **20/30 non-null linear** and **5/5 null
correct**. Do not compare the denominator directly with Luotain2's original
35 *non-null* trials, which include nonlinear worlds and exclude the echo's
measurement-noise condition.

The **diagonal** failures are repeatable, not random scatter: the recovered
model has two dominant lag-one diagonal terms (~0.238), an inflated
self-term (~0.469), and two small lag-two opposite-diagonal terms (~0.0275).
It omits the four true immediate cardinal diffusion terms. The **wormhole**
failure is similarly systematic: about **+1.77*u[t-1] -0.79*u[t-2]** and
a tiny false lag-two spatial term (instead of six true terms including
the +17,-9 remote link). This is a concrete **predictor/law
non-identifiability** problem in low-excitation smooth fields, not a
missing coefficient solver.

### Combined-noise stress (same frozen settings)

With **2% process noise plus 1% sensor noise**, seeds 201–202:

| World | Exact |
|---|---:|
| Heat | 2/2 |
| Wave | 1/2 (one run included an extra diagonal term) |
| Wormhole | **2/2** |
| Echo | 2/2 |
| Null | 2/2 |

**9/10 exact.** The wormhole's remote coefficient was **0.09976** and
**0.10058**, compared with generating truth **0.10**, after the search
found its spatial address. No oracle support was provided to the search.

A plausible interpretation is **persistent process excitation**:
independent disturbances prevent the dynamics from falling rapidly into a
smooth low-dimensional subspace in which different lag/shift combinations
look equivalent. The two-seed result does **not** establish that causal
mechanism. A frozen amplitude sweep with independent seeds is needed
before drawing that conclusion. Further, the combined-noise trial
is **not a replication of WAIVES** or its continuous SDE results; in our
discrete autoregression IV can have greater variance than OLS.

**Research status:** useful synthetic method; no novel theorem, no natural
sensor validation, no claim to general nonlinear law discovery.

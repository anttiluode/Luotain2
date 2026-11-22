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
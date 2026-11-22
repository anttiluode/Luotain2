# Luotain2 / IV Gate 001 — Is measurement noise hiding the true law?

**Status: experimental, oracle-support only. This does NOT yet solve the original 1% observation-noise law-discovery failure.**

Source inspiration: Hamza Ruzayqat, David M. Bortz and Vanja Dukić,
*WAIVES: Weak-form Asymptotically-unbiased Instrumental-Variable Estimation
for SDEs*, arXiv:2611.23951v2 (November 2026).

The scientific distinction matters. WAIVES proves consistency for certain
**continuous-time Itô stochastic differential equations** using integrated
weak-form regressors and temporally disjoint instruments. Luotain2 models
**discrete-time spatial fields** whose next frame is a function of earlier
frames; its data can contain independent noise added only to the camera
observation. Our estimator below uses the classical instrumental-variable
principle and a discrete average. It is **not WAIVES**, does not inherit
WAIVES's asymptotic results, and is not an SPDE estimator.

## Why this experiment exists

The original Luotain2 benchmark (see [RESULTS.md](RESULTS.md)) recovers
44/45 exact known laws + one equivalent under process noise at each of two
levels. With **1% observation noise**, it recovers **no exact non-null
generating law** on its 35 tested non-null world/seed trials. Ordinary
prediction can improve by mixing noisy neighboring or historical
observations even when that produces the wrong generating equation.

We must now test two propositions separately:

1. **Coefficient estimation:** when someone has told us the exact list of
   terms, can a time-disjoint instrument fit their coefficients more
   accurately than OLS? **Gate IV-001 tests only this.**
2. **Unknown law discovery:** can Luotain2's residue/FFT proposals be
   selected and validated under measurement noise without learning spurious
   denoisers? **Unanswered.** Do not merge this experiment into Discoverer
   until the coefficient test earns that promotion.

## A simple geometric picture

At equation time t, current output Y[t] is explained by terms such as
Y[t-1] at different spatial offsets. If the earlier observed image
contains a camera-noise speck, OLS partially explains it as though it were
part of the world's actual dynamics. Instead, ask a more distant frame
about the same term. It is still correlated with the underlying world, but
its *separate sensor speck* cannot be the same random noise as the one
being explained.

For a discrete time window of W consecutive equations and max candidate
delay L, we set the instrument shift

    shift = L + W - 1 + gap.

For each equation time s in the target window, we construct:

    b = mean_s observed_state[s]
    A_j = mean_s debiased_candidate_j(observed_state, s)
    Z_j = mean_s debiased_candidate_j(observed_state, s - shift)

The *latest* sensor reading of an instrument occurs before the *earliest*
sensor reading used by any regressor, including terms with long delays.
With one candidate term, b ≈ A beta + error, and

    beta_OLS = least_squares(A, b)
    beta_IV  = solve(Zᵀ A, Zᵀ b).

The intercept is fitted as its own column. We only solve if the normalized
moment matrix has enough rank and an acceptable condition number; otherwise
the estimator explicitly returns **not identified**. This numerical
condition check is not a Cragg–Donald first-stage F statistic. Neighboring
pixels are highly correlated and should not be counted as independent
instrument replicates.

The method assumes i.i.d. sensor noise over space and time, known or
calibrated measurement variance, and a correctly specified discrete
candidate support. It does not fix model misspecification, aliases,
unobserved states, correlated detector noise or nonstationarity.

## Nonlinear sensor-noise corrections

The manuscript's Gaussian corrected-score idea carries over algebraically
under the assumed observation model Y=X+eta with variance R:

- Quadratic: Y² → Y² − R.
- Cubic: Y³ → Y³ − 3 R Y.
- Products at *the same* pixel/time: YiYj → YiYj − covariance(eta_i,eta_j).
- Different pixels under the i.i.d. spatial-sensor model have zero cross
  covariance, so their product needs no subtraction.

These are implemented in [luotain2/iv.py](luotain2/iv.py), not in the old
[luotain2/grammar.py](luotain2/grammar.py); the normal discovery
algorithm remains unchanged.

**Important: we have NOT copied the paper's six-lag variogram.** The
paper's short-lag quadratic/intercept expansion is for a continuous-time
diffusion. A field making an entire discrete update per frame generally
does not satisfy that expansion. The 1% benchmark explicitly knows its
corruption ratio; Gate IV-001 calibrates sensor variance using that known
ratio and the observed trajectory standard deviation. This is not a
blind noise-variance estimate.

## Reproduce

    pip install numpy scipy pytest
    python -m pytest -q tests
    python scripts/run_iv_gate.py --worlds heat diagonal wave wormhole allen_cahn burgers drift --seeds 101 102 103 --out results/iv_gate.json

The new CI workflow [iv_gate.yml](.github/workflows/iv_gate.yml) runs
the above 21 comparison trials on the original 1% observation-noise
generator and uploads the JSON result as a workflow artifact. The gate
reports OLS/IV coefficient distance from the **supplied** true law and
prediction on a separately simulated **clean** synthetic trajectory.
That clean trajectory is used only for evaluation, never training.

The 1% baseline in RESULTS.md has 35 non-null cases (seven worlds × five
seeds), while the initial CI diagnostic uses only 21 of these. Run the
script with all five seeds for the full coefficient-estimation audit.
Neither run can count as an exact discovery because the support was
supplied.

## Stop signs and promotion criteria

1. **No improvement over matched OLS:** record the failure, do not inject
   IV into the open-grammar search.
2. **IV improves coefficient recovery but is often weak/refused:** run
   finite gap/window sweeps, report attempted and refused conditions, and
   check independently held-out clean dynamics; do not silently exclude
   failed windows from the denominator.
3. **IV improves established supports across the original worlds:** then
   adapt proposal/acceptance scoring to IV moments, with separate
   training/validation/test trajectories and sample-split noise
   calibration, and retest all 35 old measurement-noise cases. Crucially,
   exact support, not merely low prediction error, is the endpoint.
4. **If combined process and observation noise are introduced:** generate
   and evaluate both kinds of noise explicitly; paper's process-noise
   exogeneity applies only under its stochastic assumptions, not by
   analogy alone.
5. **If false terms pass or weak instruments are frequent:** preserve the
   original Luotain2 as the valid positive control, and mark the new
   estimator unidentifiable.

The next gate should also probe long-range wormholes and lag-three
echoes, as these are what distinguish Luotain2's open grammar from a
small pre-enumerated local stencil.

**Bottom line:** estimate the right known law first; only then try to
discover the unknown one.

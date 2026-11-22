# Gate IV-003 — Does excitation reveal the law, or merely improve SNR?

**Method frozen before evaluation, 22 November 2026.**

The previous *relative-noise* excitation sweep reached exact recovery 0/6 without process disturbances and 6/6 at process ratio 1%, but the simulator scaled the observation noise with the state variance. Changing process noise therefore changed both the latent spatial spectrum and absolute sensor noise. The earlier sweep cannot by itself isolate persistent excitation.

This experiment holds both the **absolute sensor standard deviation** and the **global latent state standard deviation** constant across excitation settings and compares **IV and OLS blind support search with every other component matched**.

- **Worlds:** wormhole, diagonal diffusion (both previously failed without process excitation)
- **New seeds:** 501, 502, 503
- **Process noise levels:** 0, 0.005, 0.01, 0.02, 0.04
- **Fixed sensor noise:** independent Gaussian noise with absolute standard deviation **0.01** (variance **R = 0.0001**)
- **Matched latent power:** rescale each simulated space-time field to standard deviation **0.25** before adding sensor noise; the linear world's support and coefficients are invariant under this scalar multiplication
- **Effective noise-to-total-field ratio:** **4%** in every setting. This is a deliberately more demanding sensor condition than the previous 1% relative-noise trial and cannot be combined with its results
- **Search:** unchanged linear-shift FFT residual proposals plus declared conventional local stencils, fixed sparsity penalty and exchange moves
- **Estimator:** disjoint-time IV or aligned matched-row OLS from the *same* moment-matrix builder; **both** use exactly the same corrected validation risk and train/validation/test separation
- **Truth:** support and coefficients withheld until scoring; this is still a restricted synthetic simulation with known generator family
- **Count:** 2 worlds × 3 seeds × 5 process levels × 2 estimators = **60** blind selections.

[Workflow](.github/workflows/fixed_r_ablation.yml) · [selector](luotain2/blind_iv.py) · [driver](scripts/run_blind_iv_gate.py) · [regression tests](tests/test_fixed_r_ols_iv.py).

**Why normalize total field power?** Holding R fixed alone would let increasing process noise increase the amount of latent signal, making higher SNR a sufficient alternative explanation. Normalizing space-time standard deviation keeps overall signal power matched. It does *not* guarantee identical mode-by-mode power spectra, temporal persistence, spatial rank, or effective instrumental relevance. Those differences are what the hypothesis is about.

**Mechanistic predictions, without an after-the-fact threshold:**

- If both methods improve sharply at nonzero excitation, signal excitation is likely the main bottleneck.
- If IV improves while matched OLS does not, disjoint-time instruments are helping beyond excitation and corrected validation.
- If neither improves at fixed R and matched power, the previous cliff may have been primarily a measurement-SNR effect.
- If results vary by seed and model, no universal 1% transition should be claimed.

A true additional ablation would compare the same corrected score with IV/OLS under candidate grammars **with and without** the hand-declared four-neighbor stencil, then transfer across wholly different linear field generators and real sensors. The existing EEG pilot is not equivalent: scalp electrodes measure mixtures of hidden sources, not a periodic 2-D spatial law.


## Verified results — both OLS and IV jobs passed

[Successful paired workflow](https://github.com/anttiluode/Luotain2/actions/runs/45022617505) · [IV per-case receipts](https://github.com/anttiluode/Luotain2/actions/runs/45022617505/artifacts/14753649828) · [OLS per-case receipts](https://github.com/anttiluode/Luotain2/actions/runs/45022617505/artifacts/14755120626) · [Persistent JSON summary](results/fixed_r_ols_iv_summary.json).

| Process-noise ratio | IV exact support | OLS exact support |
|---|---:|---:|
| 0% | 0/6 | 0/6 |
| 0.5% | 3/6 | 0/6 |
| 1% | **6/6** | 0/6 |
| 2% | **6/6** | **6/6** |
| 4% | **6/6** | **6/6** |
| **Total** | **21/30** | **12/30** |

The 0.5% IV row comprises two wormhole and one diagonal exact
recoveries. At 1%, IV recovered both worlds for all three seeds;
matched OLS recovered neither. At 2% and 4%, both recovered every law.
All 30 IV cases and all 30 OLS cases were evaluated under **the same
fixed absolute sensor standard deviation 0.01 and matched global latent
SD 0.25**; the per-row receipts confirm the latter to numerical precision.

**What this supports:**

1. The synthetic wormhole and diagonal law identification failures are
   sensitive to how the system is excited even when overall signal
   variance and sensor-noise variance are controlled.
2. With identical simulated inputs, disjoint-time IV outperforms
   matched-row OLS on exact support recovery in the marginal-excitation
   setting tested. In this model family, IV succeeded at 1% process
   noise and OLS required 2%.
3. As excitation increases further, the advantage vanishes: both
   methods reach 6/6.

**What it does NOT support:** a universal process-noise threshold,
unbiased estimation of any general SDE/SPDE, automatic selection of
nonlinear laws, or the inference of neural wiring from scalp EEG.
Global variance matching cannot control the spectral shape or
condition of the instrument matrix. Only two designed field generators,
three independent seeds each, a fixed known observation variance and
one declared local-stencil proposal prior were tested. The first fitted
residual can differ between IV and OLS, so although the algorithmic
proposal rule is identical, the exact candidate sequences later on
need not be.

**Next scientific diagnostic:** measure the eigen/singular spectrum
of the centered instrument–regressor cross moment for the *true*
candidate support, evaluate before the fitter sees the true terms.
Examine the signal's mode-by-mode excitation and compare it to the
frequency bands where the wormhole's shifted feature carries
distinguishing information. Then repeat with novel random offsets,
random law coefficients and correlated sensor noise, including false
connection controls. A stable observation-dependent confidence
boundary would be a more useful instrument than a single impressive
success percentage.

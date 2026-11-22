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

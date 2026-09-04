Revise and extend the neural-timescales paper/codebase in the following order. Keep the implementation simple and produce publication-quality figures/results. Do not rewrite the manuscript until the analyses are complete. Rather than editing existing files, make sub-module or sub-directory to self contain all of your work. Start by reading /home/rph/timescale-methods/paper.tex for context. Then continue:

## 1. Make AR(p) interpretable

Develop a rigorous interpretation of higher-order AR models in terms of poles/modes.

For

$$
x_t=\sum_{k=1}^p \phi_k x_{t-k}+\epsilon_t,
$$

compute the poles \(\lambda_k\) and characterize each mode by:

$$
\tau_k=-\frac{1/f_s}{\log |\lambda_k|}
$$

and, for complex poles,

$$
f_k=\frac{f_s|\arg(\lambda_k)|}{2\pi}.
$$

Important:

* Combine complex-conjugate poles into one damped oscillatory mode.
* Clarify that the AR(p) PSD denominator factorizes into first-order pole factors; do **not** claim the signal is literally a product of independent AR(1) processes.
* Develop a way to identify the **dominant subset of modes/timescales** in a fitted AR(p). Rank modes by their importance to reconstructing the full fitted PSD, e.g. remove each real pole or conjugate pair, refit overall scale, and measure degradation in log-PSD error or likelihood.
* Do not call this “percent variance explained” unless mathematically justified.
* Compare dominant-mode interpretation against simply using raw \(\phi_k\) coefficients.
* Add tests showing exact reconstruction of the original AR(p) PSD from its pole factorization.

The desired output of an AR(p) fit should be interpretable approximately as:

`[(tau, frequency, importance), ...]`

rather than `[phi1, ..., phip]`.

## 2. Build a systematic simulation benchmark

Compare three standard single-timescale estimators:

1. direct time-domain AR(1)
2. AR(1) spectral/PSD fitting
3. exponential/AR(1) ACF fitting

Also fit AR(p), with model order selected objectively where useful, to determine when AR(1) is inadequate.

Simulate:

1. **AR(1)**
   Positive control. All methods should recover the known \(\tau\).

2. **AR(1) + sinusoid**
   Test oscillatory contamination of timescale estimation.

3. **AR(2) damped oscillator**
   Test a process whose dynamics intrinsically contain a complex pole pair. Show that AR(p) separates decay time and oscillation frequency.

4. **Sum of two independent AR(1) processes**
   Known \(\tau_0,\tau_1\). Test what a generic AR(p) representation does when the true process contains multiple additive relaxation timescales. Do not assume fitted AR(p) poles must recover the two latent taus exactly.

5. **Power-law / long-memory process**
   Negative control where a single characteristic \(\tau\) is fundamentally questionable. Show how AR(1)/ACF/PSD timescale estimates become model- or observation-window-dependent.

6. **Finite-sample truncation**
   For the above models, systematically shorten the recording. Parameterize this primarily by recording duration relative to true timescale, \(T/\tau\).

These forms should all have _theoretical_ PSD forms. The only form that does not have a clearn form is the finite-sample truncation. This should be performed by computing a confidence interval around the true phi/tau (analytical form), and sampling deviation from ground truth. The shorter the signal, the wider the confidence and noiser the sampling should be.

For each relevant simulation quantify:

* estimation bias
* relative/absolute error
* variability across repeated simulations
* goodness-of-fit
* AR model order/model mismatch
* stability as recording duration decreases
* dominant AR(p) modes where applicable

Produce a small number of figures that clearly show **when a scalar neural timescale is recoverable and when it becomes misleading**.

## 3. Apply the framework across diverse real neural signals

See here for how to load the datasets: /home/rph/oscillatory_eigenvectors/notebooks/06_multimodal_eigenspectra_2s.ipynb
The /home/rph/oscillatory_eigenvectors repository contains all data and loading functions. Use the existing dataset registry there to demonstrate that these estimation problems are not specific to one species, modality, or oscillation.

Prioritize these default datasets:

* `human_lemon_alpha` — human scalp EEG, resting posterior alpha
* `human_cole_m1_beta` — human M1 ECoG, beta activity
* `human_ssvep_12hz` — human EEG, strong externally driven 12-Hz oscillation
* `mouse_neuropixels_natural_movie` — mouse Neuropixels LFP
* `rat_medial_septum_theta` — rat theta
* `rat_hc3_ca1_ca3` — rat hippocampal CA1/CA3 LFP
* `macaque_visual_grating_ecog` — macaque visual ECoG

Optional validation datasets already in `DATASET_REGISTRY` can be added only if they contribute a genuinely different signal regime. You decide.

For each dataset:

* fit ACF AR(1), PSD AR(1), direct AR(1), and selected AR(p)
* quantify model fit/mismatch
* extract interpretable AR(p) modes
* determine whether a single dominant relaxation timescale is adequate
* visualize representative PSD/ACF/model fits consistently across datasets

The goal is **not** to claim biologically novel effects in every dataset. Use the diversity to demonstrate how oscillations, model mismatch, and multiple dynamical modes appear across real neuroscience modalities. So the goal is to find datasets that have these deviations: oscillations, AR(1) model is insufficient, in fact, this notebook shows that high-order AR(p) forms are required for sufficient fits: /home/rph/oscillatory_eigenvectors/notebooks/06_multimodal_eigenspectra_2s.ipynb

## 4. Add empirical finite-sampling analysis

For sufficiently long real recordings, repeatedly truncate/window the signal over increasing durations.

Measure:

* convergence of estimated \(\tau\)
* between-window variance
* differences between ACF, PSD, and direct AR estimators
* stability of AR(p) model order and dominant modes

The practical question should be:

**How much data is required before a neural-timescale estimate becomes stable?**

## 5. Write these sections in a new .tex file

Reorganize the paper (/home/rph/timescale-methods/paper.tex) around the scientific problem rather than as an AR tutorial. Write to a new .tex file. Preserve the old paper.tex. Write in the style of paper.tex.

### Introduction

* Neural timescales as biologically interpreted quantities
* Examples of their use in hierarchy, cognition, state, etc.
* Central problem: estimates depend on finite sampling, oscillations, stationarity, and model assumptions
* Introduce AR models as a framework for exposing/modeling these assumptions, not as a novel invention

Remove/reduce the long temperature/AR primer.

### Results

Suggested organization:

1. **When is a single neural timescale identifiable?**
   Core simulation benchmark.

2. **Oscillations and richer dynamics violate the single-timescale model**
   AR(1)+sinusoid, damped oscillators, power laws, mixtures.

3. **Interpreting higher-order autoregressive models**
   Poles, decay times, oscillation frequencies, pole-factorized PSD, dominant-mode extraction.

4. **Finite recordings bias neural-timescale estimates**
   Simulation + empirical truncation.

5. **Real neural signals frequently deviate from a single-timescale model**
   Cross-species/cross-modality dataset benchmark.

### Methods

Move mathematical derivations, fitting details, simulations, dataset descriptions, model selection, and diagnostics here.

### Discussion

Center the interpretation around:

**A neural timescale is a model-dependent summary of temporal dynamics, and a single scalar \(\tau\) is only meaningful when the corresponding model assumptions are adequate.**

Discuss when AR(1) is sufficient, when richer AR dynamics are needed, and when even AR(p) should be regarded only as a descriptive approximation rather than recovery of literal latent biological timescales.

## General constraints

* Preserve the existing useful theory where correct.
* Remove the current Weierstrass/universal-approximation argument.
* Check all AR equations carefully for sign/polynomial conventions.
* Distinguish clearly between spectral poles/modes and independent latent processes.
* Prefer interpretable figures over adding many analyses.
* Keep the code modular so every simulation, estimator, and dataset uses the same fitting/evaluation API.
* Do the work sequentially in the order above; use results from each stage to determine the exact manuscript claims in the final rewrite.

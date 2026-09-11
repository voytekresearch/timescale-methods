.. _api_documentation:

=================
API Documentation
=================

API reference for the timescales module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 1

Models
------

An AR(p) model fit to a power spectrum or to an autocorrelation.  Both default
to AR(1), the familiar single timescale, and take any order or a grid of orders
to select from.  Neither computes its own input: pass whichever spectrum or
autocorrelation you trust.

.. currentmodule:: timescales.ar

.. autosummary::
   :toctree: generated/

   ARPSD
   ARACF

Fitting
~~~~~~~

The free functions behind the two models.

.. currentmodule:: timescales.ar.core

.. autosummary::
   :toctree: generated/

   fit_ar_psd
   fit_ar_acf
   select_order_psd
   select_order_acf
   acf_effective_n
   init_ar_from_psd
   init_ar_from_acf

Time domain
~~~~~~~~~~~

Estimating AR coefficients from the samples themselves.

.. currentmodule:: timescales.ar

.. autosummary::
   :toctree: generated/

   burg
   ar_ols
   ar_burg
   population_ar
   select_order_time
   compute_ar_spectrum
   ar_to_psd

Goodness of fit
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   spectral_gof
   acf_fit_gof
   r2_logpsd
   rmse_logpsd
   is_divergence

Decomposition
-------------

A fitted AR(p) model read as a set of modes: one relaxation per real pole and
one damped oscillation per conjugate pair, each with a decay time, a frequency
and an importance.

.. currentmodule:: timescales.decomposition

.. autosummary::
   :toctree: generated/

   ARModes
   Mode
   ar_to_poles
   poles_to_ar
   group_poles
   ar_psd
   ar_acf
   acf_mode_components
   psd_mode_components
   impulse_mode_components
   mode_importance
   variance_decomposition
   stabilize

Finite Samples
--------------

Analytic bias, variance and confidence intervals of an AR(1) timescale at
finite recording length.

.. currentmodule:: timescales.finite

.. autosummary::
   :toctree: generated/

   ar1_tau_moments
   ar1_tau_ci
   ar1_phi_ci
   ar1_phi_bias
   ar1_phi_sd
   relative_sd_asymptotic
   relative_bias_asymptotic
   duration_for_relative_sd
   duration_for_relative_bias
   bias_corrected_phi

Simulations
-----------

Processes
~~~~~~~~~

Generative processes carrying their own closed-form PSD and ACF, so an
estimator can be scored against a known timescale.

.. currentmodule:: timescales.sim

.. autosummary::
   :toctree: generated/

   Process
   AR1
   AR2Damped
   ARpModes
   AR1PlusSine
   AR1PlusAR2
   SumAR1
   AR1Mixture

Signals
~~~~~~~

.. autosummary::
   :toctree: generated/

   sim_ar
   sim_ar_process
   sim_ou
   sim_branching
   sim_spikes_synaptic
   sim_spikes_prob
   sim_poisson
   sim_acf_cos
   sim_exp_decay
   sim_damped_cos
   sim_lorentzian

Utilities
---------

Conveniences for producing the inputs the models take.

.. currentmodule:: timescales.utils

.. autosummary::
   :toctree: generated/

   compute_psd
   compute_acf
   normalize
   create_windows
   get_distinct_windows
   resample_logspace

Conversions
-----------

.. currentmodule:: timescales.conversions

.. autosummary::
   :toctree: generated/

   convert_knee
   convert_tau
   knee_to_tau
   tau_to_knee
   tau_to_phi
   phi_to_tau
   psd_to_acf
   acf_to_psd

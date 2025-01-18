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

Model Objects
-------------

Objects for computing timescales from PSD or ACF.

PSD
~~~

The PSD object fits power spectra and extracts timescales.

.. currentmodule:: timescales.fit

.. autosummary::
   :toctree: generated/

   PSD

ACF
~~~

The ACF object fits autocorrelation functions and extracts timescales.

.. autosummary::
   :toctree: generated/

   ACF

ARPSD
~~~~~

The ARPSD object fits power spectra using the AR(p) form.

.. autosummary::
   :toctree: generated/

   ARPSD

Autoregressive
--------------

Spectral
~~~~~~~~

Autoregressive functions.

.. currentmodule:: timescales.autoreg

.. autosummary::
   :toctree: generated/

   compute_ar_spectrum
   burg
   ar_to_psd


Simulations
-----------

LFPs
~~~~

Local field potentials as AR, branching, or Ornstein-Uhlenbeck processes.

.. currentmodule:: timescales.sim

.. autosummary::
   :toctree: generated/

   sim_ar
   sim_ou
   sim_branching


Spikes
~~~~~~

Spike simulations use exponentially decaying probability kernels, convolved with a Poisson.

.. autosummary::
   :toctree: generated/

   sim_spikes_synaptic
   sim_spikes_prob
   sim_poisson


ACF
~~~

.. autosummary::
   :toctree: generated/

   sim_acf_cos
   sim_exp_decay
   sim_damped_cos

PSD
~~~

.. autosummary::
   :toctree: generated/

   sim_ar_spectrum
   sim_lorentzian


Conversions
-----------

Conversion functions are usefull for convert PSD to/from ACF, and to convert knee frequencies
to taus.

.. currentmodule:: timescales.conversions

.. autosummary::
   :toctree: generated/

   psd_to_acf
   acf_to_psd
   tau_to_knee
   tau_to_phi
   phi_to_tau
   knee_to_tau

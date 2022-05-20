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

Simulations
-----------

Spikes
~~~~~~

Spike simulations use exponentially decaying probability kernels, convolved with a Poisson.

.. currentmodule:: timescales.sim

.. autosummary::
   :toctree: generated/

   sim_spikes_synaptic
   sim_spikes_prob
   sim_poisson

LFPs
~~~~

Local field potentials are simulated branching and Ornstein-Uhlenbeck processes.

.. autosummary::
   :toctree: generated/

   sim_branching
   sim_ou

Autoregression
--------------

Spectral
~~~~~~~~

Autoregressive models are availble to compute PSD from.

.. currentmodule:: timescales.autoreg

.. autosummary::
   :toctree: generated/

    compute_ar_spectrum


Pipelines
---------

Pipes
~~~~~

Pipe objects are used to reproducible simulate, transform, and/or fit timescales.

.. currentmodule:: timescales.pipe

.. autosummary::
   :toctree: generated/

    Pipe


Conversions
-----------

Conversion functions are usefull for convert PSD to/from ACF, and to convert knee frequencies
to taus.

.. currentmodule:: timescales.conversions

.. autosummary::
   :toctree: generated/

   convert_knee
   convert_tau
   psd_to_acf
   acf_to_psd

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

.. currentmodule:: timescales.fit

Model Objects
-------------

Objects that manage data and fit the model to parameterize neural power spectra.

PSD
~~~

The PSD object fits power spectra and extracts timescales.

.. autosummary::
   :toctree: generated/

   PSD

ACF
~~~

The ACF object fits autocorrelation functions and extracts timescales.

.. autosummary::
   :toctree: generated/

   ACF


.. currentmodule:: timescales.sim

Simulations
-----------

Spikes
~~~~~~

Spike simulations use exponentially decaying probability kernels, convolved with a Poisson.

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


.. currentmodule:: timescales.autoreg

Autoregression
--------------

Spectral
~~~~~~~~

Autoregressive models are availble to compute PSD from.

.. autosummary::
   :toctree: generated/

    compute_ar_spectrum


.. currentmodule:: timescales.pipe

Pipelines
---------

Pipes
~~~~~

Pipe objects are used to reproducible simulate, transform, and/or fit timescales.


.. autosummary::
   :toctree: generated/

    Pipe

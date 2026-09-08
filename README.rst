==========
timescales
==========

This module provides timescale simulation and fitting methods.
These simulations allow the validation and comparison of different timescale estimation methods
including PSD and ACF.

|ProjectStatus|_ |BuildStatus|_ |Coverage|_

.. |ProjectStatus| image:: http://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |BuildStatus| image:: https://github.com/voytekresearch/timescale-methods/actions/workflows/build.yml/badge.svg
.. _BuildStatus: https://github.com/voytekresearch/timescale-methods/actions/workflows/build.yml

.. |Coverage| image:: https://codecov.io/gh/voytekresearch/timescale-methods/branch/main/graph/badge.svg?token=vxK8jrHm9O
.. _Coverage: https://codecov.io/gh/voytekresearch/timescale-methods

Dependencies
------------

timescales is written in Python, and requires Python >= 3.8 to run.

It has the following required dependencies:

- numpy >= 1.18.5
- scipy >= 1.4.1
- spectrum >= 0.8.0
- fooof @ git+https://github.com/ryanhammonds/fooof.git@knee
- neurodsp @ git+https://github.com/neurodsp-tools/neurodsp.git@main

And the following optional dependencies:

- matplotlib >= 3.0.3
- seaborn >= 0.11.2
- statsmodels >= 0.12.2
- tqdm >= 4.62.3


Installation
------------

**Development Version**

To get the current development version, first clone this repository:

.. code-block:: shell

    $ git clone https://github.com/voytekresearch/timescale-methods

To install this cloned copy, move into the directory you just cloned, and run:

.. code-block:: shell

    $ pip install -e .

That is the base install: numpy, scipy and matplotlib, which is everything
needed to fit, decompose and plot a model. One extra, ``all``, adds what the
test suite, the simulations outside the core model (spikes,
Ornstein-Uhlenbeck, branching), the analysis pipeline and the documentation
build need:

.. code-block:: shell

    $ pip install -e .[all]


Usage
-----

Fit an AR(p) model to a power spectrum, or to an autocorrelation, and read the
fitted model as a set of modes:

.. code-block:: python

    from timescales import ARPSD, ARACF, compute_psd, compute_acf

    freqs, powers = compute_psd(sig, fs, nperseg=int(2 * fs))
    ar = ARPSD(order=range(1, 11), fs=fs).fit(freqs, powers)
    ar.decompose()
    ar.plot(plot_modes=True)
    ar.tau        # aperiodic timescale, seconds
    ar.taus       # every mode's decay time
    ar.freqs_hz   # and its frequency; zero for a relaxation

    lags, corrs = compute_acf(sig, nlags=250)
    ARACF(order=range(1, 11), fs=fs).fit(lags, corrs).tau

``order=1`` is the default and reproduces the usual single-timescale fit: a
Lorentzian in the spectrum, an exponential in the autocorrelation.  Passing a
range of orders selects one by BIC.  The models never compute their own input,
so any spectral or autocorrelation estimator can be used.

Reproducing the paper
---------------------

The analyses in the accompanying methods paper -- the simulation benchmark, the
fits to eight public recordings, and the scripts behind every figure and
supplementary table -- live on the ``analysis`` branch, tracked with
`DVC <https://dvc.org>`_ so that the whole pipeline can be rerun from the raw
recordings.

They are kept off ``main`` on purpose.  Installing ``timescales`` should give
you the method, not one particular application of it; keeping the analysis on
its own branch also pins the exact code and settings the published numbers came
from.

.. code-block:: shell

    $ git checkout analysis
    $ pip install -e .[all]
    $ cd analysis
    $ dvc pull      # the extracted recordings
    $ dvc repro     # every fit, table and figure
    $ dvc status    # silence means the results match the code

``analysis/README.md`` documents the pipeline and what ``dvc status``
does and does not catch.

Funding
-------

Supported by NIH award R01 GM134363 from the
`NIGMS <https://www.nigms.nih.gov/>`_.

.. image:: https://www.nih.gov/sites/all/themes/nih/images/nih-logo-color.png
  :width: 400

|

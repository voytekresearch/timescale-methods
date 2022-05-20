==========
timescales
==========

This module provides timescale simulation and fitting methods.
These simulations allow the validation and comparison of different timescale estimation methods
including PSD and ACF.

|ProjectStatus|_ |BuildStatus|_ |Coverage|_

.. |ProjectStatus| image:: http://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |BuildStatus| image:: https://github.com/github/docs/actions/workflows/main.yml/badge.svg
.. _BuildStatus: https://github.com/voytekresearch/timescale-methods/actions/workflows/build.yml

.. |Coverage| image:: https://codecov.io/gh/voytekresearch/timescale-methods/branch/main/graph/badge.svg
.. _Coverage: https://codecov.io/gh/voytekresearch/timescale-methods


Dependencies
------------

timescales is written in Python, and requires Python >= 3.6 to run.

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


Funding
-------

Supported by NIH award R01 GM134363 from the
`NIGMS <https://www.nigms.nih.gov/>`_.

.. image:: https://www.nih.gov/sites/all/themes/nih/images/nih-logo-color.png
  :width: 400

|

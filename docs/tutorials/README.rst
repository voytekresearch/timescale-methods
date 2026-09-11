
Tutorials
=========

Four short tutorials, in order.  The first fits the familiar single-timescale
AR(1) model in both the spectral and lag domains.  The second raises the model
order, which is what lets one fit describe several decays and a rhythm at once.
The third reads a fitted high-order model back out as timescales and
frequencies.  The last applies all of it to eight real recordings.

The first three use simulated data, where the answer is known and the estimate
can be checked against it.

Editing and re-running
----------------------

The notebooks are the source.  Edit them in Jupyter and commit them with their
outputs: the documentation build does not execute anything
(``nb_execution_mode = 'off'`` in ``docs/conf.py``), so the site renders
whatever outputs are committed.

That makes it possible for a committed notebook to disagree with the package,
which is worth checking after any change to an estimator or its defaults.
Re-execute them in place and inspect the diff:

.. code-block:: shell

    $ jupyter nbconvert --to notebook --execute --inplace docs/tutorials/tutorial_*.ipynb

A clean diff means the tutorials still match the code.  Tutorial 4 fits eight
recordings over a grid of orders and takes a couple of minutes; the others are
seconds.

"""Code specific to this paper, on top of the ``timescales`` package.

The generalizable machinery -- the AR(p) estimators, the pole/mode
decomposition, the generative processes and the finite-sample theory -- lives
in ``timescales`` and is imported from there.  What remains here is what only
this paper needs:

``constants``  the particular AR(7), AR(5) and AR(2) the figures simulate
``datasets``   the registry and common preprocessing of the eight recordings
``benchmark``  the Monte-Carlo harness that writes the result tables
``plotting``   the figure style and shared panels
"""

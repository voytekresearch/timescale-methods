"""Choosing which modes to report, by the variance they account for.

:meth:`ARModes.resolvable` answers whether the analysed band could measure a
mode at all.  This answers a different question: of the modes it could measure,
which ones are worth reporting?

The two decompositions of an AR(p) model behave differently here.  In the
spectrum the mode factors *multiply*, so every factor is a full-band curve and
none of them has a share of anything: a factor can look prominent and
contribute nothing to the fitted spectrum, because the other factors fall
through the same frequencies and cancel it.  In the lag domain the partial
fractions *add*, they sum to the model autocorrelation, and each one's value
at lag zero is that mode's share of the process variance.

Those partial fractions are the same objects in both domains -- they sum to
the model spectrum, and their transforms sum to the model autocorrelation --
so their lag-zero values are the one quantity that is a share of the fit in
the spectrum and in the autocorrelation at once.  That is what makes them the
right basis for a single selection used by both.

The share is also, directly, the size of the curve a mode draws: a mode
holding 0.2% of the variance contributes an autocorrelation component whose
amplitude is 0.002, which is indistinguishable from zero on an axis that spans
-0.5 to 1.  Drawing it asserts structure that the figure cannot show and the
fit does not need.

Shares are non-negative and sum to one, so a threshold on them is a statement
about the process rather than about the plotting.
"""

import numpy as np

#: Smallest share of the variance a mode must hold to be worth reporting.
#: One percent draws an autocorrelation component of amplitude 0.01, at the
#: edge of what is visible; below it the curve cannot be told from zero.
MIN_VARIANCE_SHARE = 0.01


def variance_shares(modes, components):
    """Each mode's share of the variance, from its additive ACF component.

    Parameters
    ----------
    modes : sequence of Mode
        The modes whose components were computed, in the same order.
    components : 2d array, (n_modes, n_lags + 1)
        Additive autocorrelation components, one row per mode, normalised so
        that the rows sum to the model autocorrelation.  Pass the same array
        the figure draws, so that the selection and the drawing cannot
        disagree.

    Returns
    -------
    shares : 1d array
        Value of each component at lag zero.
    """
    components = np.asarray(components, dtype=float)
    if len(components) != len(modes):
        raise ValueError(f'{len(modes)} modes but {len(components)} components.')
    return components[:, 0].copy()


def dominant_modes(modes, shares, min_share=MIN_VARIANCE_SHARE, n_max=None):
    """The modes worth reporting, most variance first.

    Parameters
    ----------
    modes : sequence of Mode
    shares : 1d array
        Each mode's share of the variance, aligned with ``modes``.
    min_share : float
        Smallest share to keep.  Defaults to :data:`MIN_VARIANCE_SHARE`.
    n_max : int, optional
        Keep at most this many.

    Returns
    -------
    kept : list of (Mode, float)
        Mode and its share, ranked by share, highest first.

    Notes
    -----
    Ranking by share rather than by spectral importance is deliberate.
    Importance measures how much the model's *log* spectrum changes when a
    mode is deleted, which weights every decade equally and therefore ranks
    high-frequency modes far above the variance they carry; ranking one way
    and drawing the result in a linear-power panel is what makes a mode look
    prominent in the spectrum and absent from the autocorrelation.  A single
    ranking, in the currency both panels share, keeps them consistent.
    """
    shares = np.asarray(shares, dtype=float)
    if len(shares) != len(modes):
        raise ValueError(f'{len(modes)} modes but {len(shares)} shares.')
    kept = [(m, float(s)) for m, s in zip(modes, shares)
            if np.isfinite(s) and s >= min_share]
    kept.sort(key=lambda ms: -ms[1])
    return kept if n_max is None else kept[:n_max]

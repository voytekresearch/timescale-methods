"""Spectral estimation methods."""

import numpy as np


def convert_knee_val(knee, exponent=2.):
    """Convert knee parameter(s) to frequency and time-constant value.

    Parameters
    ----------
    knee : float or array
        Knee of the aperiodic spectral fit.
    exponent : float, optional, default: 2/.
        Used for more accurate frequency estimation when PSD is Lorentzian.

    Returns
    -------
    knee_freq : float
        Frequency where the knee occurs.
    knee_tau : float
        Timescale, in seconds.
    """

    knee_freq = knee**(1./exponent)
    knee_tau = 1./(2*np.pi*knee_freq)

    return knee_freq, knee_tau

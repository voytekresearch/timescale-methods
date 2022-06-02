"""Resampling functions."""

import numpy as np
from scipy.interpolate import interp1d


def resample_logspace(freqs, powers, n_samples):
    """Evenly resamples PSD in log-log space.

    Parameters
    ---------
    freqs : 1d array
        Frequencies.
    powers : 1d or 2d array
        Power spectral density.
    n_samples : int
        Number of samples to resample to. A large number,
        relative to the range of frequencies, is required
        to prevent loss of high frequency resolution.

    Notes
    -----
    Even log-log resampling allows for more robust regression.
    """

    freqs_log = np.log10(freqs)
    powers_log = np.log10(powers)

    lin_interp = interp1d(freqs_log, powers_log, kind='linear')

    freqs_even = np.linspace(freqs_log.min(), freqs_log.max(), n_samples)
    powers_even = lin_interp(freqs_even)

    freqs_even = 10**freqs_even
    powers_even = 10**powers_even

    return freqs_even, powers_even
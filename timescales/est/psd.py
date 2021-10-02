"""Spectral estimation methods."""

import numpy as np


def fit_psd(spikes, fs, f_range, fooof_init=None, n_jobs=-1):

    freqs, powers = compute_spectrum(spikes, fs, f_range=f_range)

    if fooof_init is not None:
        fm = FOOOF(aperiodic_mode='knee', verbose=False, **fooof_init)
    else:
        fm = FOOOF(aperiodic_mode='knee', verbose=False)

    fm.fit(freqs, powers, f_range)

    knee = fm.get_params('aperiodic', 'knee')
    exp = fm.get_params('aperiodic', 'exponent')

    knee_freq, knee_tau = convert_knee_val(knee, exponent=exp)

    return fm, knee_freq, knee_tau


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

"""Spectral estimation methods."""

import numpy as np

from fooof import FOOOF, FOOOFGroup


def fit_psd(freqs, powers, f_range, fooof_init=None,
            knee_bounds=None, mode=None, n_jobs=-1, progress=None):
    """Fit a PSD, using SpecParam, and estimate tau.

    Parameters
    ----------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spectrum : 1d or 2d array
        Power spectral density.
    f_range : tuple of (float, float)
        Frequency range of interest.
    fooof_init : dict, optional, default: None
        Fooof initialization arguments.
    knee_bounds : tuple of (float, float)
        Aperiodic knee bounds bounds.
    mode : {None, 'mean', 'median'}
        How to combine 2d spectra.
    n_jobs : int
        Number of jobs to run in parralel, when spikes is 2d.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    fm : fooof.FOOOF or fooof.FOOOFGroup
        Fit spectral parameterization model.
    knee_freq : float or 1d array
        Knee frequencies.
    knee_tau : float or 1d array
        Estimated timescales.
    """

    if fooof_init is None:
        fooof_init = {}

    if mode == 'mean' and powers.ndim == 2:
        powers = np.mean(powers, axis=0)
    elif mode == 'median' and powers.ndim == 2:
        powers = np.median(powers, axis=0)

    fooof_init_cp = fooof_init.copy()
    ap_mode = fooof_init_cp.pop('aperiodic_mode', 'knee')

    # Init FOOOF
    if powers.ndim == 1:
        fm = FOOOF(aperiodic_mode=ap_mode, verbose=False, **fooof_init_cp)
    elif powers.ndim == 2:
        fm = FOOOFGroup(aperiodic_mode=ap_mode, verbose=False, **fooof_init_cp)

    # Overwite bounds and guess
    if knee_bounds is not None and ap_mode == 'knee_constant':
        fm._ap_bounds = ((-np.inf, knee_bounds[0], -np.inf, 0),
                         (np.inf, knee_bounds[1], np.inf, np.inf))
        fm._ap_guess = [None, (knee_bounds[1] - knee_bounds[0])/2, None, 0]
    elif knee_bounds is not None and ap_mode == 'knee':
        fm._ap_bounds = ((-np.inf, knee_bounds[0], -np.inf),
                         (np.inf, knee_bounds[1], np.inf))
        fm._ap_guess = [None, (knee_bounds[1] - knee_bounds[0])/2, None]

    # Fit
    if powers.ndim == 1:
        fm.fit(freqs, powers, f_range)
        knee_freq = fm.get_params('aperiodic', 'knee')
    else:
        fm.fit(freqs, powers, f_range, n_jobs, progress)
        knee_freq = fm.get_params('aperiodic', 'knee')

    return fm, knee_freq


def convert_knee_val(knee_freq):
    """Convert knee parameter(s) to frequency and time-constant value.

    Parameters
    ----------
    knee : float or array
        Knee of the aperiodic spectral fit.
    exponent : float, optional, default: 2.
        Used for more accurate frequency estimation when PSD is Lorentzian.

    Returns
    -------
    knee_freq : float
        Frequency where the knee occurs.
    knee_tau : float
        Timescale, in seconds.
    """

    knee_tau = 1./(2*np.pi*knee_freq)

    return knee_tau

"""Spectral estimation methods."""

import numpy as np

from neurodsp.spectral import compute_spectrum

from fooof import FOOOF, FOOOFGroup


def fit_psd(spikes, fs, f_range, fooof_init=None, n_jobs=-1, progress=None):
    """Fit a PSD, using SpecParam, and estimate tau.

    Parameters
    ----------
    spikes : 1d or 2d array
        Spike counts or probabilities.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range of interest.
    fooof_init : dict, optional, default: None
        Fooof initialization arguments.
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

    freqs, powers = compute_spectrum(spikes, fs, f_range=f_range)

    if fooof_init is None:
        fooof_init = {}

    if spikes.ndim == 1:
        fm = FOOOF(aperiodic_mode='knee', verbose=False, **fooof_init)
    elif spikes.ndim == 2:
        fm = FOOOFGroup(aperiodic_mode='knee', verbose=False, **fooof_init)

    if spikes.ndim == 1:
        fm.fit(freqs, powers, f_range)

        knee = fm.get_params('aperiodic', 'knee')
        exp = fm.get_params('aperiodic', 'exponent')

        knee_freq, knee_tau = convert_knee_val(knee, exponent=exp)
    else:
        fm.fit(freqs, powers, f_range, n_jobs, progress)

        knees = fm.get_params('aperiodic', 'knee')
        exps = fm.get_params('aperiodic', 'exponent')

        knee_freq = np.zeros(len(knees))
        knee_tau = np.zeros(len(knees))

        for ind, (knee, exp) in enumerate(zip(knees, exps)):
            _knee_freq, _knee_tau = convert_knee_val(knee, exponent=exp)
            knee_freq[ind] = _knee_freq
            knee_tau[ind] = _knee_tau

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

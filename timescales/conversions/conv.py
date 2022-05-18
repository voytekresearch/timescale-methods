"""Conversion utilities."""

import numpy as np
from scipy.fft import ifft, fftfreq

from timescales.utils import normalize as _normalize


def convert_knee(knee_freq):
    """Convert knee parameter(s) to time-constant value.

    Parameters
    ----------
    knee : float or array
        Knee of the aperiodic spectral fit.

    Returns
    -------
    tau : float
        Timescale, in seconds.
    """

    tau = 1. / (2 * np.pi * knee_freq)

    return tau


def convert_tau(tau):
    """Convert tau(s) to knee frequency.

    Parameters
    ----------
    tau : float
        Timescale, in seconds.

    Returns
    -------
    knee : float or array
        Knee of the aperiodic spectral fit.
    """

    return convert_knee(tau)


def psd_to_acf(freqs, powers, fs, normalize=None):
    """Convert a PSD to ACF.

    Parameters
    ----------
    freqs : 1d array
        Frequency definition.
    powers : 1d
        Power spectral density.
    fs : float
        Sampling rate, in Hertz.
    normalize : tuple of (float, float), default: None
        Normalize from (min, max).
    """
    corrs = ifft(powers).real
    corrs = corrs[1:len(powers)//2]

    if isinstance(normalize, (tuple, list)):
        corrs = _normalize(corrs, *normalize)

    f_res = freqs[1] - freqs[0]

    lags = fftfreq(len(powers), 1)[1:len(powers)//2] / f_res * fs

    return lags, corrs


def acf_to_psd(lags, corrs, fs, normalize=None):
    """Convert ACF to PSD.

    Parameters
    ----------
    lags : 1d array
        Lag definitions.
    corrs : 1d array
        Auto-correlations.
    fs : float
        Sampling rate, in Hertz.
    normalize : tuple of (float, float), default: None
        Normalize from (min, max).

    Returns
    -------
    freqs : 1d array
        Frequency definition.
    powers : 1d
        Power spectral density.
    """
    powers = ifft(corrs).real
    freqs = fftfreq(len(powers), 1 / fs * (lags[1] - lags[0]))

    powers = powers[1:len(powers)//2]
    freqs = freqs[1:len(freqs)//2]

    if isinstance(normalize, (tuple, list)):
        powers = _normalize(powers, *normalize)

    return freqs, powers

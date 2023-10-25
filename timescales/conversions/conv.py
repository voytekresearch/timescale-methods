"""Conversion utilities."""

import numpy as np
from scipy.fft import ifft, fftfreq
from spectrum import arma2psd
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

    Returns
    -------
    lags : 1d array
        Lag definitions.
    corrs : 1d array
        Correlation coefficients.
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

def ar_to_psd(ar, fs, nfft, f_range=None):
    """Convert AR coefficients to PSD.

    Parameters
    ----------
    ar : 1d array
        Positive AR coefficients in descending order.
    fs : float
        Sampling rate, in Hz.
    nfft : int
        Number of samples to use in the fft.
        Determines frequency resolution.
    f_range : tuple of (float, float)
        Lower and upper frequency bounds.

    Returns
    -------
    freqs : 1d array
        Frequency definition.
    powers : 1d array
        AR power spectral density.
    """
    powers = arma2psd(A=-ar, rho=1., T=fs, NFFT=nfft)
    freqs = fftfreq(nfft, 1/fs)
    powers = powers[:len(freqs)//2]
    freqs = freqs[:len(freqs)//2]

    if f_range is not None:
        inds = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))
        freqs = freqs[inds]
        powers = powers[inds]

    return freqs, powers
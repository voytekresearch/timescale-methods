"""Conversion utilities."""

import numpy as np
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

def knee_to_tau(knee_freq):
    # Alias
    return convert_knee(knee_freq)
def tau_to_knee(tau):
    # Alias
    return convert_knee(tau)

def tau_to_phi(tau, fs):
    """The AR(1) coefficient of a relaxation time, ``exp(-1 / (tau fs))``."""
    return np.exp(-1.0 / (np.asarray(tau, dtype=float) * fs))


def phi_to_tau(phi, fs):
    """The relaxation time of an AR(1) coefficient, ``-(1/fs) / log(phi)``.

    NaN unless ``0 < phi < 1``: a coefficient at or above one does not decay
    and a negative one alternates in sign, so neither has a relaxation time
    to report.
    """
    phi = np.asarray(phi, dtype=float)
    out = np.full(phi.shape, np.nan) if phi.ndim else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        val = -1.0 / (fs * np.log(np.abs(phi)))
    good = (phi > 0) & (phi < 1)
    return np.where(good, val, out) if phi.ndim else (float(val) if good else np.nan)

def psd_to_acf(freqs, powers, fs):
    """Convert a PSD to ACF.

    Parameters
    ----------
    freqs : 1d array
        Frequency definition.
    powers : 1d
        Power spectral density.
    fs : float
        Sampling rate, in Hertz.

    Returns
    -------
    lags : 1d array
        Lag definitions.
    corrs : 1d array
        Correlation coefficients.
    """
    powers = powers / np.sum(powers)

    corrs = np.fft.ifft(powers, norm='forward')[:len(powers)//2].real
    lags = np.arange(len(corrs)) * (fs/freqs[-1])

    return lags, corrs


def acf_to_psd(corrs, fs):
    """Convert ACF to PSD.

    Parameters
    ----------
    corrs : 1d array
        Auto-correlations.
    fs : float
        Sampling rate, in Hertz.

    Returns
    -------
    freqs : 1d array
        Frequency definition.
    powers : 1d
        Power spectral density.
    """
    # Compute power
    powers = np.abs(np.fft.fft(corrs))**2
    freqs = np.fft.fftfreq(len(corrs), 1/fs)

    # Drop imaginary
    freqs = freqs[:len(corrs)//2]
    powers = powers[:len(corrs)//2]

    return freqs, powers

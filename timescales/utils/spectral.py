"""Convenience estimators for the inputs the AR models fit.

The AR estimators take ``(freqs, powers)`` or ``(lags, corrs)`` and do not
care how they were produced.  These are the two most common ways of producing
them, offered so that simple use does not require assembling them by hand.
Neither carries a default chosen for any particular recording: pass the
segment length and lag range the data justify.
"""

import numpy as np
from scipy.signal import welch


def compute_psd(sig, fs, nperseg=None, noverlap=None, f_range=None, **kwargs):
    """Welch power spectral density.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, Hz.
    nperseg : int, optional
        Segment length in samples.  Defaults to the whole record, which is a
        periodogram: pick a shorter segment to trade frequency resolution for
        variance.
    noverlap : int, optional
        Overlap in samples, default half of ``nperseg``.
    f_range : tuple of (float, float), optional
        Restrict the returned frequencies to this band.

    Returns
    -------
    freqs, powers : 1d arrays
    """
    sig = np.asarray(sig, dtype=float)
    nperseg = len(sig) if nperseg is None else int(nperseg)
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    freqs, powers = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          **kwargs)
    if f_range is not None:
        keep = (freqs >= f_range[0]) & (freqs <= f_range[1])
        freqs, powers = freqs[keep], powers[keep]
    return freqs, powers


def compute_acf(sig, nlags, mean_center=True):
    """Biased sample autocorrelation, lags 0 to ``nlags``.

    The biased (divide by n) estimator is used because it is positive
    semi-definite, so the autocorrelation it returns is one an AR model can
    actually have.

    Returns
    -------
    lags, corrs : 1d arrays
    """
    x = np.asarray(sig, dtype=float)
    if mean_center:
        x = x - x.mean()
    nlags = int(nlags)
    if nlags >= len(x):
        raise ValueError('`nlags` must be shorter than the signal.')
    nfft = int(2 ** np.ceil(np.log2(2 * len(x) - 1)))
    spec = np.fft.rfft(x, nfft)
    corrs = np.fft.irfft(spec * np.conj(spec), nfft)[:nlags + 1]
    return np.arange(nlags + 1), corrs / corrs[0]

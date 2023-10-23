"""Compute spectra from autoregssive models."""

import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.fft import fftfreq

from statsmodels.regression.linear_model import yule_walker
from spectrum import arma2psd


def compute_ar_spectrum(sig, fs, order, f_range=None, method='burg', nfft=4096, n_jobs=1):
    """Compute an autoregressive power spectrum.

    Parameters
    ----------
    sig : 1d or 2d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    order : int
        Number of autogressive coefficients to fit.
    f_range : tuple
        Lower and upper frequency bounds.
    method : str, {'burg', 'yule_walker'}
        Coefficient estimation method.
    nfft : int, optional, default: 4096
        Window length.
    n_jobs : optional, int, default: 1
        Number of jobs to run in parallel.

    Returns
    -------
    freqs : 1d array
        Frequency definitions.
    powers : 1d array
        Power values.
    """

    # 2d
    if sig.ndim == 2:

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with Pool(processes=n_jobs) as pool:

            pfunc = partial(compute_ar_spectrum, fs=fs, order=order,
                            nfft=nfft, f_range=f_range, n_jobs=n_jobs)

            mapping = pool.map(pfunc, sig)

            results = np.array(mapping)

            freqs = results[0, 0]
            powers = results[:, 1, :]

        return freqs, powers

    # 1d
    if method == 'burg':
        ar = burg(sig, order=order)
    else:
        ar, _ = yule_walker(sig, order=order)

    powers = arma2psd(A=-ar, rho=1., T=fs, NFFT=nfft)
    freqs = fftfreq(nfft, 1/fs)
    powers = powers[:len(freqs)//2]
    freqs = freqs[:len(freqs)//2]

    if f_range is not None:
        inds = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))
        freqs = freqs[inds]
        powers = powers[inds]

    return freqs, powers

def burg(sig, order, demean=True):

    if demean:
        sig = sig - sig.mean()

    # Initalize arrays for reflection coefficients
    #   and ar coefficients
    k = np.zeros(order)
    ar = np.zeros(order)

    # Loop to solve reflection coefficients, k
    _f = sig
    _b = sig
    for i in range(order):

        # Forward and backward shifts
        f = _f[1:]
        b = _b[:-1]

        # Density, sum of squares
        if i == 0:
            den = (f@f) + (b@b)
        else:
            # Faster via recursion
            den = (
                (1 - k[i-1]**2) *
                den - (_f[0]**2) - (_b[-1]**2)
            )

        # Reflection coefficient
        k[i] = (-2 * b @ f) / den

        # AR coefficient
        ar[i] = -k[i]
        if i > 0:
            prev = ar[:i]
            ar[:i] = prev - ar[i] * prev[::-1]

        # Update forward and backward
        _f = f + k[i] * b
        _b = b + k[i] * f

    return ar
"""Compute spectra from autoregssive models."""

import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.fft import fftfreq

from statsmodels.regression.linear_model import yule_walker, burg
from spectrum import arma2psd

from neurodsp.filt import filter_signal



def ar_psd(sig, fs, order, method='burg', nfft=4096, n_jobs=1):
    """Compute an autoregressive power spectrum.

    Parameters
    ----------
    sig : 1d or 2d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    order : int
        Number of autogressive coefficients to fit.
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

            pfunc = partial(ar_psd, fs=fs, order=order,
                            nfft=nfft, n_jobs=n_jobs)

            mapping = pool.map(pfunc, sig)

            results = np.array(mapping)

            freqs = results[0, 0]
            powers = results[:, 1, :]

        return freqs, powers

    # 1d
    if method == 'burg':
        ar, rho = burg(sig, order=order)
    else:
        ar, rho = yule_walker(sig, order=order)

    powers = arma2psd(A=-ar, rho=rho, T=fs, NFFT=nfft)

    freqs = fftfreq(nfft, 1/fs)
    powers = powers[:len(freqs)//2]
    freqs = freqs[:len(freqs)//2]

    return freqs, powers


def ar_psds_bandstop(sig, fs, band_ranges, order,
                     filter_kwargs=None, ar_psd_kwargs=None):
    """Compute autoregressive PSDs for a set of bandstop filters.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    order : int
        Number of autogressive coefficients to fit.
    filter_kwargs : dict, optional, default: None
        Additional arguments to pass to neurodsp.filt.filter_signal.
    ar_psd_kwargs : dict, optional, default: None
        Additional arguments to pass to ar.psd.

    Returns
    -------
    freqs : 1d array
        Frequency definitions.
    powers : 2d array
        Power values.
    sig_filt : 2d array
        Bandstopped signals.
    """

    # Make kwargs unpackable
    filter_kwargs = {} if filter_kwargs is None else filter_kwargs
    ar_psd_kwargs = {} if ar_psd_kwargs is None else ar_psd_kwargs

    filter_kwargs['remove_edges'] = False

    # Determine size of psd array
    nfft = ar_psd_kwargs.pop('nfft', 4096)

    if nfft % 2 != 0:
        nfft += 1

    powers = np.zeros((len(band_ranges), nfft//2))

    # Filter and compute PSD
    sigs_filt = np.zeros((len(band_ranges), len(sig)))

    for ind, b_range in enumerate(band_ranges):

        with warnings.catch_warnings():

            warnings.filterwarnings('ignore')

            _sig_filt = filter_signal(sig, fs, 'bandstop', b_range,
                                      **filter_kwargs)
            sigs_filt[ind] = _sig_filt

        freqs, _powers = ar_psd(_sig_filt, fs, order, **ar_psd_kwargs)
        powers[ind] = _powers

    return freqs, powers, sigs_filt

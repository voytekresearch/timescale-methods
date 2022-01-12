"""Multi-processing wrapper functions."""

from functools import partial
from importlib import import_module
from multiprocessing import cpu_count, Pool

import numpy as np

from neurodsp.spectral import compute_spectrum
from mne.time_frequency import psd_array_multitaper

from timescales.fit.psd import fit_psd
from timescales.fit.acf import ACF


def compute_taus(iterable, method='acf', fit_kwargs=None, low_mem=False,
                 n_jobs=-1, chunksize=20, progress='tqdm.notebook'):
    """Compute and fit ACF or PSD in parallel.

    Parameters
    ----------
    iterable : 1d array-like
        Signal start indices (or correlation coefficients if pre-computed) to fit
        in parallel.
    method : {'acf', 'psd'}
        Method to compute the timscale with.
    fit_kwargs : dict, optional, default: None
        Addition fit kwargs to pass to either mp_fit_acf or mp_fit_psd.
    low_mem : bool, optional, default: False
        Reduce memory usage.
    n_jobs : int, optional, default: -1
        Number of jobs to run in parallel. -1 defaults to max cpus.
    chunksize : int, optional, default: 20
        Number of jobs to submit per chunk.

    Returns
    -------
    taus : 1d array
        Timescales.
    rsq : 1d array
        Goodness of fit.
    result_class : ACF or FOOOF
        Access to intermediate arrays or parameters.
    """

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    try:
        tqdm = import_module(progress).tqdm
    except:
        tqdm = None

    with Pool(processes=n_jobs) as pool:

        if method == 'acf':
            mapping = pool.imap(partial(mp_fit_acf, low_mem=low_mem, **fit_kwargs),
                                iterable, chunksize=chunksize)
        elif method == 'psd':
            mapping = pool.imap(partial(mp_fit_psd, **fit_kwargs),
                                iterable, chunksize=chunksize)

        if tqdm is None:
            result = list(mapping)
        else:
            result = list(tqdm(mapping, total=len(iterable), dynamic_ncols=True))

    taus, rsq, result_class = sort_result(result)

    return taus, rsq, result_class


def mp_fit_psd(iterable, sig=None, fs=None, win_len=None, f_range=None,
               compute_spectrum_kwargs=None, fit_kwargs=None, rsq_type='linear'):
    """Multiprocessing wrapper from computing and fitting PSDs.

    Parameters
    ----------
    ind : 1d array
        Start index to compute spectrum.
    sig : 1d array, optional, default: None
        Voltage time series.
    fs : float, optional, default: None
        Sampling rate, in Hz.
    win_len : int, optional, default: None
        Number of samples per window.
    f_range : tuple, optional, default: None
        Lower and upper frequency range bounds.
    compute_spectrum_kwargs : dict, optional, default: None
        Additional keyword arguments to pass to compute_spectrum.
        Should also contain win_len to determine window size.
    fit_kwargs : dict, optional, default: None
        Additional keyword arguments to pass fooof initalization.
        May also include knee_bounds.
    rsq_type : {'linear', 'log'}
        Compute goodness of fit in either linear or log space.

    Returns
    -------
    tau : float
        Timescale.
    rsq : float
        Goodness of fit.
    fm : fooof.FOOOF
        Class containing intermediate objects.
    """

    compute_spectrum_kwargs = {} if compute_spectrum_kwargs is None else compute_spectrum_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    win_len = fs if win_len is None else win_len

    if 'method' not in compute_spectrum_kwargs.keys():
        compute_spectrum_kwargs['method'] = 'welch'

    if compute_spectrum_kwargs['method'] != 'multitaper':
        freqs, powers = compute_spectrum(sig[iterable:iterable+win_len], fs, f_range=f_range,
                                        **compute_spectrum_kwargs)
    else:
        _compute_spectrum_kwargs = compute_spectrum_kwargs.copy()
        del _compute_spectrum_kwargs['method']
        powers, freqs = psd_array_multitaper(sig[iterable:iterable+win_len], fs, verbose=0,
                                             **_compute_spectrum_kwargs)

    _fit_kwargs = fit_kwargs.copy()

    knee_bounds = _fit_kwargs.pop('knee_bounds', None)

    fm, _, tau = fit_psd(freqs, powers, f_range, fooof_init=_fit_kwargs,
                         knee_bounds=knee_bounds, n_jobs=1)

    if rsq_type == 'log':
        rsq = fm.r_squared_
    elif rsq_type == 'linear':
        rsq = np.corrcoef(10**fm.power_spectrum, 10**fm.fooofed_spectrum_)[0][1] ** 2

    return tau, rsq, fm


def mp_fit_acf(ind, sig=None, lags=None, fs=None, win_len=None, method='cos',
               compute_acf_kwargs=None, fit_kwargs=None, low_mem=False):
    """Multiprocessing wrapper for computing and fitting ACFs.

    Parameters
    ----------
    ind : int or float
        Start index when a sig is passed, or correleation coefficients when sig is None.
    sig : 1d array, optional, default: None
        Voltage time series.
    lags : 1d array, optional, default: None
        Lag steps if ind contains correlation coefficients and sig is None.
        Used in cases where in is pre-computed correlation coefficients.
    fs : float, optional, default: None
        Sampling rate, in Hz.
    win_len : int, optional, default: None
        Number of samples per window.
    method : {'cos', 'decay'}
        ACF fitting method.
    compute_acf_kwargs : dict, optional, default: None
        Additional keyword arguments to pass to compute_acf:

        - nlags
        - from_psd
        - psd_kwargs

    fit_kwargs: dict, optional, default: None
        Additional keyword agruments for either fit_acf of fit_acf_cos:

        - lags
        - guess
        - bounds
        - n_jobs
        - maxfev
        - progress

        Note, the shape of guess and bounds is dependent on method.

    low_mem : bool, optional, default: False
        Reduces memory usage.

    Returns
    -------
    tau : float
        Timescale.
    rsq : float
        Goodness of fit.
    acf : ACF
        ACF class containing intermediate arrays.
    """

    win_len = fs if win_len is None else win_len
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    if sig is not None:
        acf = ACF(low_mem=low_mem)
        acf.compute_acf(sig, fs, start=ind, win_len=win_len, **compute_acf_kwargs)
    elif sig is None:
        lags = np.arange(1, len(ind)) if lags is None else lags
        acf = ACF(ind, lags, fs, low_mem)

    if method == 'cos':
        acf.fit_cos(**fit_kwargs)
    elif method == 'decay':
        acf.fit(**fit_kwargs)

    if np.isnan(acf.params).any():
        return np.nan, np.nan, acf

    tau = acf.params[0]
    rsq = acf.rsq

    return tau, rsq, acf


def sort_result(result):
    """Sort multiprocessing outputs."""

    taus = np.array([r[0] for r in result])
    rsq = np.array([r[1] for r in result])
    result_class = [r[2] for r in result]

    return taus, rsq, result_class

"""Autocorrelation estimation methods."""

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf


def fit_acf(spikes, nlags, guess=None, bounds=None, mode='mean', n_jobs=-1, maxfev=1000):
    """Compute and fit ACF.

    Parameters
    ----------
    spikes : 1d or 2d array
        Spike counts. Optionally bin prior if desired.
        If a 2d array, the autocorrelations are fit per 1d array.
    nlags : float
        Number of lags to shift spikes by.
    guess : list, optional, default: None
        Estimated parameters as [height, tau, offset].
    bounds : list, optional, default: None
        Parameters bounds as [(*lower_bounds), (*upper_bounds)].

    Returns
    -------
    corrs : 1d or 2d array
        Correlation coefficients, per lag.
        A 2d array is return is spikes is 2d.
    params : 1d or 2d array
        Exponential decay parameters as [height, tau, offset].
        If mode is None, then a 2d array of these
    """

    # Compute acf
    if spikes.ndim == 2:
        # 2d case
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with Pool(processes=n_jobs) as pool:
            results = pool.map(partial(acf, nlags=nlags, qstat=False, fft=True), spikes)

        corrs_2d = np.array(results)[:, 1:]

        # Take mean or median of 2d acf results; if None, fit each acf separately
        if mode == 'mean':
            corrs = np.mean(corrs_2d, axis=0)
        elif mode == 'median':
            corrs = np.median(corrs_2d, axis=0)
        elif mode is None:
            corrs = corrs_2d
        else:
            raise ValueError('Requested mode must be either {\'mean\', \'median\', None}.')

        n_samples = len(spikes[0])

    elif spikes.ndim == 1:
        # 1d case
        corrs = acf(spikes, nlags=nlags, qstat=False, fft=True)[1:]
        n_samples = len(spikes)
    else:
        raise ValueError('Spike counts must be either 1d or 2d.')

    # Fit acf
    if guess is None:
        target_tau = np.argmin(np.abs(corrs - np.max(corrs) * (1/np.exp(1))))
        guess = [np.max(corrs), target_tau, 0.]

    if bounds is None:
        bounds = [
            (0, 0, -2),
            (2*np.max(corrs), target_tau * 10, 2)
        ]

    if corrs.ndim == 2:
        # Define global wrapper for curve_fit for compatibility with mp.Pool
        global _curve_fit

        def _curve_fit(corrs):
            params, _ = curve_fit(exp_decay_func, np.arange(0, len(corrs)), corrs,
                                  p0=guess, bounds=bounds, maxfev=maxfev)
            return params

        with Pool(processes=n_jobs) as pool:
            params = pool.map(_curve_fit, corrs)

        params = np.array(params)

    else:
        params, _ = curve_fit(exp_decay_func, np.arange(1, len(corrs)+1), corrs,
                              p0=guess, bounds=bounds, maxfev=maxfev)

    if spikes.ndim == 2:
        return corrs_2d, params
    else:
        return corrs, params


def exp_decay_func(delta_t, amplitude, tau, offset):
    """Exponential function to fit to autocorrelation.

    Parameters
    ----------
    delta_t : 1d array
        Time lags, acf x-axis definition.
    ampltidue : float
        Height of the exponential.
    tau : float
        Timescale.
    offset : float
        Y-intercept of the exponential.
    """

    return amplitude * (np.exp(-(delta_t / tau)) + offset)

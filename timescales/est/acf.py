"""Autocorrelation estimation methods."""

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf

from neurodsp.filt import filter_signal
from neurodsp.spectral import compute_spectrum

from timescales.sim import sim_acf_cos, exp_decay_func
from timescales.est.utils import progress_bar, check_guess_and_bounds


def compute_acf(spikes, nlags, mode=None, n_jobs=-1, progress=None):
    """Compute the autocorrelation for a spiking series.

    Parameters
    ----------
    spikes : 1d or 2d array
        Spike counts or probabilities.
    nlags : float
        Number of lags to compute autocorrelation over.
    mode : {None, 'mean', 'median'}
        Used when spikes is 2d. Specifices computing the ACF as either the mean or median of the
        ACF. Returns the 2d array of correlation when None.
    n_jobs : int
        Number of jobs to run in parralel, when corrs is 2d.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    corrs : 1d or 2d array
        Autcorrelation.
    """

    # Compute acf
    if spikes.ndim == 1:
        corrs = acf(spikes, nlags=nlags, qstat=False, fft=True)[1:]
    elif spikes.ndim == 2:
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with Pool(processes=n_jobs) as pool:

            mapping = pool.map(partial(acf, nlags=nlags, qstat=False, fft=True), spikes)

            results = list(progress_bar(mapping, progress, len(spikes), 'Computing ACF'))

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
    else:
        raise ValueError('Spike counts must be either 1d or 2d.')

    return corrs


def fit_acf(corrs, fs, guess=None, bounds=None, n_jobs=-1, maxfev=1000, progress=None):
    """Compute and fit ACF.

    Parameters
    ----------
    corrs : 1d or 2d array
        Autocorrelation coefficients.
    fs : float
        Sampling rate, in Hz.
    guess : list, optional, default: None
        Estimated parameters as [height, tau, offset].
    bounds : list, optional, default: None
        Parameters bounds as [(*lower_bounds), (*upper_bounds)].
    n_jobs : int
        Number of jobs to run in parralel, when corrs is 2d.
    maxfev : int
        Maximum number of fitting iterations.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    params : 1d or 2d array
        Exponential decay parameters as [height, tau, offset].
        If mode is None, then a 2d array of these
    """

    if corrs.ndim == 1:
        params = _fit_acf(corrs, fs, guess, bounds, maxfev)
    elif corrs.ndim == 2:
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Ensure guess and bounds are zipable
        guess, bounds = check_guess_and_bounds(corrs, guess, bounds)

        # Proxy function to organize args
        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_acf_proxy, fs=fs, maxfev=maxfev),
                                zip(corrs, guess, bounds))

            params = list(progress_bar(mapping, progress, len(corrs)))

        params = np.array(params)

    return params


def fit_acf_cos(corrs, fs, guess=None, bounds=None, maxfev=1000, n_jobs=-1, progress=None):
    """Fit an autocorraltion as the sum of exponential and cosine components.

    Parameters
    ----------
    corrs : 1d or 2d array
        Autocorrelation coefficients.
    fs : float
        Sampling rate, in Hz.
    guess : list, optional, default: None
        Estimated parameters as [height, tau, offset].
    bounds : list, optional, default: None
        Parameters bounds as [(*lower_bounds), (*upper_bounds)].
    n_jobs : int
        Number of jobs to run in parralel, when corrs is 2d.
    maxfev : int
        Maximum number of fitting iterations.
    n_jobs : int
        Number of jobs to run in parralel, when corrs is 2d.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    params : 1d or 2d array
        Fit params as [tau, gamma, var_exp, var_cos, var_cos_exp, freq].
    """

    if corrs.ndim == 1:
        params = _fit_acf_cos(corrs, fs, guess, bounds, maxfev)

    elif corrs.ndim == 2:

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Ensure guess and bounds are zipable
        guess, bounds = check_guess_and_bounds(corrs, guess, bounds)

        # Proxy function to organize args
        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_acf_cos_proxy, fs=fs, maxfev=maxfev),
                                zip(corrs, guess, bounds))

            params = list(progress_bar(mapping, progress, len(corrs)))

        params = np.array(params)

    return params


def _fit_acf(corrs, fs, guess=None, bounds=None, maxfev=1000):
    """Fit 1d ACF."""

    if guess is not None:
        target_tau = guess[1]
    else:
        inds = np.argsort(np.abs(corrs - np.max(corrs) * (1/np.exp(1))))
        target_tau = inds[1] if inds[0] == 0 else inds[0]

    if guess is None:
        guess = [target_tau, np.max(corrs), 0.]

    _bounds = [
        (0, 0, -2),
        (target_tau * 10, 2*np.max(corrs), 2)
    ]

    if bounds is None:
        bounds = _bounds
    else:
        _bounds = np.array(_bounds)

        xinds, yinds = np.where(bounds == None)
        if len(xinds) != 0:
            for x, y in zip(xinds, yinds):
                bounds[x, y] = _bounds[x, y]

        bounds = [tuple(b) for b in bounds.tolist()]

    # If guess is outside of bounds,
    #   set to midpoint of bounds
    for ind, g in enumerate(guess):
        if g <= bounds[0][ind] or g >= bounds[1][ind]:
            guess[ind] = (bounds[0][ind] + bounds[1][ind]) / 2

    try:
        params, _ = curve_fit(
            lambda xs, t, amp, off : exp_decay_func(xs, fs, t, amp, off),
            np.arange(1, len(corrs)+1), corrs, p0=guess, bounds=bounds, maxfev=maxfev
        )
    except RuntimeError:
        params = np.nan

    return params


def _fit_acf_cos(corrs, fs, guess=None, bounds=None, maxfev=1000):
    """Fit 1d ACF with cosine."""

    xs = np.arange(1, len(corrs)+1)

    if guess is None or bounds is None:

        # Compute spectrum of autocorrs to determine cos freq
        f, p = compute_spectrum(corrs, len(corrs))
        freq = int(np.argmax(p))

        # Tau estimation
        inds = np.where(np.diff(np.sign(np.diff(corrs))) < 0)[0] + 1

        exp_est = corrs[inds].copy()
        exp_est -= np.min(exp_est)
        exp_est_interp = np.interp(np.arange(xs[inds][0], xs[inds][-1]+1), xs[inds], exp_est)
        exp_est_bl = exp_est_interp - exp_est_interp[0] / np.exp(1)

        _inds = np.where(exp_est_bl < 0)[0]

        if len(_inds) == 0:
            tau_guess = inds[0] / fs
        else:
            pts = [_inds[0]-2, _inds[0]-1]
            tau_guess = (pts[np.argmin(exp_est_bl[pts])] + inds[0]) / fs

        # Fit
        _guess = [tau_guess, 1, 0, 1, 1, .5]

        _bounds = [
            (tau_guess * .1, 0, -.5, .1, .1, 0),
            (tau_guess *  1, 1, .5,  1,  1, 1)
        ]

    if bounds is None:
        bounds = _bounds
    else:
        _bounds = np.array(_bounds)

        xinds, yinds = np.where(bounds == None)
        if len(xinds) != 0:
            for x, y in zip(xinds, yinds):
                bounds[x, y] = _bounds[x, y]

        bounds = [tuple(b) for b in bounds.tolist()]

    if guess is None:
        guess = _guess

    # If guess is outside of bounds,
    #   set to midpoint of bounds
    for ind, g in enumerate(guess):
        if g <= bounds[0][ind] or g >= bounds[1][ind]:
            guess[ind] = (bounds[0][ind] + bounds[1][ind]) / 2

    try:
        params, _ = curve_fit(
            lambda xs, t, amp, off, g, vc, vce : sim_acf_cos(xs, fs, t, amp, off,
                                                             g, vc, vce, freq, True),
            xs, corrs, p0=guess, bounds=bounds, maxfev=maxfev
        )
        params = np.append(params, freq)
    except RuntimeError:
        params = np.nan

    return params


def _acf_proxy(args, fs, maxfev):
    corrs, guess, bounds = args
    params = _fit_acf(corrs, fs, guess, bounds, maxfev)
    return params


def _acf_cos_proxy(args, fs, maxfev):
    corrs, guess, bounds = args
    params = _fit_acf_cos(corrs, fs, guess, bounds, maxfev)
    return params

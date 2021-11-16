"""Autocorrelation estimation methods."""

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf

from neurodsp.filt import filter_signal
from neurodsp.spectral import compute_spectrum

from timescales.sim.acf import sim_acf_cos, sim_exp, sim_osc
from timescales.sim.exp import exp_decay_func
from timescales.est.utils import progress_bar, check_guess_and_bounds


class ACF:
    """Autocorrelation function class.

    Parameters
    ----------
    sig : 1d array
        LFP or spike counts or spike probabilities.
    fs : float
        Sampling rate, in Hz.
    corrs : 1d array
        Autocorrelation coefficients.
    corrs_fit : 1d array
        Autocorrelation coefficient full fit.
    corrs_fit_cos : 1d array
        Damped oscillation fit. Only defined if the fit_cos method is called.
    params : 1d array
        Fit parameters. If using the fit method: [tau, height, offset]
    cos_params : 1d array
        If using the sequential cosine method, this attribute will contain:
        [osc_tau, osc_amp, osc_gamma, osc_freq].
    rsq : float
        R-squared of the full fit.
    """

    def __init__(self):
        """Initialize object."""

        self.sig = None
        self.fs = None

        self.params = None
        self.cos_params = None
        self.corrs = None
        self.corrs_fit = None
        self.corrs_fit_cos = None
        self.rsq = None


    def compute_acf(self, sig, fs, start, win_len, nlags=None):
        """Compute autocorrelation

        Parameters
        ----------
        sig : 1d array
            LFP or spike counts or spike probabilities.
        fs : float
            Sampling rate, in Hz.
        start : int
            Index of the starting point to compute the ACF.
        win_len : int
            Window length, in samples.
        nlags : int, optional, default: None
            Number of lags to compute. None defaults to the sampling rate, fs.
        """
        self.sig = sig
        self.fs = fs
        nlags = self.fs if nlags is None else nlags
        self.corrs = compute_acf(self.sig[start:start+win_len], nlags)


    def fit(self, **fit_kwargs):
        """Fit without an oscillitory component."""

        self.params = fit_acf(self.corrs, self.fs, **fit_kwargs)

        self.corrs_fit = exp_decay_func(np.arange(1, len(self.corrs)+1),
                                        self.fs, *self.params)

        self.rsq = np.corrcoef(self.corrs, self.corrs_fit)[0][1] ** 2


    def fit_cos(self, **fit_kwargs):
        """Fit with an oscillitory component."""

        xs = np.arange(1, len(self.corrs)+1)

        params = fit_acf_cos(self.corrs, self.fs, **fit_kwargs)

        if 'sequential' not in fit_kwargs.keys():
            fit_kwargs['sequential'] = True

        if 'sequential' in fit_kwargs.keys() and fit_kwargs['sequential'] == True:
            self.cos_params = params[0]
            self.params = params[1]

            cos = sim_osc(xs, self.fs, *self.cos_params)
            exp = sim_exp(xs, self.fs, *self.params)

            self.corrs_fit = cos + exp
            self.corrs_fit_cos = cos
            self.corrs_fit_exp = exp
        else:
            self.params = params
            self.corrs_fit = sim_acf_cos(np.arange(1, len(self.corrs)+1), self.fs, *self.params)

        self.rsq = np.corrcoef(self.corrs, self.corrs_fit)[0][1] ** 2


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
        Estimated parameters as [tau, height, offset].
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


def fit_acf_cos(corrs, fs, sequential=True, guess=None, bounds=None,
                maxfev=1000, n_jobs=-1, progress=None):
    """Fit an autocorraltion as the sum of exponential and cosine components.

    Parameters
    ----------
    corrs : 1d or 2d array
        Autocorrelation coefficients.
    fs : float
        Sampling rate, in Hz.
    sequential : bool, optional, default:True
        Whether to fit the damped cosine and exponential decay sequentially (True),
        rather than simulataneously (False).
    guess : list or list of list, optional, default: None
        Estimated parameters as [height, tau, offset], or as
        [[cos_guess...], [exp_guess...]].
    bounds : list or list of list, optional, default: None
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
        Fit params as [exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma, offset, osc_freq].
    """

    if sequential and guess is not None:
        osc_guess, exp_guess = guess[0], guess[1]
    elif sequential:
        osc_guess, exp_guess = None, None

    if sequential and bounds is not None:
        osc_bounds, exp_bounds = bounds[0], bounds[1]
    elif sequential:
        osc_bounds, exp_bounds = None, None

    if corrs.ndim == 2:
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    # 1D
    if corrs.ndim == 1 and not sequential:
        params = _fit_acf_cos(corrs, fs, guess, bounds, maxfev)
    elif corrs.ndim == 1 and sequential:
        cos_params, exp_params = _fit_acf_cos_seq(corrs, fs, osc_guess, osc_bounds,
                                                  exp_guess, exp_bounds, maxfev)
        params = (cos_params, exp_params)
    # 2D
    elif corrs.ndim == 2 and not sequential:

        # Ensure guess and bounds are zipable
        guess, bounds = check_guess_and_bounds(corrs, guess, bounds)

        # Proxy function to organize args
        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_acf_cos_proxy, fs=fs, maxfev=maxfev),
                                zip(corrs, guess, bounds))

            params = list(progress_bar(mapping, progress, len(corrs)))

        params = np.array(params)

    elif corrs.ndim == 2 and not sequential:

        # Ensure guess and bounds are zipable
        osc_guess, osc_bounds = check_guess_and_bounds(corrs, guess[0], bounds[0])
        exp_guess, exp_bounds = check_guess_and_bounds(corrs, guess[1], bounds[1])

        # Proxy function to organize args
        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_acf_cos_proxy, fs=fs, maxfev=maxfev),
                                zip(corrs, osc_guess, osc_bounds, exp_guess, exp_bounds))

            params = list(progress_bar(mapping, progress, len(corrs)))

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


def _fit_acf_cos_seq(corrs, fs, osc_guess=None, osc_bounds=None,
                     exp_guess=None, exp_bounds=None, maxfev=1000):
    """Fit 1d ACF with cosine, seqentially."""

    xs = np.arange(1, len(corrs)+1)

    # Fit a damped cosine and remove it from the acf
    osc_bounds = [[1e-3, 0, .01, 0], [100, 10, .2, 10]] if osc_bounds is None else osc_bounds

    osc_guess = [1, 1, .1, 1] if osc_guess is None else osc_guess

    cos_params, _ = curve_fit(
        lambda xs, osc_tau, osc_amp, osc_gamma, osc_freq : sim_osc(xs, fs, osc_tau, osc_amp,
                                                                   osc_gamma, osc_freq),
        np.arange(1, len(corrs)+1), corrs, p0=osc_guess, bounds=osc_bounds, maxfev=maxfev
    )

    cos_fit = sim_osc(xs, fs, *cos_params)

    corrs_cos_rm = corrs - cos_fit

    # Fit an exponential decay to the damped cosine removed acf
    exp_bounds = [[1e-4, 1e-3, -.1], [5, 1, .1]] if exp_bounds is None else exp_bounds

    exp_guess = [.1, 1, 0]

    exp_params, _ = curve_fit(
            lambda xs, t, amp, off : sim_exp(xs, fs, t, amp, off),
            np.arange(1, len(corrs)+1), corrs_cos_rm, p0=exp_guess,
            bounds=exp_bounds, maxfev=maxfev
    )

    return (cos_params, exp_params)


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
        _guess = [tau_guess, 1, .5, 1, 0, .5, freq]

        _bounds = [
            (tau_guess * .01, 0, 0, 0, 0, 0, 0),
            (tau_guess *  1, 1, 1, 1, .1, 10, 100)
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
            lambda xs, exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma, freq, offset: \
                sim_acf_cos(xs, fs, exp_tau, exp_amp, osc_tau, osc_amp,
                            osc_gamma, freq, offset),
            xs, corrs, p0=guess, bounds=bounds, maxfev=maxfev
        )

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

def _fit_acf_cos_seq_proxy(args, fs, maxfev):
    corrs, osc_guess, osc_bounds, exp_guess, exp_bounds = args

    params = _fit_acf_cos_seq(corrs, fs, osc_guess, osc_bounds, exp_guess, exp_bounds, maxfev)

    return params

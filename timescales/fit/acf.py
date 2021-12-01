"""Autocorrelation estimation methods."""

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import ifft
from statsmodels.tsa.stattools import acf

from neurodsp.spectral import compute_spectrum

from timescales.sim.acf import sim_acf_cos, sim_exp_decay, sim_damped_cos
from timescales.fit.utils import progress_bar, check_guess_and_bounds


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
        Exponential decay fit params, and optionally damped cosine parameters, along with an offset
        parameter.
    params_exp : 1d array
        Exponential decay fit parameters: [tau, height, offset]
    params_cos : 1d array
        Damped cosine fit parameters:
        [osc_tau, osc_amp, osc_gamma, osc_freq].
    rsq : float
        R-squared of the full fit.
    low_mem : bool, optional, default: False
        If true, only correlation coefficient arrays are stored.
    """

    def __init__(self, corrs=None, lags=None, fs=None, low_mem=False):
        """Initialize object."""

        self.sig = None
        self.fs = None if fs is None else fs
        self.corrs = None if corrs is None else corrs
        self.lags = None if lags is None else lags

        self.guess = None
        self.bounds = None

        self.params = None
        self.param_exp = None
        self.params_cos = None


        self.corrs_fit = None
        self.corrs_fit_exp = None
        self.corrs_fit_cos = None

        self.rsq = None
        self.low_mem = low_mem


    def compute_acf(self, sig, fs, start=0, win_len=None, nlags=None,
                    from_psd=False, psd_kwargs=None):
        """Compute autocorrelation.

        Parameters
        ----------
        sig : 1d array
            LFP or spike counts or spike probabilities.
        fs : float
            Sampling rate, in Hz.
        start : int, optional, default: 0
            Index of the starting point to compute the ACF.
        win_len : int, optional, default: None
            Window length, in samples. If None, default to fs.
        nlags : int, optional, default: None
            Number of lags to compute. None defaults to the sampling rate, fs.
        from_psd : bool, optional, default: False
            Compute correlations from the inverse FFT of the PSD.
        psd_kwargs : dict, optional, default: None
            Compute spectrum kwargs. Only used if from_psd is True.
        """
        self.sig = sig
        self.fs = fs

        win_len = fs if win_len is None else win_len
        nlags = self.fs if nlags is None else nlags

        if not from_psd:
            self.corrs = compute_acf(self.sig[start:start+win_len], nlags)
        else:
            psd_kwargs = {} if psd_kwargs is None else psd_kwargs
            _, _powers = compute_spectrum(self.sig[start:start+win_len], self.fs, **psd_kwargs)
            self.corrs = ifft(_powers).real
            self.corrs = self.corrs[:len(self.corrs)//2]

        if self.low_mem:
            self.sig = None


    def fit(self, **fit_kwargs):
        """Fit without an oscillitory component."""

        self.lags = np.arange(1, len(self.corrs)+1) if self.lags is None else self.lags

        self.params, self.guess, self.bounds = fit_acf(self.corrs, self.fs, **fit_kwargs)

        if not np.isnan(self.params).any():
            self.corrs_fit = sim_exp_decay(self.lags, self.fs, *self.params)
            self.rsq = np.corrcoef(self.corrs, self.corrs_fit)[0][1] ** 2

        if self.low_mem:
            self.lags = None
            self.corrs_fit = None


    def fit_cos(self, **fit_kwargs):
        """Fit with an oscillitory component."""

        self.lags = np.arange(1, len(self.corrs)+1) if self.lags is None else self.lags

        self.params, self.guess, self.bounds = fit_acf_cos(self.corrs, self.fs, **fit_kwargs)

        if not np.isnan(self.params).any():
            self.params_exp = self.params[:2]
            self.params_cos = self.params[2:-1]

            self.corrs_fit = sim_acf_cos(self.lags, self.fs, *self.params)
            self.corrs_fit_exp = sim_exp_decay(self.lags, self.fs, *self.params_exp)
            self.corrs_fit_cos = sim_damped_cos(self.lags, self.fs, *self.params_cos)

            self.rsq = np.corrcoef(self.corrs, self.corrs_fit)[0][1] ** 2

        if self.low_mem:
            self.lags = None
            self.corrs_fit = None
            self.corrs_fit_exp = None
            self.corrs_fit_cos = None

    def gen_corrs_fit(self):

        if len(self.params) == 3:
            return sim_exp_decay(np.arange(1, len(self.corrs)+1), self.fs, *self.params)
        else:
            return sim_acf_cos(np.arange(1, len(self.corrs)+1), self.fs, *self.params)


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


def fit_acf(corrs, fs, lags=None, guess=None, bounds=None, n_jobs=-1, maxfev=1000, progress=None):
    """Compute and fit ACF.

    Parameters
    ----------
    corrs : 1d or 2d array
        Autocorrelation coefficients.
    fs : float
        Sampling rate, in Hz.
    lags : 1d array, optional, default: None
        Lags each coefficient was computed for.
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
        Exponential decay parameters as [tau, height, offset].
    guess : 1d or 2d array
        Curve fit initial guess parameters.
    bounds : 1d or 2d array
        Curve fit parameter bounds.
    """



    if corrs.ndim == 1:

        lags = np.arange(1, len(corrs)+1) if lags is None else lags
        params, guess, bounds = _fit_acf(corrs, lags, fs, guess, bounds, maxfev)

    elif corrs.ndim == 2:

        lags = np.arange(1, len(corrs[0])+1) if lags is None else lags

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Ensure guess and bounds are zipable
        guess, bounds = check_guess_and_bounds(corrs, guess, bounds)

        # Proxy function to organize args
        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_acf_proxy, lags=lags, fs=fs, maxfev=maxfev),
                                zip(corrs, guess, bounds))

            params = list(progress_bar(mapping, progress, len(corrs)))

        guess = np.array([i[1] for i in params])
        bounds = np.array([i[2] for i in params])
        params = np.array([i[0] for i in params])

    return params, guess, bounds


def fit_acf_cos(corrs, fs, lags=None, guess=None, bounds=None,
                maxfev=1000, n_jobs=-1, progress=None):
    """Fit an autocorraltion as the sum of exponential and cosine components.

    Parameters
    ----------
    corrs : 1d or 2d array
        Autocorrelation coefficients.
    fs : float
        Sampling rate, in Hz.
    lags : 1d array, optional, default: None
        Lags each coefficient was computed for.
    guess : list or list of list, optional, default: None
        Estimated parameters as [height, tau, offset].
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
        Fit params as [exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma, osc_freq, offset].
    """



    if corrs.ndim == 1:

        lags = np.arange(1, len(corrs)+1) if lags is None else lags
        params, guess, bounds = _fit_acf_cos(corrs, lags, fs, guess, bounds, maxfev)
        params = np.array(params)

    elif corrs.ndim == 2:

        lags = np.arange(1, len(corrs[0])+1) if lags is None else lags

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Ensure guess and bounds are zipable
        guess, bounds = check_guess_and_bounds(corrs, guess, bounds)

        # Proxy function to organize args
        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_acf_cos_proxy, lags=lags, fs=fs, maxfev=maxfev),
                                zip(corrs, guess, bounds))

            params = list(progress_bar(mapping, progress, len(corrs)))

        guess = np.array([i[1] for i in params])
        bounds = np.array([i[2] for i in params])
        params = np.array([i[0] for i in params])

    return params, guess, bounds


def _fit_acf(corrs, lags, fs, guess=None, bounds=None, maxfev=1000):
    """Fit 1d ACF."""

    if guess is not None:
        target_tau = guess[1]
    else:
        inds = np.argsort(np.abs(corrs - np.max(corrs) * (1/np.exp(1))))
        target_tau = inds[1] if inds[0] == 0 else inds[0]

    target_tau /= fs

    if guess is None:
        guess = [target_tau, np.max(corrs), 0.]

    _bounds = [
        (0, 0, -.5),
        (target_tau * 10, 2, .5)
    ]

    if bounds is None:
        bounds = _bounds
    else:
        _bounds = np.array(_bounds)
        bounds = np.array(bounds)

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
            lambda lags, t, amp, off : sim_exp_decay(lags, fs, t, amp, off),
            np.arange(1, len(corrs)+1), corrs, p0=guess, bounds=bounds, maxfev=maxfev
        )
    except RuntimeError:
        params = np.nan

    return params, guess, bounds


def _fit_acf_cos(corrs, lags, fs, guess=None, bounds=None, maxfev=1000):
    """Fit 1d ACF with cosine."""

    if guess is None or bounds is None:

        # Compute spectrum of autocorrs to determine cos freq
        f, p = compute_spectrum(corrs, len(corrs))
        freq = int(np.argmax(p))

        # Tau estimation
        inds = np.where(np.diff(np.sign(np.diff(corrs))) <= 0)[0] + 1

        if len(inds) == 0:
            inds = np.arange(len(corrs))

        exp_est = corrs[inds].copy()
        exp_est -= np.min(exp_est)
        exp_est_interp = np.interp(np.arange(lags[inds][0], lags[inds][-1]+1), lags[inds], exp_est)
        exp_est_bl = exp_est_interp - exp_est_interp[0] / np.exp(1)

        _inds = np.where(exp_est_bl < 0)[0]

        if len(_inds) == 0:
            tau_guess = inds[0] / fs
        else:
            pts = [_inds[0]-2, _inds[0]-1]
            tau_guess = (pts[np.argmin(exp_est_bl[pts])] + inds[0]) / fs

        # Fit
        _guess = [tau_guess, 5, 0, freq, .5, np.max(corrs), 0]

        _bounds = [
            (0,              0,  0, 0,  0, 0, -.5),
            (tau_guess * 10, 1, .1, 10, 1, 1, .5)
        ]

    if bounds is None:
        bounds = _bounds
    else:
        bounds = np.array(bounds)

        xinds, yinds = np.where(bounds == None)
        if len(xinds) != 0:
            for x, y in zip(xinds, yinds):
                bounds[x, y] = _bounds[x][y]

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
            lambda lags, exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma, freq, offset: \
                sim_acf_cos(lags, fs, exp_tau, exp_amp, osc_tau, osc_amp,
                            osc_gamma, freq, offset),
            lags, corrs, p0=guess, bounds=bounds, maxfev=maxfev
        )

    except RuntimeError:
        params = np.nan

    return params, guess, bounds


def _acf_proxy(args, lags, fs, maxfev):
    corrs, guess, bounds = args
    params, guess, bounds = _fit_acf(corrs, lags, fs, guess, bounds, maxfev)
    return params, guess, bounds


def _acf_cos_proxy(args, lags, fs, maxfev):
    corrs, guess, bounds = args
    params, guess, bounds = _fit_acf_cos(corrs, lags, fs, guess, bounds, maxfev)
    return params, guess, bounds

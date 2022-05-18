"""Autocorrelation estimation methods."""

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import ifft
from statsmodels.tsa.stattools import acf

from neurodsp.spectral import compute_spectrum
from timescales.autoreg import compute_ar_spectrum

from timescales.sim.acf import sim_acf_cos, sim_exp_decay, sim_damped_cos
from timescales.utils import normalize as normalize_acf
from timescales.conversions import convert_knee
from timescales.fit.utils import progress_bar, check_guess_and_bounds


class ACF:
    """Autocorrelation function class.

    Parameters
    ----------
    lags : 1d array
        Time lag definitions.
    corrs : 1d or 2d array
        Autocorrelation coefficients.
    fs : float
        Sampling rate, in Hz.

    Attributes
    ----------
    corrs_fit : 1d array
        Autocorrelation full fit.
    params : 1d array
        Optimized parameters.
    param_names : list of str
        Parameter names in order of params.
    rsq : float
        R-squared of the full fit.
    guess : list, optional, default: None
        Estimated parameters as either:
        [tau, height, offset] when with_cos is False, or
        [exp_tau, osc_tau, osc_gamma, osc_freq, amp_ratio, height, offset].
    bounds : list, optional, default: None
        Parameters bounds as [(*lower_bounds), (*upper_bounds)].

    Notes
    -----
    Parameters may be set on initialization or using the compute_acf method.
    """

    def __init__(self, lags=None, corrs=None, fs=None):
        """Initialize object."""

        self.lags = lags
        self.corrs = corrs
        self.fs = fs

        # Set via other methods
        self.guess = None
        self.bounds = None

        self.params = None
        self.param_names = None
        self.param_exp = None
        self.params_cos = None
        self.tau = None
        self.knee_freq = None

        self.corrs_fit = None

        self.corrs_fit_exp = None
        self.corrs_fit_cos = None
        self.params_exp = None
        self.params_cos = None

        self.rsq = None

        # For comparison to PSD models
        self.tau = None
        self.knee_freq = None


    def compute_acf(self, sig, fs, nlags=None, normalize=True, from_psd=False,
                    psd_kwargs=None, n_jobs=-1, progress=None):
        """Compute autocorrelation.

        Parameters
        ----------
        sig : 1d or 2d array
            Voltage time series or spike counts.
        fs : float
            Sampling rate, in Hz.
        nlags : int, optional, default: None
            Number of lags to compute. None defaults to the sampling rate, fs.
        normalize : bool, optional, default: True
            Normalizes from zero to one when True.
        from_psd : bool, optional, default: False
            Compute correlations from the inverse FFT of the PSD.
        psd_kwargs : dict, optional, default: None
            Compute spectrum kwargs. Only used if from_psd is True.
        n_jobs : int
            Number of jobs to run in parralel, when corrs is 2d.
            Default is equal to multiprocessing's cpu_count().
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
        """

        self.fs = fs
        nlags = self.fs if nlags is None else nlags

        if not from_psd:

            # Compute ACF
            if sig.ndim == 1:
                self.corrs = acf(sig, nlags=nlags, qstat=False, fft=True)[1:]
                self.lags = np.arange(1, len(self.corrs)+1)
            elif sig.ndim == 2:

                n_jobs = cpu_count() if n_jobs == -1 else n_jobs

                with Pool(processes=n_jobs) as pool:
                    mapping = pool.map(partial(acf, nlags=nlags, qstat=False, fft=True), sig)
                    results = list(progress_bar(mapping, progress, len(sig), 'Computing ACF'))

                self.corrs = np.array(results)[:, 1:]
                self.lags = np.arange(1, len(self.corrs[0])+1)
            else:
                raise ValueError('sig must be either 1d or 2d.')

        else:

            # Handle kwargs
            psd_kwargs = {} if psd_kwargs is None else psd_kwargs
            _psd_kwargs = psd_kwargs.copy()
            norm_range = _psd_kwargs.pop('norm_range', None)

            # Compute spectrum
            if 'ar_order' in psd_kwargs:
                ar_order = _psd_kwargs.pop('ar_order')
                _, powers = compute_ar_spectrum(sig, self.fs, ar_order, **_psd_kwargs)
            else:
                _, powers = compute_spectrum(sig, self.fs, **_psd_kwargs)

            # Normalize power if requested
            if norm_range is not None:
                powers = normalize_acf(powers, *norm_range)

            # Take inverse fft to get acf
            if sig.ndim == 2:

                for ind in range(len(powers)):

                    _corrs = ifft(powers[ind]).real

                    if ind == 0:
                        self.corrs = np.zeros((len(powers), len(_corrs)//2))

                    self.corrs[ind] = _corrs[:len(_corrs)//2]

                self.lags = np.arange(1, len(self.corrs[0])+1)
                self.corrs = self.corrs[:, :nlags]

            else:

                self.corrs = ifft(powers).real
                self.corrs = self.corrs[:len(self.corrs)//2]
                self.lags = np.arange(1, 2*len(self.corrs)+1, 2)
                self.corrs = self.corrs[:nlags]

            self.lags = self.lags[:nlags]

        if normalize:
            self.corrs = normalize_acf(self.corrs, 0, 1)


    def fit(self, gen_fits=True, gen_components=False, with_cos=False,
            guess=None, bounds=None, maxfev=1000, n_jobs=-1, progress=None):
        """Fit without an oscillitory component.

        Parameters
        ----------
        gen_fits : bool, optional, default: False
            Generates fit array and r-squared when True.
            Does not generate full fits when False to prevent OOM.
        gen_components : bool, optional, default: False
            When gen_fits and with_cos are True, the exponential decay and cosine
            components are be generated separately when this parameter is True.
        with_cos : bool, optional, default: False
            Includes oscillatory component as a damped cosine.
        guess : list, optional, default: None
            Estimated parameters as either:
            [tau, height, offset] when with_cos is False, or
            [exp_tau, osc_tau, osc_gamma, osc_freq, amp_ratio, height, offset].
        bounds : list, optional, default: None
            Parameters bounds as [(*lower_bounds), (*upper_bounds)].
        maxfev : int
            Maximum number of fitting iterations.
        n_jobs : int
            Number of jobs to run in parralel, when corrs is 2d.
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
        """

        if self.corrs is None or self.lags is None:
            raise ValueError('Initialize with lags, corrs, and fs. '
                             'Or call compute_acf prior to fitting.')

        if not with_cos:
            # Non-oscillatory model
            self.param_names = ['tau', 'height', 'offset']
            self.params, self.guess, self.bounds = fit_acf(self.corrs, self.fs, self.lags, guess=guess,
                bounds=bounds, maxfev=maxfev, n_jobs=n_jobs, progress=progress)
        else:
            # Oscillatory model
            self.param_names = ['exp_tau', 'osc_tau', 'osc_gamma', 'osc_freq',
                                'amp_ratio', 'height', 'offset']
            self.params, self.guess, self.bounds = fit_acf_cos(self.corrs, self.fs, self.lags,
                guess=guess, bounds=bounds, maxfev=maxfev, n_jobs=n_jobs, progress=progress)

        if gen_fits:
            self.gen_corrs_fit(gen_components)

        self.tau = self.params[0]
        self.knee_freq = convert_knee(self.tau)


    def gen_corrs_fit(self, gen_components=False):
        """Generate fit and r-squared.

        Parameters
        ----------
        gen_components : bool, optional, default: True
            Generates oscillatory and exponential components separately, in additon to
            combined, when True.
        """

        sim_func = sim_exp_decay if self.params.shape[-1] == 3 else sim_acf_cos

        if self.corrs.ndim == 2:

            self.corrs_fit = np.zeros((len(self.params), len(self.corrs[0])))
            self.rsq = np.zeros(len(self.params))

            for ind in range(len(self.params)):
                self.corrs_fit[ind] = sim_func(self.lags, self.fs, *self.params[ind])
                self.rsq[ind] = np.corrcoef(self.corrs, self.corrs_fit[ind])[0][1] ** 2

        else:
            self.corrs_fit = sim_func(self.lags, self.fs, *self.params)
            self.rsq = np.corrcoef(self.corrs, self.corrs_fit)[0][1] ** 2

        # Separate oscillatory and exponential decay compoents
        n_params = len(self.params) if self.corrs.ndim == 1 else len(self.params[0])
        n_corrs = 1 if self.corrs.ndim == 1 else len(self.corrs)

        if gen_components and n_params != 3:

            if n_corrs > 1:

                self.params_exp = np.zeros((n_corrs, 2))
                self.params_cos = np.zeros((n_corrs, 4))
                self.corrs_fit_exp = np.zeros((n_corrs, len(self.corrs[0])))
                self.corrs_fit_cos = np.zeros((n_corrs, len(self.corrs[0])))

                for ind in range(n_corrs):

                    exp_tau, osc_tau, osc_gamma, osc_freq, amp_ratio, _, _ = self.params[ind]

                    self.params_exp[ind] = np.array([exp_tau, amp_ratio])
                    self.params_cos[ind] = np.array([osc_tau, 1-amp_ratio, osc_gamma, osc_freq])

                    self.corrs_fit_exp[ind] = sim_exp_decay(self.lags, self.fs, exp_tau, amp_ratio)
                    self.corrs_fit_cos[ind] = sim_damped_cos(self.lags, self.fs, osc_tau,
                        1-amp_ratio, osc_gamma, osc_freq)


            else:
                exp_tau, osc_tau, osc_gamma, osc_freq, amp_ratio, _, _ = self.params

                self.params_exp = np.array([exp_tau, amp_ratio])
                self.params_cos = np.array([osc_tau, 1-amp_ratio, osc_gamma, osc_freq])

                exp = sim_exp_decay(self.lags, self.fs, exp_tau, amp_ratio)
                osc = sim_damped_cos(self.lags, self.fs, osc_tau, 1-amp_ratio, osc_gamma, osc_freq)

                self.corrs_fit_exp = exp
                self.corrs_fit_cos = osc

        elif gen_components:
            raise ValueError('Call the fit method with fit_cos=True for separable components.')


    def plot(self, ax):
        """Plot ACF.

        Parameters
        ----------
        ax : AxesSubplot, optional, default: None
            Axis to plot on.
        """

        if self.corrs is None:
            raise ValueError('corrs and lags are undefined.')
        else:
            ax.plot(self.lags, self.corrs, label='ACF')

        if self.corrs_fit is not None:
            ax.plot(self.lags, self.corrs_fit, label='Fit', ls='--')

        ax.legend()
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Lags')


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
            lags, corrs, p0=guess, bounds=bounds, maxfev=maxfev
        )
    except RuntimeError:
        params = np.nan

    return params, guess, bounds


def _fit_acf_cos(corrs, lags, fs, guess=None, bounds=None, maxfev=1000):
    """Fit 1d ACF with cosine."""

    if (guess is None or bounds is None) or (None in guess or None in bounds):

        # Compute spectrum of autocorrs to determine cos freq
        _, p = compute_spectrum(corrs, len(corrs))
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
    else:
        guess = np.array(guess)
        inds = np.where(guess == None)
        if len(inds) != 0:
            for ind in inds[0]:
                guess[ind] = _guess[ind]
        guess = guess.tolist()

    # If guess is outside of bounds,
    #   set to midpoint of bounds
    for ind, g in enumerate(guess):
        if g <= bounds[0][ind] or g >= bounds[1][ind]:
            guess[ind] = (bounds[0][ind] + bounds[1][ind]) / 2

    try:
        params, _ = curve_fit(
            lambda lags, exp_tau, osc_tau, osc_gamma, freq, amp_ratio, height, offset: \
                sim_acf_cos(lags, fs, exp_tau, osc_tau, osc_gamma, freq,
                            amp_ratio, height, offset),
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

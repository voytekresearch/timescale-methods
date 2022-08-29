"""Spectral estimation methods."""

from itertools import repeat
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from neurodsp.spectral import compute_spectrum
from timescales.autoreg import compute_ar_spectrum

from fooof import FOOOF, FOOOFGroup
from fooof.core.funcs import expo_const_function, expo_double_const_function

from timescales.conversions import convert_knee
from timescales.utils import normalize as normalize_psd
from timescales.utils import resample_logspace
from timescales.fit.utils import progress_bar


class PSD:
    """Power spectral density class.

    Parameters
    ----------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    powers : 1d or 2d array
        Power spectral density.

    Attributes
    ----------
    powers_fit : 1d or 2d array
        Aperiodic fit.
    params : 1d or 2d array
        Parameters as [offset, knee_freq, exp, const].
    param_names : list of str
        Parameter names in order of params.
    knee_freq : float or 1d array
        Knee frequency.
    rsq : float
        R-squared of the aperiodic fit.
    guess : list, optional, default: None
        Inital parameter estimates.
    bounds : list, optional, default: None
        Parameters bounds as [(*lower_bounds), (*upper_bounds)].
    """

    def __init__(self, freqs=None, powers=None):
        """Initialize object."""

        self.freqs = freqs
        self.powers = powers
        self.powers_fit = None

        # Set via other methods
        self.params = None
        self.param_names = ['offset', 'knee_freq', 'exp', 'const']
        self.knee_freq = None
        self.tau = None
        self.rsq = None
        self.rsq_full = None
        self.guess=None
        self.bounds = None

        # For comparison to PSD models
        self.tau = None
        self.knee_freq = None


    def compute_spectrum(self, sig, fs, ar_order=None, f_range=None,
                         norm_range=None, n_jobs=1, **kwargs):
        """Compute powers spectral density.

        Parameters
        ----------
        sig : 1d or 2d array
            Voltage time series or spike counts.
        fs : float
            Sampling rate, in Hz.
        ar_order : int, optional, default: None
            Compute an autoregressive spectra when not None.
        f_range : tuple of (float, float)
            Frequency range of interest, inclusive.
        norm_range : tuple of (float, float), optional, default: None
            The lower and upper normalization range.
        n_jobs : int, optional, default: -1
            Number of jobs to run in parralel, when powers is 2d.
            Only available when using an ar_order.
        **kwargs
            Additional keyword arguments to pass to compute_spectrum or compute_ar_spectrum.
        """
        if ar_order is not None:
            self.freqs, self.powers = compute_ar_spectrum(sig, fs, ar_order, n_jobs=n_jobs,
                                                          f_range=f_range, **kwargs)
        else:
            self.freqs, self.powers = compute_spectrum(sig, fs, f_range=f_range, **kwargs)

        if norm_range is not None:
            self.powers = normalize_psd(self.powers, *norm_range)


    def fit(self, freqs=None, powers=None, f_range=None, ap_mode='single', method='huber', fooof_init=None, bounds=None,
            guess=None, n_resample=None, f_scale=.1, r_thresh=None, maxfev=1000, sigma=None, n_jobs=1, progress=None):
        """Fit power spectra.

        Parameters
        ----------
        freqs : 1d array
            Frequencies at which the measure was calculated.
        powers : 1d or 2d array
            Power spectral density.
        f_range : tuple of (float, float)
            Frequency range of interest, inclusive.
        ap_mode : {'single', 'double'}
            Aperiodic mode as a single or double timescales (knee) process.
            Only availble for non-fooof methods.
        method : {'huber', 'cauchy', 'soft_l1', 'arctan', 'fooof'}
            Fit using a single scipy curve_fit call using robust regression or use the
            fooof model.
        fooof_init : dict, optional, default: None
            Fooof initialization arguments.
            Only used if method is 'fooof'.
        bounds : 2d array-like
            Parameter bounds.
        guess : 1d array-like
            Initial parameter estimates.
        n_resample : int, optional, default: None:
            Evenly resample in log-log space to improve robust regression.
        f_scale : float or 1d array-like, optional, default: 0.1
            Value of soft margin between inlier and outlier residuals.
            Only used if method is in {'huber', 'cauchy', 'soft_l1', 'arctan'}.
        r_thresh : float, optional, default: None
            Minimum r-squared required to accept f_scale. Only used when f_scale is a 1d array.
            When None, all f_scale values are attempted and the highest resulting r-squared is accepted.
        maxfev : int
            Maximum number of fitting iterations. Only availble for huber method.
        sigma : dict, optional, default: None
            Target error per sample.
        n_jobs : int
            Number of jobs to run in parralel, when powers is 2d.
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
        """

        if freqs is not None:
            self.freqs = freqs

        if powers is not None:
            self.powers = powers

        self.bounds = bounds
        self.guess = guess

        if f_range is not None:
            inds = np.where((self.freqs >= f_range[0]) & \
                            (self.freqs <= f_range[1]))[0]
            self.freqs = self.freqs[inds]
            self.powers = self.powers[:, inds] if self.powers.ndim == 2 else self.powers[inds]

        # Skip 0 hz if using fooof or resampling
        if (method == 'fooof' or n_resample is not None) and self.freqs[0] == 0:
            self.freqs = self.freqs[1:]
            self.powers = self.powers[1:] if self.powers.ndim == 1 else self.powers[:, 1:]

        # Fit
        if method != 'fooof' and fooof_init is None:
            # Robust regression (aperiodic only)
            self.params, self.powers_fit = fit_psd_robust(
                self.freqs, self.powers, ap_mode=ap_mode, loss=method, f_scale=f_scale, r_thresh=r_thresh,
                n_resample=n_resample, bounds=bounds, guess=guess, maxfev=maxfev, sigma=sigma,
                n_jobs=n_jobs, progress=progress
            )
        elif method == 'fooof' or fooof_init is not None:
            # Aperiodic and periodic model
            self.params, self.powers_fit, self.rsq_full = fit_psd_fooof(
                self.freqs, self.powers, fooof_init=fooof_init, return_rsq=True,
                ap_bounds=bounds, ap_guess=guess, n_jobs=n_jobs, progress=progress
            )
        else:
            raise ValueError('method must be in \{\'huber\', \'cauchy\', \'soft_l1\', \'arctan\', \'fooof\'\}.')

        # Get r-squared, knee freqs, and taus
        if self.powers_fit.ndim == 1:
            self.rsq = np.corrcoef(np.log10(self.powers),
                                   np.log10(self.powers_fit))[0][1] ** 2

            self.knee_freq = self.params[1]
            self.tau = convert_knee(self.knee_freq)
        else:
            self.rsq = np.zeros(len(self.powers_fit))

            for ind in range(len(self.powers_fit)):
                self.rsq[ind] = np.corrcoef(np.log10(self.powers[ind]),
                                            np.log10(self.powers_fit[ind]))[0][1] ** 2
            self.knee_freq = self.params[:, 1]
            self.tau = convert_knee(self.knee_freq)


    def plot(self, ax=None, title=None):
        """Plot spectra.

        Parameters
        ----------
        ax : AxesSubplot, optional, default: None
            Axis to plot on.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        if self.freqs is None or self.powers is None:
            raise ValueError('freqs and powers are undefined.')

        # Plot spectra
        if self.powers.ndim == 1:
            ax.loglog(self.freqs, self.powers, label='PSD', color='C0')
        elif self.powers.ndim == 2:
            ax.loglog(self.freqs, self.powers.mean(axis=0), label='PSD', color='C0')
            for power in self.powers:
                ax.loglog(self.freqs, power, color='C0', alpha=.1)

        # Plot fits
        if self.powers_fit is not None and self.powers_fit.ndim == 1:
            ax.loglog(self.freqs, self.powers_fit, label='Fit', ls='--', color='C1')
        elif self.powers_fit is not None and self.powers_fit.ndim == 2:
            ax.loglog(self.freqs, self.powers_fit.mean(axis=0),
                      ls='--', color='C1', label='Mean Fit')

        ax.legend()
        ax.set_ylabel('Powers')
        ax.set_xlabel('Frequencies')

        title = 'Aperiodic Model Fit' if title is None else title
        ax.set_title(title)


def fit_psd_fooof(freqs, powers, f_range=None, fooof_init=None, return_rsq=False,
                  ap_bounds=None, ap_guess=None, n_jobs=-1, progress=None):
    """Fit a PSD, using SpecParam, and estimate tau.

    Parameters
    ----------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    powers : 1d or 2d array
        Power spectral density.
    f_range : tuple of (float, float), optional, default: None
        Frequency range of interest.
    fooof_init : dict, optional, default: None
        Fooof initialization arguments.
    return_rsq, optional, default: False
        Returns the model's full r-squared if True.
    ap_bounds : 2d array-like
        Aperiodic bounds.
    ap_guess : 1d array-like
        Initial aperiodic parameter estimates.
    n_jobs : int
        Number of jobs to run in parralel, when powers is 2d.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    params : 1d or 2d array
        Parameters as [offset, knee_freq, exp, const].
    powers_fit : 1d or 2d array
        Aperiodic fit.
    full_rsq : float or 1d array, optional
        Full model's r-squared.
    """

    if fooof_init is None:
        fooof_init = {}

    fooof_init_cp = fooof_init.copy()
    ap_mode = fooof_init_cp.pop('aperiodic_mode', 'knee_constant')

    # Init FOOOF
    if powers.ndim == 1:
        fm = FOOOF(aperiodic_mode=ap_mode, verbose=False, **fooof_init_cp)
    elif powers.ndim == 2:
        fm = FOOOFGroup(aperiodic_mode=ap_mode, verbose=False, **fooof_init_cp)

    # Parameter bounds and guess
    if ap_bounds is None:
        ap_bounds = [[-np.inf, 1e-6,       0,      0],
                     [ np.inf, freqs.max(),  np.inf, np.inf]]

    if ap_guess is None:
        ap_guess =  [0, 1, 1, 1e-6]

    if ap_mode == 'knee':
        ap_bounds[0] = ap_bounds[0][:-1]
        ap_bounds[1] = ap_bounds[1][:-1]
        ap_guess = ap_guess[:-1]

    fm._ap_guess = ap_guess
    fm._ap_bounds = ap_bounds

    # Fit
    if powers.ndim == 1:
        fm.fit(freqs, powers, f_range)
    else:
        fm.fit(freqs, powers, f_range, n_jobs, progress)

    if not fm.has_model:
        return np.nan, np.nan

    powers_fit = 10**fm._ap_fit if powers.ndim == 1 else \
        np.array([10**fm.get_fooof(i)._ap_fit for i in range(len(fm))])

    params = fm.get_params('aperiodic')

    if return_rsq:
        return params, powers_fit, fm.get_params('r_squared')
    else:
        return params, powers_fit


def fit_psd_robust(freqs, powers, f_range=None, ap_mode='single', loss='huber', f_scale=.1, r_thresh=None,
                   n_resample=None, bounds=None, guess=None, maxfev=1000, sigma=None, n_jobs=-1, progress=None):
    """Fit the aperiodic spectrum using robust regression.

    Parameters
    ----------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    powers : 1d or 2d array
        Power spectral density.
    f_range : tuple of (float, float)
        Frequency range of interest.
    ap_mode : {'single', 'double'}
        Aperiodic mode as a single or double timescales (knee) process.
    loss : {'huber', 'soft_l1', 'cauchy', 'arctan'}
        Loss function.
    f_scale : float or 1d array-like, optional, default: 0.1
        Value of soft margin between inlier and outlier residuals.
    r_thresh : float, optional, default: None
        Minimum r-squared required to accept f_scale. When None, all f_scale values are
        attempted and the highest resulting r-squared is accepted.
    n_resample : int, optional, default: None:
        Evenly resample in log-log space to improve robust regression.
    bounds : 2d array-like
        Parameter bounds.
    guess : 1d array-like
        Initial parameter estimates.
    maxfev : int
        Maximum number of fitting iterations.
    sigma : dict, optional, default: None
        Target error per sample.
    n_jobs : int
        Number of jobs to run in parralel, when powers is 2d.
    progress : tqdm, optional, default: None
        Progress bar.

    Returns
    -------
    params : 1d or 2d array
        Parameters as [offset, knee_freq, exp, const, (f_scale, r_squared)].
        Note: f_scale and r_squared only returned when optimizing f_scale using an array input.
    powers_fit : 1d or 2d array
        Aperiodic fit.
    """

    # Unpack when in a mp pool
    if powers is None:
        freqs, powers = freqs

    # Bound to freq range
    if f_range is not None:
        inds = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[0]
        freqs = freqs[inds]
        powers = powers[:, inds] if powers.ndim == 2 else powers[inds]

    # Parameter bounds and guess
    fmax = freqs.max()

    if bounds is None and ap_mode == 'single':
        bounds = [[-np.inf, 1e-3,  0,  0],
                  [ np.inf, fmax, 10, 10]]
    elif bounds is None and ap_mode == 'double':
        bounds = [[-np.inf, 1e-3,  0,  0, -np.inf, 1e-3,  0,  0],
                  [ np.inf, fmax, 10, 10,  np.inf, fmax, 10, 10]]

    if guess is None and ap_mode == 'single':
         guess = [1, 1, 2, 1e-3]
    elif guess is None and ap_mode == 'double':
         guess = [1,  1, 2, 1e-3,
                  0, 10, 2, 1e-3]

    if ap_mode == 'double':
        # Double knee model
        expo_func = expo_double_const_function
    else:
        # Single knee model
        expo_func = expo_const_function

    if powers.ndim == 2:
        # Recursively call 1d in parallel
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        _freqs = repeat(freqs)

        with Pool(processes=n_jobs) as pool:
            mapping = pool.imap(
                partial(fit_psd_robust, powers=None, ap_mode=ap_mode,  n_resample=n_resample,
                        loss=loss, bounds=bounds, guess=guess, maxfev=maxfev, n_jobs=n_jobs),
                zip(_freqs, powers)
            )
            results = list(progress_bar(mapping, progress, len(powers), pbar_desc='Fitting PSD'))

        params = np.array([r[0] for r in results])
        powers_fit = np.array([r[1] for r in results])

    else:
        # 1d
        if n_resample is not None:
            # Evenly resample in log-log space for improved robust regression
            freqs_orig = freqs.copy()
            freqs, powers = resample_logspace(freqs, powers, n_resample)

        if isinstance(f_scale, (float, int)):
            params, _ = curve_fit(expo_func, freqs, np.log10(powers),
                                  loss=loss, f_scale=f_scale, maxfev=maxfev,
                                  p0=guess, bounds=bounds, sigma=sigma)
        else:
            rsq = -np.inf

            for f in f_scale:

                # Catch non-convergence exceptions
                try:
                    _params, _ = curve_fit(expo_func, freqs, np.log10(powers),
                                           loss=loss, f_scale=f, maxfev=maxfev,
                                           p0=guess, bounds=bounds, sigma=sigma)
                except RuntimeError:
                    continue

                # Get fit
                _powers_fit = 10**expo_func(freqs, *_params)

                # Get r-squared from fit
                _rsq = np.corrcoef(np.log10(powers),
                                   np.log10(_powers_fit))[0][1] ** 2

                if r_thresh is not None and _rsq > r_thresh:
                    # Above r-squared threshold
                    params = _params
                    powers_fit = _powers_fit
                    break
                elif _rsq > rsq:
                    # Track the current best r-squared
                    params = _params
                    powers_fit = _powers_fit
                    rsq = _rsq

         # Move back to original frequency space if resampling
        if n_resample is not None:
            powers_fit = 10**expo_func(freqs_orig, *params)
        else:
            powers_fit = 10**expo_func(freqs, *params)

        # Sort parameters using ascending knee frequency
        if ap_mode == 'double':
            if params[1] > params[5]:
                params = np.hstack((params[4:], params[:4]))

    return params, powers_fit

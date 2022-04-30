"""Spectral estimation methods."""

from itertools import repeat
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import curve_fit

from neurodsp.spectral import compute_spectrum
from timescales.autoreg import compute_ar_spectrum

from fooof import FOOOF, FOOOFGroup
from fooof.core.funcs import expo_const_function

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
    knees : float or 1d array
        Knee frequency.
    rsq : float
        R-squared of the full fit.
    guess : list, optional, default: None
        Inital parameter estimates.
    bounds : list, optional, default: None
        Parameters bounds as [(*lower_bounds), (*upper_bounds)].
    """

    def __init__(self, freqs=None, powers=None):
        """Initialize object."""

        self.freqs = freqs
        self.powers = powers

        # Set via other methods
        self.models = None
        self.params = None
        self.param_names = ['offset', 'knee_freq', 'exp', 'const']
        self.knees = None
        self.rsq = None
        self.guess=None
        self.bounds = None


    def compute_spectrum(self, sig, fs, ar_order=None, f_range=None, n_jobs=-1, **kwargs):
        """Compute powers spectral density.
        Parameters
        ---------
        sig : 1d or 2d array
            Voltage time series or spike counts.
        fs : float
            Sampling rate, in Hz.
        ar_order : int, optional, default: None
            Compute an autoregressive spectra when not None.
        f_range : tuple of (float, float)
            Frequency range of interest, inclusive.
        n_jobs : int
            Number of jobs to run in parralel, when powers is 2d.
            Only available when using an ar_order.
        **kwargs
            Additional keyword arguments to pass to neurodsp's compute_spectrum.
        """
        if ar_order is not None:
            self.freqs, self.powers = compute_ar_spectrum(sig, fs, ar_order, n_jobs=n_jobs,
                                                          **kwargs)
        else:
            self.freqs, self.powers = compute_spectrum(sig, fs, f_range=f_range)


    def fit(self, f_range=None, method='huber', bounds=None,
            maxfev=1000, guess=None, n_jobs=1, progress=None):
        """Fit power spectra.

        Parameters
        ----------
        f_range : tuple of (float, float)
            Frequency range of interest, inclusive.
        method : {'huber', 'fooof'}
            Fit using a single scipy curve_fit call using robust regression ('huber') or use the
            fooof model ('fooof').
        fooof_init : dict, optional, default: None
            Fooof initialization arguments.
        bounds : 2d array-like
            Parameter bounds.
        guess : 1d array-like
            Initial parameter estimates.
        maxfev : int
            Maximum number of fitting iterations. Only availble for huber method.
        n_jobs : int
            Number of jobs to run in parralel, when powers is 2d.
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
        """
        self.bounds = bounds
        self.guess = guess

        if f_range is not None:
            inds = np.where((self.freqs >= f_range[0]) & \
                            (self.freqs <= f_range[1]))[0]
            self.freqs = self.freqs[inds]
            self.powers = self.powers[:, inds] if self.powers.ndim == 2 else self.powers[inds]

        if method == 'huber':
            # Robust regression (aperiodic only)
            self.params, self.powers_fit = fit_psd_huber(
                self.freqs, self.powers, bounds=bounds,
                  guess=guess, maxfev=maxfev, n_jobs=n_jobs, progress=progress
            )
        elif method == 'fooof':

            # Fooof can't, but should, handle 0 hertz
            if self.freqs[0] == 0:
                self.freqs = self.freqs[1:]
                self.powers = self.powers[1:] if self.powers.ndim == 1 else self.powers[:, 1:]

            # Aperiodic and periodic model
            self.params, self.powers_fit = fit_psd_fooof(
                self.freqs, self.powers, fooof_init=None, ap_bounds=bounds,
                ap_guess=guess, n_jobs=n_jobs, progress=progress
            )
        else:
            raise ValueError('method must be in [\'huber\', \'fooof\'].')

        # Get r-squared
        if self.powers_fit.ndim == 1:
            self.rsq = np.corrcoef(self.powers, self.powers_fit)[0][1] ** 2
        else:
            self.rsq = np.zeros(len(self.powers_fit))

            for ind in range(len(self.powers_fit)):
                self.rsq[ind] = np.corrcoef(self.powers, self.powers_fit[ind])[0][1] ** 2


def fit_psd_fooof(freqs, powers, f_range=None, fooof_init=None,
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
        ap_bounds = [[-np.inf,   1e-6, -np.inf,      0],
                     [ np.inf, np.inf,  np.inf, np.inf]]

    if ap_guess is None:
        ap_guess =  [None, 1, None, 1e-6]

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

    return params, powers_fit


def fit_psd_huber(freqs, powers, f_range=None, bounds=None,
                  guess=None, maxfev=1000, n_jobs=-1, progress=None):
    """Fit the aperiodic spectrum using robust regression.

    Parameters
    ----------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    powers : 1d or 2d array
        Power spectral density.
    f_range : tuple of (float, float)
        Frequency range of interest.
    bounds : 2d array-like
        Parameter bounds.
    guess : 1d array-like
        Initial parameter estimates.
    maxfev : int
        Maximum number of fitting iterations.
    n_jobs : int
        Number of jobs to run in parralel, when powers is 2d.

    Returns
    -------
    params : 1d or 2d array
        Parameters as [offset, knee_freq, exp, const].
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
    if bounds is None:
        bounds = [[-np.inf,   1e-6, 0,      0],
                  [ np.inf, np.inf, 5, np.inf]]

    if guess is None:
        guess = [0, 1, 1, 1e-6]

    if powers.ndim == 2:
        # Recursively call 1d in parallel
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        _freqs = repeat(freqs)

        with Pool(processes=n_jobs) as pool:
            mapping = pool.imap(
                partial(fit_psd_huber, powers=None, bounds=bounds,
                        guess=guess, maxfev=maxfev, n_jobs=n_jobs),
                zip(_freqs, powers)
            )
            results = list(progress_bar(mapping, progress, len(powers)))

        params = np.array([r[0] for r in results])
        powers_fit = np.array([r[1] for r in results])

    else:
        # 1d
        params, _ = curve_fit(expo_const_function, freqs, np.log10(powers),
                              loss='huber', maxfev=maxfev, p0=guess, bounds=bounds)

        powers_fit = 10**expo_const_function(freqs, *params)

    return params, powers_fit


def convert_knee_val(knee_freq):
    """Convert knee parameter(s) to frequency and time-constant value.

    Parameters
    ----------
    knee : float or array
        Knee of the aperiodic spectral fit.
    exponent : float, optional, default: 2.
        Used for more accurate frequency estimation when PSD is Lorentzian.

    Returns
    -------
    knee_freq : float
        Frequency where the knee occurs.
    knee_tau : float
        Timescale, in seconds.
    """

    knee_tau = 1./(2*np.pi*knee_freq)

    return knee_tau

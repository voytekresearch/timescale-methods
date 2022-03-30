"""Autoregressive deomposition."""

import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.fft import fftfreq

from statsmodels.regression.linear_model import burg

from spectrum import eigen

from timescales.sim import sim_asine_oscillation, sim_autoregressive

class CAD:
    """Canonical Autoregressive Decomposition.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hertz.
    osc_order : int
        Oscillatory order. Number of oscillatory terms.
    ar_order : int
        Autoregressive order. Number of autoregressive terms.

    Attributes
    ----------
    full_fit : 1d array
        Sum of oscillatory and autoregressive fits.
    osc_fit : 1d array
        Oscillatory fit.
    ar_fit : 1d array
        Autoregressive fit.
    params : 1d array
        Model parameters.
    params_dict : dict
        Model parameters with parameter names as keys.
    xs : 1d array
        Samples indices.
    guess : list
        Parameter initial estimates.
    bounds : list of list
        Parameter lower and upper bounds.
    """

    def __init__(self, sig, fs, osc_order, ar_order):

        self.sig = sig
        self.fs = fs
        self.osc_order = osc_order
        self.ar_order = ar_order

        self.full_fit = None
        self.osc_fit = None
        self.ar_fit = None
        self.params = None
        self.params_dict = None

        self.xs = np.arange(len(sig))

        self.guess = None
        self.bounds = None


    def fit(self, use_freq_est=True, n_eig=20, freq_pad=10,
            bounds=None, guess=None, maxfev=1000):
        """Fit model and decompose components.

        Parameters
        ----------
        use_freq_est : bool, optional, default: True
            Uses robust frequency estimation when True.
        n_eig : int
            Number of eigenvalues to compute the pseudo spectrum.
        freq_pad : int
            Frequency search range.
        guess : list
            Parameter initial estimates.
        bounds : list of list
            Parameter lower and upper bounds.
        maxfev : int, optional, default: 1000
            Maximum number of fitting iterations.
        """

        if use_freq_est:
            self.iter_freq_estimation(n_eig, freq_pad)
        else:
            self.bounds = bounds
            self.guess = guess

        osc_params = _decompose_ar(self.xs, self.sig, self.fs, self.osc_order, self.ar_order,
                                   bounds=self.bounds, guess=self.guess, maxfev=maxfev)

        self.osc_fit, self.ar_fit, self.params = _fit(
            self.xs, self.sig, self.fs, self.osc_order,
            self.ar_order, *osc_params, return_params=True
        )

        self.full_fit = self.osc_fit + self.ar_fit


    def iter_freq_estimation(self, n_eig, freq_pad):
        """Iterative oscillatory frequency search.

        Parameters
        ----------
        n_eig : int
            Number of eigenvalues to compute the pseudo spectrum.
        freq_pad : int
            Frequency search range.
        """

        # Compute pseudo spectral peaks
        psd, _ = eigen(self.sig, n_eig, method='ev')

        freqs = fftfreq(4096, 1/self.fs)
        freqs = freqs[:4096//2]

        psd = psd[4096//2:]

        peaks, _ = find_peaks(psd)

        if len(peaks) == 0:
            warnings.warn('No pseudo-spectral peaks detected.')

            self.bounds = [
                [0, -1, 1, -1] * self.osc_order,
                [2,  1, 100, 1] * self.osc_order
            ]
            self.guess = [1, 0, 1, 0]

        else:

            freq_guess = freqs[peaks[0]]

            lower_freq = freq_guess - freq_pad
            lower_freq = 1 if lower_freq < 0 else lower_freq
            upper_freq = freq_guess + freq_pad

            bounds = [
                [0, -1, lower_freq, -1] * self.osc_order,
                [2,  1, upper_freq, 1] * self.osc_order
            ]

            guess = [1, 0, freq_guess, 0] * self.osc_order

            f_range = np.arange(lower_freq, upper_freq+1)

            rsq = -1

            # Step through range of frequencies
            for f in f_range:

                guess = np.array(guess).reshape(self.osc_order, -1)
                guess[:, 2] = f
                guess = guess.flatten()

                osc_params = _decompose_ar(self.xs, self.sig, self.fs, self.osc_order, 0,
                                        bounds=bounds, guess=guess, maxfev=1000)

                osc_fit = _fit(self.xs, self.sig, self.fs, self.osc_order, 0,
                                            *osc_params, return_params=False)

                _rsq = np.corrcoef(osc_fit, self.sig)[0][1]

                if _rsq > rsq:
                    rsq = _rsq
                    params = osc_params
                    bounds = bounds

            self.guess = params
            self.bounds = bounds


class CADGroup:
    """2D Canonical Autoregressive Decomposition.
    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hertz.
    osc_order : int
        Oscillatory order. Number of oscillatory terms.
    ar_order : int
        Autoregressive order. Number of autoregressive terms.

    Attributes
    ----------
    models : list
        List of fit CAD objects.
    """
    def __init__(self, sigs, fs, osc_order, ar_order):

        self.sigs = sigs
        self.fs = fs
        self.osc_order = osc_order
        self.ar_order = ar_order

        self.models = None

    def __len__(self):
        return len(self.models)


    def __iter__(self):
        """Allow for iterating across the object by stepping across model fit results."""

        for result in self.models:
            yield result


    def __getitem__(self, index):
        return self.models[index]


    def fit(self, use_freq_est=True, n_eig=20, freq_pad=10,
            bounds=None, guess=None, maxfev=1000, n_jobs=-1, progress=None):
        """Fit model and decompose components.

        Parameters
        ----------
        use_freq_est : bool, optional, default: True
            Uses robust frequency estimation when True.
        n_eig : int
            Number of eigenvalues to compute the pseudo spectrum.
        freq_pad : int
            Frequency search range.
        guess : list
            Parameter initial estimates.
        bounds : list of list
            Parameter lower and upper bounds.
        maxfev : int, optional, default: 1000
            Maximum number of fitting iterations.
        n_jobs : int, optional, default: -1
            Number of jobs to run in parallel. Defaults to all cores.
        progress : func, {tqdm.notebook.tqdm or tqdm.tqdm}
            Progress bar.
        """

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with Pool(n_jobs) as pool:
            fit_kwargs = dict(use_freq_est=use_freq_est, n_eig=n_eig, freq_pad=freq_pad,
                          bounds=bounds, guess=guess, maxfev=maxfev)

            mapping = pool.imap(partial(self._fit, **fit_kwargs), self.sigs)

            if progress is None:
                models = list(mapping)
            else:
                models = list(progress(mapping, total=len(self.sigs)))

        self.models = models

    def _fit(self, sig, **fit_kwargs):

        cad = CAD(sig, self.fs, self.osc_order, self.ar_order)
        cad.fit(**fit_kwargs)

        return cad


def _decompose_ar(xs, sig, fs, osc_order, ar_order, bounds=None, guess=None, maxfev=1000):
    """Decompose a signal in periodic and aperiodic components.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    osc_order : int
        Oscillatory model order.
    ar_order : int
        Autoregressive order.
    bounds : list, optional, default: None
        Parameter bounds for curve_fit.
    guess : list, optional, default: None
        Parameter guess for curve_fit.
    maxfev : int, optional, default: 1000
        Maximum number of fitting iterations.

    Returns
    -------
    params : 1d array
        Oscillatory and/or autoregressive parameters.
    """

    if bounds is None:
        bounds = [
            [.2, -1, 0, -1] * osc_order,
            [ 2,  1, 100, 1] * osc_order
        ]

    if guess is None:
        guess = [1, 0, 1, 0] * osc_order

    # Inital oscillatory fit
    if ar_order > 0:

        guess, _ = curve_fit(
            lambda xs, *args : _fit(xs, sig, fs, osc_order, 0, *args),
            xs, sig, bounds=bounds, p0=guess, maxfev=maxfev
        )

    # Final Fit
    params, _ = curve_fit(
        lambda xs, *args : _fit(xs, sig, fs, osc_order, ar_order, *args),
        xs, sig, bounds=bounds, p0=guess, maxfev=maxfev
    )

    return params


def _fit(xs, sig, fs, osc_order, ar_order, *fit_args, return_params=False):
    """Fit a signal decomposition model.

    Parameters
    ----------
    xs : 1d array
        Sample definition.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    osc_order : int
        Oscillatory model order.
    ar_order : int
        Autoregressive order.
    fit_args : list
        Fit parameters as [heights, phis, freqs, rdsyms].
    return_params : bool, optional, default: False
        Return fit parameters when True. Returns only signal fit when False.
        If true,

    Returns
    -------
    full_fit : 1d array, optional
        Full model fit. Returned if return_params is False.
    osc_fit : 1d array, optional
        Oscillatory fit. Returned if osc_order is > 0 and return_paras is True.
    ar_fit : 1d array, opttional
        Autoregressive fit. Returned if ar_order is > 0 and return_params is True.
    params : 1d array, optional
        Oscillatory and/or autoregressive parameters Returned if return_params is True.
    """

    if osc_order <= 0:
        raise ValueError('Oscillatory order must be greater than zero.')

    heights, phis, freqs, rdsyms = np.array(fit_args).reshape(osc_order, -1).T
    params = {}

    # Construct (asym) oscillation
    if osc_order > 0:

        osc_fit = sim_asine_oscillation(xs, fs, freqs, rdsyms, phis, heights)

        params['params_osc'] = dict(heights=heights, phis=phis, freqs=freqs, rdsyms=rdsyms)

    if ar_order == 0 and not return_params:
        return osc_fit
    elif ar_order == 0:
        return osc_fit, params

    # Fit AR Burg
    ar_params, _ = burg(sig-osc_fit, order=ar_order, demean=True)

    params['params_ar'] = {'ar_' + str(ind):v for ind, v in enumerate(ar_params)}

    ar_fit = sim_autoregressive(sig-osc_fit, ar_params)

    if not return_params:
        return osc_fit + ar_fit
    else:
        return osc_fit, ar_fit, params

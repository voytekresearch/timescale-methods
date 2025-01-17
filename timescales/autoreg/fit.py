"""Spectral AR fitting."""
import warnings
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from timescales.sim import sim_ar

class ARPSD:
    """Fits AR(p) model to PSD."""
    def __init__(self, order, fs, bounds=None, ar_bounds=None, guess=None,
                 maxfev=100, loss_fn='linear', scaling="log", f_scale=None,
                 curve_fit_kwargs=None):
        """Intialize object.

        Parameters
        ----------
        order : int
            Autoregressive order.
        fs : float
            Sampling rate, in Hertz.
        bounds : 2d tuple or list, optional, default: None
            Bounds on the AR weights as (lower, uper).
            Defaults to (-0.9999, 0.9999). In some cases, (0, 0.9999)
            may be more appropriate.
        ar_bounds : tuple of (float, float):
            Sets bounds across all AR weights.
        guess : list, optional, default: None
            Inital AR weights. Defaults to zeros.
        maxfev : int, optional, default: None
            Max number of optimization iterations.
        loss_fn : str, optional, default: 'linear'
            Name of loss function supported by curve_fit.
        scaling : str, {"log", "linear"}
            Scale of power to fit.
        f_scale : float, optional, default: None
            Robust regression. Determines inliers/outliers. Between [0, 1].
        curve_fit_kwargs : dict, optional, default: None
            Additonal kwargs to pass to curve_fit.
        """
        self.order = order
        self.fs = fs

        self.freqs = None
        self.powers = None

        self.bounds = bounds
        self.ar_bounds = ar_bounds
        self.guess = guess
        self.f_scale = f_scale
        self.maxfev = maxfev
        self.loss_fn = loss_fn
        self.scaling = scaling
        self.params = None
        self.param_names = [f"phi_{i}" for i in range(order)]
        self.param_names.append("offset")
        self.curve_fit_kwargs = {} if curve_fit_kwargs is None else curve_fit_kwargs

    def fit(self, freqs, powers):
        """Fit PSD.

        Parameters
        ----------
        freqs : 1d array
            Frequencies.
        powers : 1d or 2d array
            Power.
        """

        # Constants
        self.freqs = freqs
        self.powers = powers
        k = np.arange(1, self.order+1)
        self._exp = np.exp(-2j * np.pi * np.outer(freqs, k) / self.fs).T

        # Inital parameters and bounds
        if self.bounds is None:
            if self.ar_bounds is not None:
                l = [self.ar_bounds[0]] * self.order
                u = [self.ar_bounds[1]] * self.order
            else:
                l = [-1+1e-9] * self.order
                u = [1-1e-9] * self.order

            self.bounds = [
                [*l, 1e-16],
                [*u, 1e16],
            ]

        if self.guess is None:
            guess = [0] * self.order
            self.guess = [*guess, 1.]

        # Fit
        with warnings.catch_warnings():

            warnings.simplefilter("ignore") # log of negative may occur

            if self.scaling == "log":
                t = np.log10
                f = lambda freqs, *params : t(_ar_spectrum(self._exp, *params))
            else:
                t = lambda i : i # identity
                f = lambda freqs, *params : t(_ar_spectrum(self._exp, *params))

            if powers.ndim == 1:

                self.params, _ = curve_fit(
                    f, freqs, t(powers), p0=self.guess, bounds=self.bounds,
                    maxfev=self.maxfev, f_scale=self.f_scale, loss=self.loss_fn,
                    **self.curve_fit_kwargs
                )

                self.powers_fit = _ar_spectrum(self._exp, *self.params)

            else:

                self.params = np.zeros((len(powers), self.order+1))
                self.powers_fit = np.zeros_like(powers)

                for i, p in enumerate(powers):

                    self.params[i], _ = curve_fit(
                        f, freqs, t(p), p0=self.guess, bounds=self.bounds,
                        maxfev=self.maxfev, f_scale=self.f_scale, loss=self.loss_fn,
                        **self.curve_fit_kwargs
                    )

                    self.powers_fit[i] = _ar_spectrum(self._exp, *self.params[i])

    def plot(self, ax=None):
        """Plot model fit."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

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
        ax.set_title('Aperiodic Model Fit')

    def simulate(self, n_seconds, fs, init=None, error=None, index=None):
        """Simulate a signal based on learned parameters."""
        if self.params is not None and index is None:
            return sim_ar(n_seconds, fs, self.params[:-1][::-1], init=init, error=error)
        elif self.params is not None and index is not None:
            return sim_ar(n_seconds, fs, self.params[index][:-1][::-1], init=init, error=error)
        else:
            raise ValueError("Must call .fit prior to simulating.")

    @property
    def is_stationary(self):
        """Determines if the learned coefficients give a stationary process."""
        if self.params is not None and self.params.ndim == 1:
            roots = np.polynomial.Polynomial(np.insert(-self.params[:-1], 0, 1.)).roots()
            return np.all(np.abs(roots) > 1.)
        elif self.params is not None and self.params.ndim == 2:
            _is_stationary = np.zeros(len(self.params), dtype=bool)
            for i in range(len(self.params)):
                roots = np.polynomial.Polynomial(np.insert(-self.params[i][:-1], 0, 1.)).roots()
                _is_stationary[i] = np.all(np.abs(roots) > 1.)
            return _is_stationary
        else:
            raise ValueError("Must call .fit to check stationarity.")


def _ar_spectrum(exp, *params):
    """Spectral form of an AR(p) model.

    Notes
    -----
    This func is for fitting efficiency.
    Use timescales.sim.autoreg.sim_ar_spectrum otherwise.
    """
    phi = params[:-1]
    offset = params[-1]

    denom = 1 - (phi @ exp)
    powers_fit = offset / np.abs(denom)**2

    return powers_fit

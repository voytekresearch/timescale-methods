"""Spectral AR fitting."""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class ARPSD:
    """Fits AR(p) model to PSD."""
    def __init__(self, order, fs, bounds=None, ar_bounds=None,
                 guess=None, maxfev=100, loss_fn='linear', f_scale=None):
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
        f_scale : float, optional, default: None
            Robust regression. Determines inliers/outliers. Between [0, 1].
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
        self.params = None

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
                [*l, 1e-6],
                [*u, 1e6],
            ]

        if self.guess is None:
            guess = [l[0]] * self.order
            self.guess = [*guess, 1.]

        # Fit
        f = lambda freqs, *params : np.log10(ar_spectrum(self._exp, *params))

        if powers.ndim == 1:

            self.params, _ = curve_fit(
                f, freqs, np.log10(powers), p0=self.guess, bounds=self.bounds,
                maxfev=self.maxfev, f_scale=self.f_scale, loss=self.loss_fn
            )

            self.powers_fit = ar_spectrum(self._exp, *self.params)

        else:

            self.params = np.zeros((len(powers), self.order))
            self.powers_fit = np.zeros_like(powers)

            for i, p in enumerate(powers):

                self.params[i], _ = curve_fit(
                    f, freqs, np.log10(p), p0=self.guess, bounds=self.bounds,
                    maxfev=self.maxfev, f_scale=self.f_scale, loss=self.loss_fn
                )

                self.powers_fit[i] = ar_spectrum(self._exp, *self.params[i])

    def plot(self):
        """Plot model fit."""
        if self.params is not None:
            plt.loglog(self.freqs, self.powers, label="Target")
            plt.loglog(self.freqs, ar_spectrum(self._exp, *self.params), label="Fit", ls='--')
            plt.title("AR Spectral Model Fit")
            plt.legend()
        else:
            raise ValueError("Must call .fit prior to plotting.")

    def simulate(self, n_seconds, fs, init=None, error=None, index=None):
        """Simulate a signal based on learned parameters."""
        if self.params is not None and index is None:
            return simulate_ar(n_seconds, fs, self.params[:-1][::-1], init=init, error=error)
        elif self.params is not None and index is None:
            return simulate_ar(n_seconds, fs, self.params[index][:-1][::-1], init=init, error=error)
        else:
            raise ValueError("Must call .fit prior to simulating.")

    @property
    def is_stationary(self, index=None):
        """Determines if the learned coefficients give a stationary process."""
        if self.params is not None and index is None:
            roots = np.polynomial.Polynomial(np.insert(-self.params[:-1], 0, 1.)).roots()
            return np.all(np.abs(roots) > 1.)
        elif self.params is not None and index is None:
            roots = np.polynomial.Polynomial(np.insert(-self.params[index][:-1], 0, 1.)).roots()
            return np.all(np.abs(roots) > 1.)
        else:
            raise ValueError("Must call .fit to check stationarity.")


def ar_spectrum(exp, *params):
    """Spectral form of an AR(p) model."""
    phi = params[:-1]
    offset = params[-1]

    denom = 1 - (phi @ exp)
    powers_fit = offset / np.abs(denom)**2

    return powers_fit


def simulate_ar(n_seconds, fs, phi, init=None, error=None):
    """Simulate a signal given AR coefficients, phi."""
    p = len(phi)

    sig = np.zeros((n_seconds * fs) + p)

    if init is None:
        init = np.random.randn(p)

    sig[:p] = init

    if error is None:
        error = np.random.randn(len(sig))

    for i in range(p, len(sig)):
        sig[i] = (sig[i-p:i] @ phi) + error[i-p]

    sig = sig[p:]

    return sig

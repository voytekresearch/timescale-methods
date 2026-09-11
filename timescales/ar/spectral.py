"""Compute spectra from autoregssive models."""

import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.fft import fftfreq
from scipy.linalg import solve_toeplitz

from timescales.decomposition.modes import ar_psd


def yule_walker(sig, order):
    """AR coefficients and innovation scale by the Yule-Walker equations.

    Uses the adjusted (denominator ``n - k``) autocovariance, matching
    ``statsmodels.regression.linear_model.yule_walker`` to machine precision;
    it is implemented here so that the package needs only numpy and scipy.

    Returns
    -------
    rho : 1d array
        AR coefficients, ``order`` of them.
    sigma : float
        Standard deviation of the innovations.
    """
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    n = len(x)
    if order >= n:
        raise ValueError(f'order {order} needs more than {n} samples.')
    acov = np.array([np.dot(x[:n - k], x[k:]) / (n - k) for k in range(order + 1)])
    rho = solve_toeplitz(acov[:-1], acov[1:])
    return rho, np.sqrt(max(acov[0] - np.dot(acov[1:], rho), 0.0))


def compute_ar_spectrum(sig, fs, order, f_range=None, method='burg', nfft=4096, n_jobs=1):
    """Compute an autoregressive power spectrum.

    Parameters
    ----------
    sig : 1d or 2d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    order : int
        Number of autogressive coefficients to fit.
    f_range : tuple
        Lower and upper frequency bounds.
    method : str, {'burg', 'yule_walker'}
        Coefficient estimation method.
    nfft : int, optional, default: 4096
        Window length.
    n_jobs : optional, int, default: 1
        Number of jobs to run in parallel.

    Returns
    -------
    freqs : 1d array
        Frequency definitions.
    powers : 1d array
        Power values.
    """

    # 2d
    if sig.ndim == 2:

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with Pool(processes=n_jobs) as pool:

            pfunc = partial(compute_ar_spectrum, fs=fs, order=order,
                            nfft=nfft, f_range=f_range, n_jobs=n_jobs)

            mapping = pool.map(pfunc, sig)

            results = np.array(mapping)

            freqs = results[0, 0]
            powers = results[:, 1, :]

        return freqs, powers

    # 1d
    if method == 'burg':
        ar = burg(sig, order=order)
    else:
        ar, _ = yule_walker(sig, order=order)

    freqs, powers = ar_to_psd(ar, fs, nfft, f_range)

    return freqs, powers


def ar_to_psd(ar_coeffs, fs, nfft, f_range=None):
    """Compute PSD from AR coefficients."""

    freqs = fftfreq(nfft, 1/fs)
    freqs = freqs[:len(freqs)//2]
    # The model spectrum on that grid.  `spectrum.arma2psd(A=-ar, rho=1, T=fs)`
    # was used here and is this same quantity divided by fs; the package's own
    # ar_psd reproduces it to 1e-15, so the dependency is not needed.
    powers = ar_psd(ar_coeffs, freqs, fs, 1.0) / fs

    if f_range is not None:
        inds = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))
        freqs = freqs[inds]
        powers = powers[inds]

    return freqs, powers


def burg(sig, order, demean=True):
    """Burg AR(p) coefficient estimates.

    Returned in the ``x_t = sum_k phi_k x_{t-k} + eps_t`` convention.

    The prediction-error energy is recomputed from the current forward and
    backward errors at each order rather than carried forward by the usual
    recursion.  The two agree to about one part in 1e8; recomputing avoids
    accumulating that error across orders, which matters at the high orders
    the AR(p) estimators select.
    """
    x = np.asarray(sig, dtype=float)
    if demean:
        x = x - x.mean()
    if order == 0:
        return np.zeros(0)

    ef, eb = x[1:], x[:-1]
    phi = np.zeros(0)
    for _ in range(order):
        den = ef @ ef + eb @ eb
        k = 2.0 * (eb @ ef) / den if den > 0 else 0.0
        phi = np.concatenate([phi - k * phi[::-1], [k]]) if len(phi) else np.array([k])
        ef, eb = ef[1:] - k * eb[1:], eb[:-1] - k * ef[:-1]
    return phi

"""Autocorrelation simulation functions."""

import numpy as np
from neurodsp.utils.data import create_times


def sim_acf_cos(xs, fs, freq, tau, cos_gamma, var_exp, var_cos,
                var_cos_exp, return_sum=True):
    """Simulate an autocorrelation with an oscillitory component.

    Parameters
    ----------
    xs : 1d array
        Lag definitions.
    fs : float
        Sampling rate, in Hz.
    freq : float
        Frequency of the cosine component.
    tau : float
        Timescale of the exponential component.
    cos_gamma : float
        Inverse timescale of the damped cosine's exponential component.
    var_exp : float
        Variance of the the ACF's exponential component.
    var_cos : float
        Variance of the damped cosine's component.
    var_cos_exp : float
        Variance of the damped cosine's exponential component.
    return_sum : bool, optional, default: True
        Returns the sum of exponential and consine components when True. Or the two components
        separately when False.

    Returns
    ------
    acf : 1d array
        Sum of exponential and damped cosine components.
    exp, cos : 1d arrays, optional
        Separate exponential and cosine compoents.
    """
    exp = np.exp(-np.arange(len(xs))/(tau * fs)) * var_exp
    cos = sim_damped_oscillation(1, len(xs), freq, cos_gamma, var_cos, var_cos_exp)

    if return_sum:
        acf = exp + cos
        return acf

    return exp, cos


def sim_damped_oscillation(n_seconds, fs, freq, gamma, var_cos, var_cos_exp):
    """Simulate an oscillation as a damped cosine.

    Parameters
    ----------
    n_seconds : float
        Length of timeseries, in seconds.
    fs : float
        Sampling rate, in Hz.
    freq : float
        Frequency of the cosine.
    gamma : float
        Decay of the exponential.
    var_cos : float
        Variance of the damped cosine's component.
    var_cos_exp : float
        Variance of the damped cosine's exponential component.

    Returns
    -------
    sig : 1d array
        Timeseries of the damped oscillation.
    """
    times = create_times(n_seconds, fs)

    exp = np.exp(-1 * gamma * times) * var_cos
    cos = np.cos(2 * np.pi * freq * times) * var_cos_exp

    sig = exp * cos

    return sig

"""Autocorrelation simulation functions."""

import numpy as np
from neurodsp.utils.data import create_times

from .exp import exp_decay_func

def sim_acf_cos(xs, fs, exp_tau, exp_amp, osc_tau, osc_amp,
                osc_gamma,  offset, osc_freq):
    """Simulate an autocorrelation with an oscillitory component.

    Parameters
    ----------
    xs : 1d array
        Lag definitions.
    fs : float
        Sampling rate, in Hz.
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
    freq : float
        Frequency of the cosine component.

    Returns
    ------
    acf : 1d array
        Sum of exponential and damped cosine components.
    exp, cos : 1d arrays, optional
        Separate exponential and cosine compoents.
    """

    xs = np.arange(1, len(xs) + 1)

    exp = exp_amp * (np.exp(-(xs / (exp_tau * fs))))
    osc = osc_amp * (np.exp(-(xs / osc_tau * fs) ** osc_gamma)) * \
        np.cos(2 * np.pi * osc_freq * (xs/len(xs)))

    return exp + osc + offset


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

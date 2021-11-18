"""Autocorrelation simulation functions."""

import numpy as np
from neurodsp.utils.data import create_times


def sim_acf_cos(xs, fs, exp_tau, exp_amp, osc_tau, osc_amp,
                osc_gamma, osc_freq, offset):
    """Simulate an autocorrelation with an oscillitory component.

    Parameters
    ----------
    xs : 1d array
        Lag definitions.
    fs : float
        Sampling rate, in Hz.
    exp_tau : float
        Timescale of the exponential component.
    exp_amp : float
        Amplitude of the exponential component.
    osc_tau : float
        Timescale of the damped cosine.
    osc_amp : float
        Amplitude of the damped cosine.
    osc_gamma : float
        Exponential constant of the damped cosine.
    osc_freq : float
        Frequency of the damped cosine.
    offset : float
        Constant add to translate along the y-axis

    Returns
    ------
    acf : 1d array
        Sum of exponential and damped cosine components.
    exp, cos : 1d arrays, optional
        Separate exponential and cosine components.
    """

    xs = np.arange(1, len(xs) + 1)

    exp = sim_exp_decay(xs, fs, exp_tau, exp_amp)
    osc = sim_damped_cos(xs, fs, osc_tau, osc_amp, osc_gamma, osc_freq)

    return exp + osc + offset


def sim_exp_decay(xs, fs, exp_tau, exp_amp, offset=0):
    return exp_amp * (np.exp(-(xs / (exp_tau * fs))) + offset)


def sim_damped_cos(xs, fs, osc_tau, osc_amp, osc_gamma, osc_freq):
    return osc_amp * (np.exp(-(xs / osc_tau * fs) ** osc_gamma)) * \
        np.cos(2 * np.pi * osc_freq * (xs/len(xs)))

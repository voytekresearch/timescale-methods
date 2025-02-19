"""Autocorrelation simulation functions."""

import numpy as np


def sim_acf_cos(xs, fs, exp_tau, osc_tau, osc_gamma,
                osc_freq, amp_ratio, height, offset):
    """Simulate an autocorrelation with an oscillitory component.

    Parameters
    ----------
    xs : 1d array
        Lag definitions.
    fs : float
        Sampling rate, in Hz.
    exp_tau : float
        Timescale of the exponential component.
    osc_tau : float
        Timescale of the damped cosine.
    osc_gamma : float
        Exponential constant of the damped cosine.
    osc_freq : float
        Frequency of the damped cosine.
    amp_ratio : float
        Ratio between the amplitude of the exponetial decay and damped cosine components,
        respectively.
    height : float
        Total height of the combined components.
    offset : float
        Y-axis intercept.

    Returns
    ------
    acf : 1d array
        Sum of exponential and damped cosine components.
    """

    xs = np.arange(len(xs))

    exp = sim_exp_decay(xs, fs, exp_tau, amp_ratio)
    osc = sim_damped_cos(xs, fs, osc_tau, 1-amp_ratio, osc_gamma, osc_freq)

    exp_osc = exp + osc

    exp_max = np.max(exp_osc)
    if exp_max > 0:
        exp_osc /= exp_max
    exp_osc *= height

    return exp_osc + offset


def sim_exp_decay(xs, fs, exp_tau, exp_amp, offset=0.):
    """Simulate an exponential decay.

    Parameters
    ----------
    xs : 1d array
        Lag definitions.
    fs : float
        Sampling rate, in Hz.
    exp_tau : float
        Timescale of the exponential decay.
    exp_amp : float
        Height of exponential decay.
    offset : float, optional, default: 0.
        Y-axis interecept.

    Returns
    -------
    exp_decay : 1d array
        Autocorrelation with exponential decay.
    """

    exp_decay = (np.exp(-(xs / (exp_tau * fs))) + offset)

    exp_max = np.max(exp_decay)
    if exp_max > 0:
        exp_decay /= exp_max

    return exp_amp * exp_decay


def sim_damped_cos(xs, fs, osc_tau, osc_amp, osc_gamma, osc_freq):
    """Simulate a damped cosine.

    Parameters
    ----------
    xs : 1d array
        Lag definitions.
    fs : float
        Sampling rate, in Hz.
    osc_tau : float
        Timescale of the damped cosine.
    osc_gamma : float
        Exponential constant of the damped cosine.
    osc_freq : float
        Frequency of the damped cosine.

    Returns
    -------
    damped_cos : 1d array
        Autocorrelation with damped cosine.
    """
    damped_cos = np.exp(-(xs / (osc_tau * fs)) ** osc_gamma) * \
        np.cos(2 * np.pi * osc_freq * (xs/len(xs)))
    damped_max = np.max(damped_cos)

    if damped_max > 0:
        damped_cos /= damped_max

    return osc_amp * damped_cos

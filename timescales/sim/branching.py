"""Branching simulations."""

import numpy as np


def sim_branching_spikes(n_seconds, fs, tau, lambda_h, lambda_a=None, n_neurons=100):
    """Simulate spikes from a branching Poisson process.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    tau : float
        Timescale, in seconds. Determines branching parameter, m.
    lambda_h : float
        Poisson lambda constant.
    lamda_a : float
        Initial Poisson lambda weight.
        If None, default to (tau * fs) * lambda_h.

    Returns
    -------
    probs : 1d array
        Probability distribution of spikes.
    spikes : 1d or 2d array
        Sum of spike counts across neurons.
    """

    n_samples = int(n_seconds * fs)

    probs = sim_branching(n_seconds, fs, tau, lambda_h, lambda_a)
    probs = (probs - np.min(probs)) / np.ptp(probs)

    spikes = np.zeros((n_neurons, n_samples))

    for ind in range(n_neurons):
        spikes[ind] = (probs > np.random.rand(*probs.shape))

    return probs, spikes


def sim_branching(n_seconds, fs, tau, lambda_h, lambda_a=None):
    """Simulate a branching Poisson process.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    tau : float
        Timescale, in seconds. Determines branching parameter, m.
    lambda_h : float
        Poisson lambda constant.
    lamda_a : float
        Initial Poisson lambda weight.
        If None, default to (tau * fs) * lambda_h.

    Returns
    -------
    sig : 1d array
        Timseries containing timescale process.

    Notes
    -----
    Simplified implentation based on MR. Estimator:

    - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0249447
    - https://github.com/Priesemann-Group/mrestimator

    """

    n_samples = int(n_seconds * fs)

    sig = np.zeros(n_samples)

    # Branching parameter
    m = np.exp(-1/(tau * fs))

    # Initial Point
    if lambda_a is None:
        lambda_a = (tau * fs) * lambda_h

    sig[0] = np.random.poisson(lam=m * lambda_a + lambda_h)

    # Poisson with memory
    for ind in range(1, len(sig)):
        sig[ind] = np.random.poisson(lam=m * sig[ind-1] + lambda_h)

    return sig

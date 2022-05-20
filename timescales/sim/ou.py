"""Ornstein-Uhlenbeck (OU) simulations."""

import numpy as np

from neurodsp.utils.norm import normalize_sig

def sim_ou(n_seconds, fs, tau, mean=0., variance=1.):
    """Simulate spikes as an Ornstein-Uhlenbeck process.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    tau : float
        Timescale, in seconds.

    Returns
    -------
    sig : 1d or 2d array
        Signal containing timescale of interest.
    """
    std = variance ** .5

    n_samples = int(n_seconds * fs)

    # Determine spiking probabilities
    sigma = std * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(1/fs)

    sig = np.zeros(n_samples)

    # Simulate the OU process
    for ind in range(n_samples - 1):
        sig[ind + 1] = sig[ind] + (1/fs) * \
            (-(sig[ind] - mean) / tau) + sigma * sqrtdt * np.random.randn()

    # Enforce exact mean and variance
    sig = normalize_sig(sig, mean=mean, variance=variance)

    return sig

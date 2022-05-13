"""Ornstein-Uhlenbeck (OU) simulations."""

import numpy as np


def sim_ou(n_seconds, fs, tau, std=.1, mu=10.):
    """Simulate spikes as an Ornstein-Uhlenbeck process.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    tau : float
        Timescale, in seconds.
    std : float, optional, default: 1
        Standard deviation of the OU process.
    mu : float, optional, default: None
        Mean of of the OU process.

    Returns
    -------
    sig : 1d or 2d array
        Signal containing timescale of interest.
    """

    n_samples = int(n_seconds * fs)

    # Determine spiking probabilities
    sigma = std * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(1/fs)

    sig = np.zeros(n_samples)

    # Simulate the OU process
    for ind in range(n_samples - 1):
        sig[ind + 1] = sig[ind] + (1/fs) * \
            (-(sig[ind] - mu) / tau) + sigma * sqrtdt * np.random.randn()

    return sig

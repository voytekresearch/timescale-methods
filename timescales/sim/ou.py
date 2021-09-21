"""Ornstein-Uhlenbeck (OU) simulations."""

import numpy as np


def sim_spikes_ou(n_seconds, fs, tau, n_neurons=100, std=.1, mu=10., return_sum=True):
    """Simulate spikes as an Ornstein-Uhlenbeck process.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    tau : float
        Timescale, in seconds.
    n_neurons : int, optional, default: 100
        Number of neurons to simulate.
    std : float, optional, default: 1
        Standard deviation of the OU process.
    mu : float, optional, default: None
        Mean of of the OU process.
    return_sum : bool, optional, default: True
        Returns sum of neurons if True. If False, a 2d binary array is returned.

    Returns
    -------
    spikes : 1d or 2d array
        Sum of spike counts across neurons.
    """

    n_samples = int(n_seconds * fs)

    # Determine spiking probabilities
    sigma = std * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(1/fs)

    probs = np.zeros(n_samples)

    # Simulate the OU process and convert to probabilities
    for ind in range(n_samples - 1):
        probs[ind + 1] = probs[ind] + (1/fs) * \
            (-(probs[ind] - mu) / tau) + sigma * sqrtdt * np.random.randn()

    probs = (probs - np.min(probs)) / np.ptp(probs)

    # Select [0=no spike, 1=spike] using probabilities
    spikes = np.zeros((n_neurons, len(probs)), dtype=bool)

    for ind in range(n_neurons):
        spikes[ind] = (probs > np.random.rand(*probs.shape))

    if return_sum:
        spikes = spikes.sum(axis=0)

    return spikes

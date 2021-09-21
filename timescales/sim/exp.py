"""Exponential decay kernel simulations."""

import warnings

import numpy as np
from neurodsp.sim import sim_synaptic_kernel


def sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=100, mu=None,
                        isi=None, var_noise=0., return_sum=True):
    """Simulate a spiking autocorrelation as a synaptic kernel.

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
    mu : float, optional, default: None
        Mean of the isi exponential distribuion. Only used if isi is None.
    isi : 1d array, optional, default: None
        Interspike intervals to randomly sample from.
    var_noise : float, optional, default: 0.
        Variance of gaussian noise to be added to spike probabilities.
        Larger values, approaching 1, will produce smaller spectral exponents.
    return_sum : bool, optional, default: True
        Returns sum of neurons if True. If False, a 2d binary array is returned.

    Returns
    -------
    spikes : 1d or 2d array
        Sum of spike counts across neurons.
    """

    if (tau * fs) < 1:
        warnings.warn('Requested tau and fs are too small. This may result in inaccurate '
                      'simulation of tau. Increase tau or fs.')

    # Simulate a synaptic kernel
    kernel = sim_synaptic_kernel(5 * tau, fs, 0, tau)

    # Simulate probabilities of spiking
    probs = sim_poisson_distribution(n_seconds, fs, kernel, isi=isi, mu=mu, var_noise=var_noise)

    # Select [0=no spike, 1=spike] using probabilities
    spikes = np.zeros((n_neurons, len(probs)), dtype=bool)

    for ind in range(n_neurons):
        spikes[ind] = (probs > np.random.rand(*probs.shape))

    if return_sum:
        spikes = spikes.sum(axis=0)

    return spikes


def sim_poisson_distribution(n_seconds, fs, kernel, isi=None, mu=None, var_noise=None):
    """Simulate spike trains using a poisson distribution.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    n_neurons : int, optional, default: 100
        Number of neurons to simulate.
    isi : 1d array, optional, default: None
        Interspike intervals to randomly sample from.
    mu : float, optional, default: None
        Mean of the isi exponential distribuion. Only used if isi is None.
    var_noise : float, optional, default: 0.
        Variance of gaussian noise to be added to spike probabilities.
        Larger values, approaching 1, will produce smaller spectral exponents.

    Returns
    -------
    probs : 2d array, optional
        Probablility of spiking at each sample.
    """

    # Pad n_seconds to account for convolution
    times = np.arange(0, int(n_seconds + (len(kernel) * 2)), 1/fs)

    if isi is None:
        mu = fs * .1  if mu is None else mu
        isi = np.round_(np.random.exponential(scale=mu, size=len(times))).astype(int)

    # Randomly sample isi's
    last_ind = np.where(isi.cumsum() >= len(times))[0][0]
    inds = isi[:last_ind].cumsum()

    poisson = np.zeros(len(times), dtype=bool)
    poisson[inds] = True

    # Convolve the binary poisson array with the kernel
    #   and scale to probabilities from 0-1
    n_samples = int(n_seconds * fs)

    probs = np.convolve(poisson, kernel, 'valid')[:n_samples]
    probs = (probs - np.min(probs)) / np.ptp(probs)

    # Add gaussian noise if requested
    if var_noise is not None:
        n_rand = int(n_samples * var_noise)
        inds = np.arange(n_samples)

        for _ in range(len(probs)):

            if n_rand > 0:
                rand_inds = np.random.choice(inds, n_rand, replace=False)
                probs[rand_inds] = .5

    return probs

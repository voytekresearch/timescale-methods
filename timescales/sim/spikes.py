"""Exponential decay kernel simulations."""

import warnings

import numpy as np
from scipy.signal import convolve

from neurodsp.sim import sim_synaptic_kernel
from neurodsp.utils.norm import normalize_sig


def sim_spikes_synaptic(n_seconds, fs, tau, mu=None, refract=None, isi=None, var_noise=None):
    """Simulate a spiking autocorrelation as a synaptic kernel.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    tau : float
        Timescale, in seconds.
    mu : float, optional, default: None
        Mean of the isi exponential distribuion. Only used if isi is None.
    refract : int, optional, default: None
        Minimum distances between spikes (i.e. refactory period).
    isi : 1d array, optional, default: None
        Interspike intervals to randomly sample from.
    var_noise : float, optional, default: None
        Variance of gaussian noise to be added to spike probabilities.
        Larger values, approaching 1, will produce smaller spectral exponents.
    return_sum : bool, optional, default: True
        Returns sum of neurons if True. If False, a 2d binary array is returned.

    Returns
    -------
    spikes : 1d
        Spike counts.
    """

    if (tau * fs) < 1:
        warnings.warn('Requested tau and fs are too small. This may result in inaccurate '
                      'simulation of tau. Increase tau or fs.')

    # Simulate a synaptic kernel
    kernel = sim_synaptic_kernel(10 * tau, fs, 0, tau)

    # Simulate probabilities of spiking
    probs = sim_spikes_prob(n_seconds, fs, kernel, isi, mu, refract, var_noise)

    # Select [0=no spike, 1=spike] using probabilities
    spikes = (probs > np.random.rand(*probs.shape))

    return spikes


def sim_spikes_prob(n_seconds, fs, kernel, isi=None, mu=None, refract=None, var_noise=None):
    """Simulate spiking probability.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    kernel : 1d or 2d array
        Synaptic kernel to convolve with Poisson.
    n_neurons : int, optional, default: 100
        Number of neurons to simulate.
    isi : 1d array, optional, default: None
        Interspike intervals to randomly sample from.
    mu : float, optional, default: None
        Mean of the isi exponential distribuion. Only used if isi is None.
    refract : int, optional, default: None
        Minimum distances between spikes (i.e. refactory period).
    var_noise : float, optional, default: 0.
        Variance of gaussian noise to be added to spike probabilities.
        Larger values, approaching 1, will produce smaller spectral exponents.

    Returns
    -------
    probs : 2d array, optional
        Probablility of spiking at each sample.
    """

    n_samples = int(n_seconds * fs)

    poisson = sim_poisson(n_seconds, fs, kernel, isi, mu, refract)

    # Single kernel
    if kernel.ndim == 1:

        # Convolve the binary poisson array with the kernel
        probs = convolve(poisson, kernel)[:n_samples]

    # Multi-kernel
    elif kernel.ndim == 2:

        probs = np.zeros((len(kernel), n_samples))

        for kernel_ind in range(len(kernel)):

            probs[kernel_ind] = np.convolve(poisson, kernel[kernel_ind])[:n_samples]

        probs = probs.sum(axis=0)

    # Scale probabilities from 0-1
    probs = (probs - np.min(probs)) / np.ptp(probs)

    # Add gaussian noise if requested
    if var_noise is not None:
        n_rand = int(n_samples * var_noise)
        inds = np.arange(n_samples)

        if n_rand > 0:
            rand_inds = np.random.choice(inds, n_rand, replace=False)
            probs[rand_inds] = .5

    return probs


def sim_poisson(n_seconds, fs, kernel, isi=None, mu=None, refract=None):
    """Simulate a poisson distribution.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    kernel : 1d or 2d array
        Synaptic kernel to convolve with Poisson.
    n_neurons : int, optional, default: 100
        Number of neurons to simulate.
    isi : 1d array, optional, default: None
        Interspike intervals to randomly sample from.
    mu : float, optional, default: None
        Mean of the isi exponential distribuion. Only used if isi is None.
    refract : int, optional, default: None
        Minimum distances between spikes (i.e. refactory period).

    Returns
    -------
    poisson : 2d array, optional
        Probablility of spiking at each sample.
    """

    # Pad n_seconds to account for convolution
    kern_len = len(kernel[0]) if kernel.ndim == 2 else len(kernel)
    times = np.arange(0, int(n_seconds + (kern_len * 2)), 1/fs)

    if isi is None:
        mu = fs * .1  if mu is None else mu
        isi = np.round_(np.random.exponential(scale=mu, size=len(times))).astype(int)

    if refract is not None:
        isi = isi[np.where(isi > refract)[0]]

    # Randomly sample isi's
    n_samples = int(n_seconds * fs)
    last_ind = np.where(isi.cumsum() >= n_samples)[0]
    inds = isi.cumsum() if len(last_ind) == 0 else isi[:last_ind[0]].cumsum()

    # If kernel is 2d, one kernel is expected per isi
    if kernel.ndim == 2 and len(kernel) != len(inds):
        raise ValueError('Mismatch between 2d kernel length and ISIs. '
                         'Explicitly pass isi arg with length == 2d kernel.')

    # Single kernel
    if kernel.ndim == 1:
        poisson = np.zeros(len(times), dtype=bool)
        poisson[inds] = True

    # Multi-kernel
    elif kernel.ndim == 2:

        for sample_ind in inds:

            poisson = np.zeros(len(times), dtype=bool)
            poisson[sample_ind] = True

    return poisson


def sample_spikes(probs):
    """Randomly sample spikes from a probability array.

    Parameters
    ----------
    probs : 1d or 2d array
        Probabilities to sample. Assumed to be between zero and one.

    Returns
    -------
    spikes : 1d or 2d array
        Spike counts.
    """

    if probs.ndim == 2:
        spikes = np.zeros((len(probs), len(probs[0])), dtype=bool)
        for ind in range(len(probs)):
            spikes[ind] = sample_spikes(probs[ind])
    else:
        spikes = (probs > np.random.rand(*probs.shape))

    return spikes


def bin_spikes(spikes, fs, bin_size, mean=None, variance=None):
    """Bin spikes.

    Parameters
    ----------
    spikes : 1d or 2d array
        Spike counts.
    fs : float
        Sampling rate, in Hertz.
    bin_size : int
        Number of samples per bin.
    mean : float, optional, default: None
        Mean to normalize signal to.
    variance : float, optional, default: None
        Variance to normalize signal to.

    Returns
    -------
    spikes_bin : 1d or 2d array
        Binned spike counts
    fs_bin : float
        Updated sampling rate.
    """

    if spikes.ndim == 2:
        for ind in range(len(spikes)):

            _spikes_bin, fs_bin = bin_spikes(spikes, bin_size, fs)

            if ind == 0:
                spikes_bin = np.zeros((len(spikes), len(_spikes_bin)))

            spikes_bin[ind] = _spikes_bin
    else:

        spikes_bin = spikes.reshape(-1, bin_size).sum(axis=1)

        if mean is not None or variance is not None:
            spikes_bin = normalize_sig(spikes_bin, mean=mean, variance=variance)

        fs_bin = fs / bin_size

    return spikes_bin, fs_bin

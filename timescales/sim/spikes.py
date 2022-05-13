"""Exponential decay kernel simulations."""

import warnings

import numpy as np
from scipy.signal import convolve

from neurodsp.sim import sim_synaptic_kernel, sim_oscillation
from neurodsp.utils.norm import normalize_sig


def sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=1, mu=None,
                        refract=None, isi=None, var_noise=None, return_sum=True):
    """Simulate a spiking autocorrelation as a synaptic kernel.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    fs : float
        Sampling rate, in hz.
    tau : float
        Timescale, in seconds.
    n_neurons : int, optional, default: 1
        Number of neurons to simulate.
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
    probs : 1d array
        Probability distribution of spikes.
    spikes : 1d or 2d array
        Sum of spike counts across neurons.
    """

    if (tau * fs) < 1:
        warnings.warn('Requested tau and fs are too small. This may result in inaccurate '
                      'simulation of tau. Increase tau or fs.')

    # Simulate a synaptic kernel
    kernel = sim_synaptic_kernel(10 * tau, fs, 0, tau)

    # Simulate probabilities of spiking
    probs = sim_spikes_prob(n_seconds, fs, kernel, isi, mu, refract, var_noise)

    # Select [0=no spike, 1=spike] using probabilities
    spikes = np.zeros((n_neurons, len(probs)), dtype=bool)

    for ind in range(n_neurons):
        spikes[ind] = (probs > np.random.rand(*probs.shape))

    if return_sum:
        spikes = spikes.sum(axis=0)

    return probs, spikes


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


def sim_probs_combined(n_seconds, fs, ap_freq, pe_freq,
                        ap_sim_kwargs=None, heights=None):
    """Simulate spiking probabilities with aperiodic and periodic timescales.

    Parameters
    ----------
    n_seconds : float
        Length of the signal, in seconds.
    ap_freq : float
        Aperiodic frequency. Determines aperiodic timescale.
    pe_freq : float
        Periodic frequency. Determines periodic timescale.
    ap_sim_kwargs : dict, optional, default: None
        Keyword arguments to pass to sim_spike_prob.
    heights : tuple of float, optional, default: None
        Max probabilities as tuple of (aperiodic, periodic).

    Returns
    -------
    probs_ap : 1d array
        Probability of aperiodic spiking.
    probs_pe : 1d array
        Probability of periodic spiking.
    """

    from timescales.fit import convert_knee_val

    if isinstance(heights, (list, tuple, np.ndarray)):
        height_ap, height_pe = heights
    else:
        height_ap = .5
        height_pe = .5

    if ap_sim_kwargs is None:
        ap_sim_kwargs = {}

    # Aperiodic
    ap_tau = convert_knee_val(ap_freq)

    kernel = sim_synaptic_kernel(10 * ap_tau, fs, 0, ap_tau)
    kernel = kernel.round(6)

    probs_ap = sim_spikes_prob(n_seconds, fs, kernel=kernel, **ap_sim_kwargs)
    probs_ap -= probs_ap.min()
    probs_ap /= probs_ap.max()
    probs_ap *= height_ap

    # Periodic
    probs_pe = sim_oscillation(n_seconds, fs, pe_freq)
    probs_pe -= probs_pe.min()
    probs_pe /= probs_pe.max()
    probs_pe *= height_pe

    return probs_ap, probs_pe

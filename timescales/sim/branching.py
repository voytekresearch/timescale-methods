"""Branching simulations."""

from neurodsp.utils.norm import normalize_sig

import numpy as np


def sim_branching(n_seconds, fs, tau, lambda_h, lambda_a=None, mean=None, variance=None):
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
    mean : float, optional, default: None
        Mean to normalize signal to.
    variance : float, optional, default: None
        Variance to normalize signal to.

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

    # Normalize
    if mean is not None or variance is not None:
        sig = normalize_sig(sig, mean=mean, variance=variance)

    return sig

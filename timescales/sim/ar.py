"""Simulate an AR(p) process."""

import numpy as np

def sim_ar(n_seconds, fs, phi, init=None, error=None):
    """Simulate a signal given AR coefficients, phi.

    Parameters
    ----------
    n_seconds : float
        Number of seconds to simulate.
    fs : float
        Sampling rate, in Hertz.
    phi : 1d array
        Autoregressive coefficients.
    init : 1d array, default: None
        First p values of the signal to begin convolutional
        with weights. Default samples a standard normal.
    error : 1d array, default: None
        Epsilon term added at each convolutional step.
        Should have length of int(fs * n_seconds).
        Default samples a standard normal.

    Returns
    -------
    sig : 1d array
        Simulated signal.
    """
    # Order
    p = len(phi)

    # Initalize arrays
    sig = np.zeros(int(n_seconds * fs) + p)

    if init is None:
        init = np.random.randn(p) * np.sqrt(1/(1-phi[0]**2))

    sig[:p] = init

    if error is None:
        error = np.random.randn(len(sig))

    for i in range(p, len(sig)):
        sig[i] = (sig[i-p:i] @ phi) + error[i-p]

    sig = sig[p:]

    return sig

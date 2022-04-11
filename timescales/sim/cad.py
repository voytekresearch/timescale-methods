"""Oscillatory and autoregressive simulations."""

from itertools import repeat
import numpy as np


def sim_autoregressive(sig, ar_params):
    """Generate autoregressive fit.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    ar_params : 1d array
        Autoregressive parameters.

    Returns
    -------
    sig_ar : 1d array
        Autoregressive signal.
    """
    ar_params = ar_params[::-1]
    ar_order = len(ar_params)

    _sig = np.pad(sig, ar_order)

    sig_ar = np.zeros(len(_sig))

    for i in range(ar_order, len(_sig)):
        sig_ar[i] = sum(_sig[i-ar_order:i] * ar_params)

    sig_ar = sig_ar[ar_order:-ar_order]

    return sig_ar


def sim_asine_oscillation(xs, fs, freq, rdsym, phi, height):
    """Simulate an asymmetrical sinusoidal wave.

    Parmeters
    ---------
    xs : 1d array
        Sample indices.
    fs : float
        Sampling rate, in Hz.
    freq : float or array-like
        Oscillatory frequency, in Hz.
    rdsym : float or array-like
        Rise-decay symmetry of oscillation.
    phi : float or array-like
        Phase of oscillation.
    height : float or array-like
        Height of oscillations.

    Returns
    -------
    sig_osc : 1d array
        Oscillatory signal.
    """

    if any([isinstance(i, (list, np.ndarray)) for i in [freq, rdsym, phi, height]]):

        freq = repeat(freq) if isinstance(freq, (float, int)) else freq
        rdsym = repeat(rdsym) if isinstance(rdsym, (float, int)) else rdsym
        phi = repeat(phi) if isinstance(phi, (float, int)) else phi
        height = repeat(height) if isinstance(height, (float, int)) else height

        params = zip(freq, rdsym, phi, height)

        sig_osc = np.zeros(len(xs))

        for p in params:
            sig_osc += sim_asine_oscillation(xs, fs, *p)

    else:

        pha = np.exp(1.j * np.pi * phi)

        sig_osc = height * pha * np.exp(2.j * np.pi * (freq/fs) * xs)

        sig_osc = (sig_osc * np.exp(1.j * rdsym * sig_osc)).real

    return sig_osc

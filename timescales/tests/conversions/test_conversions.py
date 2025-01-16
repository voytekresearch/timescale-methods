"""Tests for conversion functions."""

import numpy as np

from timescales.sim import sim_exp_decay, sim_lorentzian
from timescales.conversions import conv, convert_knee, convert_tau, psd_to_acf, acf_to_psd


def test_convert_knee():

    knee_freq_lo = 10
    knee_freq_hi = 100

    knee_tau_long = convert_knee(knee_freq_lo)
    knee_tau_short = convert_knee(knee_freq_hi)

    assert knee_tau_long > knee_tau_short
    assert knee_tau_long < knee_freq_lo
    assert knee_tau_short < knee_freq_hi

    assert convert_knee(10) == convert_tau(10)


def test_psd_to_acf():

    # Reference ACF
    fs = 1000
    knee_freq = 10
    tau = convert_knee(knee_freq)
    phi = 0.95
    tau = -1 / (np.log(phi) * fs)
    lags = np.arange(0, 251).astype(np.float64)
    corrs = sim_exp_decay(lags, fs, tau, 1)

    # PSD to ACF via iFFT
    freqs = np.linspace(0.01, fs//2, 20000)
    powers = 1 / (1 - 2 * phi * np.cos(2 * np.pi * freqs * 1/fs) + phi**2)
    _, corrs_ifft = psd_to_acf(freqs, powers, fs)

    corrs = corrs[::2]
    corrs_ifft = corrs_ifft[:len(corrs)]

    np.testing.assert_almost_equal(corrs, corrs_ifft, 3)


def test_acf_to_psd():

    # Reference PSD
    fs = 1000
    phi = 0.95
    freqs = np.arange(0, 500).astype(np.float64)
    powers = 1 / (1 - 2 * phi * np.cos(2 * np.pi * freqs * 1/fs) + phi**2)

    # ACF to PSD via iFFT
    tau = -1 / (np.log(phi) * fs)
    lags = np.arange(0, 1000)
    corrs = sim_exp_decay(lags, fs, tau, 1)
    _, powers_ifft = acf_to_psd(corrs, fs)

    np.testing.assert_almost_equal(powers, powers_ifft, 4)

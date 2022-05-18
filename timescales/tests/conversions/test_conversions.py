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

    lags = np.linspace(0, 250, 500)
    corrs = sim_exp_decay(lags, fs, tau, 1)

    # PSD to ACF via iFFT
    freqs = np.linspace(0, 2001, 1002)
    powers = sim_lorentzian(freqs, knee_freq, constant=1e-4)

    _, corrs_ifft = psd_to_acf(freqs, powers, fs, (0, 1))

    np.testing.assert_almost_equal(corrs, corrs_ifft, 3)


def test_acf_to_psd():

    # Reference PSD
    fs = 1000
    knee_freq = 10
    tau = convert_knee(knee_freq)
    freqs = np.arange(1, 500)
    powers = sim_lorentzian(freqs, knee_freq, constant=1e-4)

    # ACF to PSD via iFFT
    lags = np.arange(1, 1001)
    corrs = sim_exp_decay(lags, fs, tau, 1)
    _, powers_ifft = acf_to_psd(lags, corrs, fs, (powers.min(), powers.max()))

    np.testing.assert_almost_equal(powers, powers_ifft, 4)

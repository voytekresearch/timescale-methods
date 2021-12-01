"""Tests for PSD estimation."""

import pytest

import numpy as np

from fooof import FOOOF, FOOOFGroup
from neurodsp.spectral import compute_spectrum

from timescales.fit.psd import fit_psd, convert_knee_val
from timescales.sim import sim_spikes_synaptic


@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('init', [True, False])
def test_fit_psd(ndim, init):

    n_seconds = 1
    fs = 1000
    tau = .01
    n_neurons = 2

    probs, _ = sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=n_neurons, return_sum=True)

    if ndim == 2:
        probs = np.tile(probs, (2, 1))

    freqs, powers = compute_spectrum(probs, fs)

    if init:
        fooof_init = {'max_n_peaks': 0}
        fm, knee_freq, knee_tau = fit_psd(freqs, powers, (1, 100), fooof_init=fooof_init)
    else:
        fm, knee_freq, knee_tau = fit_psd(freqs, powers, (1, 100))

    if np.isnan(knee_freq).any() or np.isnan(knee_tau).any():
        assert isinstance(fm, (FOOOF, FOOOFGroup))
        return

    if ndim == 1:
        assert isinstance(fm, FOOOF)
        assert isinstance(knee_freq, float)
        assert isinstance(knee_tau, float)
    elif ndim == 2:
        assert isinstance(fm, FOOOFGroup)
        assert (knee_freq[0] == knee_freq[1]).all()
        assert (knee_tau[0] == knee_tau[1]).all()


def test_convert_knee_val():

    knee_freq_lo = 10
    knee_freq_hi = 100

    knee_tau_long = convert_knee_val(knee_freq_lo)
    knee_tau_short = convert_knee_val(knee_freq_hi)

    assert knee_tau_long > knee_tau_short
    assert knee_tau_long < knee_freq_lo
    assert knee_tau_short < knee_freq_hi

"""Tests for PSD estimation."""

import pytest

import numpy as np

from fooof.core.funcs import expo_const_function
from neurodsp.spectral import compute_spectrum

from timescales.fit.psd import PSD, fit_psd_fooof, fit_psd_robust
from timescales.sim import sim_spikes_synaptic, sim_ou


def test_psd_init():
    freqs = [0, 1]
    powers = [2, 3]

    psd = PSD(freqs, powers)
    assert (psd.freqs == freqs)
    assert (psd.powers == powers)

@pytest.mark.parametrize('ar_order', [None, 1])
def test_psd_compute_spectrum(ar_order):

    n_seconds = 1
    fs = 1000
    tau = .01

    sig = sim_spikes_synaptic(n_seconds, fs, tau)

    psd = PSD()

    psd.compute_spectrum(sig, fs, ar_order=ar_order)

    assert isinstance(psd.freqs, np.ndarray)
    assert isinstance(psd.powers, np.ndarray)

@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('f_range', [None, (1, 100)])
@pytest.mark.parametrize('method', ['huber', 'fooof', 'invalid'])
def test_psd_fit(ndim, f_range, method):

    freqs = np.arange(0, 101)
    powers = 10**expo_const_function(freqs, 0, 10, 2, .001)

    if ndim == 2:
        powers = np.vstack((powers, powers))

    psd = PSD(freqs, powers)

    if method == 'invalid':
        with pytest.raises(ValueError):
            psd.fit(f_range=f_range, method=method)
    else:
        psd.fit(f_range=f_range, method=method)
        assert psd.powers_fit.shape == psd.powers.shape

@pytest.mark.parametrize('ap_mode', ['knee', 'knee_constant'])
@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('init', [True, False])
def test_fit_psd_fooof(ap_mode, ndim, init):

    n_seconds = 1
    fs = 1000
    tau = .01

    sig = sim_ou(n_seconds, fs, tau)

    if ndim == 2:
        sig = np.tile(sig, (2, 1))

    freqs, powers = compute_spectrum(sig, fs, f_range=(1, 100))

    if init:
        fooof_init = {'max_n_peaks': 0, 'aperiodic_mode': ap_mode}
        params, powers_fit = fit_psd_fooof(freqs, powers, (1, 100), fooof_init=fooof_init)
    else:
        params, powers_fit = fit_psd_fooof(freqs, powers, (1, 100))

    if ap_mode == 'knee_constant' and ndim == 1:
        assert len(powers_fit) == len(powers)
        assert len(params) == 4
    elif ap_mode == 'knee_constant' and ndim == 2:
        assert len(powers_fit[0]) == len(powers[0])
        assert len(powers_fit) == len(powers)
        assert len(params) == 2
        assert len(params[0]) == 4


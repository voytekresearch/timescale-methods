"""Tests for autocorrelation functions."""

import pytest
import numpy as np

from timescales.est.acf import compute_acf, fit_acf, fit_acf_cos, _acf_proxy, _acf_cos_proxy
from timescales.sim import sim_spikes_synaptic, sim_acf_cos, exp_decay_func


@pytest.mark.parametrize('ndim_mode',
    [(1, None), (2, None), (2, 'mean'), (2, 'median'), (2, 'invalid'), (3, None)]
)
def test_compute_acf(ndim_mode):

    ndim, mode = ndim_mode
    n_seconds = 1
    fs = 1000
    tau = .01
    nlags = 10
    n_neurons = 2

    probs, _ = sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=n_neurons, return_sum=True)

    if ndim == 2:
        probs = np.tile(probs, (2, 1))
    if ndim == 3:
        probs = np.tile(probs, (3, 2, 1))

    if mode == 'invalid' or ndim == 3:
        with pytest.raises(ValueError):
            corrs = compute_acf(probs, nlags, mode=mode, n_jobs=-1, progress=None)
        return
    else:
        corrs = compute_acf(probs, nlags, mode=mode, n_jobs=-1, progress=None)


    if ndim == 2 and mode is None:
        assert corrs.ndim == 2
        corrs = corrs.mean(axis=0)
    else:
        assert corrs.ndim == 1

    assert len(corrs) == nlags
    assert np.max(corrs <= 1)
    assert np.min(corrs >= -1)


@pytest.mark.parametrize('ndim', [1, 2])
def test_fit_acf(ndim):

    n_seconds = 1
    fs = 1000
    tau = .01
    nlags = 10
    n_neurons = 2

    probs, _ = sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=n_neurons, return_sum=True)
    corrs = compute_acf(probs, nlags, n_jobs=-1, progress=None)

    if ndim == 2:
        corrs = np.tile(corrs, (2, 1))

    params = fit_acf(corrs, fs, n_jobs=-1, maxfev=1000, progress=None)

    if ndim == 1:
        assert len(params) == 3
    elif ndim == 2:
        assert (params[0] == params[1]).all()
        assert len(params[0]) == 3


@pytest.mark.parametrize('ndim', [1, 2])
def test_fit_acf_cos(ndim):

    xs = np.arange(1000)
    fs = 1000
    freq = 10
    exp_tau = .01
    exp_amp = 1
    osc_tau = .01
    osc_amp = .5
    osc_gamma = .01
    osc_freq = 5
    offset = 0

    corrs = sim_acf_cos(xs, fs, exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma, offset, osc_freq)

    if ndim == 2:
        corrs = np.tile(corrs, (2, 1))

    params = fit_acf_cos(corrs, fs, maxfev=1000, n_jobs=-1, progress=None)

    if ndim == 1:
        assert len(params) == 7
    elif ndim == 2:
        assert (params[0] == params[1]).all()
        assert len(params[0]) == 7


def test_proxies():

    xs = np.arange(1000)
    fs = 1000
    freq = 10
    exp_tau = .01
    exp_amp = 1
    osc_tau = .01
    osc_amp = .5
    osc_gamma = .01
    osc_freq = 5
    offset = 0

    corrs = exp_decay_func(np.arange(100), fs, .001, 1, 0)
    params = _acf_proxy([corrs, None, None], fs, 1000)
    assert params is not None

    corrs = sim_acf_cos(xs, fs, exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma, offset, osc_freq)
    params = _acf_cos_proxy([corrs, None, None], fs, 1000)
    assert params is not None

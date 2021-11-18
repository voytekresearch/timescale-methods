"""Tests for autocorrelation functions."""

import pytest
import numpy as np

from timescales.est import ACF, compute_acf, fit_acf, fit_acf_cos
from timescales.est.acf import _acf_proxy, _acf_cos_proxy
from timescales.sim import sim_spikes_synaptic, sim_acf_cos, sim_exp_decay



def test_ACF_init():
    corrs = np.random.rand(100)
    lags = np.arange(1, 101)
    fs = 100
    acf = ACF(corrs, lags, fs)

    assert (acf.corrs == corrs).all()
    assert (acf.lags == lags).all()
    assert acf.fs == fs

@pytest.mark.parametrize('from_psd', [True, False])
def test_ACF_compute_acf(from_psd):

    sig = np.random.rand(1000)
    fs = 100
    start = 100
    win_len = 100

    acf = ACF()
    acf.compute_acf(sig, fs, start, win_len, from_psd=from_psd)

    assert acf.corrs is not None
    assert abs(np.mean(acf.corrs)) < .25


def test_ACF_fit():
    lags = np.arange(1, 101)
    tau = .01
    amp = 1
    fs = 100
    corrs = sim_exp_decay(lags, fs, tau, amp)
    acf = ACF(corrs, lags, fs)
    acf.fit()

    assert (acf.params[0] - tau) < (tau * .1)
    assert acf.rsq > .5

def test_ACF_fit_cos():
    lags = np.arange(1, 101)
    tau = .01
    fs = 100

    corrs = sim_acf_cos(lags, fs, .01, 1, .25, .5, .01, 5, 0)


    acf = ACF(corrs, lags, fs)
    acf.fit_cos()

    assert (acf.params[0] - tau) < (tau * .1)
    assert acf.rsq > .5


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
@pytest.mark.parametrize('bounds', [True, None])
def test_fit_acf(ndim, bounds):

    n_seconds = 1
    fs = 1000
    tau = .01
    nlags = 10
    n_neurons = 2

    if bounds is True:
        bounds = [
            (.01, 0, 0, 0, 0, 0, -.5),
            (1  ,  1, 1, 1, .1, 10, .5)
        ]

    # Fit spikes
    probs, _ = sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=n_neurons, return_sum=True)
    corrs = compute_acf(probs, nlags, n_jobs=-1, progress=None)

    if ndim == 2:
        corrs = np.tile(corrs, (2, 1))

    if bounds:
        bounds = [[.001, 0, None], [1, 1, None]]

    params, _, _ = fit_acf(corrs, fs, bounds=bounds, n_jobs=-1, maxfev=1000, progress=None)

    if ndim == 1:
        assert len(params) == 3
    elif ndim == 2:
        assert (params[0] == params[1]).all()
        assert len(params[0]) == 3

    # Fit exponential decay
    corrs = sim_exp_decay(np.arange(1, 1000), fs, tau, 1)
    params, _, _ = fit_acf(corrs, fs, n_jobs=-1, progress=None)
    assert abs(params[0] - tau) < (tau * .1)



@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('bounds', [True, None])
def test_fit_acf_cos(ndim, bounds):

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

    if bounds is True:
        bounds = [
            (.01, 0, 0, 0, 0, 0, None),
            (1  ,  1, 1, 1, .1, 10, None)
        ]

    params, _, _ = fit_acf_cos(corrs, fs, bounds=bounds, maxfev=1000, n_jobs=-1, progress=None)

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

    corrs = sim_exp_decay(xs, fs, .001, 1, 0)
    params = _acf_proxy([corrs, None, None], xs, fs, 1000)
    assert params is not None

    corrs = sim_acf_cos(xs, fs, exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma, offset, osc_freq)
    params = _acf_cos_proxy([corrs, None, None], xs, fs, 1000)
    assert params is not None

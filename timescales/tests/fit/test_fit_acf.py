"""Tests for autocorrelation functions."""

import pytest
import numpy as np

from timescales.fit import ACF, fit_acf, fit_acf_cos
from timescales.fit.acf import _acf_proxy, _acf_cos_proxy
from timescales.sim import sim_acf_cos, sim_exp_decay



def test_ACF_init():
    corrs = np.random.rand(100)
    lags = np.arange(1, 101)
    fs = 100
    acf = ACF(lags, corrs, fs)

    assert (acf.corrs == corrs).all()
    assert (acf.lags == lags).all()
    assert acf.fs == fs

@pytest.mark.parametrize('from_psd', [True, False])
def test_ACF_compute_acf(from_psd):

    sig = np.random.rand(1000)
    fs = 100

    acf = ACF()
    acf.compute_acf(sig, fs, from_psd=from_psd)
    assert acf.corrs is not None
    assert abs(np.mean(acf.corrs)) < .25

    acf = ACF()
    acf.compute_acf(np.vstack((sig, sig)), fs, from_psd=from_psd, n_jobs=2)
    print(np.vstack((sig, sig)).shape)
    print(acf.corrs.shape)
    assert (acf.corrs[0] == acf.corrs[1]).all()

    # Expect error
    if from_psd is False:
        sig = np.random.rand(2, 2, 1000)
        acf = ACF()

        with pytest.raises(ValueError):
            acf.compute_acf(sig, fs, from_psd=from_psd)


@pytest.mark.parametrize('with_cos', [True, False])
@pytest.mark.parametrize('gen_fits', [True, False])
@pytest.mark.parametrize('ndim', [1, 2])
def test_ACF_fit(with_cos, gen_fits, ndim):
    lags = np.arange(1, 101)
    tau = .01
    amp = 1
    fs = 1000
    corrs = sim_exp_decay(lags, fs, tau, amp)

    if ndim == 2:
        corrs = np.vstack((corrs, corrs))

    acf = ACF(lags, corrs, fs)
    acf.fit(gen_fits=gen_fits)

    if ndim == 1:
        assert (acf.params[0] - tau) < (tau * .1)
    else:
        assert (acf.params[0, 0] - tau) < (tau * .1)

    if gen_fits:
        assert acf.corrs_fit is not None

        if ndim == 1:
            assert acf.rsq > .5
        else:
            assert acf.rsq.mean() > .5

    # Errors
    acf = ACF()
    with pytest.raises(ValueError):
        acf.fit()


@pytest.mark.parametrize('with_cos', [True, False])
@pytest.mark.parametrize('gen_components', [True, False])
@pytest.mark.parametrize('ndim', [1, 2])
def test_ACF_gen_corrs_fit(with_cos, gen_components, ndim):

    lags = np.arange(1, 101)
    tau = .01
    amp = 1
    fs = 1000
    corrs = sim_exp_decay(lags, fs, tau, amp)

    if ndim == 2:
        corrs = np.vstack((corrs, corrs))

    acf = ACF(lags, corrs, fs)
    # Note:
    # Can't sparate components when there is only one
    if gen_components and not with_cos:
        with pytest.raises(ValueError):
            acf.fit(with_cos=with_cos, gen_fits=True, gen_components= gen_components)
    else:
        acf.fit(with_cos=with_cos, gen_fits=True, gen_components= gen_components)


@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('bounds', [True, None])
def test_fit_acf(ndim, bounds):

    fs = 1000
    tau = .01
    corrs = sim_exp_decay(np.arange(1, 100), fs, tau, 1)

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
    osc_tau = .01
    osc_gamma = .01
    amp_ratio = .5
    height = 1
    offset = 0

    corrs = sim_acf_cos(xs, fs, exp_tau, osc_tau, osc_gamma, freq, amp_ratio, height, offset)

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
    osc_tau = .01
    osc_gamma = .01
    amp_ratio = .5
    height = 1
    offset = 0

    corrs = sim_exp_decay(xs, fs, .001, 1, 0)
    params = _acf_proxy([corrs, None, None], xs, fs, 1000)
    assert params is not None

    corrs = sim_acf_cos(xs, fs, exp_tau, osc_tau, osc_gamma, freq, amp_ratio, height, offset)
    params = _acf_cos_proxy([corrs, None, None], xs, fs, 1000)
    assert params is not None

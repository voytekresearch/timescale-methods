"""Tests for ACF simulation."""

import numpy as np

import pytest

from timescales.sim.acf import sim_acf_cos, sim_damped_oscillation


@pytest.mark.parametrize('return_sum', [True, False])
def test_sim_acf_cos(return_sum):

    xs = np.arange(1000)
    fs = 1000
    freq = 10
    tau = .01
    cos_gamma = 1
    var_exp = 1
    var_cos = .5
    var_cos_exp = .5

    acf = sim_acf_cos(xs, fs, freq, tau, cos_gamma, var_exp, var_cos,
                      var_cos_exp, return_sum=return_sum)

    if return_sum:
        assert len(acf) == len(xs)
        assert acf.max() >= 0
        assert acf.min() >= -1
        assert (acf[int(tau * fs)] < 1) & (acf[int(tau * fs)] > 0)
    else:
        exp, cos = acf
        assert len(exp) == len(cos) == len(xs)
        assert exp.max() > cos.max()
        assert exp.min() > cos.min()
        assert (exp > 0).all()


def test_sim_damped_oscillation():

    n_seconds = 10
    fs = 1000
    freq = 10
    gamma = 1
    var_cos = 1
    var_cos_exp = 1

    sig = sim_damped_oscillation(n_seconds, fs, freq, gamma, var_cos, var_cos_exp)

    assert sig.max() <= 1
    assert sig.min() >= -1
    np.testing.assert_almost_equal(np.mean(sig), 0, 4)

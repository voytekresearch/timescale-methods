"""Tests for ACF simulation."""

import numpy as np

from timescales.sim.acf import sim_acf_cos, sim_damped_cos, sim_exp_decay


def test_sim_acf_cos():

    xs = np.arange(1000)
    fs = 1000
    freq = 10
    exp_tau = .01
    osc_tau = .01
    osc_gamma = .01
    osc_freq = 5
    offset = 0
    amp_ratio = .5
    height = 1

    acf =  sim_acf_cos(xs, fs, exp_tau, osc_tau, osc_gamma, osc_freq, amp_ratio, height, offset)

    assert len(acf) == len(xs)
    assert acf.max() >= 0
    assert acf.min() >= -1
    assert (acf[int(exp_tau * fs)] < 1) & (acf[int(exp_tau * fs)] > 0)


def test_sim_exp_decay():

    xs = np.arange(1000)
    fs = 1000
    amplitude = 1
    tau = .01
    offset = 0

    exp = sim_exp_decay(xs, fs, amplitude, tau, offset)

    assert exp.ndim == 1
    assert len(exp) == len(xs)
    assert np.argmax(exp) == 0
    assert exp.max() <= 1
    assert exp.min() >= 0


def test_sim_damped_cos():

    xs = np.arange(1000)
    fs = 1000
    amplitude = 1
    tau = .01
    gamma = 0.05
    freq = 10

    cos = sim_damped_cos(xs, fs, tau, amplitude, gamma, freq)

    assert cos.ndim == 1
    assert len(cos) == len(xs)
    assert cos.max() <= 1
    assert cos.min() >= -1

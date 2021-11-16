"""Tests for ACF simulation."""

import numpy as np

import pytest

from timescales.sim.acf import sim_acf_cos, sim_damped_oscillation


def test_sim_acf_cos():

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

    acf =  sim_acf_cos(xs, fs, exp_tau, exp_amp, osc_tau, osc_amp, osc_gamma,  offset, osc_freq)

    assert len(acf) == len(xs)
    assert acf.max() >= 0
    assert acf.min() >= -1
    assert (acf[int(exp_tau * fs)] < 1) & (acf[int(exp_tau * fs)] > 0)

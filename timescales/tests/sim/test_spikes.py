"""Tests for exponential spiking simulations."""

import numpy as np

import pytest

from neurodsp.sim import sim_synaptic_kernel

from timescales.sim.spikes import (sim_spikes_synaptic, sim_spikes_prob,
    sim_poisson)


@pytest.mark.parametrize('tau', [.1, 1e-6])
def test_sim_spikes_synaptic(tau, return_sum):

    n_seconds = 10
    fs = 1000

    spikes = sim_spikes_synaptic(n_seconds, fs, tau)

    assert spikes.ndim == 1
    assert spikes.dtype == bool


def test_sim_spikes_prob():

    n_seconds = 1
    fs = 1000
    kernel =  sim_synaptic_kernel(5 * .1, fs, 0, .1)
    kernel = np.tile(kernel, (2, 1))
    isi = np.array([100, 100])
    mu = 100
    var_noise = .1

    with pytest.raises(ValueError):
        probs = sim_spikes_prob(n_seconds, fs, kernel, mu=mu, var_noise=var_noise)

    probs = sim_spikes_prob(n_seconds, fs, kernel, isi=isi, mu=mu, var_noise=var_noise)

    assert probs.ndim  == 1
    assert probs.max() <= 1
    assert probs.min() >= 0


def test_sim_poisson():

    n_seconds = 1
    fs = 1000
    kernel =  sim_synaptic_kernel(5 * .1, fs, 0, .1)
    kernel = np.tile(kernel, (2, 1))
    isi = np.array([100, 100])
    mu = 100

    with pytest.raises(ValueError):
        poisson = sim_poisson(n_seconds, fs, kernel, mu=mu)

    poisson = sim_poisson(n_seconds, fs, kernel, isi=isi, mu=mu)

    assert np.min(poisson) == 0
    assert np.max(poisson) == 1

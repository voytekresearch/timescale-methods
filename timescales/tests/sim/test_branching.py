"""Tests for branching simulations."""

import pytest

from timescales.sim.branching import sim_branching, sim_branching_spikes


@pytest.mark.parametrize('lambda_a', [None, 100])
def test_sim_branching(lambda_a):

    n_seconds = 10
    fs = 1000
    tau = .01
    lambda_h = 10

    sig = sim_branching(n_seconds, fs, tau, lambda_h, lambda_a)

    assert len(sig) == int(n_seconds * fs)
    assert sig.mean() > 0


@pytest.mark.parametrize('lambda_a', [None, 100])
def test_sim_branching_spikes(lambda_a):

    n_seconds = 10
    fs = 1000
    tau = .01
    lambda_h = 10

    probs, spikes = sim_branching_spikes(n_seconds, fs, tau, lambda_h, lambda_a, n_neurons=10)

    assert len(probs) == int(n_seconds * fs)
    assert probs.min() >= 0
    assert probs.max() <= 1

    assert spikes.dtype == bool
    assert len(spikes) == 10

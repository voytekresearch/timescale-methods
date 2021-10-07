"""Test OU simulations."""

import pytest

from timescales.sim.ou import sim_spikes_ou


@pytest.mark.parametrize('return_sum', [True, False])
def test_sim_spikes_ou(return_sum):

    n_seconds = 1
    fs = 1000
    tau = .1
    n_neurons = 10

    spikes = sim_spikes_ou(n_seconds, fs, tau, n_neurons=n_neurons, return_sum=return_sum)

    if return_sum:
        assert spikes.dtype == int
        assert len(spikes) == int(n_seconds * fs)
    else:
        assert spikes.dtype == bool
        assert len(spikes) == n_neurons
        assert len(spikes[0]) == int(n_seconds * fs)

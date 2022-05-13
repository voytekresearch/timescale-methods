"""Test OU simulations."""

import pytest

from timescales.sim.ou import sim_ou


@pytest.mark.parametrize('return_sum', [True, False])
def test_sim_ou(return_sum):

    n_seconds = 1
    fs = 1000
    tau = .1
    n_neurons = 10

    spikes = sim_ou(n_seconds, fs, tau)

    assert len(spikes) == int(n_seconds * fs)

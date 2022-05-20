"""Tests for branching simulations."""

import pytest

from timescales.sim.branching import sim_branching


@pytest.mark.parametrize('lambda_a', [None, 100])
def test_sim_branching(lambda_a):

    n_seconds = 10
    fs = 1000
    tau = .01
    lambda_h = 10

    sig = sim_branching(n_seconds, fs, tau, lambda_h, lambda_a)

    assert len(sig) == int(n_seconds * fs)
    assert sig.mean() > 0

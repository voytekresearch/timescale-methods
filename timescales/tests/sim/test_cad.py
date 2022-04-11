"""Test simulations functions."""

import numpy as np

from neurodsp.utils.norm import normalize_sig

from timescales.fit import convert_knee_val
from timescales.decompose import CAD
from timescales.sim import sim_asine_oscillation, sim_autoregressive, sim_branching

def test_sim_autoregressive():

    fs = 1000
    n_seconds = 1
    tau = convert_knee_val(10)

    sig = sim_branching(n_seconds, fs, tau, 100)
    sig = normalize_sig(sig, 0, .5)

    sig_fit = sim_autoregressive(sig, [.8, .4])

    assert len(sig_fit) == len(sig)


def test_sim_asine_oscillations():

    fs = 1000
    xs = np.arange(fs)
    freq = 10
    rdsym = 0
    phi = 0
    height = 1

    sig = sim_asine_oscillation(xs, fs, freq, rdsym, phi, height)

    freqs = [10, 20]
    sig_double_freq = sim_asine_oscillation(xs, fs, freqs, rdsym, phi, height)

    assert len(sig_double_freq) == len(sig) == len(xs)

"""Tests for autoregressive decomposition."""

from scipy.fftpack import fftfreq
from tqdm.notebook import tqdm
import numpy as np

import pytest

from neurodsp.utils.norm import normalize_sig

from timescales.sim import sim_branching
from timescales.fit import convert_knee_val


from timescales.decompose.decompose import (decompose_ar, decompose_ar_windows, gen_ar_fit,
    sim_asine, iter_estimate_freq, estimate_freq)



def test_decompose_ar():

    osc_order = 2
    ar_order = 2
    fs = 1000
    n_seconds = 1
    tau = convert_knee_val(10)
    xs = np.arange(fs)

    lfp = sim_branching(n_seconds, fs, tau, 100)
    osc = sim_asine(xs, fs, 20, .25, -.49, 1)

    lfp = normalize_sig(lfp, 0, .5)
    osc = normalize_sig(osc, 0, .5)

    sig = lfp + osc

    params = decompose_ar(sig, fs, osc_order, ar_order)

    assert len(params) == (ar_order * 4)


def test_decompose_ar_windows():

    osc_order = 1
    ar_order = 1
    fs = 1000
    n_seconds = 1
    tau = convert_knee_val(10)
    xs = np.arange(fs)

    lfp = sim_branching(n_seconds, fs, tau, 100)
    osc = sim_asine(xs, fs, 20, .25, -.49, 1)

    lfp = normalize_sig(lfp, 0, .5)
    osc = normalize_sig(osc, 0, .5)

    sig = lfp + osc

    nperseg = 100
    noverlap = 0

    osc_fit, ar_fit, params, t_def = decompose_ar_windows(
        sig, fs, osc_order, ar_order, nperseg, noverlap, progress=tqdm)

    assert len(osc_fit) == len(ar_fit)
    assert len(osc_fit[0]) == len(ar_fit[0]) == nperseg

    osc_fit, params, t_def = decompose_ar_windows(
        sig, fs, osc_order, 0, nperseg, noverlap, progress=tqdm)

    assert len(osc_fit) == len(ar_fit)


def test_gen_ar_fit():

    osc_order = 2
    ar_order = 2
    fs = 1000
    n_seconds = 1
    tau = convert_knee_val(10)
    xs = np.arange(fs)

    lfp = sim_branching(n_seconds, fs, tau, 100)
    osc = sim_asine(xs, fs, 20, .25, -.49, 1)

    lfp = normalize_sig(lfp, 0, .5)
    osc = normalize_sig(osc, 0, .5)

    sig = lfp + osc

    params = decompose_ar(sig, fs, osc_order, ar_order)

    sig_fit = gen_ar_fit(sig, params)

    assert len(sig_fit) == len(sig)

def test_sim_asine():

    fs = 1000
    xs = np.arange(fs)
    freq = 10
    rdsym = 0
    phi = 0
    height = 1

    sig = sim_asine(xs, fs, freq, rdsym, phi, height)
    assert len(sig) == len(xs)

def test_iter_estimate_freq():

    fs = 1000
    xs = np.arange(fs)
    freq = 10
    rdsym = 0
    phi = 0
    height = 1

    sig = sim_asine(xs, fs, freq, rdsym, phi, height)
    osc_order = 1
    n_eig = 20

    guess, bounds = iter_estimate_freq(sig, fs, osc_order, n_eig, freq_pad=10)

    assert len(guess) == len(bounds[0]) == len(bounds[1])
    for g, lb, up in zip(guess, bounds[0], bounds[1]):
        assert (g >= lb) & (g <= up)


def test_estimate_freq():

    fs = 1000
    xs = np.arange(fs)
    freq = 10
    rdsym = 0
    phi = 0
    height = 1

    sig = sim_asine(xs, fs, freq, rdsym, phi, height)

    freq_guess = estimate_freq(sig, fs, 20)

    assert freq_guess > 0 and freq_guess < 100

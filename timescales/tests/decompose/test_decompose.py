"""Tests for autoregressive decomposition."""

from scipy.fftpack import fftfreq
from tqdm.notebook import tqdm
import numpy as np

import pytest

from neurodsp.utils.norm import normalize_sig

from timescales.sim import sim_branching, sim_asine_oscillation
from timescales.conversions import convert_knee
from timescales.decompose.decompose import CAD, CADGroup


def test_CAD():

    np.random.seed(0)

    n_seconds = 1
    fs = 1000

    tau = convert_knee(10)
    xs = np.arange(fs)

    lfp = sim_branching(n_seconds, fs, tau, 100)
    osc = sim_asine_oscillation(xs, fs, 20, .25, -.49, 1)
    sig = normalize_sig(lfp, 0, .5) + normalize_sig(osc, 0, .5)

    osc_order = 1
    ar_order = 2

    cad = CAD(sig, fs, osc_order, ar_order)

    assert (sig == cad.sig).all()
    assert fs == cad.fs
    assert osc_order == cad.osc_order
    assert ar_order == cad.ar_order

    cad.fit(use_freq_est=True)

    assert (cad.full_fit == cad.osc_fit + cad.ar_fit).all()
    assert isinstance(cad.params, dict)

    cad_no_freq_est = CAD(sig, fs, osc_order, ar_order)
    cad_no_freq_est.fit(use_freq_est=False)

    assert cad_no_freq_est.bounds is None
    assert cad_no_freq_est.guess is None

    cad_no_freq_est = CAD(sig, fs, osc_order, ar_order)
    cad_no_freq_est.iter_freq_estimation(20, 10)

    assert cad.bounds == cad_no_freq_est.bounds
    assert (cad.guess == cad_no_freq_est.guess).all()


def test_CADGroup():

    np.random.seed(0)

    n_seconds = 1
    fs = 1000

    tau = convert_knee(10)
    xs = np.arange(fs)

    lfp = sim_branching(n_seconds, fs, tau, 100)
    osc = sim_asine_oscillation(xs, fs, 20, .25, -.49, 1)
    sig = normalize_sig(lfp, 0, .5) + normalize_sig(osc, 0, .5)

    sigs = np.tile(sig, (2, 1))

    osc_order = 1
    ar_order = 2

    cg = CADGroup(sigs, fs, osc_order, ar_order)
    cg.fit()

    assert (cg[0].full_fit == cg[1].full_fit).all()
    assert len(cg) == 2
    for i in cg:
        assert isinstance(i, CAD)

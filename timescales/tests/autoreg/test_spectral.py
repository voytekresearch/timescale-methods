"""Test for computing AR spectra."""

import numpy as np
import pytest

from statsmodels.regression.linear_model import burg as burg_sm

from neurodsp.utils.norm import normalize_sig

from timescales.sim import sim_branching
from timescales.conversions import convert_knee
from timescales.autoreg import compute_ar_spectrum, burg


@pytest.mark.parametrize('method', ['burg', 'yules_walker'])
def test_compute_ar_spectrum(method):

    # 1d
    n_seconds = 1
    fs = 1000
    knee_freq = 10
    tau = convert_knee(knee_freq)
    sig = normalize_sig(sim_branching(n_seconds, fs, tau, 200), 0, 1)
    nfft=4096

    freqs, powers = compute_ar_spectrum(sig, fs, 5, None, method, nfft, n_jobs=1)

    assert len(freqs) == len(powers) == nfft//2
    assert powers[:len(powers)//2].sum() > powers[len(powers)//2].sum()

    # 2d
    sig = np.stack((sig, sig))
    freqs, powers = compute_ar_spectrum(sig, fs, 5, None, method, nfft, n_jobs=1)

    assert len(freqs) == len(powers[0]) == len(powers[1])
    assert (powers[0] == powers[1]).all()


def test_burg():

    np.random.seed(0)
    sig = sim_branching(1, 1000, .01, 200, 0, 1)
    sig -= sig.mean()
    sig /= sig.std()

    order = 5
    ar_burg, _ = burg_sm(sig, order)
    ar = burg(sig, order)
    np.testing.assert_almost_equal(ar_burg, ar)

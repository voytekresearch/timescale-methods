"""Test for multiprocessing wrapper functions."""

import pytest

from timescales.mp import compute_taus, mp_fit_psd, mp_fit_acf, sort_result

@pytest.mark.parametrize('method', ['PSD', 'ACF'])
@pytest.mark.parametrize('progress', ['tdqm.notebook', pytest.mark.xfail('fail')])
def test_compute_taus(method, progress):
    pass

def test_mp_fit_psd():
    pass

def test_mp_fit_acf():
    pass

def sort_results():
    pass

"""Test for multiprocessing wrapper functions."""

import pytest

import numpy as np

from fooof import FOOOF

from timescales.sim import sim_acf_cos, sim_spikes_synaptic
from timescales.fit import compute_acf, ACF
from timescales.mp import compute_taus, mp_fit_psd, mp_fit_acf, sort_result

@pytest.mark.parametrize('method', ['psd', 'acf'])
@pytest.mark.parametrize('progress', ['tdqm.notebook', pytest.mark.xfail('fail')])
def test_compute_taus(method, progress):

    if method == 'acf':
        xs = np.arange(1, 100)
        fs = 1000
        freq = 10
        tau = .01
        osc_tau = .25
        osc_gamma = .01
        amp_ratio = .8
        height = 1
        offset = 0

        corrs = sim_acf_cos(xs, fs, tau, osc_tau, osc_gamma, freq, amp_ratio, height, offset)
        corrs = np.array([corrs] * 2)

        fit_kwargs={'fs':fs, 'lags':xs}
        taus, rsq, result_class = compute_taus(corrs, method=method, fit_kwargs=fit_kwargs,
                                               progress=None)

    elif method == 'psd':

        n_seconds = 10
        fs = 1000
        sig, _ = sim_spikes_synaptic(n_seconds, fs, .01, n_neurons=1)

        fooof_kwargs = dict(knee_bounds=(.2, 40), max_n_peaks=3,
                    peak_threshold=3, peak_width_limits=(2, 10))

        compute_spectrum_kwargs = dict(method='welch', avg_type='mean', nperseg=int(1 * fs))

        fit_kwargs = dict(
            sig=sig,
            fs=fs,
            win_len=fs,
            f_range=(.2, 40),
            compute_spectrum_kwargs=compute_spectrum_kwargs,
            fit_kwargs=fooof_kwargs,
            rsq_type='linear'
        )

        win_starts = np.array([0, 100], dtype=int)

        taus, rsq, result_class = compute_taus(win_starts, method=method, fit_kwargs=fit_kwargs,
                                               progress=None)

    assert len(taus) == len(rsq) == len(result_class)


def test_mp_fit_psd():

    n_seconds = 10
    fs = 1000

    np.random.seed(0)
    sig, _ = sim_spikes_synaptic(n_seconds, fs, .01, n_neurons=1)

    fooof_kwargs = dict(knee_bounds=(.2, 40), max_n_peaks=3,
                peak_threshold=3, peak_width_limits=(2, 10))

    compute_spectrum_kwargs = dict(method='welch', avg_type='mean', nperseg=int(1 * fs))

    fit_kwargs = dict(
        sig=sig,
        fs=fs,
        win_len=fs,
        f_range=(.2, 40),
        compute_spectrum_kwargs=compute_spectrum_kwargs,
        fit_kwargs=fooof_kwargs,
        rsq_type='linear'
    )

    taus, rsq, result_class = mp_fit_psd(0, **fit_kwargs)

    assert isinstance(taus, float)
    assert isinstance(rsq, float)
    assert isinstance(result_class, FOOOF)


@pytest.mark.parametrize('method', ['cos', 'decay'])
@pytest.mark.parametrize('sig_bool', [True, False])
def test_mp_fit_acf(method, sig_bool):

    n_seconds = 10
    fs = 1000

    np.random.seed(0)
    sig, _ = sim_spikes_synaptic(n_seconds, fs, .01, n_neurons=1)

    fit_kwargs = dict(
        sig = sig,
        fs = fs,
        win_len = fs,
        method = method,
        compute_acf_kwargs = dict(nlags=int(.5 * fs)),
        fit_kwargs = dict(maxfev=10000)
    )

    if not sig_bool:
        del fit_kwargs['sig']

        corrs = compute_acf(sig, 1000)
        fit_kwargs['lags'] = np.arange(1, len(corrs)+1)

        taus, rsq, result_class = mp_fit_acf(corrs, **fit_kwargs)
    else:
        taus, rsq, result_class = mp_fit_acf(0, **fit_kwargs)

    assert isinstance(taus, float)
    assert isinstance(rsq, float)
    assert isinstance(result_class, ACF)

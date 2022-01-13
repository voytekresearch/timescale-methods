"""Tests for fit optimization."""

import pytest
from pytest_cov.embed import cleanup_on_sigterm
from itertools import combinations
import numpy as np

from fooof import FOOOF

from neurodsp.sim import sim_oscillation

from timescales.sim import sim_spikes_synaptic
from timescales.fit import ACF, convert_knee_val
from timescales.optimize  import fit_grid


@pytest.mark.parametrize('type', [None, pytest.param('fail_rsq', marks=pytest.mark.xfail),
                                  pytest.param('fail_fooof', marks=pytest.mark.xfail)])
def test_fit_grid_psd(type):

    cleanup_on_sigterm()

    n_seconds = 10
    fs = 1000
    tau = convert_knee_val(10)
    n_neurons = 1

    sig, _ = sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=n_neurons, return_sum=True)
    cos = sim_oscillation(n_seconds, fs, 10)
    sig = (sig * .95) + (cos * .05)

    grid = {
        # Init kwargs
        'peak_width_limits': np.array(list(combinations(np.arange(1, 10, 2), 2))),
        'max_n_peaks': np.arange(1, 13, 3),
        'peak_threshold': np.arange(1, 5),
        # Self kwargs (post-init)
        'knee_freq_bounds': np.column_stack((np.arange(0, 100, 10), np.arange(0, 100, 10)+20))[:-1],
        'exp_bounds': np.column_stack((np.array([.01, 1, 2]), np.arange(2, 5))),
        # Compute spectra kwargs
        'nperseg': (np.linspace(.5, 1.5, 5) * fs).astype(int),
        'noverlap': np.arange(0, 10, 2)/10,
        # Fit kwargs
        'freq_range': np.array([[0, 100]])
    }

    max_n_params = 100

    if type == None:
        results = fit_grid(sig, fs, grid, mode='psd', max_n_params=max_n_params, rsq_thresh=0)

        for result in results:
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], float)
            assert isinstance(result[2], float)
            assert isinstance(result[3], (FOOOF, float))

    elif type == 'fail_rsq':
        results = fit_grid(sig, fs, grid, mode='psd', max_n_params=max_n_params, rsq_thresh=np.inf)
    elif type == 'fail_fooof':
        _grid = grid.copy()
        _grid['freq_range'] = np.array([[-10, -1]])
        results = fit_grid(sig, fs, _grid, mode='psd', max_n_params=max_n_params)


    assert len(results) <= max_n_params

@pytest.mark.parametrize('type', [None, pytest.param('fail_rsq', marks=pytest.mark.xfail)])
def test_fit_grid_acf(type):

    cleanup_on_sigterm()

    n_seconds = 10
    fs = 1000
    tau = convert_knee_val(10)
    n_neurons = 1

    sig, _ = sim_spikes_synaptic(n_seconds, fs, tau, n_neurons=n_neurons, return_sum=True)
    cos = sim_oscillation(n_seconds, fs, 10)
    sig = (sig * .95) + (cos * .05)

    grid = {
        # ACF params
        'nlags': (np.linspace(0, 1, 9)[1:] * fs).astype(int),
        # Exponential decay param
        'exp_tau': np.array([[convert_knee_val(20), convert_knee_val(1)]]),
        # Oscillatory params
        'osc_tau': np.array([
            [.2, .6],
            [.4, .8],
            [.6, 1.],
            [.8, 1.2]
        ]),
        'osc_gamma': np.array([
            [.4, .8],
            [.6, 1],
            [.8, 1.2],
            [1, 1.4]
        ]),
        'freq': np.array([[None, None]]),
        # Scaling and offset params
        'amp_ratio': np.array([
            [.2, .6],
            [.6, .8],
            [.8, 1.]
        ]),
        'height': np.array([
            [.2, .8],
            [.4, 1],
            [.6, 1.2],
            [.8, 1.4]
        ]),
        'offset': np.array([[-1, 1]])
    }

    max_n_params = 100

    if type == None:

        results = fit_grid(sig, fs, grid, mode='acf', max_n_params=max_n_params)

        assert len(results) <= max_n_params

        for result in results:
            assert isinstance(result[0], np.ndarray)
            assert isinstance(result[1], float)
            assert isinstance(result[2], float)
            assert isinstance(result[3], (ACF, float))
    else:
        results = fit_grid(sig, fs, grid, mode='acf', max_n_params=max_n_params, rsq_thresh=np.inf)

        assert len(results) <= max_n_params

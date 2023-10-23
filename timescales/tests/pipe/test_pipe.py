"""Pipeline tests."""

import pytest
import numpy as np

from timescales.sim import sim_spikes_prob
from neurodsp.sim import sim_oscillation, sim_synaptic_kernel

from timescales.conversions import convert_knee
from timescales.pipe import Pipe

def test_pipe_init():

    seeds = np.arange(5)
    n_seconds = 10
    fs = 1000

    pipe = Pipe(n_seconds, fs, seeds)
    assert pipe.n_seconds == n_seconds
    assert pipe.fs == fs
    assert (pipe.seeds == seeds).all()


@pytest.mark.parametrize('operator', ['+', '-', '*', '/'])
@pytest.mark.parametrize('rescale', [0, 1, 2, 3])
def test_pipe_simulate(operator, rescale):
    """Test simulations."""
    n_seconds = 1
    fs = 1000
    freqs = [10, 100]

    pipe = Pipe(n_seconds, fs)

    mean = None
    variance = None
    if rescale == 0:
        rescale = None
    elif rescale == 1:
        rescale = (-1, 1)
    elif rescale == 2:
        rescale = None
        mean = 0
        variance = 1

    if rescale == 3:
        # Expects either rescale or (mean, variance), but shouldn't be defined
        with pytest.raises(ValueError):
            pipe.simulate(sim_oscillation, freqs[0], operator=operator, rescale=(-1, 1), mean=0,
                variance=1)
    else:
        pipe.simulate(sim_oscillation, freqs[0], rescale=rescale, mean=mean,
            variance=variance)

        pipe.simulate(sim_oscillation, freqs[0], operator=operator, rescale=rescale, mean=mean,
            variance=variance)


def test_pipe_spikes():
    """Tests the sample and bin methods for spiking signals."""

    # Settings
    n_seconds = 5
    fs = 5000
    tau = convert_knee(10)
    kernel = sim_synaptic_kernel(5 * tau, fs, 0, tau)

    # Initialize pipeline
    pipe = Pipe(n_seconds, fs)

    # Define steps in the pipeline
    pipe.simulate(sim_spikes_prob, kernel, rescale=(0, .8))
    pipe.simulate(sim_oscillation, 10, rescale=(0, .2))

    pipe.sample(10000 * n_seconds)
    assert pipe.sig.min() == 0 and pipe.sig.max() == 1
    _sig = pipe.sig.copy()

    pipe.bin(10)
    assert len(_sig) > len(pipe.sig)
    assert pipe.sig.max() >= 1


def test_pipe_run():
    """Test adding steps and fitting."""

    # Settings
    n_seconds = 2
    fs = 1000
    tau = convert_knee(10)
    kernel = sim_synaptic_kernel(5 * tau, fs, 0, tau)

    # PSD
    pipe_psd = Pipe(n_seconds, fs, seeds=list(range(5)))
    pipe_psd.add_step('simulate', sim_spikes_prob, kernel, rescale=(0, 1))
    pipe_psd.add_step('transform', 'PSD', ar_order=5)
    pipe_psd.add_step('fit', ['knee_freq', 'rsq', 'tau'], f_scale=1)
    pipe_psd.run()

    # ACF
    pipe_acf = Pipe(n_seconds, fs, seeds=list(range(5)))
    pipe_acf.add_step('simulate', sim_spikes_prob, kernel, rescale=(0, 1))
    pipe_acf.add_step('transform', 'ACF')
    pipe_acf.add_step('fit', ['knee_freq', 'rsq', 'tau'])
    pipe_acf.run()

    # AR-PSD is more Lorentzian than ACF
    assert pipe_psd.results[:, 1].mean() > pipe_acf.results[:, 1].mean()

    # AR-PSD timescales are more accurate (MAE)
    assert np.abs(pipe_psd.results[:, 2] - tau).mean() < \
        np.abs(pipe_acf.results[:, 2] - tau).mean()


def test_pipe__run():

    # Settings
    n_seconds = 2
    fs = 1000
    tau = .016
    kernel = sim_synaptic_kernel(5 * tau, fs, 0, tau)

    # PSD
    pipe = Pipe(n_seconds, fs)
    pipe.add_step('simulate', sim_spikes_prob, kernel, rescale=(0, .8))
    pipe.add_step('transform', 'ACF')
    pipe.add_step('fit', 'knee_freq')
    pipe._run(0)

    pipe = Pipe(n_seconds, fs)
    pipe.add_step('simulate', sim_spikes_prob, kernel, rescale=(0, .8))
    pipe.add_step('transform', 'PSD')
    pipe.add_step('fit', 'knee_freq')
    pipe._run(0)


    assert pipe.model is not None
    assert pipe.result is not None

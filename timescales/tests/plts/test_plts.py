"""Test plotting functions."""

import pytest

import matplotlib.pyplot as plt
import numpy as np

from timescales.plts import plot_connected_scatter


@pytest.mark.parametrize('paired', [True, False])
@pytest.mark.parametrize('twin', [True, False])
@pytest.mark.parametrize('fill_nans', [True, False])
@pytest.mark.parametrize('scatter_jit', [0.05, (.1, .05)])
def test_plot_connected_scatter(paired, twin, fill_nans, scatter_jit):

    taus_a = np.random.uniform(0, 1, 100)
    taus_b = np.random.uniform(1, 10, 100)

    _, ax = plt.subplots()

    plot_connected_scatter(taus_a, taus_b, ax, paired=paired, twin=twin, fill_nans=fill_nans,
                           scatter_jit=scatter_jit)

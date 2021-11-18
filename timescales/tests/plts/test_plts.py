"""Test plotting functions."""

import pytest

import matplotlib.pyplot as plt
import numpy as np

from timescales.plts import plot_connected_scatter


@pytest.mark.parametrize('paired', [True, False])
def test_plot_connected_scatter(paired):

    taus_a = np.random.uniform(0, 1, 100)
    taus_b = np.random.uniform(1, 10, 100)

    fig, ax = plt.subplots()

    plot_connected_scatter(taus_a, taus_b, ax, '', paired=paired)

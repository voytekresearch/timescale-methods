"""Test normaliztaion function."""

import numpy as np
from timescales.utils import normalize


def test_normalize():

    x = np.random.rand(1000)

    x_norm = normalize(x, 0, 1)

    assert x_norm.min() >= 0
    assert x_norm.max() <= 1

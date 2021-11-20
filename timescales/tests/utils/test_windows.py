"""Test window utility functions."""

import pytest

import numpy as np

from timescales.utils import create_windows


@pytest.mark.parametrize('dtype', ['array', 'int'])
def test_create_windows(dtype):

    if dtype == 'array':
        samples = np.array([[0, 1000]])
    else:
        samples = 1000

    win_len = 100
    win_spacing = 100

    starts, mids, ends = create_windows(samples, win_len, win_spacing)

    assert (mids == np.arange(100, 1000, 100)).all()
    assert (starts + win_spacing//2 == mids).all()
    assert (starts + win_spacing == ends).all()

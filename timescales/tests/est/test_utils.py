"""Test estimation utilitiy functions."""

import pytest

import numpy as np

from timescales.est.utils import progress_bar, check_guess_and_bounds


@pytest.mark.parametrize('progress', [None, 'tqdm', pytest.param('invalid',
                                                                 marks=pytest.mark.xfail)])
def test_progress_bar(progress):

    n_iterations = 10
    iterable = [0] * n_iterations

    pbar = progress_bar(iterable, progress, n_iterations)

    assert len(pbar) == n_iterations


@pytest.mark.parametrize('guess_', [None, [0, 0, 0]])
@pytest.mark.parametrize('bounds_', [None, [0, 0, 0]])
def test_check_guess_and_bounds(guess_, bounds_):

    corrs = np.random.rand(3, 100)

    guess, bounds = check_guess_and_bounds(corrs, guess_, bounds_)

    assert len(corrs) == len(guess) == len(bounds)
    assert isinstance(guess, np.ndarray)
    assert isinstance(bounds, np.ndarray)

    if guess_ is None:
        assert guess[0] == guess[1]
    else:
        assert (guess[0] == guess[1]).all()

    if bounds_ is None:
        assert bounds[0] == bounds[1]
    else:
        assert (bounds[0] == bounds[1]).all()


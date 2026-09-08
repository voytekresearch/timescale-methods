"""Choosing which modes to report, by variance share."""

import numpy as np
import pytest

from timescales.decomposition.select import (MIN_VARIANCE_SHARE, dominant_modes,
                                             variance_shares)


class _Mode:
    """Stand-in: dominant_modes only ever passes modes through."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'_Mode({self.name})'


def _modes(n):
    return [_Mode(i) for i in range(n)]


class TestVarianceShares:

    def test_the_share_is_the_component_at_lag_zero(self):
        modes = _modes(3)
        comps = np.array([[0.70, 0.3, 0.1],
                          [0.25, -0.1, 0.0],
                          [0.05, 0.01, 0.0]])
        np.testing.assert_allclose(variance_shares(modes, comps), [0.70, 0.25, 0.05])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            variance_shares(_modes(2), np.zeros((3, 4)))


class TestDominantModes:

    def test_ranked_by_share_not_by_input_order(self):
        modes = _modes(3)
        kept = dominant_modes(modes, [0.05, 0.70, 0.25], min_share=0.0)
        assert [m.name for m, _ in kept] == [1, 2, 0]

    def test_modes_below_the_floor_are_dropped(self):
        modes = _modes(4)
        kept = dominant_modes(modes, [0.60, 0.30, 0.009, 0.001], min_share=0.01)
        assert [m.name for m, _ in kept] == [0, 1]

    def test_the_default_floor_is_one_percent(self):
        assert MIN_VARIANCE_SHARE == 0.01
        modes = _modes(2)
        assert len(dominant_modes(modes, [0.5, 0.011])) == 2
        assert len(dominant_modes(modes, [0.5, 0.009])) == 1

    def test_n_max_truncates_after_ranking(self):
        modes = _modes(4)
        kept = dominant_modes(modes, [0.1, 0.5, 0.3, 0.05], min_share=0.0, n_max=2)
        assert [m.name for m, _ in kept] == [1, 2]

    def test_shares_come_back_with_their_modes(self):
        modes = _modes(2)
        kept = dominant_modes(modes, [0.25, 0.60], min_share=0.0)
        assert kept[0][1] == pytest.approx(0.60)
        assert kept[1][1] == pytest.approx(0.25)

    def test_non_finite_shares_are_dropped(self):
        modes = _modes(3)
        kept = dominant_modes(modes, [0.5, np.nan, 0.2], min_share=0.0)
        assert [m.name for m, _ in kept] == [0, 2]

    def test_everything_below_the_floor_gives_nothing(self):
        assert dominant_modes(_modes(2), [0.004, 0.002]) == []

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            dominant_modes(_modes(2), [0.5, 0.3, 0.2])

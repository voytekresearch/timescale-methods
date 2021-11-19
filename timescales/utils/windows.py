"""Functions for windowing a timeseries."""

import numpy as np


def create_windows(samples, win_len, win_spacing):
    """Create windows timings for a sliding window analysis.

    Parameters
    ----------
    samples : 2d array or int
        Contains start index and end index for periods of interest.
        An integer is interpreted as a range from (0, int).
    win_len : int
        Number of samples each window contains
    win_spacing : int
        Number of samples between window starts.

    Returns
    -------
    win_starts : 1d array
        Start indices for each window.
    mid_points : 1d array
        Mid point indices for each window.
    win_ends : 1d array
        End indices for each window.
    """

    if isinstance(samples, int):
        samples = [[0, samples]]

    win_starts = []

    for lower, upper in samples:
        inds = np.arange(lower, upper, win_spacing) - (win_len//2)
        win_starts.append(inds[np.where((inds >= 0) & (inds < upper))[0]])

    win_starts = np.concatenate(win_starts)
    mid_points = win_starts + (win_len//2)
    win_ends = win_starts + win_len

    return win_starts, mid_points, win_ends

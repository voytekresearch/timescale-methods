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
        win_starts.append(inds[np.where((inds >= lower) & (inds <= upper))[0]])

    win_starts = np.concatenate(win_starts)
    mid_points = win_starts + (win_len//2)
    win_ends = win_starts + win_len

    return win_starts, mid_points, win_ends


def get_distinct_windows(starts, ends):
    """Return non-overlapping windows.

    Parameters
    ----------
    starts : 1d array
        Window starts, in samples.
    ends : 1d array
        Window ends, in samples.

    Returns
    -------
    starts : 1d array
        Non-overlapping window starts, in samples.
    ends : 1d array
        Non-overlapping window ends, in samples.
    """

    starts_nooverlap = []
    ends_nooverlap = []

    for ind, (s, e) in enumerate(zip(starts, ends)):

        if ind == 0:
            starts_nooverlap.append(s)
            ends_nooverlap.append(e)
            continue

        if s >= ends_nooverlap[-1]:
            starts_nooverlap.append(s)
            ends_nooverlap.append(e)

    return np.array(starts_nooverlap), np.array(ends_nooverlap)

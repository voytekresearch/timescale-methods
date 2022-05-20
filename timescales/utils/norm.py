"""Normalization functions."""

import numpy as np


def normalize(arr, min, max):
    """Normalize to range from min to max.

    Parameters
    ----------
    arr : 1d or 2d array
        Powers or correlations to normalize.
    min : float
        Minimum value.
    max : float
        Maximum value.

    Returns
    -------
    arr : 1d or 2d array
        Normalized array.
    """

    if arr.ndim == 2:
        for ind in range(len(arr)):
            arr[ind] = normalize(arr[ind], min, max)
    else:
        arr = np.interp(arr, (arr.min(), arr.max()), (min, max))

    return arr

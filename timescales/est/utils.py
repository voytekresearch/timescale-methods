"""Utiliity fitting functions."""


from importlib import import_module

import numpy as np


def progress_bar(iterable, progress, n_to_run, pbar_desc='Fitting ACF'):
    """Add a progress bar to an iterable to be processed.

    Parameters
    ----------
    iterable : list or iterable
        Iterable object to potentially apply progress tracking to.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Which kind of progress bar to use. If None, no progress bar is used.
    n_to_run : int
        Number of jobs to complete.
    pbar_desc: str, optional
        Display text for the progress bar.

    Returns
    -------
    pbar : iterable or tqdm object
        Iterable object, with tqdm progress functionality, if requested.

    Raises
    ------
    ValueError
        If the input for `progress` is not understood.
    """

    # Check progress specifier is okay
    tqdm_options = ['tqdm', 'tqdm.notebook']
    if progress is not None and progress not in tqdm_options:
        raise ValueError("Progress bar option not understood.")

    # Use a tqdm, progress bar, if requested
    if progress:

        # Try loading the tqdm module
        try:
            tqdm = import_module(progress)

            # If tqdm loaded, apply the progress bar to the iterable
            pbar = tqdm.tqdm(iterable, desc=pbar_desc, total=n_to_run, dynamic_ncols=True)

        except ImportError:

            # If tqdm isn't available, proceed without a progress bar
            print(("A progress bar requiring the 'tqdm' module was requested, "
                   "but 'tqdm' is not installed. \nProceeding without using a progress bar."))
            pbar = iterable

    # If progress is None, return the original iterable without a progress bar applied
    else:
        pbar = iterable

    return pbar


def check_guess_and_bounds(corrs, guess, bounds):
    """Check the dimensions of guess and bounds for 2d fitting."""

    # Ensure guess and bounds are zipable
    if isinstance(guess, list):
        guess = np.array(guess)
    if isinstance(bounds, list):
        bounds = np.array(bounds)

    if guess is None:
        guess = np.array([None] * len(corrs))
    elif guess.ndim == 1 and corrs.ndim == 2:
        guess = np.tile(guess, (len(corrs), 1))

    if bounds is None:
        bounds = np.array([None] * len(corrs))
    elif bounds.ndim == 1 and corrs.ndim == 2:
        bounds = np.tile(bounds, (len(corrs), 1))

    return guess, bounds

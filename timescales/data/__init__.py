"""Example spectra and autocorrelations from public neural recordings.

Eight recordings spanning fMRI, MEG, EEG, ECoG, LFP and spike counts, from
four species, with sampling rates three orders of magnitude apart.  Each is
stored as one median power spectrum and one median autocorrelation, taken
across the traces of that recording, which is enough to demonstrate and test
the estimators without shipping the recordings themselves.

The band and lag range of each were chosen for the modality: fMRI is analysed
over 0.01-0.24 Hz and electrophysiology over 1-100 Hz, so the set is a fair
test of whether one configuration of an estimator generalizes.

Examples
--------
>>> from timescales.data import load_spectrum, DATASETS
>>> freqs, powers, fs = load_spectrum(DATASETS[0])
"""

from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parent / 'real_spectra.npz'

#: Keys of the available recordings.
DATASETS = ('human_fmri_rest', 'human_lemon_alpha', 'human_meg_beta',
            'human_cole_m1_beta', 'macaque_visual_grating_ecog',
            'mouse_neuropixels_natural_movie', 'mouse_neuropixels_spikes',
            'rat_hc3_ca1_ca3')

#: Short, human-readable labels.
LABELS = {
    'human_fmri_rest': 'Human fMRI (precuneus, rest)',
    'human_lemon_alpha': 'Human EEG (posterior, eyes closed)',
    'human_meg_beta': 'Human MEG (sensorimotor)',
    'human_cole_m1_beta': 'Human ECoG (motor cortex)',
    'macaque_visual_grating_ecog': 'Macaque ECoG (visual cortex)',
    'mouse_neuropixels_natural_movie': 'Mouse LFP (visual cortex)',
    'mouse_neuropixels_spikes': 'Mouse spike counts (visual cortex)',
    'rat_hc3_ca1_ca3': 'Rat LFP (hippocampus)',
}


def _check(key):
    if key not in DATASETS:
        raise KeyError(f'Unknown recording {key!r}. Options: {list(DATASETS)}')


def load_spectrum(key):
    """Power spectrum of one recording.

    Returns
    -------
    freqs, powers : 1d arrays
    fs : float
        Sampling rate of the recording, Hz.
    """
    _check(key)
    with np.load(_PATH) as handle:
        return (handle[f'{key}/freqs'], handle[f'{key}/powers'],
                float(handle[f'{key}/fs']))


def load_acf(key):
    """Autocorrelation of one recording.

    Returns
    -------
    lags : 1d array
        Lags in samples.
    corrs : 1d array
    fs : float
        Sampling rate of the recording, Hz.
    """
    _check(key)
    with np.load(_PATH) as handle:
        return (handle[f'{key}/lags'], handle[f'{key}/corrs'],
                float(handle[f'{key}/fs']))

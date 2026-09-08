"""Loading and common preprocessing for the real neural recordings.

``scripts/extract_datasets.py`` writes one ``.npz`` per dataset holding
continuous traces (channels, blocks or trials) drawn from the dataset registry
in ``oscillatory_eigenvectors/notebooks/06_multimodal_eigenspectra_2s.ipynb``,
extended here with MEG and fMRI.

One recipe is applied to all electrophysiological datasets so that species and
modalities are compared on equal terms: resample to a common rate (250 Hz), fit
spectra over 1-100 Hz, and exclude line noise and its harmonics from the
spectral fits.  fMRI is sampled three orders of magnitude more slowly, so it
carries its own analysis band; those settings travel with the dataset rather
than being hard-coded here.
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from tsmodes.prep import N_LAGS, prepare

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

DATASET_KEYS = [
    'human_fmri_rest',
    'human_lemon_alpha',
    'human_meg_beta',
    'human_cole_m1_beta',
    'macaque_visual_grating_ecog',
    'mouse_neuropixels_natural_movie',
    'mouse_neuropixels_spikes',
    'rat_hc3_ca1_ca3',
]

# Short names used in figures and tables.
SHORT_NAMES = {
    'human_fmri_rest': 'fMRI precuneus',
    'human_lemon_alpha': 'EEG alpha',
    'human_meg_beta': 'MEG beta',
    'human_cole_m1_beta': 'ECoG beta',
    'human_ssvep_12hz': 'EEG SSVEP',
    'macaque_visual_grating_ecog': 'Macaque ECoG',
    'mouse_neuropixels_natural_movie': 'Mouse LFP',
    'mouse_neuropixels_spikes': 'Mouse spikes',
    'rat_medial_septum_theta': 'Rat septal',
    'rat_hc3_ca1_ca3': 'Rat CA1/CA3',
}

# Row labels for the real-data figure: "<Species> <modality>", the coarse
# modality only, so that rows are comparable across recordings.
SHORT_MODALITY = {
    'human_fmri_rest': 'fMRI',
    'human_lemon_alpha': 'EEG',
    'human_meg_beta': 'MEG',
    'human_cole_m1_beta': 'ECoG',
    'human_ssvep_12hz': 'EEG',
    'macaque_visual_grating_ecog': 'ECoG',
    'mouse_neuropixels_natural_movie': 'LFP',
    'mouse_neuropixels_spikes': 'spikes',
    'rat_medial_septum_theta': 'LFP',
    'rat_hc3_ca1_ca3': 'LFP',
}


def panel_name(key, meta):
    """``"Human fMRI"``-style label from the species and coarse modality."""
    species = str(meta.get('species', '')).capitalize()
    modality = SHORT_MODALITY.get(key) or str(meta.get('modality', ''))
    return f'{species} {modality}'.strip()


DEFAULT_ANALYSIS = {
    'fs_target': 250.0,
    'f_range': (1.0, 100.0),
    'welch_win_s': 4.0,
    'welch_overlap': 0.5,
    'method': 'welch',
    'nw': 4.0,
    'nlags': N_LAGS,
}
# Per-recording departures from the defaults.  The MEG carries a sharp peak at
# 100 Hz, an artefact rather than brain, so its fit band stops short of it.
ANALYSIS_OVERRIDES = {
    'human_meg_beta': {'f_range': (1.0, 95.0)},
}
LINE_HALF_WIDTH = 2.0


def line_bands(line_hz, f_max, half_width=LINE_HALF_WIDTH):
    """Line-noise fundamental and harmonics as exclusion bands."""
    if not line_hz:
        return ()
    bands, freq = [], float(line_hz)
    while freq < f_max:
        bands.append((freq - half_width, freq + half_width))
        freq += line_hz
    return tuple(bands)


def _ratio(target, source, max_denominator=2000):
    from fractions import Fraction
    frac = Fraction(target / source).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


def load(key, data_dir=DATA_DIR, max_traces=None, max_seconds=None,
         overrides=None):
    """Load one dataset, resampled to its analysis rate."""
    path = Path(data_dir) / f'{key}.npz'
    with np.load(path, allow_pickle=False) as handle:
        X = np.asarray(handle['X'], dtype=float)
        fs = float(handle['fs'])
        meta = json.loads(str(handle['meta']))

    analysis = dict(DEFAULT_ANALYSIS)
    analysis.update(meta.get('analysis', {}))
    analysis.update(ANALYSIS_OVERRIDES.get(key, {}))
    analysis.update(overrides or {})
    analysis['f_range'] = tuple(analysis['f_range'])

    if max_traces is not None:
        X = X[:max_traces]
    fs_target = analysis['fs_target']
    if fs_target is not None and abs(fs - fs_target) > 1e-6:
        up, down = _ratio(fs_target, fs)
        X = resample_poly(X, up, down, axis=-1)
        fs = fs * up / down
    if max_seconds is not None:
        X = X[:, : int(max_seconds * fs)]

    X = X - X.mean(axis=1, keepdims=True)
    X = X / X.std(axis=1, keepdims=True)
    meta['fs_analysis_hz'] = fs
    meta['duration_analysis_s'] = X.shape[1] / fs
    return {'key': key, 'X': X, 'fs': fs, 'meta': meta, 'analysis': analysis,
            'label': meta.get('label', key),
            'short': SHORT_NAMES.get(key, key),
            'panel_name': panel_name(key, meta),
            'band_hz': tuple(meta.get('band_hz', (8, 13))),
            'line_hz': meta.get('line_noise_hz')}


def prepare_trace(trace, data, name='', **overrides):
    """The common ``Signal`` preparation used for every real recording.

    ``data`` is a dataset dict from :func:`load`.  The Welch segment is
    shortened for short recordings and the lower edge of the fit band is raised
    to the resulting frequency resolution, so a short window is never fit below
    the frequencies it can resolve.
    """
    settings = dict(data['analysis'])
    settings.update(overrides)
    fs = data['fs']
    duration = len(trace) / fs

    welch_win_s = settings['welch_win_s']
    # At most half the record, so that at least two segments are averaged.
    welch_win_s = min(welch_win_s, max(duration / 2, min(duration, 2.0)))
    nperseg = max(8, int(round(welch_win_s * fs)))
    f_lo, f_hi = settings['f_range']
    f_hi = min(f_hi, fs / 2)
    f_lo = max(f_lo, fs / nperseg)
    nlags = int(min(settings['nlags'], len(trace) // 3))
    return prepare(trace, fs, welch_win_s=welch_win_s, nlags=nlags,
                   welch_overlap=settings['welch_overlap'],
                   method=settings['method'], nw=settings['nw'],
                   f_range=(f_lo, f_hi), name=name,
                   exclude_bands=line_bands(data['line_hz'], f_hi))


def available(data_dir=DATA_DIR):
    return [key for key in DATASET_KEYS if (Path(data_dir) / f'{key}.npz').exists()]

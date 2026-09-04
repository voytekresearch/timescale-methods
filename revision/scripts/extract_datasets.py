"""Extract continuous single-channel traces from the multimodal dataset registry.

Run with the ``oscillatory_eigenvectors`` environment, which has the readers
(mne, h5py, allensdk) and the cached downloads:

    /home/rph/oscillatory_eigenvectors/.venv/bin/python extract_datasets.py

Data access mirrors ``oscillatory_eigenvectors/notebooks/06_multimodal_eigenspectra_2s.ipynb``
(``DATASET_REGISTRY``), but returns *continuous* traces rather than 2 s windows,
because the timescale and truncation analyses need long uninterrupted segments.

Each dataset is written to ``revision/data/<key>.npz`` with
``X`` (n_traces, n_samples), ``fs`` and a JSON ``meta`` string.
"""

import io
import json
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

EIG = Path('/home/rph/oscillatory_eigenvectors')
DATA_ROOT = EIG / 'notebooks' / 'data' / 'multimodal_cache'
ALLEN_CACHE = EIG / 'notebooks' / 'data' / 'ecephys_cache_dir'
HC3_EEG = EIG / 'data' / 'hc-3' / 'pin01-11-04' / '11-05_0-06-51' / '11-05_0-06-51.eeg'
OUT = Path(__file__).resolve().parents[1] / 'data'

MAX_SECONDS = 480.0
MAX_TRACES = 8

ABIDE_ROOT = ('https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
              '/Outputs/cpac/nofilt_noglobal/rois_ho')
# UM_1 has the longest resting runs in ABIDE (296 volumes at TR = 2 s).
ABIDE_SUBJECTS = ['UM_1_0050327', 'UM_1_0050329', 'UM_1_0050330',
                  'UM_1_0050331', 'UM_1_0050332', 'UM_1_0050333']


def download_to_cache(url, out_path):
    """Fetch a file once and reuse it."""
    from urllib.request import urlretrieve
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        print(f'  downloading {url}', flush=True)
        urlretrieve(url, out_path)
    return out_path


def save(key, X, fs, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float32)
    X = X[np.isfinite(X).all(axis=1)]
    X = X - X.mean(axis=1, keepdims=True)
    meta = dict(meta)
    meta.update({'key': key, 'fs_hz': float(fs), 'n_traces': int(X.shape[0]),
                 'n_samples': int(X.shape[1]),
                 'duration_s': float(X.shape[1] / fs)})
    np.savez_compressed(OUT / f'{key}.npz', X=X, fs=np.float64(fs),
                        meta=json.dumps(meta))
    print(f'  saved {key}: {X.shape} @ {fs:g} Hz '
          f'({X.shape[1] / fs:.0f} s per trace)', flush=True)


def crop(data, fs, max_seconds=MAX_SECONDS, max_traces=MAX_TRACES):
    n = min(data.shape[1], int(max_seconds * fs))
    return np.asarray(data[:max_traces, :n])


# ---------------------------------------------------------------------------

def extract_lemon(subject='sub-032302',
                  channels=('O1', 'Oz', 'O2', 'POz'), marker='S210',
                  max_blocks=3):
    import mne
    archive = DATA_ROOT / 'lemon' / f'{subject}.tar.gz'
    extract_dir = archive.with_suffix('').with_suffix('')
    if not extract_dir.exists():
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(extract_dir)
    vhdrs = sorted(extract_dir.rglob(f'{subject}.vhdr'))
    vhdr = vhdrs[0]
    fixed = vhdr.with_name(f'{subject}_fixed.vhdr')
    if not fixed.exists():
        vmrk = vhdr.with_name(f'{subject}.vmrk')
        fixed_vmrk = vhdr.with_name(f'{subject}_fixed.vmrk')
        text = vmrk.read_text(encoding='utf-8', errors='replace')
        fixed_vmrk.write_text(re.sub(r'^(DataFile=).*$', rf'\1{subject}.eeg',
                                     text, flags=re.MULTILINE), encoding='utf-8')
        text = vhdr.read_text(encoding='utf-8', errors='replace')
        text = re.sub(r'^(DataFile=).*$', rf'\1{subject}.eeg', text, flags=re.MULTILINE)
        text = re.sub(r'^(MarkerFile=).*$', rf'\1{fixed_vmrk.name}', text,
                      flags=re.MULTILINE)
        fixed.write_text(text, encoding='utf-8')

    raw = mne.io.read_raw_brainvision(fixed, preload=True, verbose=False)
    raw.pick(list(channels))
    fs = raw.info['sfreq']

    onsets = np.sort([o for o, d in zip(raw.annotations.onset,
                                        raw.annotations.description)
                      if str(d).endswith(marker)])
    segments, start, prev = [], onsets[0], onsets[0]
    stop_time = raw.n_times / fs
    for onset in onsets[1:]:
        if onset - prev <= 3.0:
            prev = onset
        else:
            segments.append((start, min(prev + 2.0, stop_time)))
            start = prev = onset
    segments.append((start, min(prev + 2.0, stop_time)))
    segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)[:max_blocks]

    length = int(min(stop - beg for beg, stop in segments) * fs)
    traces = []
    for beg, _ in segments:
        i0 = int(beg * fs)
        traces.append(raw.get_data(start=i0, stop=i0 + length))
    X = np.concatenate(traces, axis=0)
    save('human_lemon_alpha', X, fs, {
        'species': 'human', 'modality': 'scalp EEG', 'region': 'posterior',
        'state': 'eyes-closed rest', 'band_hz': [8, 13], 'line_noise_hz': 50,
        'subject': subject, 'channels': list(channels),
        'source': 'MPI-LEMON raw resting EEG (10.15387/fcp_indi.mpi_lemon)',
        'trace_definition': 'eyes-closed block x posterior channel',
        'label': 'Human EEG (LEMON): posterior alpha'})


def extract_cole(condition='B'):
    from scipy.io import loadmat
    mat = loadmat(DATA_ROOT / 'Cole_2017' / 'data.mat', squeeze_me=True,
                  struct_as_record=False)
    traces = [np.asarray(t, dtype=np.float32).squeeze()
              for t in np.ravel(mat[condition])]
    X = np.vstack(traces)
    save('human_cole_m1_beta', X[:MAX_TRACES], 1000.0, {
        'species': 'human', 'modality': 'M1 ECoG', 'region': 'M1',
        'state': 'pre-DBS Parkinson rest', 'band_hz': [13, 30],
        'line_noise_hz': 60, 'subject': f'{len(traces)} participants',
        'source': 'Cole 2017 (10.1523/JNEUROSCI.2208-16.2017)',
        'trace_definition': 'one M1 ECoG trace per participant',
        'label': 'Human M1 ECoG: Parkinson beta'})


def extract_ssvep(subject='01', channel='Oz', trial_type='stim/12hz'):
    import mne
    import pandas as pd
    vhdrs = sorted((DATA_ROOT / 'mne').rglob(
        f'sub-{subject}_ses-01_task-ssvep_eeg.vhdr'))
    vhdr = vhdrs[0]
    events = pd.read_csv(vhdr.with_name(vhdr.name.replace('_eeg.vhdr',
                                                          '_events.tsv')),
                         sep='\t')
    events = events[events['trial_type'] == trial_type]
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick([channel])
    fs = raw.info['sfreq']
    length = int(28.0 * fs)
    offset = int(2.0 * fs)
    traces = []
    for event in events.itertuples(index=False):
        i0 = int(event.sample) + offset
        if i0 + length <= raw.n_times:
            traces.append(raw.get_data(start=i0, stop=i0 + length)[0])
    save('human_ssvep_12hz', np.vstack(traces)[:MAX_TRACES], fs, {
        'species': 'human', 'modality': 'scalp EEG', 'region': 'occipital',
        'state': '12 Hz checkerboard SSVEP', 'band_hz': [11, 13],
        'line_noise_hz': 50, 'subject': f'sub-{subject}',
        'source': 'MNE SSVEP example data',
        'trace_definition': 'one 28 s stimulation trial per trace',
        'label': 'Human EEG: 12 Hz SSVEP'})


def extract_macaque(channels=range(16, 24)):
    from scipy.io import loadmat
    zp = DATA_ROOT / 'neurotycho' / '20100805S1_Visual-Grating_K2_mat_ECoG128-Eye.zip'
    with zipfile.ZipFile(zp) as zf:
        name = [n for n in zf.namelist()
                if n.endswith('ECoG-3.mat') and not n.startswith('__MACOSX')][0]
        mat = loadmat(io.BytesIO(zf.read(name)))
    data = np.asarray(mat['X'], dtype=np.float32)[list(channels)]
    save('macaque_visual_grating_ecog', crop(data, 1000.0), 1000.0, {
        'species': 'macaque', 'modality': 'ECoG', 'region': 'visual cortex',
        'state': 'visual grating task', 'band_hz': [12, 20],
        'line_noise_hz': 50, 'subject': 'K2',
        'source': 'NeuroTycho visual grating task',
        'trace_definition': 'one ECoG channel per trace',
        'label': 'Macaque ECoG: visual grating'})


def extract_rat_septum(channels=range(0, 8)):
    import h5py
    nwb = (DATA_ROOT / 'dandi' / '001607'
           / 'sub-Banner_ses-Banner-20220123_ecephys.nwb')
    with h5py.File(nwb, 'r') as f:
        dset = f['scratch/filtered data/data']
        ts = np.asarray(f['scratch/filtered data/timestamps'][:5000])
        fs = float(1 / np.nanmedian(np.diff(ts)))
        n = min(dset.shape[0], int(MAX_SECONDS * fs))
        data = np.asarray(dset[:n, list(channels)], dtype=np.float32).T
        subject = f['general/subject/subject_id'][()].decode()
    save('rat_medial_septum_theta', data, fs, {
        'species': 'rat', 'modality': 'continuous ecephys',
        'region': 'medial septum / hippocampal', 'state': 'freely behaving',
        'band_hz': [6, 12], 'line_noise_hz': 60, 'subject': subject,
        'source': 'DANDI 001607 optogenetic theta disruption',
        'trace_definition': 'one recording channel per trace',
        'label': 'Rat LFP: medial septal theta'})


def extract_hc3(channels=range(0, 8)):
    root = ET.parse(HC3_EEG.with_suffix('.xml')).getroot()
    n_channels = int(root.findtext('./acquisitionSystem/nChannels'))
    fs = float(root.findtext('./fieldPotentials/lfpSamplingRate'))
    n_samples = HC3_EEG.stat().st_size // np.dtype('<i2').itemsize // n_channels
    raw = np.memmap(HC3_EEG, dtype='<i2', mode='r',
                    shape=(n_samples, n_channels))
    n = min(n_samples, int(MAX_SECONDS * fs))
    data = np.asarray(raw[:n, list(channels)], dtype=np.float32).T
    save('rat_hc3_ca1_ca3', data, fs, {
        'species': 'rat', 'modality': 'hippocampal LFP',
        'region': 'CA1/CA3', 'state': 'continuous recording',
        'band_hz': [6, 12], 'line_noise_hz': 60, 'subject': 'pin01',
        'source': 'CRCNS HC-3 (10.6080/K0Z60KZ9)',
        'trace_definition': 'one hippocampal channel per trace',
        'label': 'Rat hippocampus: CA1/CA3 LFP'})


def extract_neuropixels(probe_description='probeA', region='VISam',
                        max_channels=8):
    import pandas as pd
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import \
        EcephysProjectCache
    cache = EcephysProjectCache.from_warehouse(
        manifest=str(ALLEN_CACHE / 'manifest.json'))
    sessions = cache.get_session_table()
    session_id = sessions.index.values[0]
    session = cache.get_session_data(session_id, isi_violations_maximum=np.inf,
                                     amplitude_cutoff_maximum=np.inf,
                                     presence_ratio_minimum=-np.inf)
    probe_id = session.probes[
        session.probes.description == probe_description].index.values[0]
    lfp = session.get_lfp(probe_id)
    regions = np.array([session.channels.loc[c, 'ecephys_structure_acronym']
                        for c in lfp.channel.to_numpy()])
    inds = np.where(regions == region)[0][:max_channels]
    if len(inds) == 0:
        inds = np.arange(max_channels)

    times = lfp.time.to_numpy()
    fs = float(1 / np.median(np.diff(times)))
    presentations = session.stimulus_presentations[
        session.stimulus_presentations.stimulus_name == 'natural_movie_one']
    t0 = float(presentations.start_time.values[0])
    i0 = int(np.searchsorted(times, t0))
    n = min(len(times) - i0, int(MAX_SECONDS * fs))
    data = lfp.isel(channel=inds).to_numpy()[i0:i0 + n].T.astype(np.float32)
    save('mouse_neuropixels_natural_movie', data, fs, {
        'species': 'mouse', 'modality': 'Neuropixels LFP', 'region': region,
        'state': 'natural movie', 'band_hz': [2, 8], 'line_noise_hz': 60,
        'subject': str(session_id), 'probe': probe_description,
        'source': 'Allen Visual Coding Neuropixels (10.1038/s41593-019-0550-9)',
        'trace_definition': 'one LFP channel per trace',
        'label': 'Mouse Neuropixels LFP: natural movie'})


def extract_neuropixels_spikes(region='VISp', bin_s=0.05, max_units=MAX_TRACES,
                               min_rate=2.0):
    """Spiking: binned spike counts from the same Allen Neuropixels session.

    Every other recording here is a continuous field potential or BOLD, yet
    the neural-timescale literature is built largely on spike-count
    autocorrelations, so this is the one modality whose absence would be
    conspicuous.  It also adds a region: the LFP traces are from VISam and
    these units are from VISp.

    Read straight from the cached NWB rather than through the AllenSDK, which
    is a heavy dependency for one table of spike times.
    """
    import h5py
    path = ALLEN_CACHE / 'session_715093703' / 'session_715093703.nwb'
    if not path.exists():
        raise RuntimeError(f'Allen session NWB not cached at {path}')
    with h5py.File(path, 'r') as handle:
        units = handle['units']
        electrodes = handle['general/extracellular_ephys/electrodes']
        dec = lambda a: np.array([v.decode() if isinstance(v, bytes) else v
                                  for v in a])
        region_of = dict(zip(electrodes['id'][:], dec(electrodes['location'][:])))
        regions = np.array([region_of.get(c, '?')
                            for c in units['peak_channel_id'][:]])
        quality, rate = dec(units['quality'][:]), units['firing_rate'][:]
        end = units['spike_times_index'][:]
        begin = np.concatenate(([0], end[:-1]))

        # The longest uninterrupted stretch of the same stimulus the LFP
        # traces come from, so the two mouse rows are comparable.
        pres = handle['intervals/natural_movie_one_presentations']
        starts, stops = pres['start_time'][:], pres['stop_time'][:]
        breaks = np.where(starts[1:] - stops[:-1] > 1.0)[0]
        blocks = np.split(np.arange(len(starts)), breaks + 1)
        block = max(blocks, key=lambda b: stops[b[-1]] - starts[b[0]])
        t0, t1 = float(starts[block[0]]), float(stops[block[-1]])

        keep = np.where((quality == 'good') & (regions == region)
                        & (rate > min_rate))[0]
        keep = keep[np.argsort(-rate[keep])][:max_units]
        edges = np.arange(t0, t1, bin_s)
        traces, kept = [], []
        for i in keep:
            spikes = units['spike_times'][begin[i]:end[i]]
            spikes = spikes[(spikes >= t0) & (spikes < t1)]
            traces.append(np.histogram(spikes, bins=edges)[0].astype(float))
            kept.append(f'unit{int(units["id"][i])}')

    save('mouse_neuropixels_spikes', np.vstack(traces), 1.0 / bin_s, {
        'species': 'mouse', 'modality': 'Neuropixels spikes', 'region': region,
        'state': 'natural movie', 'band_hz': [2, 8], 'line_noise_hz': None,
        'subject': '715093703', 'traces': kept,
        'source': 'Allen Visual Coding Neuropixels (10.1038/s41593-019-0550-9)',
        'trace_definition': f'spike counts of one unit in {bin_s*1e3:g} ms bins',
        # Above ~1.5 Hz the binned-count spectrum is within a factor of
        # 1.5 of its flat Poisson shot-noise floor, which sigma^2/|A|^2
        # cannot represent; fit only where the process exceeds it.
        'analysis': {'f_range': [0.05, 2.0], 'welch_win_s': 60.0,
                     'welch_overlap': 0.5, 'nlags': 100, 'fs_target': None,
                     'method': 'welch', 'nw': 4.0},
        'label': 'Mouse Neuropixels spikes: natural movie'})


def extract_meg_cmc(channels=None, max_seconds=MAX_SECONDS):
    """MEG: MNE/FieldTrip cortico-muscular coherence data (CTF 151-channel)."""
    import mne
    root = DATA_ROOT / 'mne' / 'MNE-fieldtrip_cmc-data'
    ds = root / 'SubjectCMC.ds'
    if not ds.exists():
        from mne.datasets import fieldtrip_cmc
        ds = Path(fieldtrip_cmc.data_path(path=str(DATA_ROOT / 'mne'),
                                          update_path=False)) / 'SubjectCMC.ds'
    raw = mne.io.read_raw_ctf(str(ds), preload=True, verbose=False)
    raw.pick(picks='mag')
    fs = raw.info['sfreq']
    n = min(raw.n_times, int(max_seconds * fs))
    data = raw.get_data(stop=n)

    # Rank channels by relative beta power; sustained sensorimotor beta is the
    # feature of interest in this recording.
    freqs = np.fft.rfftfreq(min(n, int(8 * fs)), 1 / fs)
    seg = data[:, : len(freqs) * 2 - 2]
    spec = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    band = (freqs >= 15) & (freqs <= 30)
    broad = (freqs >= 5) & (freqs <= 45)
    score = spec[:, band].sum(axis=1) / spec[:, broad].sum(axis=1)
    order = np.argsort(-score)[:MAX_TRACES] if channels is None else np.asarray(channels)

    save('human_meg_beta', data[order], fs, {
        'species': 'human', 'modality': 'MEG', 'region': 'sensorimotor',
        'state': 'isometric wrist contraction', 'band_hz': [15, 30],
        'line_noise_hz': 50, 'subject': 'SubjectCMC',
        'source': 'MNE / FieldTrip cortico-muscular coherence dataset',
        'trace_definition': 'one MEG magnetometer per trace',
        'label': 'Human MEG: sensorimotor beta'})


# Harvard-Oxford cortical region index of the precuneus, verified against the
# atlas XML.  ABIDE's rois_ho columns are named "#<region><hemisphere>" with
# region 1-48 in atlas order and hemisphere 01 = left, 02 = right.
PRECUNEUS = 31


def extract_fmri_abide(subjects=ABIDE_SUBJECTS, tr=2.0, region=PRECUNEUS):
    """fMRI: ABIDE resting-state precuneus time series (no temporal filter).

    The ``nofilt_noglobal`` CPAC strategy is used deliberately: the usual
    0.01-0.1 Hz band-pass would remove most of the spectrum a timescale is
    estimated from.

    One named region, both hemispheres, every participant, rather than the
    highest-variance ROI per participant.  Two reasons.  A single 296-sample
    BOLD run gives a spectrum too noisy to fit, and averaging over regions is
    the obvious remedy but the wrong one here, because regions differ in
    exactly the property being measured -- precuneus, medial prefrontal and
    occipital spectra have visibly different shapes in this data, so a
    111-region average describes no region at all.  Restricting to one
    default-mode hub keeps the average interpretable, and makes a trace here
    the same kind of thing as a channel in the other recordings.
    """
    traces, kept = [], []
    for subject in subjects:
        path = download_to_cache(f'{ABIDE_ROOT}/{subject}_rois_ho.1D',
                                 DATA_ROOT / 'abide' / f'{subject}_rois_ho.1D')
        with open(path) as handle:
            ids = [int(c.lstrip('#')) for c in handle.readline().split()]
        table = np.loadtxt(path, skiprows=1)
        if table.ndim != 2 or table.shape[0] < 100:
            continue
        for hemi, name in ((1, 'L'), (2, 'R')):
            try:
                col = ids.index(region * 100 + hemi)
            except ValueError:
                continue
            series = table[:, col]
            if series.std() < 1e-9:
                continue
            traces.append(series)
            kept.append(f'{subject}:precuneus-{name}')
    if not traces:
        raise RuntimeError('No ABIDE precuneus time series could be loaded.')
    length = min(len(t) for t in traces)
    X = np.vstack([t[:length] for t in traces])
    save('human_fmri_rest', X, 1.0 / tr, {
        'species': 'human', 'modality': 'fMRI BOLD', 'region': 'precuneus',
        'state': 'resting state', 'band_hz': [0.01, 0.1], 'line_noise_hz': None,
        'subject': 'ABIDE UM_1', 'traces': kept,
        'source': 'ABIDE preprocessed (CPAC, nofilt_noglobal, Harvard-Oxford ROIs)',
        'trace_definition': 'precuneus, left and right, one trace each per participant',
        'analysis': {'f_range': [0.01, 0.24], 'welch_win_s': 128.0,
                     'nlags': 60, 'fs_target': None},
        'label': 'Human fMRI: resting-state BOLD'})


EXTRACTORS = {
    'human_lemon_alpha': extract_lemon,
    'human_cole_m1_beta': extract_cole,
    'human_ssvep_12hz': extract_ssvep,
    'mouse_neuropixels_natural_movie': extract_neuropixels,
    'rat_medial_septum_theta': extract_rat_septum,
    'rat_hc3_ca1_ca3': extract_hc3,
    'macaque_visual_grating_ecog': extract_macaque,
    'human_meg_beta': extract_meg_cmc,
    'human_fmri_rest': extract_fmri_abide,
    'mouse_neuropixels_spikes': extract_neuropixels_spikes,
}


if __name__ == '__main__':
    keys = sys.argv[1:] or list(EXTRACTORS)
    for key in keys:
        print(f'{key} ...', flush=True)
        try:
            EXTRACTORS[key]()
        except Exception as exc:
            print(f'  FAILED: {type(exc).__name__}: {exc}', flush=True)

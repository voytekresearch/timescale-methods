"""Stage 4: how much data does a neural-timescale estimate need?

Each long recording is cut into non-overlapping windows of increasing duration.
Every window is analysed with the same pipeline as the full recording, so the
convergence of the estimate, the between-window variance, and the stability of
the AR order and of the dominant modes can be read off directly.

Outputs
-------
results/empirical_truncation.csv.gz, results/empirical_truncation_summary.csv
(no figure; results feed fig_windows and figS_finite)
"""
import sys
from pathlib import Path

# `tsmodes` (this paper's own code) lives beside `scripts/`, so the
# analysis directory has to be importable however the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import sys
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool, cpu_count

from tsmodes import plotting as P
from tsmodes import datasets as D
from tsmodes.benchmark import flatten_fit
from tsmodes.estimators import ESTIMATOR_LABELS, estimate_all
from timescales.finite import relative_sd_asymptotic

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'
METHODS = ['ar1_direct', 'ar1_acf', 'ar1_psd', 'arp_acf', 'arp_psd']
DURATIONS = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128], dtype=float)
#: A window with fewer samples than this cannot support a spectral estimate
#: worth fitting, whatever the estimator.  For BOLD at TR = 2 s the shortest
#: nominal window, 16 s, is eight samples.
MIN_SAMPLES = 24
# fMRI is sampled at 0.5 Hz, so its window ladder lives on a different scale.
DURATIONS_SLOW = np.array([16, 32, 64, 128, 256, 592], dtype=float)
N_JOBS = int(os.environ.get('TSMODES_N_JOBS', '12'))


def durations_for(data):
    return DURATIONS_SLOW if data['fs'] < 5 else DURATIONS
MAX_WINDOWS = 30
STABILITY_CV = 0.10
STABILITY_BIAS = 0.10


def windows_for(data, duration, max_windows=MAX_WINDOWS, rng=None):
    """Non-overlapping windows of ``duration`` seconds, pooled across traces."""
    fs = data['fs']
    n = int(round(duration * fs))
    out = []
    for trace in data['X']:
        for start in range(0, len(trace) - n + 1, n):
            out.append(trace[start:start + n])
    if not out:
        return []
    rng = np.random.default_rng(0) if rng is None else rng
    if len(out) > max_windows:
        idx = rng.choice(len(out), max_windows, replace=False)
        out = [out[i] for i in sorted(idx)]
    return out


def _analyse_window(args, key=None, data=None, duration=None):
    w, window = args
    signal = D.prepare_trace(window, data, name=f'{key}[{w}]')
    out = []
    for fit in estimate_all(signal, methods=METHODS, robust=True).values():
        row = flatten_fit(fit, dominant_tol=0.1)
        row.update({'dataset': key, 'label': data['label'],
                    'duration': float(duration), 'window': w,
                    'band_lo': data['band_hz'][0],
                    'band_hi': data['band_hz'][1]})
        band = data['band_hz']
        row['has_band_mode'] = bool(fit.modes is not None and any(
            m.kind == 'oscillatory' and band[0] <= m.freq <= band[1]
            for m in fit.modes.modes))
        out.append(row)
    return out


def analyse(key, n_jobs=N_JOBS):
    data = D.load(key)
    trace_seconds = data['X'].shape[1] / data['fs']
    durations = [d for d in durations_for(data)
                 if d <= trace_seconds and d * data['fs'] >= MIN_SAMPLES]
    rows = []
    for duration in durations:
        wins = list(enumerate(windows_for(data, duration)))
        print(f'      T={duration:g}s  {len(wins)} windows', flush=True)
        worker = partial(_analyse_window, key=key, data=data, duration=duration)
        if n_jobs == 1:
            results = [worker(w) for w in wins]
        else:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(worker, wins)
        rows.extend(r for res in results for r in res)
    return pd.DataFrame(rows)


def summarize(df):
    """Per dataset and estimator: convergence, between-window spread, order."""
    out = []
    for (key, method), sub in df.groupby(['dataset', 'method'], sort=False):
        durations = np.sort(sub['duration'].unique())
        ref = np.nanmedian(sub.loc[sub['duration'] == durations[-1], 'tau'])
        recs = []
        for duration in durations:
            vals = sub.loc[sub['duration'] == duration, 'tau'].to_numpy(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) < 3:
                continue
            median = float(np.median(vals))
            cv = float(np.std(np.log(vals)))
            # AR(1) sampling bound at this duration, using the estimate itself
            bound = float(relative_sd_asymptotic(duration / median))
            recs.append({
                'dataset': key, 'label': sub['label'].iloc[0], 'method': method,
                'duration': float(duration), 'n_windows': len(vals),
                'median_tau': median, 'log_cv': cv,
                'theory_sd': bound, 'excess_sd': cv / bound,
                'rel_to_longest': median / ref,
                'median_order': float(np.nanmedian(
                    sub.loc[sub['duration'] == duration, 'order'])),
                'median_dominant_modes': float(np.nanmedian(
                    sub.loc[sub['duration'] == duration, 'n_dominant_modes'])),
                'frac_band_mode': float(np.mean(
                    sub.loc[sub['duration'] == duration, 'has_band_mode'])),
                'reference_tau': float(ref)})
        # "Stable from T onwards": the criterion must hold at every longer
        # duration too, so an accidental crossing at a short window does not
        # count as convergence.
        cvs = np.array([r['log_cv'] for r in recs])
        rels = np.array([abs(r['rel_to_longest'] - 1) for r in recs])
        durs = np.array([r['duration'] for r in recs])
        cv_stable = bias_stable = np.nan
        for i in range(len(recs)):
            if np.isnan(cv_stable) and np.all(cvs[i:] <= STABILITY_CV):
                cv_stable = durs[i]
            if np.isnan(bias_stable) and np.all(rels[i:] <= STABILITY_BIAS):
                bias_stable = durs[i]
        for rec in recs:
            rec['duration_cv_stable'] = cv_stable
            rec['duration_bias_stable'] = bias_stable
        out.extend(recs)
    return pd.DataFrame(out)


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    cache = RES / 'empirical_truncation.csv.gz'
    if cache.exists() and '--rerun' not in sys.argv:
        print('Loading cached truncation results (pass --rerun to recompute)')
        df = pd.read_csv(cache)
    else:
        frames = []
        for key in D.available():
            print(f'  {key} ...', flush=True)
            frames.append(analyse(key))
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(cache, index=False, compression='gzip')

    summary = summarize(df)
    summary.to_csv(RES / 'empirical_truncation_summary.csv', index=False)

    table = (summary.groupby(['dataset', 'method'])
             .agg(reference_tau=('reference_tau', 'first'),
                  longest_T=('duration', 'max'),
                  shortest_over_longest=('rel_to_longest', 'first'),
                  duration_cv_stable=('duration_cv_stable', 'first'),
                  duration_bias_stable=('duration_bias_stable', 'first'),
                  log_cv_longest=('log_cv', 'last'),
                  excess_sd_longest=('excess_sd', 'last'))
             .reset_index())
    table['T_over_tau_bias'] = (table['duration_bias_stable']
                                / table['reference_tau'])
    table.to_csv(RES / 'empirical_stability_requirements.csv', index=False)
    with pd.option_context('display.width', 220):
        print(table.round(4).to_string(index=False))

    # No figure of its own: the windowing panels live in the main
    # windows figure and in figS_finite, which is where finite-sample
    # bias and truncation bias belong together.


if __name__ == '__main__':
    main()

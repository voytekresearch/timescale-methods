"""Stage 3: apply the framework across species, modalities and oscillations.

Every dataset receives the identical pipeline: resample to 250 Hz, one Welch PSD
and one ACF per continuous trace, then ACF-exponential, AR(1)-direct, AR(1)-PSD
and order-selected AR(p) fits, followed by the mode decomposition.

Outputs
-------
results/real_data_fits.csv, results/real_data_summary.csv,
results/real_data_modes.csv
"""
import sys
from pathlib import Path

# `tsmodes` (this paper's own code) lives beside `scripts/`, so the
# analysis directory has to be importable however the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd

from tsmodes import datasets as D
from tsmodes.benchmark import flatten_fit
from tsmodes.estimators import ESTIMATOR_LABELS, estimate_all

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
METHODS = ['ar1_direct', 'arp_direct', 'ar1_acf', 'arp_acf',
           'ar1_psd', 'arp_psd']
DOMINANT_TOL = 0.1   # keep modes until log-spectral R^2 >= 0.99


def in_band(freq, data, signal):
    """Is this mode a rhythm in the band the recording is expected to show?

    A mode whose frequency falls outside the fitted range is an
    extrapolation of the model past the data, so it cannot count as a
    detected rhythm however well it sits inside the nominal band.
    """
    lo, hi = data['band_hz']
    f_lo, f_hi = signal.f_range
    return bool(lo <= freq <= hi and f_lo <= freq <= f_hi)


def analyse_dataset(key):
    data = D.load(key)
    rows, mode_rows = [], []
    for i, trace in enumerate(data['X']):
        signal = D.prepare_trace(trace, data, name=f'{key}[{i}]')
        fits = estimate_all(signal, methods=METHODS, robust=True)
        for fit in fits.values():
            row = flatten_fit(fit, dominant_tol=DOMINANT_TOL)
            row.update({'dataset': key, 'label': data['label'], 'trace': i,
                        'fs': data['fs'], 'duration': signal.duration,
                        'short': data['short'],
                        'species': data['meta'].get('species'),
                        'modality': data['meta'].get('modality'),
                        'band_lo': data['band_hz'][0],
                        'band_hi': data['band_hz'][1]})
            rows.append(row)
            if fit.modes is not None and fit.method in ('arp_psd', 'arp_acf'):
                dom = fit.modes.dominant_subset(tol=DOMINANT_TOL)
                if fit.method == 'arp_psd':
                    curve = dom['r2_curve']
                    for level in (0.90, 0.95, 0.99):
                        hit = np.where(curve >= level)[0]
                        rows[-1][f'n_modes_r2_{int(level*100)}'] = (
                            int(hit[0]) if len(hit) else len(fit.modes.modes))
                    rows[-1]['stabilized'] = bool(
                        fit.params.get('stabilized', False))
                osc_rank = 0
                for rank, mode in enumerate(fit.modes.modes):
                    if mode.kind == 'oscillatory':
                        osc_rank += 1
                    mode_rows.append({
                        'dataset': key, 'label': data['label'], 'trace': i,
                        'method': fit.method, 'rank': rank,
                        'osc_rank': osc_rank if mode.kind == 'oscillatory' else 0,
                        'kind': mode.kind, 'tau_s': mode.tau,
                        'freq_hz': mode.freq, 'n_cycles': mode.n_cycles,
                        'radius': mode.radius,
                        'importance': mode.importance,
                        'rel_importance': mode.rel_importance,
                        'in_dominant_set': rank in dom['indices'],
                        'in_band': in_band(mode.freq, data, signal)})
    return data, pd.DataFrame(rows), pd.DataFrame(mode_rows)


def summarize_dataset(df, modes):
    out = []
    for key, sub in df.groupby('dataset', sort=False):
        rec = {'dataset': key, 'label': sub['label'].iloc[0],
               'species': sub['species'].iloc[0],
               'modality': sub['modality'].iloc[0],
               'n_traces': sub['trace'].nunique(),
               'duration_s': float(sub['duration'].iloc[0])}
        for method in METHODS:
            m = sub[sub['method'] == method]
            rec[f'tau_{method}'] = float(np.nanmedian(m['tau']))
            rec[f'r2_{method}'] = float(np.nanmedian(m['r2_logpsd']))
            if method.startswith('arp'):
                rec[f'order_{method}'] = float(np.nanmedian(m['order']))
        taus = [rec[f'tau_{m}'] for m in ('ar1_direct', 'ar1_acf', 'ar1_psd')]
        taus = [t for t in taus if np.isfinite(t) and t > 0]
        rec['ar1_tau_spread'] = float(max(taus) / min(taus)) if taus else np.nan
        rec['delta_r2_arp_vs_ar1'] = rec['r2_arp_psd'] - rec['r2_ar1_psd']
        rec['unexplained_ar1'] = 1 - rec['r2_ar1_psd']
        rec['unexplained_arp'] = 1 - rec['r2_arp_psd']
        sub_modes = modes[(modes['dataset'] == key)
                          & (modes['method'] == 'arp_psd')]
        arp = sub[sub['method'] == 'arp_psd']
        rec['n_dominant_modes'] = float(np.nanmedian(arp['n_dominant_modes']))
        for level in (90, 95, 99):
            col = f'n_modes_r2_{level}'
            if col in arp:
                rec[col] = float(np.nanmedian(arp[col]))
        rec['frac_stabilized'] = float(np.nanmean(arp.get('stabilized', np.nan)))
        dom = sub_modes[sub_modes['in_dominant_set']]
        acf_modes = modes[(modes['dataset'] == key)
                          & (modes['method'] == 'arp_acf')
                          & modes['in_band'] & (modes['osc_rank'] > 0)]
        rec['acf_band_mode_freq'] = (float(np.nanmedian(acf_modes['freq_hz']))
                                     if len(acf_modes) else np.nan)
        rec['acf_frac_traces_with_band_mode'] = float(
            acf_modes['trace'].nunique() / max(sub['trace'].nunique(), 1))
        rec['n_dominant_oscillatory'] = float(
            dom.groupby('trace')['kind'].apply(
                lambda s: (s == 'oscillatory').sum()).median())
        in_band = dom[dom['in_band']]
        rec['frac_traces_with_band_mode'] = float(
            in_band['trace'].nunique() / max(sub['trace'].nunique(), 1))
        rec['band_mode_freq'] = float(np.nanmedian(in_band['freq_hz'])) \
            if len(in_band) else np.nan
        rec['band_mode_tau'] = float(np.nanmedian(in_band['tau_s'])) \
            if len(in_band) else np.nan
        # Cycles the in-band mode rings before its envelope decays by 1/e.
        # Under about one, it is a broad shoulder rather than a rhythm.
        rec['band_mode_cycles'] = float(np.nanmedian(in_band['n_cycles'])) \
            if len(in_band) else np.nan
        rec['single_timescale_adequate'] = bool(
            (rec['n_dominant_modes'] <= 1) and (rec['unexplained_ar1'] < 0.02))
        out.append(rec)
    return pd.DataFrame(out)


def main():
    RES.mkdir(exist_ok=True)

    keys = D.available()
    print(f'Datasets: {keys}', flush=True)

    frames, mode_frames = [], []
    for key in keys:
        print(f'  {key} ...', flush=True)
        data, df, mode_rows = analyse_dataset(key)
        frames.append(df)
        mode_frames.append(mode_rows)
    fits = pd.concat(frames, ignore_index=True)
    modes = pd.concat(mode_frames, ignore_index=True)
    fits.to_csv(RES / 'real_data_fits.csv', index=False)
    modes.to_csv(RES / 'real_data_modes.csv', index=False)

    summary = summarize_dataset(fits, modes)
    summary.to_csv(RES / 'real_data_summary.csv', index=False)
    cols = ['dataset', 'tau_ar1_direct', 'tau_ar1_acf', 'tau_ar1_psd',
            'ar1_tau_spread', 'r2_ar1_psd', 'r2_arp_psd', 'order_arp_psd',
            'n_modes_r2_90', 'n_modes_r2_95', 'n_dominant_modes',
            'band_mode_freq', 'band_mode_tau', 'band_mode_cycles',
            'single_timescale_adequate']
    with pd.option_context('display.width', 200):
        print(summary[cols].round(4).to_string(index=False))



if __name__ == '__main__':
    main()

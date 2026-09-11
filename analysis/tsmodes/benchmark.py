"""Monte-Carlo benchmark harness.

One condition = one generative process at one recording duration.  Every
repetition simulates a fresh realization, prepares it once, and runs every
estimator on the *same* PSD and ACF, so differences between estimators are not
confounded by differences in spectral estimation.
"""

import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from tsmodes.estimators import ESTIMATOR_LABELS, estimate_all
from tsmodes.prep import prepare

N_MODES_RECORDED = 3


def flatten_fit(fit, dominant_tol=0.05):
    """Turn a :class:`TimescaleFit` into flat columns."""
    row = {
        'method': fit.method,
        'method_label': ESTIMATOR_LABELS.get(fit.method, fit.method),
        'tau': fit.tau,
        'tau_relaxation': fit.tau_relaxation,
        'order': fit.order,
        'success': fit.success,
        'order_at_ceiling': bool(fit.params.get('order_at_ceiling', False)),
    }
    for key in ('r2_logpsd', 'rmse_logpsd', 'is_div', 'r2_acf', 'rmse_acf'):
        row[key] = fit.gof.get(key, np.nan)

    if fit.modes is not None:
        dom = fit.modes.dominant_subset(tol=dominant_tol)
        row['n_dominant_modes'] = dom['n_modes']
        row['dominant_error'] = dom['error']
        row['n_modes'] = len(fit.modes.modes)
        row['tau_slowest'] = fit.modes.tau_slowest
        row['is_stationary'] = fit.modes.is_stationary
        osc = [m for m in fit.modes.modes if m.kind == 'oscillatory']
        row['n_osc_modes'] = len(osc)
        row['top_osc_freq'] = osc[0].freq if osc else np.nan
        row['top_osc_tau'] = osc[0].tau if osc else np.nan
        for i in range(N_MODES_RECORDED):
            if i < len(fit.modes.modes):
                m = fit.modes.modes[i]
                row[f'mode{i}_tau'] = m.tau
                row[f'mode{i}_freq'] = m.freq
                row[f'mode{i}_importance'] = m.importance
                row[f'mode{i}_kind'] = m.kind
            else:
                row[f'mode{i}_tau'] = np.nan
                row[f'mode{i}_freq'] = np.nan
                row[f'mode{i}_importance'] = np.nan
                row[f'mode{i}_kind'] = ''
    else:
        for key in ('n_dominant_modes', 'dominant_error', 'n_modes',
                    'tau_slowest', 'n_osc_modes', 'top_osc_freq', 'top_osc_tau'):
            row[key] = np.nan
        row['is_stationary'] = True
        for i in range(N_MODES_RECORDED):
            row[f'mode{i}_tau'] = np.nan
            row[f'mode{i}_freq'] = np.nan
            row[f'mode{i}_importance'] = np.nan
            row[f'mode{i}_kind'] = ''
    if fit.phi is not None:
        for i in range(min(3, len(fit.phi))):
            row[f'phi{i+1}'] = float(fit.phi[i])
    return row


def _run_one(rep, process=None, fs=500.0, duration=100.0, methods=None,
             seed=0, prep_kwargs=None, est_options=None, dominant_tol=0.05):
    rng = np.random.default_rng([seed, rep])
    prep_kwargs = {} if prep_kwargs is None else prep_kwargs
    sig = process.simulate(duration, fs, rng)
    signal = prepare(sig, fs, **prep_kwargs)
    # Robust regression is switched on only where there is a rhythm to be
    # robust to; see tsmodes.estimators.loss_settings.
    options = {m: dict(robust=process.has_oscillation)
               for m in ('ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd')}
    for key, value in (est_options or {}).items():
        options.setdefault(key, {}).update(value)
    fits = estimate_all(signal, methods=methods, options=options)

    base = {
        'process': process.name,
        'process_label': process.label,
        'duration': duration,
        'fs': fs,
        'n_samples': len(sig),
        'rep': rep,
        'tau_true': process.tau_true,
        'tau_integrated': process.tau_integrated(fs),
    }
    base.update({f'param_{k}': v for k, v in process.params.items()})
    base['duration_over_tau'] = (duration / process.tau_true
                                 if np.isfinite(process.tau_true) else np.nan)

    rows = []
    for fit in fits.values():
        row = dict(base)
        row.update(flatten_fit(fit, dominant_tol=dominant_tol))
        rows.append(row)
    return rows


def run_condition(process, fs=500.0, duration=100.0, n_reps=100, methods=None,
                  seed=0, prep_kwargs=None, est_options=None, n_jobs=1,
                  dominant_tol=0.05):
    """Run one process/duration condition for ``n_reps`` repetitions."""
    worker = partial(_run_one, process=process, fs=fs, duration=duration,
                     methods=methods, seed=seed, prep_kwargs=prep_kwargs,
                     est_options=est_options, dominant_tol=dominant_tol)
    if n_jobs == 1:
        results = [worker(rep) for rep in range(n_reps)]
    else:
        n_jobs = cpu_count() if n_jobs in (-1, None) else n_jobs
        with Pool(processes=n_jobs) as pool:
            results = pool.map(worker, range(n_reps))
    return [row for rows in results for row in rows]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(df, truth_col='tau_true', value_col='tau',
              group=('process', 'duration', 'method')):
    """Bias, error and variability of an estimate against a truth column."""
    group = list(group)
    out = []
    for keys, sub in df.groupby(group, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        truth = sub[truth_col].to_numpy(dtype=float)
        est = sub[value_col].to_numpy(dtype=float)
        good = np.isfinite(est) & np.isfinite(truth)
        rec = dict(zip(group, keys))
        rec['n'] = int(good.sum())
        rec['frac_valid'] = float(good.mean()) if len(good) else np.nan
        if good.sum() == 0:
            out.append(rec)
            continue
        e, t = est[good], truth[good]
        rec['median'] = float(np.median(e))
        rec['mean'] = float(np.mean(e))
        rec['truth'] = float(np.median(t))
        rec['bias'] = float(np.mean(e - t))
        rec['rel_bias'] = float(np.median((e - t) / t))
        rec['rel_error'] = float(np.median(np.abs(e - t) / t))
        rec['sd'] = float(np.std(e))
        rec['rel_sd'] = float(np.std(e) / np.mean(t))
        rec['iqr'] = float(np.subtract(*np.percentile(e, [75, 25])))
        rec['log_ratio_median'] = float(np.median(np.log10(e / t)))
        for col in ('r2_logpsd', 'rmse_logpsd', 'is_div', 'order',
                    'n_dominant_modes', 'n_osc_modes'):
            if col in sub:
                rec[f'{col}_median'] = float(np.nanmedian(sub[col].to_numpy(dtype=float)))
        out.append(rec)
    return pd.DataFrame(out)

"""Stage 2c: finite recordings bias timescale estimates.

Everything is parameterized by the dimensionless ratio T/tau.  Monte-Carlo
estimates are compared against the closed-form AR(1) sampling distribution
(bias, standard error, confidence interval) from ``tsmodes.finite``, and the
same durations are applied to the non-AR(1) processes to show which of them
converge to a stable timescale and which remain window-dependent.

Outputs
-------
results/finite_sampling.csv.gz, results/finite_summary.csv,
results/finite_analytic.csv
figures/figS_finite.svg
"""

import os
import sys
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmodes import plotting as P
from tsmodes.benchmark import run_condition, summarize
from tsmodes.estimators import ESTIMATOR_LABELS
from tsmodes.finite import (ar1_tau_ci, ar1_tau_moments, phi_of_tau,
                            relative_bias_asymptotic, relative_sd_asymptotic,
                            sample_tau_finite, duration_for_relative_sd)
from tsmodes.sims import AR1, AR1PlusSine, AR2Damped, ARpModes, PowerLaw

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'

FS = 500.0
TAU = 0.05
RATIOS = np.array([10, 20, 40, 100, 200, 400, 1000, 2000, 4000], dtype=float)
DURATIONS = RATIOS * TAU
N_REPS_AR1 = 150
N_REPS_OTHER = 60
N_JOBS = int(os.environ.get('TSMODES_N_JOBS', '12'))
METHODS = ['ar1_direct', 'arp_direct', 'ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']

OTHER = [
    AR1PlusSine(tau=TAU, f_osc=10.0, snr=1.0),
    AR2Damped(tau=TAU, f_osc=10.0),
    ARpModes(),
    PowerLaw(d=0.45),
]
OTHER_LABELS = ['AR(1) + sine', 'AR(2) damped', 'AR(8), 5 modes',
                'Power law']


def prep_for(duration):
    """Analysis settings that scale with the recording.

    The Welch segment is a quarter of the recording and the ACF is taken out to
    a quarter of the recording, so that the lowest analysed frequency is
    ``4/T``.  This is what makes the analysis window, rather than a fixed
    segment length, the thing that changes as recordings are shortened -- the
    situation a shorter experiment actually puts an analyst in.
    """
    return {'welch_win_s': duration / 4, 'nlags_s': duration / 4}


def _cached(name):
    """Load a cached result table, or None when it has not been computed."""
    path = RES / name
    return pd.read_csv(path) if path.exists() else None


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    # --figures-only replots from the cached table; see run_05_real_data.
    cached = _cached('finite_sampling.csv.gz') if '--figures-only' in sys.argv else None
    if cached is not None:
        print('Loading cached finite-sampling results '
              '(omit --figures-only to recompute)', flush=True)
        df = cached
        analytic = pd.read_csv(RES / 'finite_analytic.csv')
        make_figure(df, analytic)
        return

    rows = []
    print('AR(1) duration sweep ...', flush=True)
    for duration in DURATIONS:
        print(f'  T={duration:g}s (T/tau={duration/TAU:g})', flush=True)
        rows.extend(run_condition(AR1(tau=TAU), fs=FS, duration=float(duration),
                                  n_reps=N_REPS_AR1, seed=41, n_jobs=N_JOBS,
                                  methods=METHODS,
                                  prep_kwargs=prep_for(duration)))
    print('Other processes ...', flush=True)
    for proc in OTHER:
        for duration in DURATIONS:
            rows.extend(run_condition(proc, fs=FS, duration=float(duration),
                                      n_reps=N_REPS_OTHER, seed=51,
                                      n_jobs=N_JOBS, methods=METHODS,
                                      prep_kwargs=prep_for(duration)))
        print(f'  {proc.name} done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / 'finite_sampling.csv.gz', index=False, compression='gzip')

    summary = summarize(df, truth_col='tau_true', value_col='tau_relaxation')
    summary.to_csv(RES / 'finite_summary.csv', index=False)

    analytic = pd.DataFrame({
        'duration_over_tau': RATIOS,
        'duration_s': DURATIONS,
        'rel_bias': [ar1_tau_moments(TAU, d, FS)[0] / TAU for d in DURATIONS],
        'rel_sd': [ar1_tau_moments(TAU, d, FS)[1] / TAU for d in DURATIONS],
        'rel_bias_asymptotic': relative_bias_asymptotic(RATIOS),
        'rel_sd_asymptotic': relative_sd_asymptotic(RATIOS),
        'ci_lo': [ar1_tau_ci(TAU, d, FS)[0] for d in DURATIONS],
        'ci_hi': [ar1_tau_ci(TAU, d, FS)[1] for d in DURATIONS],
    })
    analytic.to_csv(RES / 'finite_analytic.csv', index=False)
    print(analytic.to_string(index=False))
    print('\nT/tau for 10% / 20% relative standard error: '
          f'{duration_for_relative_sd(1.0, 0.1):.0f} / '
          f'{duration_for_relative_sd(1.0, 0.2):.0f}')

    make_figure(df, analytic)


def method_curve(df, process, method, column='tau_relaxation'):
    med, lo, hi, sd = [], [], [], []
    for duration in DURATIONS:
        sub = df[(df['process'] == process) & (df['method'] == method)
                 & np.isclose(df['duration'], duration)]
        v = sub[column].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        med.append(np.median(v) if len(v) else np.nan)
        lo.append(np.percentile(v, 16) if len(v) else np.nan)
        hi.append(np.percentile(v, 84) if len(v) else np.nan)
        sd.append(np.std(v) / np.mean(v) if len(v) > 2 else np.nan)
    return np.array(med), np.array(lo), np.array(hi), np.array(sd)


def make_figure(df, analytic):
    """Four panels: the analytic finite-sample theory checked against Monte
    Carlo, and the same question asked of real recordings by windowing them.

    The main-text figure reports the headline numbers; this one shows the
    per-estimator and per-dataset detail behind them.
    """
    from tsmodes import datasets as D
    emp_path = RES / 'empirical_truncation_summary.csv'
    emp = pd.read_csv(emp_path) if emp_path.exists() else None

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.60, wspace=0.36)
    durations = np.sort(df['duration'].unique())

    # (a) analytic vs simulated -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    sub = df[(df['process'] == 'ar1') & (df['method'] == 'ar1_direct')]
    med = sub.groupby('duration')['tau'].median().reindex(durations)
    ax.plot(durations / TAU, med / TAU, 'o-', ms=3.5, lw=1.2, color='k',
            label='simulated')
    ax.plot(analytic['duration_over_tau'],
            1 + analytic['rel_bias'], '--', lw=1.2,
            color=P.MODE_COLORS['oscillatory'], label='analytic')
    ax.fill_between(analytic['duration_over_tau'],
                    analytic['ci_lo'] / TAU, analytic['ci_hi'] / TAU,
                    color=P.LIGHT, alpha=0.5, lw=0, label='95% interval')
    ax.axhline(1.0, color=P.GREY, lw=0.8, zorder=0)
    ax.set_xscale('log')
    ax.set_ylim(0.4, 1.6)
    ax.set_xlabel(r'Recording length / $\tau$')
    ax.set_ylabel(r'$\hat\tau\,/\,\tau$')
    ax.set_title('Theory vs simulation', fontsize=9)
    ax.legend(fontsize=6.2, loc='lower right', handlelength=1.5)
    P.panel_label(ax, 'a', dx=-0.24)

    # (b) the sampling distribution ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for tot, colour in zip([20, 100, 1000],
                           ['#B3402F', '#E08A2E', '#3B6FA0']):
        dur = TAU * tot
        near = durations[np.argmin(np.abs(durations - dur))]
        vals = df[(df['process'] == 'ar1') & (df['method'] == 'ar1_direct')
                  & (df['duration'] == near)]['tau'].to_numpy(float) / TAU
        vals = vals[np.isfinite(vals)]
        if len(vals) > 5:
            ax.hist(vals, bins=40, range=(0, 3), density=True, histtype='step',
                    color=colour, lw=1.2, label=rf'$T={tot}\tau$')
    ax.axvline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xlabel(r'$\hat\tau\,/\,\tau$')
    ax.set_ylabel('Density')
    ax.set_title('Sampling distribution', fontsize=9)
    ax.legend(fontsize=6.2, loc='upper right')
    P.panel_label(ax, 'b', dx=-0.24)

    if emp is None:
        P.save(fig, FIG / 'figS_finite.svg')
        return

    keys = [k for k in D.DATASET_KEYS if k in set(emp['dataset'])]
    colours = dict(zip(keys, plt.cm.viridis(np.linspace(0, 0.9, len(keys)))))

    # (c) the same question, asked of real recordings ----------------------
    ax = fig.add_subplot(gs[1, 0])
    for key in keys:
        sub = emp[(emp['dataset'] == key) & (emp['method'] == 'ar1_psd')]
        ax.plot(sub['duration'], sub['rel_to_longest'], 'o-', ms=2.5, lw=1.0,
                color=colours[key], label=D.SHORT_NAMES.get(key, key))
    ax.axhline(1.0, color='k', ls=':', lw=0.9)
    ax.axhspan(0.9, 1.1, color=P.LIGHT, alpha=0.4, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(0.2, 3.2)
    ax.set_yticks([0.25, 0.5, 1, 2]); ax.set_yticklabels(['0.25', '0.5', '1', '2'])
    ax.set_xlabel('Window duration (s)')
    ax.set_ylabel(r'$\hat\tau(T)\,/\,\hat\tau$(longest)')
    ax.set_title('Windowing, spectral fit', fontsize=9)
    ax.legend(fontsize=4.8, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), columnspacing=0.5,
              handlelength=1.2, frameon=False)
    P.panel_label(ax, 'c', dx=-0.26)

    # (d) model order needs data too ---------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for key in keys:
        sub = emp[(emp['dataset'] == key) & (emp['method'] == 'arp_psd')]
        if 'median_order' in sub:
            ax.plot(sub['duration'], sub['median_order'], 'o-', ms=2.5, lw=1.0,
                    color=colours[key])
    ax.set_xscale('log')
    ax.set_xlabel('Window duration (s)')
    ax.set_ylabel('AR order selected')
    ax.set_title('Order needs data', fontsize=9)
    P.panel_label(ax, 'd', dx=-0.26)

    P.save(fig, FIG / 'figS_finite.svg')


if __name__ == '__main__':
    main()

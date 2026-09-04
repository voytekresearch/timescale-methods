"""Main-text figure: how much data a timescale estimate needs.

Combines the simulated duration sweep (``run_04_finite.py``) with the empirical
windowing analysis (``run_06_empirical_truncation.py``), which are two halves of
the same question.  Both scripts must have been run first.
"""

import os
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmodes import plotting as P
from tsmodes import datasets as D
from tsmodes.estimators import ESTIMATOR_LABELS
from tsmodes.finite import relative_bias_asymptotic, relative_sd_asymptotic

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'
TAU = 0.05
SIM_METHODS = ['ar1_direct', 'ar1_acf', 'ar1_psd', 'arp_psd']
SCALE_FREE = {'power_law': 'power law', 'arp_modes': 'AR(8), 5 modes'}


def sim_curve(df, process, method, column='tau'):
    durations = np.sort(df['duration'].unique())
    med, sd = [], []
    for duration in durations:
        vals = df.loc[(df['process'] == process) & (df['method'] == method)
                      & np.isclose(df['duration'], duration),
                      column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        med.append(np.median(vals) if len(vals) else np.nan)
        sd.append(np.std(vals) / np.mean(vals) if len(vals) > 2 else np.nan)
    return durations, np.asarray(med), np.asarray(sd)


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)

    sim = pd.read_csv(RES / 'finite_sampling.csv.gz')
    analytic = pd.read_csv(RES / 'finite_analytic.csv')
    emp = pd.read_csv(RES / 'empirical_truncation_summary.csv')
    durations = np.sort(sim['duration'].unique())

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34)

    # (a) how short a recording biases the estimate ------------------------
    ax = fig.add_subplot(gs[0, 0])
    for method in ('ar1_direct', 'ar1_acf', 'ar1_psd'):
        _, med, _ = sim_curve(sim, 'ar1', method)
        ax.plot(durations / TAU, med / TAU, 'o-', ms=3, lw=1.1,
                color=P.METHOD_COLORS[method],
                label=ESTIMATOR_LABELS[method])
    ax.plot(analytic['duration_over_tau'], analytic['ci_lo'] / TAU, 'k:', lw=1.0)
    ax.plot(analytic['duration_over_tau'], analytic['ci_hi'] / TAU, 'k:', lw=1.0,
            label='95% interval')
    ax.axhline(1.0, color=P.LIGHT, lw=1.0, zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'Recording length / $\tau$')
    ax.set_ylabel(r'$\hat\tau\,/\,\tau$')
    ax.set_title('Bias', fontsize=9)
    ax.set_ylim(0.4, 1.6)
    ax.legend(fontsize=6, loc='lower right', handlelength=1.4)
    P.panel_label(ax, 'a', dx=-0.22)

    # (b) and how imprecise ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for method in ('ar1_direct', 'ar1_acf', 'ar1_psd'):
        _, _, sd = sim_curve(sim, 'ar1', method)
        ax.plot(durations / TAU, sd, 'o-', ms=3, lw=1.1,
                color=P.METHOD_COLORS[method])
    grid = np.logspace(1, 3.7, 50)
    ax.plot(grid, relative_sd_asymptotic(grid), 'k--', lw=1.2,
            label=r'$\sqrt{2\tau/T}$')
    ax.axhline(0.1, color=P.LIGHT, lw=1.0)
    ax.axvline(200, color=P.LIGHT, lw=1.0)
    ax.annotate('10% error\nneeds $T\\approx200\\tau$', xy=(200, 0.1),
                xytext=(250, 0.30), fontsize=6.5,
                arrowprops=dict(arrowstyle='->', lw=0.7, color=P.GREY))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Recording length / $\tau$')
    ax.set_ylabel(r'Relative s.d. of $\hat\tau$')
    ax.set_title('Precision', fontsize=9)
    ax.legend(fontsize=6.5, loc='lower left', handlelength=1.6)
    P.panel_label(ax, 'b', dx=-0.24)

    # (c) real recordings, cut into windows --------------------------------
    keys = [k for k in D.DATASET_KEYS if k in set(emp['dataset'])]
    colours = dict(zip(keys, plt.cm.viridis(np.linspace(0, 0.9, len(keys)))))
    ax = fig.add_subplot(gs[1, 0])
    for key in keys:
        sub = emp[(emp['dataset'] == key) & (emp['method'] == 'ar1_acf')]
        ax.plot(sub['duration'] / sub['reference_tau'], sub['rel_to_longest'],
                'o-', ms=2.5, lw=1.0, color=colours[key],
                label=D.SHORT_NAMES.get(key, key))
    ax.axhline(1.0, color='k', ls=':', lw=0.9)
    ax.axhspan(0.8, 1.25, color=P.LIGHT, alpha=0.35, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.8, 3e4); ax.set_ylim(0.2, 3.2)
    ax.set_yticks([0.25, 0.5, 1, 2])
    ax.set_yticklabels(['0.25', '0.5', '1', '2'])
    ax.set_xlabel(r'Window length / $\hat\tau$')
    ax.set_ylabel(r'$\hat\tau(T)\,/\,\hat\tau$(longest)')
    ax.set_title('Window dependence', fontsize=9)
    ax.legend(fontsize=4.8, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), columnspacing=0.5,
              handlelength=1.2, frameon=False)
    P.panel_label(ax, 'c', dx=-0.26)

    # (d) the spread between windows is not sampling noise -----------------
    ax = fig.add_subplot(gs[1, 1])
    shown = ['ar1_acf', 'ar1_psd']
    height = 0.36
    for i, method in enumerate(shown):
        vals = []
        for key in keys:
            sub = emp[(emp['dataset'] == key) & (emp['method'] == method)]
            vals.append(sub['excess_sd'].iloc[-1] if len(sub) else np.nan)
        ax.barh(np.arange(len(keys)) + (i - 0.5) * height, vals, height=height,
                color=P.METHOD_COLORS[method], label=ESTIMATOR_LABELS[method])
    ax.axvline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xscale('log')
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([D.SHORT_NAMES.get(k, k) for k in keys], fontsize=6)
    ax.set_xlabel('Observed spread / sampling bound')
    ax.set_title('Beyond sampling noise', fontsize=9)
    ax.legend(fontsize=6, loc='lower right', handlelength=1.2,
              borderpad=0.25, framealpha=0.9)
    P.panel_label(ax, 'd', dx=-0.42)

    P.save(fig, FIG / 'fig_windows.svg')


if __name__ == '__main__':
    main()

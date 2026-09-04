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

    fig = plt.figure(figsize=(7.4, 4.8))
    gs = fig.add_gridspec(2, 3, hspace=0.66, wspace=0.40)

    # (a) short recordings underestimate -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for method in ('ar1_direct', 'ar1_acf', 'ar1_psd'):
        _, med, _ = sim_curve(sim, 'ar1', method)
        ax.plot(durations / TAU, med / TAU, 'o-', ms=3, lw=1.1,
                color=P.METHOD_COLORS[method], label=ESTIMATOR_LABELS[method])
    ax.axhline(1.0, color=P.LIGHT, lw=1.0, zorder=0)
    ax.set_xscale('log')
    ax.set_ylim(0.4, 1.6)
    ax.set_xlabel(r'Recording length / $\tau$')
    ax.set_ylabel(r'$\hat\tau\,/\,\tau$')
    ax.set_title('Bias', fontsize=9)
    ax.legend(fontsize=5.8, loc='lower right', handlelength=1.3)
    P.panel_label(ax, 'a', dx=-0.28)

    # (b) and the analytic prediction says by how much ----------------------
    ax = fig.add_subplot(gs[0, 1])
    sub = sim[(sim['process'] == 'ar1') & (sim['method'] == 'ar1_direct')]
    med = sub.groupby('duration')['tau'].median().reindex(durations)
    ax.fill_between(analytic['duration_over_tau'],
                    analytic['ci_lo'] / TAU, analytic['ci_hi'] / TAU,
                    color=P.LIGHT, alpha=0.5, lw=0, label='95% interval')
    ax.plot(analytic['duration_over_tau'], 1 + analytic['rel_bias'], '--',
            lw=1.3, color=P.MODE_COLORS['oscillatory'], label='analytic')
    ax.plot(durations / TAU, med / TAU, 'o-', ms=3, lw=1.1, color='k',
            label='simulated')
    ax.axhline(1.0, color=P.GREY, lw=0.8, zorder=0)
    ax.set_xscale('log')
    ax.set_ylim(0.4, 1.6)
    ax.set_xlabel(r'Recording length / $\tau$')
    ax.set_ylabel(r'$\hat\tau\,/\,\tau$')
    ax.set_title('Theory vs simulation', fontsize=9)
    ax.legend(fontsize=5.8, loc='lower right', handlelength=1.4)
    P.panel_label(ax, 'b', dx=-0.28)

    # (c) precision improves only as sqrt(T) --------------------------------
    ax = fig.add_subplot(gs[0, 2])
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
                xytext=(260, 0.32), fontsize=6,
                arrowprops=dict(arrowstyle='->', lw=0.7, color=P.GREY))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Recording length / $\tau$')
    ax.set_ylabel(r'Relative s.d. of $\hat\tau$')
    ax.set_title('Precision', fontsize=9)
    ax.legend(fontsize=6, loc='lower left', handlelength=1.5)
    P.panel_label(ax, 'c', dx=-0.30)

    # (d) the shape of the sampling distribution ---------------------------
    ax = fig.add_subplot(gs[1, 0])
    for tot, colour in zip([20, 100, 1000],
                           ['#D55E00', '#E69F00', '#0072B2']):
        near = durations[np.argmin(np.abs(durations - TAU * tot))]
        vals = sim[(sim['process'] == 'ar1') & (sim['method'] == 'ar1_direct')
                   & (sim['duration'] == near)]['tau'].to_numpy(float) / TAU
        vals = vals[np.isfinite(vals)]
        if len(vals) > 5:
            ax.hist(vals, bins=40, range=(0, 3), density=True,
                    histtype='step', color=colour, lw=1.2,
                    label=rf'$T={tot}\tau$')
    ax.axvline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xlabel(r'$\hat\tau\,/\,\tau$')
    ax.set_ylabel('Density')
    ax.set_title('Sampling distribution', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    P.panel_label(ax, 'd', dx=-0.28)

    # (e) real recordings, cut into windows ---------------------------------
    keys = [k for k in D.DATASET_KEYS if k in set(emp['dataset'])]
    colours = dict(zip(keys, plt.cm.viridis(np.linspace(0, 0.9, len(keys)))))
    ax = fig.add_subplot(gs[1, 1])
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
    ax.legend(fontsize=4.6, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.34), columnspacing=0.5,
              handlelength=1.1, frameon=False)
    P.panel_label(ax, 'e', dx=-0.30)

    # (f) and the spread between them is not sampling noise -----------------
    ax = fig.add_subplot(gs[1, 2])
    height = 0.36
    for i, method in enumerate(('ar1_acf', 'ar1_psd')):
        vals = []
        for key in keys:
            sub = emp[(emp['dataset'] == key) & (emp['method'] == method)]
            vals.append(sub['excess_sd'].iloc[-1] if len(sub) else np.nan)
        ax.barh(np.arange(len(keys)) + (i - 0.5) * height, vals, height=height,
                color=P.METHOD_COLORS[method], label=ESTIMATOR_LABELS[method])
    ax.axvline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xscale('log')
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([D.SHORT_NAMES.get(k, k) for k in keys], fontsize=5.6)
    ax.set_xlabel('Observed spread /\nsampling bound')
    ax.set_title('Beyond sampling noise', fontsize=9)
    ax.legend(fontsize=5.6, loc='lower right', handlelength=1.2,
              borderpad=0.25, framealpha=0.9)
    P.panel_label(ax, 'f', dx=-0.34)

    P.save(fig, FIG / 'fig_windows.svg')


if __name__ == '__main__':
    main()

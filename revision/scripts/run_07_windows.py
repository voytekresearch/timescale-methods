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
    sim = pd.read_csv(RES / 'finite_sampling.csv.gz')
    analytic = pd.read_csv(RES / 'finite_analytic.csv')
    emp = pd.read_csv(RES / 'empirical_truncation_summary.csv')

    fig = plt.figure(figsize=(7.3, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.42)
    ratios = analytic['duration_over_tau'].to_numpy()

    # ---------------------------------------------------------------- row 1
    ax = fig.add_subplot(gs[0, 0])
    for method in SIM_METHODS:
        _, med, _ = sim_curve(sim, 'ar1', method)
        ax.plot(ratios, med / TAU, 'o-', ms=3, lw=1.1,
                color=P.METHOD_COLORS[method],
                label=ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'))
    ax.plot(ratios, analytic['ci_lo'] / TAU, color='k', ls=':', lw=0.9)
    ax.plot(ratios, analytic['ci_hi'] / TAU, color='k', ls=':', lw=0.9,
            label='95% interval')
    ax.axhline(1.0, color=P.LIGHT, lw=0.9, zorder=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Recording length / $\\tau$')
    ax.set_ylabel('$\\hat\\tau\\,/\\,\\tau$')
    ax.set_title('Short recordings\nunderestimate $\\tau$', fontsize=8.5)
    ax.set_ylim(0.35, 2.6)
    ax.legend(fontsize=5.6, loc='lower right', handlelength=1.4,
              borderpad=0.2, labelspacing=0.25)
    P.panel_label(ax, 'a', dx=-0.28)

    ax = fig.add_subplot(gs[0, 1])
    for method in SIM_METHODS:
        _, _, sd = sim_curve(sim, 'ar1', method)
        ax.plot(ratios, sd, 'o-', ms=3, lw=1.1, color=P.METHOD_COLORS[method])
    ax.plot(ratios, relative_sd_asymptotic(ratios), 'k--', lw=1.2,
            label='$\\sqrt{2\\tau/T}$')
    ax.axhline(0.1, color=P.LIGHT, lw=0.9)
    ax.axvline(200, color=P.LIGHT, lw=0.9)
    ax.text(230, 0.115, '10% error\nneeds $T\\approx200\\,\\tau$', fontsize=6.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Recording length / $\\tau$')
    ax.set_ylabel('Spread of $\\hat\\tau$')
    ax.set_title('Precision improves only\nas $\\sqrt{T}$', fontsize=8.5)
    ax.legend(fontsize=6.5, loc='lower left')
    P.panel_label(ax, 'b', dx=-0.30)

    ax = fig.add_subplot(gs[0, 2])
    durations, med, _ = sim_curve(sim, 'ar1', 'ar1_acf')
    ax.plot(durations, med, 'o--', ms=3, lw=1.1, color='k', label='AR(1)')
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(SCALE_FREE)))
    for (proc, lab), colour in zip(SCALE_FREE.items(), colors):
        _, med, _ = sim_curve(sim, proc, 'ar1_acf')
        ax.plot(durations, med, 'o-', ms=3, lw=1.1, color=colour, label=lab)
    ax.plot(durations, durations * 0.035, color=P.GREY, ls=':', lw=1.0)
    ax.text(durations[-1] * 1.15, durations[-1] * 0.035,
            '$\\hat\\tau\\propto T$', fontsize=6.5, color=P.GREY,
            ha='left', va='center', clip_on=False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Recording length (s)')
    ax.set_ylabel('$\\hat\\tau$ (s)')
    ax.set_title('Without a true $\\tau$, the answer\nfollows the window',
                 fontsize=8.5)
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0.0, 0.98))
    P.panel_label(ax, 'c', dx=-0.30)

    # ---------------------------------------------------------------- row 2
    keys = [k for k in D.DATASET_KEYS if k in set(emp['dataset'])]
    colours = dict(zip(keys, plt.cm.viridis(np.linspace(0, 0.9, len(keys)))))

    ax = fig.add_subplot(gs[1, 0])
    for key in keys:
        sub = emp[(emp['dataset'] == key) & (emp['method'] == 'ar1_acf')]
        # Window length in units of the timescale being measured.  Plotted in
        # seconds the curves say nothing, because a 1 s window is generous for
        # a 4 ms timescale and hopeless for a 1 s one.
        ax.plot(sub['duration'] / sub['reference_tau'], sub['rel_to_longest'],
                'o-', ms=2.5, lw=1.0, color=colours[key],
                label=D.SHORT_NAMES.get(key, key))
    grid = np.logspace(np.log10(4.0), 4.2, 60)
    ax.plot(grid, 1 + relative_bias_asymptotic(grid), 'k--', lw=1.2,
            label='expected from\nfinite data alone')
    ax.axhline(1.0, color='k', ls=':', lw=0.8)
    ax.axhspan(0.8, 1.25, color=P.LIGHT, alpha=0.35, lw=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Window length / $\\hat\\tau$')
    ax.set_ylabel('$\\hat\\tau(T)$ / $\\hat\\tau$(longest)')
    ax.set_title('The median estimate hardly\nmoves with window length',
                 fontsize=8.5)
    ax.set_ylim(0.2, 3.2)
    ax.set_xlim(0.8, 3e4)
    ax.set_yticks([0.25, 0.5, 1, 2])
    ax.set_yticklabels(['0.25', '0.5', '1', '2'])
    ax.legend(fontsize=4.6, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.32), columnspacing=0.5,
              handlelength=1.2, frameon=False)
    P.panel_label(ax, 'd', dx=-0.28)

    ax = fig.add_subplot(gs[1, 1])
    for key in keys:
        sub = emp[(emp['dataset'] == key) & (emp['method'] == 'ar1_psd')]
        ax.plot(sub['duration'] / sub['median_tau'], sub['log_cv'], 'o-',
                ms=2.5, lw=1.0, color=colours[key])
    grid = np.logspace(0.6, 4.6, 50)
    ax.plot(grid, relative_sd_asymptotic(grid), 'k--', lw=1.2,
            label='expected from\nfinite data alone')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Window length / $\\hat\\tau$')
    ax.set_ylabel('Spread across windows')
    ax.set_title('Windows disagree more than\nsampling explains', fontsize=8.5)
    ax.legend(fontsize=5.6, loc='lower left', handlelength=1.4)
    P.panel_label(ax, 'e', dx=-0.30, dy=1.16)

    ax = fig.add_subplot(gs[1, 2])
    shown = ['ar1_acf', 'ar1_psd']
    height = 0.36
    for i, method in enumerate(shown):
        vals = []
        for key in keys:
            sub = emp[(emp['dataset'] == key) & (emp['method'] == method)]
            vals.append(sub['excess_sd'].iloc[-1] if len(sub) else np.nan)
        ax.barh(np.arange(len(keys)) + (i - 0.5) * height, vals, height=height,
                color=P.METHOD_COLORS[method],
                label=ESTIMATOR_LABELS[method])
    ax.axvline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xscale('log')
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([D.SHORT_NAMES.get(k, k) for k in keys], fontsize=5.5)
    ax.set_xlabel('Observed spread /\nexpected from finite data')
    ax.set_title('Non-stationarity, not noise', fontsize=8.5)
    ax.legend(fontsize=5.6, loc='center right', handlelength=1.2,
              borderpad=0.25, framealpha=0.9)
    P.panel_label(ax, 'f', dx=-0.34, dy=1.20)

    P.save(fig, FIG / 'fig_windows.svg')


if __name__ == '__main__':
    main()

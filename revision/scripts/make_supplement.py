"""Supplementary figures: the technical detail kept out of the main text.

Reads the cached result tables written by the stage scripts.
"""

import os
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmodes import plotting as P
from tsmodes.estimators import ESTIMATOR_LABELS
from tsmodes.modes import ARModes, poles_to_ar

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'
FS = 500.0
TAU = 0.05
SNRS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
FREQS = [4.0, 6.0, 10.0, 16.0, 25.0, 40.0]
SHORT = ['AR(1)', 'AR(1) + sine', 'AR(2) oscillator', 'AR(8), 5 modes',
         'Power law']
PROCESS_KEYS = ['ar1', 'ar1_sine', 'ar2_damped', 'arp_modes', 'power_law']


def build_true_model():
    spec = [(0.100, 0.0), (0.050, 10.0), (0.010, 40.0), (0.002, 0.0),
            (0.003, 180.0)]
    poles = []
    for tau, freq in spec:
        r = np.exp(-1.0 / (tau * FS))
        if freq == 0:
            poles.append(r + 0j)
        else:
            theta = 2 * np.pi * freq / FS
            poles += [r * np.exp(1j * theta), r * np.exp(-1j * theta)]
    return ARModes.from_phi(poles_to_ar(np.array(poles)), FS)


def fig_population():
    """What AR(p) converges to with infinite data, and how order is chosen."""
    df = pd.read_csv(RES / 'benchmark_main.csv.gz')
    orders = pd.read_csv(RES / 'oscillation_population_orders.csv')

    fig = plt.figure(figsize=(5.4, 2.4))
    gs = fig.add_gridspec(1, 2, wspace=0.40)

    # (a) what order finite data actually buys ----------------------------
    ax = fig.add_subplot(gs[0, 0])
    methods = ['arp_direct', 'arp_psd']
    for i, method in enumerate(methods):
        vals, lo, hi = [], [], []
        for key in PROCESS_KEYS:
            o = df[(df['process'] == key) & (df['method'] == method)]['order']
            vals.append(o.median())
            lo.append(o.quantile(0.1))
            hi.append(o.quantile(0.9))
        ax.errorbar(np.arange(len(PROCESS_KEYS)) + (i - 0.5) * 0.24, vals,
                    yerr=[np.array(vals) - np.array(lo),
                          np.array(hi) - np.array(vals)],
                    fmt='o', ms=3, lw=1.0, color=P.METHOD_COLORS[method],
                    label=ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'))
    ax.set_xticks(np.arange(len(PROCESS_KEYS)))
    ax.set_xticklabels(SHORT, rotation=32, ha='right', fontsize=6)
    ax.set_ylabel('Order chosen by BIC')
    ax.set_title('Order selected', fontsize=9)
    ax.legend(fontsize=5.8, loc='center left', bbox_to_anchor=(0.02, 0.62))
    P.panel_label(ax, 'a', dx=-0.30)

    # (c) --------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    firsts = orders.groupby('f_osc')['first_order_with_osc'].first()
    ax.plot(firsts.index, firsts.values, 'o-', color='k', lw=1.1, ms=4)
    ax.axvline(1 / (2 * np.pi * TAU), color=P.GREY, ls=':', lw=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('Oscillation frequency (Hz)')
    ax.set_ylabel('Smallest order with\nan oscillatory mode')
    ax.set_title('Cost of a rhythm', fontsize=9)
    P.panel_label(ax, 'b', dx=-0.34)

    P.save(fig, FIG / 'figS_population.svg')


def fig_oscillation_grid():
    """Full frequency x amplitude grid for the additive-sinusoid simulation."""
    sweep = pd.read_csv(RES / 'oscillation_sweep.csv.gz')
    show = ['ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']

    fig, axes = plt.subplots(1, len(show), figsize=(7.4, 2.1))
    for ax, method in zip(axes, show):
        grid = np.full((len(SNRS), len(FREQS)), np.nan)
        for i, snr in enumerate(SNRS):
            for j, f_osc in enumerate(FREQS):
                sel = ((sweep['method'] == method)
                       & np.isclose(sweep['param_snr'], snr)
                       & np.isclose(sweep['param_f_osc'], f_osc))
                vals = sweep.loc[sel, 'tau_relaxation'].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    grid[i, j] = np.median(vals) / TAU
        # Colour and printed number are the same quantity in the same units:
        # the ratio of the estimate to the truth.  The colour scale is spaced
        # logarithmically so that a factor of two too big and a factor of two
        # too small are equally far from white.
        im = ax.imshow(np.log10(grid), origin='lower', cmap='RdBu_r',
                       vmin=-1, vmax=1, aspect='auto')
        for i in range(len(SNRS)):
            for j in range(len(FREQS)):
                value = grid[i, j]
                txt = f'{value:.2f}' if np.isfinite(value) else 'none'
                ax.text(j, i, txt, ha='center', va='center', fontsize=4.4,
                        color='k' if abs(np.log10(max(value, 1e-9))) < 0.6
                        else 'white')
        ax.set_xticks(range(len(FREQS)))
        ax.set_xticklabels([f'{f:g}' for f in FREQS], fontsize=6)
        ax.set_yticks(range(len(SNRS)))
        ax.set_yticklabels([f'{v:g}' for v in SNRS] if ax is axes[0] else [],
                           fontsize=6)
        ax.set_xlabel('Oscillation freq. (Hz)', fontsize=7)
        ax.set_title(ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'),
                     fontsize=8)
    axes[0].set_ylabel('Oscillation /\naperiodic power', fontsize=7)
    ticks = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                      ticks=np.log10(ticks))
    cb.ax.set_yticklabels([f'{t:g}' for t in ticks], fontsize=6)
    cb.set_label('estimated $\\tau$ / true $\\tau$', fontsize=7)
    P.save(fig, FIG / 'figS_oscillation_grid.svg')


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    fig_population()
    fig_oscillation_grid()


if __name__ == '__main__':
    main()

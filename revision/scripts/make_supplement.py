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

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'
FS = 500.0
TAU = 0.05
SNRS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
FREQS = [4.0, 6.0, 10.0, 16.0, 25.0, 40.0]


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
    fig_oscillation_grid()


if __name__ == '__main__':
    main()

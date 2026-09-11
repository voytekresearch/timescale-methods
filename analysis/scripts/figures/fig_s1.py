"""Fig. S1 -- the full oscillation sweep (figS_oscillation_grid.svg).

Run on its own from any directory: python scripts/figures/fig_s1.py
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig_common import (
    ESTIMATOR_LABELS, RES, TAU, TEXT_WIDTH, save)

# %%
# The grid is defined once, by the stage that swept it.
from run_03_oscillations import FREQS, SNRS

sweep = pd.read_csv(RES / 'oscillation_sweep.csv.gz')
show = ['ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']

fig, axes = plt.subplots(1, len(show), figsize=(TEXT_WIDTH, TEXT_WIDTH * 2.5 / 8.4))
for ax, method in zip(axes, show):
    grid = np.full((len(SNRS), len(FREQS)), np.nan)
    for i, snr in enumerate(SNRS):
        for j, f_osc in enumerate(FREQS):
            sel = ((sweep['method'] == method) & np.isclose(sweep['param_snr'], snr)
                   & np.isclose(sweep['param_f_osc'], f_osc))
            vals = sweep.loc[sel, 'tau_relaxation'].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                grid[i, j] = np.median(vals) / TAU
    # Colour and number are the same quantity: estimate / truth, log-spaced so
    # a factor of two too big and too small are equally far from white.
    im = ax.imshow(np.log10(grid), origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(SNRS)):
        for j in range(len(FREQS)):
            value = grid[i, j]
            txt = ((f'{value:.2f}'.lstrip('0') if value < 1 else f'{value:.1f}')
                   if np.isfinite(value) else 'none')
            ax.text(j, i, txt, ha='center', va='center', fontsize=4.5,
                    color='k' if abs(np.log10(max(value, 1e-9))) < 0.6 else 'white')
    ax.set_xticks(range(len(FREQS)))
    ax.set_xticklabels([f'{f:g}' for f in FREQS])
    ax.set_yticks(range(len(SNRS)))
    ax.set_yticklabels([f'{v:g}' for v in SNRS] if ax is axes[0] else [])
    ax.set_title(ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'), fontsize=8)
axes[0].set_ylabel('Oscillation /\naperiodic power')
fig.supxlabel('Oscillation frequency (Hz)', fontsize=7, y=-0.08)
ticks = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, ticks=np.log10(ticks))
cb.ax.set_yticklabels([f'{t:g}' for t in ticks])
cb.set_label('estimated $\\tau$ / true $\\tau$')

save(fig, 'figS_oscillation_grid.svg')

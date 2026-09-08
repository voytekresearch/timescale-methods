"""Fig. S2 -- real recordings cut into windows (figS_windows.svg).

Run on its own from any directory: python scripts/figures/fig_s2.py
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig_common import (
    D, ESTIMATOR_LABELS, P, RES, TEXT_WIDTH, save)

# %%
emp = pd.read_csv(RES / 'empirical_truncation_summary.csv')
keys = [k for k in D.DATASET_KEYS if k in set(emp['dataset'])]
colours = dict(zip(keys, plt.cm.viridis(np.linspace(0, 0.9, len(keys)))))

fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH * 3.4 / 8.4), gridspec_kw=dict(wspace=0.55))

# (a) the median estimate against window length -------------------------------
ax = axes[0]
for key in keys:
    sub = emp[(emp['dataset'] == key) & (emp['method'] == 'ar1_acf')]
    ax.plot(sub['duration'] / sub['reference_tau'], sub['rel_to_longest'], 'o-',
            ms=3, lw=1.1, color=colours[key], label=D.SHORT_NAMES.get(key, key))
ax.axhline(1.0, color='k', ls=':', lw=1.0)
ax.axhspan(0.8, 1.25, color=P.LIGHT, alpha=0.35, lw=0)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.8, 3e4); ax.set_ylim(0.2, 3.2)
ax.set_yticks([0.25, 0.5, 1, 2])
ax.set_yticklabels(['0.25', '0.5', '1', '2'])
ax.set_xlabel(r'Window length / $\hat\tau$')
ax.set_ylabel(r'$\hat\tau(T)\,/\,\hat\tau$(longest)')
ax.set_title('Window dependence')
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.28),
          columnspacing=0.8, handlelength=1.2, fontsize=6)
P.panel_label(ax, 'a', dx=-0.24, fontsize=9)

# (b) the spread between windows, against the sampling bound ------------------
ax = axes[1]
height = 0.36
for i, method in enumerate(('ar1_acf', 'ar1_psd')):
    vals = [emp[(emp['dataset'] == k) & (emp['method'] == method)]['excess_sd'].iloc[-1]
            for k in keys]
    ax.barh(np.arange(len(keys)) + (i - 0.5) * height, vals, height=height,
            color=P.METHOD_COLORS[method], label=ESTIMATOR_LABELS[method])
ax.axvline(1.0, color='k', ls='--', lw=1.1)
ax.set_xscale('log')
ax.set_yticks(np.arange(len(keys)))
ax.set_yticklabels([D.SHORT_NAMES.get(k, k) for k in keys], fontsize=6)
ax.set_xlabel('Observed spread / sampling bound')
ax.set_title('Beyond sampling noise')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=1.2,
          fontsize=6)
P.panel_label(ax, 'b', dx=-0.46, fontsize=9)

save(fig, 'figS_windows.svg')

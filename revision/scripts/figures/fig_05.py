"""Fig. 5 -- how much data a timescale needs (fig_finite.svg).

Run on its own from any directory: python scripts/figures/fig_05.py
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig_common import (
    FIT_COLOR, P, RES, TAU, TEXT_WIDTH, save)

# %%
from scipy.signal import lfilter
from scipy.stats import gaussian_kde

rng = np.random.default_rng(5)
PHIS = np.linspace(0.5, 0.99, 25)
N_SIMS = 1000
BURN = 200


def phi_hat_ls(x):
    """Mean-corrected least-squares AR(1) coefficient for each row of x."""
    x = x - x.mean(axis=1, keepdims=True)
    return np.sum(x[:, 1:] * x[:, :-1], axis=1) / np.sum(x[:, :-1] ** 2, axis=1)


def ci_panel(ax, n, letter):
    mean_hat, lo, hi = [], [], []
    for phi in PHIS:
        eps = rng.normal(size=(N_SIMS, n + BURN))
        x = lfilter([1.0], [1.0, -phi], eps, axis=1)[:, BURN:]
        ph = phi_hat_ls(x)
        # Least-squares interval, eq. (var) of the paper, around each estimate.
        half = 1.96 * np.sqrt(np.maximum(1.0 - ph ** 2, 0.0) / (n - 1))
        mean_hat.append(ph.mean())
        lo.append((ph - half).mean())
        hi.append((ph + half).mean())
    ax.fill_between(PHIS, lo, hi, color=P.LIGHT, alpha=0.6, lw=0, label='95% CI')
    ax.plot(PHIS, PHIS, color='k', ls=':', lw=1.2, label='true')
    ax.plot(PHIS, mean_hat, color=FIT_COLOR, lw=1.8, label=r'mean $\hat\varphi$')
    ax.set_xlabel(r'True $\varphi$')
    ax.set_ylabel(r'Estimated $\varphi$')
    ax.set_title(f'n = {n} samples')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.3, 1.05)
    ax.set_box_aspect(1)
    P.panel_label(ax, letter, dx=-0.30, fontsize=9)


fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, TEXT_WIDTH * 3.0 / 8.4), gridspec_kw=dict(wspace=0.5))
ci_panel(axes[0], 100, 'a')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3,
               handlelength=1.6, columnspacing=1.2)
ci_panel(axes[1], 1000, 'b')

# (c) the sampling density of tau-hat at three recording lengths ------------
sim = pd.read_csv(RES / 'finite_sampling.csv.gz')
durations = np.sort(sim['duration'].unique())
ax = axes[2]
grid = np.linspace(0, 2, 400)
for tot, colour in zip([10, 100, 1000], ['#D55E00', '#E69F00', '#0072B2']):
    near = durations[np.argmin(np.abs(durations - TAU * tot))]
    vals = sim[(sim['process'] == 'ar1') & (sim['method'] == 'ar1_direct')
               & (sim['duration'] == near)]['tau'].to_numpy(float) / TAU
    vals = vals[np.isfinite(vals)]
    ax.plot(grid, gaussian_kde(vals)(grid), color=colour, lw=1.6, label=rf'$T = {tot}\tau$')
ax.axvline(1.0, color='k', ls='--', lw=1.1)
ax.set_xlim(-0.1, 2.1)
ax.set_xlabel(r'$\hat\tau\,/\,\tau$')
ax.set_ylabel('Density')
ax.set_title('Sampling distribution')
ax.set_box_aspect(1)
# Compact and flush with the right edge: the panel is 1.3 in wide at print
# size and the T = 1000 tau peak stands at its centre.
ax.legend(loc='upper right', fontsize=5, handlelength=0.8, handletextpad=0.3, borderaxespad=0.0)
P.panel_label(ax, 'c', dx=-0.30, fontsize=9)

save(fig, 'fig_finite.svg')

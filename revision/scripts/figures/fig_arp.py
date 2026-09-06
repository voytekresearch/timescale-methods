"""An AR(p) built from several dominant relaxation modes, read mode by mode
(fig_arp.png / fig_arp.svg): the modes themselves, the spectrum they multiply
into, and the autocorrelation they add into.

Four real poles a factor of four apart.  In the spectrum each is a knee at
1 / (2 pi tau), and the slope steepens by two at every knee; the knees are
marked.  In the autocorrelation the weights are set by the poles, not chosen:
for real poles they alternate in sign and shrink with the pole spacing, so
the slowest mode carries more than the whole autocorrelation and the next
one is subtracted from it, which is the shoulder at short lags that no single
exponential has.

Run on its own from any directory: python scripts/figures/fig_arp.py
"""
# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import ARModes, ARpModes, FIG, FS, P, TEXT_WIDTH, proxy, save

# %%
# (tau in seconds, frequency in Hz); a zero frequency is a real pole.  Any
# ARP_MODES-style tuple works here, rhythms included.
MODES = (
    (0.250, 0),
    (0.02, 50),
    (0.1, 5)
)

proc = ARpModes(modes=MODES)
phi = proc.phi(FS)
model = ARModes.from_phi(phi, FS, offset=1.0, freqs=np.linspace(0, FS / 2, 1025))
order = np.argsort([-m.tau for m in model.modes])          # slowest first
shown = [model.modes[i] for i in order]
colours = [P.MODE_COLORS[m.kind] for m in shown]
print(f'{proc.label}: phi =', np.round(phi, 4).tolist())
for m in shown:
    print(f'  {m.kind:11s} tau = {m.tau * 1e3:6.1f} ms  f = {m.freq:5.1f} Hz  '
          f'importance = {m.rel_importance:.2f}')

nlags = 250
lags = np.arange(nlags + 1) / FS
freqs = np.logspace(np.log10(0.2), np.log10(240), 600)
acf_comps = model.acf_components(nlags)
weights = acf_comps[:, 0]

with plt.rc_context({'text.usetex': True}):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH * 1.2, TEXT_WIDTH * 2.5 / 9),
                             gridspec_kw=dict(wspace=0.45))
    LEG = dict(fontsize=6, handlelength=1.8, labelspacing=0.25)

    # (a) the modes themselves ---------------------------------------------------
    ax = axes[0]
    n, step = 200, 1.35
    t = np.arange(n) / FS
    comps = model.impulse_components(n)
    total = comps.sum(axis=0)
    top = len(shown) * step
    ax.plot(t, 0.5 * total / np.max(np.abs(total)) + top, color='k', lw=1.4)
    ax.text(t[-1], top + 0.6, 'response to an input', fontsize=6, ha='right', va='bottom')
    for i, (mode, c) in enumerate(zip(shown, colours)):
        base = (len(shown) - 1 - i) * step
        y = 0.5 * comps[order[i]] / np.max(np.abs(comps[order[i]])) + base
        ax.axhline(base, color=P.LIGHT, lw=0.5, zorder=0)
        ax.plot(t, y, color=c, lw=1.3)
        label = (f'$\\tau$ = {mode.tau * 1e3:.3g} ms'
                 + ('' if mode.kind == 'relaxation' else f', {mode.freq:.0f} Hz'))
        ax.text(t[-1], base + 0.4, label, fontsize=6, ha='right', va='bottom', color=c)
    ax.set_yticks([])
    ax.set_ylim(-0.9, top + 1.15)
    ax.set_xlabel('Time after input (s)')
    ax.set_title('Modes')
    ax.spines['left'].set_visible(False)
    P.panel_label(ax, 'a', dx=-0.10, dy=1.18, fontsize=9)

    # (b) the modes multiply in the spectrum; every knee marked ----------------
    ax = axes[1]
    centre = lambda power: power / np.exp(np.mean(np.log(power)))
    total_psd = centre(model.psd(freqs))
    ax.loglog(freqs, total_psd, color='k', lw=1.8, alpha=0.5, zorder=0)
    for mode, c in zip(shown, colours):
        ax.loglog(freqs, centre(mode.psd(freqs)), color=c, lw=1.2, zorder=1)
        if mode.kind == 'relaxation':
            f_knee = 1.0 / (2 * np.pi * mode.tau)
            ax.plot(f_knee, np.interp(np.log(f_knee), np.log(freqs), np.log(total_psd),
                                      ) and np.exp(np.interp(np.log(f_knee), np.log(freqs), np.log(total_psd))),
                    'o', ms=3.5, color=c, mec='k', mew=0.5, zorder=3)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Relative power')
    ax.set_title('Spectrum: modes multiply')
    ax.legend(handles=[proxy(color='k', alpha=0.5, lw=1.8, label='Product of modes'),
                       proxy(color=P.MODE_COLORS['relaxation'], lw=1.2, label='Relaxation mode'),
                       proxy(color='w', marker='o', ms=3.5, mec='k', mew=0.5,
                             mfc=P.MODE_COLORS['relaxation'], lw=0, label=r'Knee $1/(2\pi\tau)$')],
              loc='lower left', **LEG)
    P.panel_label(ax, 'b', dx=-0.30, dy=1.18, fontsize=9)

    # (c) the modes add in the autocorrelation, with their weights ---------------
    ax = axes[2]
    for i, c in zip(order, colours):
        ax.plot(lags, acf_comps[i], color=c, lw=1.2, zorder=1)
    ax.plot(lags, model.acf(nlags), color='k', alpha=0.5, lw=1.8, zorder=0)
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    # Each visible component carries its weight, the value it starts from.
    k = 12
    for i, c in zip(order, colours):
        w = weights[i]
        if abs(w) >= 0.02:
            ax.text(lags[k], acf_comps[i][k] + (0.05 if w > 0 else -0.05), f'{w:+.2f}',
                    fontsize=5.5, color=c, ha='left', va='bottom' if w > 0 else 'top')
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation: modes add')
    ax.legend(handles=[proxy(color='k', alpha=0.5, lw=1.8, label='Sum of modes')],
              loc='upper right', **LEG)
    P.panel_label(ax, 'c', dx=-0.30, dy=1.18, fontsize=9)

    save(fig, 'fig_arp.svg')

# %%

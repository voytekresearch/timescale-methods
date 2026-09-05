"""Fig. 3 -- an AR(p) read as modes (fig_modes.svg).

Run on its own from any directory: python scripts/figures/fig_03.py
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

from fig_common import (
    ARModes, FS, P, PHI7, TEXT_WIDTH, proxy, psd_from_poles, save)

# %%
model = ARModes.from_phi(PHI7, FS, offset=1.0, freqs=np.linspace(0, FS / 2, 1025))

RECOMBINED_COLOR = '#FFB000'     # product of modes (b) and sum of modes (c)
MODE_STYLE = (0, (2, 1.5))       # every individual mode

fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, TEXT_WIDTH * 2.5 / 9), gridspec_kw=dict(wspace=0.45))
freqs = np.logspace(np.log10(0.5), np.log10(150), 600)
order = np.argsort([-m.tau for m in model.modes])       # slowest first
shown = [model.modes[i] for i in order]
colours = [P.MODE_COLORS[m.kind] for m in shown]
nlags = 150
lags = np.arange(nlags + 1) / model.fs
mode_handles = [
    proxy(color=P.MODE_COLORS['relaxation'], lw=1.2, ls=MODE_STYLE, label='Relaxation mode'),
    proxy(color=P.MODE_COLORS['oscillatory'], lw=1.2, ls=MODE_STYLE, label='Oscillatory mode'),
]
LEG = dict(fontsize=6, handlelength=1.8, labelspacing=0.25)

# (a) the modes themselves ----------------------------------------------------
ax = axes[0]
n, step = 125, 1.35
t = np.arange(n) / model.fs
comps = model.impulse_components(n)
total = comps.sum(axis=0)
top = len(shown) * step
ax.plot(t, 0.5 * total / np.max(np.abs(total)) + top, color='k', lw=1.4)
ax.text(t[-1], top + 0.6, 'response to an input', fontsize=6, ha='right', va='bottom')

has_red = False
has_blue = False
for i, mode in enumerate(shown):
    base = (len(shown) - 1 - i) * step
    y = 0.5 * comps[order[i]] / np.max(np.abs(comps[order[i]])) + base
    ax.axhline(base, color=P.LIGHT, lw=0.5, zorder=0)

    if mode.kind == 'relaxation' and not has_blue:
        lab = "Relaxation mode"
        has_blue = True
    elif not has_red:
        lab = "Oscillatory mode"
        has_red = True
    else:
        lab = None

    ax.plot(t, y, color=colours[i], lw=1.3, label=lab)

    label = (f'$\\tau$ = {mode.tau * 1e3:.0f} ms'
             + ('' if mode.kind == 'relaxation' else f', {mode.freq:.0f} Hz'))

    ax.text(t[-1], base + 0.6 - .2, label, fontsize=6, ha='right', va='bottom', color=colours[i])

ax.set_yticks([])
ax.set_ylim(-2.9, top + 1.15)
ax.set_xlabel('Time after input (s)')
ax.set_title('Modes')
ax.legend(loc='lower left', handlelength=1.2, borderaxespad=0.0)
ax.spines['left'].set_visible(False)
P.panel_label(ax, 'a', dx=-0.10, dy=1.18, fontsize=9)

# (b) the modes multiply in the spectrum --------------------------------------
ax = axes[1]
centre = lambda power: power / np.exp(np.mean(np.log(power)))
for mode, c in zip(shown, colours):
    ax.loglog(freqs, centre(mode.psd(freqs)), color=c, lw=1.2, ls=MODE_STYLE, zorder=1)
product = psd_from_poles(np.concatenate([m.poles for m in model.modes]),
                         freqs, model.fs, model.offset)
ax.loglog(freqs, centre(model.psd(freqs)), color='k', lw=1.8, zorder=3)
ax.loglog(freqs, centre(product), color=RECOMBINED_COLOR, lw=1.2, ls='--', zorder=4)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Relative power')
ax.set_title('Spectrum:\nmodes multiply')
ax.legend(handles=[proxy(color='k', lw=1.8, label='AR(p) Spectrum'),
                   proxy(color=RECOMBINED_COLOR, lw=1.2, ls='--', label='Product of modes')]
          + mode_handles, loc='lower left', **LEG)
P.panel_label(ax, 'b', dx=-0.30, dy=1.18, fontsize=9)

# (c) the modes add in the autocorrelation ------------------------------------
ax = axes[2]
acf_comps = model.acf_components(nlags)
for i, c in enumerate(colours):
    ax.plot(lags, acf_comps[order[i]], color=c, lw=1.2, ls=MODE_STYLE, zorder=1)
ax.plot(lags, model.acf(nlags), color='k', lw=1.8, zorder=3)
ax.plot(lags, acf_comps.sum(axis=0), color=RECOMBINED_COLOR, lw=1.2, ls='--', zorder=4)
ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
ax.set_xlim(0, 0.3)
ax.set_xlabel('Lag (s)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation:\nmodes add')
ax.legend(handles=[proxy(color='k', lw=1.8, label='AR(p) ACF'),
                   proxy(color=RECOMBINED_COLOR, lw=1.2, ls='--', label='Sum of modes')], loc='upper right', **LEG)
P.panel_label(ax, 'c', dx=-0.30, dy=1.18, fontsize=9)

save(fig, 'fig_modes.svg')

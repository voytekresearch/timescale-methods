"""Fig. 4 -- the AR(p) estimators on five simulated cases (fig_benchmark.svg).

Run on its own from any directory: python scripts/figures/fig_04.py
"""
# %%
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig_common import (
    AR1, AR1PlusSine, AR2Damped, ARpModes, DATA_COLOR, FIT_COLOR, FIT_LW,
    FIT_STYLE, FS, P, RES, TAU, TEXT_WIDTH, ar_acf_fft, ar_psd, estimate,
    prepare, proxy, save)

# %%
DURATION, WELCH_WIN_S = 100.0, 4.0
T_FINITE = 1.0
bench = pd.read_csv(RES / 'benchmark_main.csv.gz')
finite = pd.read_csv(RES / 'finite_sampling.csv.gz')
finite = finite[(finite['process'] == 'ar1') & np.isclose(finite['duration'], T_FINITE)]

# (process, label, table, robust fit?)
COLS = [
    (AR1(tau=TAU), 'AR(1)', bench[bench['process'] == 'ar1'], False),
    # The same added rhythm as fig_bias: 20 Hz, a fifth of the aperiodic variance.
    (AR1PlusSine(tau=TAU, f_osc=20.0, snr=0.2), 'AR(1) + sine', bench[bench['process'] == 'ar1_sine'], True),
    (AR2Damped(tau=TAU, f_osc=10.0), 'AR(2)\noscillator', bench[bench['process'] == 'ar2_damped'], True),
    (ARpModes(), ARpModes().label.replace(', ', ',\n'), bench[bench['process'] == 'arp_modes'], False),
    (AR1(tau=TAU), f'Finite sample', finite, False),
]
METHODS = [('arp_acf', 'ACF'), ('arp_psd', 'PSD')]
OSCILLATORY_TRUTH = {'ar2_damped'}


def target(df, proc):
    """The timescale each AR(p) fit is asked for: the envelope decay for the
    oscillator, the most important relaxation mode otherwise."""
    col = 'top_osc_tau' if proc.name in OSCILLATORY_TRUTH else 'tau_relaxation'
    return df[col].to_numpy(float)


n = len(COLS)
fig = plt.figure(figsize=(TEXT_WIDTH, TEXT_WIDTH * 7.4 / 8.4))
gs = fig.add_gridspec(4, n, hspace=0.8, wspace=0.32, height_ratios=[0.7, 1.0, 1.0, 1.15])
box_vals, tau_axes = [], []
for j, (proc, label, df, robust) in enumerate(COLS):
    rng = np.random.default_rng(3)
    finite_col = j == n - 1
    seconds = T_FINITE if finite_col else DURATION
    signal = prepare(proc.simulate(seconds, FS, rng), FS,
                     welch_win_s=0.5 if finite_col else WELCH_WIN_S)
    fit_psd = estimate(signal, 'arp_psd', robust=robust)
    fit_acf = estimate(signal, 'arp_acf', robust=robust)

    # -- row a: the signal ---------------------------------------------------
    ax = fig.add_subplot(gs[0, j])
    t = np.arange(int(min(1.5, seconds) * FS)) / FS
    y = signal.sig[:len(t)]
    ax.plot(t, y / np.std(y), color=DATA_COLOR, lw=0.5)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-4, 4)
    ax.set_xticks([0, 1])
    ax.set_yticks([])
    ax.set_title(label)
    ax.spines['left'].set_visible(False)
    if j == 0:
        ax.set_ylabel('Signal')
        P.panel_label(ax, 'a', dx=-0.32, dy=1.35, fontsize=9)
    if j == n // 2:
        ax.set_xlabel('Time (s)')

    # -- row b: the spectrum and its AR(p) fit -------------------------------
    ax = fig.add_subplot(gs[1, j])
    m = signal.fit_mask
    f = signal.freqs[m]
    ax.loglog(f, signal.powers[m], color=DATA_COLOR, lw=1.0)
    ax.loglog(f, ar_psd(fit_psd.phi, f, FS, fit_psd.offset), color=FIT_COLOR,
              lw=FIT_LW, ls=FIT_STYLE)
    ax.set_xticks([1, 10, 100])
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if j == 0:
        ax.set_ylabel('Power')
        ax.legend(handles=[proxy(color=DATA_COLOR, lw=1.2, label='data'),
                           proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')],
                  loc='lower left', fontsize=6, handlelength=1.0, handletextpad=0.4,
                  borderaxespad=0.2)
        P.panel_label(ax, 'b', dx=-0.32, dy=1.18, fontsize=9)
    if j == n // 2:
        ax.set_xlabel('Frequency (Hz)')

    # -- row c: the autocorrelation and its AR(p) fit ------------------------
    ax = fig.add_subplot(gs[2, j])
    nlags = min(len(signal.corrs) - 1, 150)
    lags = np.arange(nlags + 1) / FS
    ax.plot(lags, signal.corrs[:nlags + 1], color=DATA_COLOR, lw=1.0)
    ax.plot(lags, fit_acf.params['height'] * ar_acf_fft(fit_acf.phi, nlags)
            + fit_acf.params['acf_offset'], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE)
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(-0.6, 1.05)
    ax.set_yticks([0, 1])
    if j == 0:
        ax.set_ylabel('Autocorrelation')
        ax.legend(handles=[proxy(color=DATA_COLOR, lw=1.2, label='data'),
                           proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')],
                  loc='upper right', fontsize=6, handlelength=1.0, handletextpad=0.4,
                  borderaxespad=0.2)
        P.panel_label(ax, 'c', dx=-0.32, dy=1.18, fontsize=9)
    else:
        ax.set_yticklabels([])
    if j == n // 2:
        ax.set_xlabel('Lag (s)')

    # -- row d: what each AR(p) fit reports ----------------------------------
    ax = fig.add_subplot(gs[3, j])
    for i, (method, name) in enumerate(METHODS):
        vals = target(df[df['method'] == method], proc) * 1e3
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if not len(vals):
            continue
        box_vals.append(np.percentile(vals, [10, 90]))
        colour = P.METHOD_COLORS[method]
        bp = ax.boxplot([vals], positions=[i], widths=0.6, whis=(10, 90),
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='k', lw=1.2),
                        whiskerprops=dict(color=colour, lw=0.9),
                        capprops=dict(color=colour, lw=0.9))
        for box in bp['boxes']:
            box.set(facecolor=colour, alpha=0.6, lw=0.7, edgecolor=colour)
    truth = proc.tau_true * 1e3
    if np.isfinite(truth):
        ax.axhline(truth, color=P.MODE_COLORS['oscillatory'] if proc.name in OSCILLATORY_TRUTH else 'k',
                   ls='--', lw=1.1)
    ax.set_yscale('log')
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([name for _, name in METHODS])
    ax.set_xlabel('AR($p$) fit to')
    if j == 0:
        ax.set_ylabel('Estimated $\\tau$ (ms)')
        ax.legend(handles=[proxy(color='k', ls='--', lw=1.1, label='true $\\tau$')],
                  loc='upper right', fontsize=6, handlelength=1.8)
        P.panel_label(ax, 'd', dx=-0.32, dy=1.14, fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    tau_axes.append(ax)

# One y-range for every box panel, covering the 10-90% whiskers of all of
# them, and never below 5 ms: the few fits that collapse to a sample or two
# are real, but they should not set the axis for every other panel.
lo, hi = max(np.min(box_vals) / 1.6, 5.0), np.max(box_vals) * 1.6
for ax in tau_axes:
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=5))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))

save(fig, 'fig_benchmark.svg')

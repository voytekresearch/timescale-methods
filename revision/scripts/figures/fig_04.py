"""Fig. 4 -- the AR(p) estimators on four simulated cases (fig_benchmark.svg).

Run on its own from any directory: python scripts/figures/fig_04.py
"""
# %%
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig_common import (
    AR1, AR2_OSC, ARP, BENCH_TAU, DATA_COLOR, FIT_COLOR, FIT_LW, FIT_STYLE,
    FS, P, RES, TEXT_WIDTH, ar_acf_fft, ar_psd, estimate, prepare, proxy, save)

# %%
DURATION, WELCH_WIN_S = 100.0, 4.0
T_FINITE = 20 * BENCH_TAU            # the finite sample: twenty timescales
FINITE_SEED = 13                     # run_02_benchmark's seed for that condition
bench = pd.read_csv(RES / 'benchmark_main.csv.gz')
long = bench[np.isclose(bench['duration'], DURATION)]
finite = bench[(bench['process'] == 'ar1') & np.isclose(bench['duration'], T_FINITE)]


def median_repetition(table):
    """The repetition whose two AR(p) estimates are jointly closest to their
    medians: the finite-sample example is a member of the distribution drawn
    under it, and the most typical one, rather than an arbitrary draw."""
    acf = table[table['method'] == 'arp_acf'].set_index('rep')['tau_relaxation']
    psd = table[table['method'] == 'arp_psd'].set_index('rep')['tau_relaxation']
    dist = (np.log(acf / acf.median()).abs() + np.log(psd / psd.median()).abs())
    return int(dist.idxmin())


FINITE_REP = median_repetition(finite)

# (process, label, table, robust fit?).  Every column has the same
# constructed timescale, BENCH_TAU = 200 ms: the AR(1)'s tau, the AR(2)
# oscillator's envelope decay (the rhythm of fig_bias b), the AR(7)'s
# dominant relaxation (the process of fig_modes; with TSMODES_ARP=5 the
# AR(5), whose target is 100 ms) and the finite sample's tau, so that row d
# can share one axis.  The finite sample is simulated by the benchmark stage
# too (run_02_benchmark, FINITE_DURATION).
COLS = [
    (AR1(tau=BENCH_TAU), 'AR(1)', long[long['process'] == 'ar1'], False),
    (AR2_OSC, 'AR(2) oscillator', long[long['process'] == 'ar2_damped'], True),
    (ARP, ARP.label, long[long['process'] == ARP.name], False),
    (AR1(tau=BENCH_TAU), 'Finite sample', finite, False),
]
METHODS = [('arp_acf', 'ACF'), ('arp_psd', 'PSD')]
OSCILLATORY_TRUTH = {'ar2_damped'}


def target(df, proc):
    """The timescale each AR(p) fit is asked for: the envelope decay for the
    oscillator, the most important relaxation mode otherwise."""
    col = 'top_osc_tau' if proc.name in OSCILLATORY_TRUTH else 'tau_relaxation'
    return df[col].to_numpy(float)


with plt.rc_context({'text.usetex': True}):
    n = len(COLS)
    fig = plt.figure(figsize=(TEXT_WIDTH, TEXT_WIDTH * 7.4 / 8.4))
    gs = fig.add_gridspec(4, n, hspace=0.8, wspace=0.5, height_ratios=[0.7, 1.0, 1.0, 1.15])
    col_vals, tau_axes = {}, []
    for j, (proc, label, df, robust) in enumerate(COLS):
        finite_col = j == n - 1
        seconds = T_FINITE if finite_col else DURATION
        # The finite sample is repetition FINITE_REP of the benchmark condition
        # (same seeding as tsmodes.benchmark), analysed exactly as the long
        # recordings, so the duration is the only difference.
        rng = (np.random.default_rng([FINITE_SEED, FINITE_REP]) if finite_col
               else np.random.default_rng(3))
        signal = prepare(proc.simulate(seconds, FS, rng), FS, welch_win_s=WELCH_WIN_S)
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
            ax.legend(handles=[proxy(color=DATA_COLOR, lw=1.2, label='Data'),
                              proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')],
                    loc='lower left', fontsize=6)
            P.panel_label(ax, 'b', dx=-0.32, dy=1.18, fontsize=9)
        ax.set_xlabel('Frequency (Hz)')

        # -- row c: the autocorrelation and its AR(p) fit ------------------------
        ax = fig.add_subplot(gs[2, j])
        nlags = len(signal.corrs) - 1               # every lag the fit used
        lags = np.arange(nlags + 1) / FS
        ax.plot(lags, signal.corrs[:nlags + 1], color=DATA_COLOR, lw=1.0)
        ax.plot(lags, fit_acf.params['height'] * ar_acf_fft(fit_acf.phi, nlags)
                + fit_acf.params['acf_offset'], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE)
        ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
        ax.set_xlim(None, 0.5)
        ax.set_xticks([0, 0.25, 0.5])
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([0, 1])
        if j == 0:
            ax.set_ylabel('Autocorrelation')
            ax.legend(handles=[proxy(color=DATA_COLOR, lw=1.2, label='Data'),
                            proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')],
                    loc='lower right', fontsize=6)
            P.panel_label(ax, 'c', dx=-0.32, dy=1.18, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Lag (s)')

        # -- row d: what each AR(p) fit reports, on one shared axis --------------
        ax = fig.add_subplot(gs[3, j], sharey=tau_axes[0] if tau_axes else None)
        for i, (method, name) in enumerate(METHODS):
            vals = target(df[df['method'] == method], proc) * 1e3
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if not len(vals):
                continue
            col_vals.setdefault(j, []).append(np.percentile(vals, [10, 90]))
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
            ax.axhline(truth, color='k', ls='--', lw=1.1)
        ax.set_yscale('log')
        ax.set_xlim(-0.7, 1.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([name for _, name in METHODS])
        ax.set_xlabel('Fit domain')

        if j == 0:
            ax.set_ylabel('Estimated $\\tau$ (ms)')
            ax.legend(handles=[proxy(color='k', ls='--', lw=1.1, label='True $\\tau$')],
                      loc='upper right', fontsize=6, handlelength=1.8)
            P.panel_label(ax, 'd', dx=-0.32, dy=1.14, fontsize=9)
        tau_axes.append(ax)      # every column keeps its own tick labels

    # One y-range for the whole row, since every column has the same
    # constructed timescale: it covers the 10-90% whiskers of every box and
    # never goes below 5 ms -- the few fits that collapse to a sample or two
    # are real, but they should not set the axis.  Every column keeps its own
    # tick labels.
    vals = np.concatenate([v for cols in col_vals.values() for v in cols])
    lo, hi = max(np.min(vals) / 1.5, 5.0), np.max(vals) * 1.5
    tau_axes[0].set_ylim(lo, hi)
    # Under a decade or two a LogLocator names almost nothing, so the round
    # values inside the range are named outright.  Once the range spans more
    # than about three decades (the finite-sample column's spectral fits reach
    # the 20 s bound of the search range) only the decades are named, or the
    # labels overlap.
    steps = (1, 2, 5) if np.log10(hi / lo) < 3.0 else (1,)
    ticks = [m * 10 ** k for k in range(0, 6) for m in steps if lo <= m * 10 ** k <= hi]
    for ax in tau_axes:
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))

    save(fig, 'fig_benchmark.svg')

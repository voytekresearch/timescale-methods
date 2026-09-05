"""Fig. 6 -- the single-timescale model in eight recordings (fig_realdata.svg).

Run on its own from any directory: python scripts/figures/fig_06.py
"""
# %%
import copy
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fig_common import (
    D, DATA_COLOR, FIT_COLOR, FIT_LW, FIT_STYLE, P, RANK_COLORS, RES,
    TEXT_HEIGHT, TEXT_WIDTH, ar_psd, decade_ticks, estimate, proxy, save,
    spectrum_acov)

# %%
fits = pd.read_csv(RES / 'real_data_fits.csv')
keys = D.available()
cache = {k: D.load(k) for k in keys}
N_MODES_SHOWN = 3


def resolvable_modes(fit, signal):
    """The fit's modes, most important first, dropping those slower than the
    fitted band can resolve: a mode with tau > 1 / (2 pi f_lo) has its knee
    below the band, so its timescale is an extrapolation, not a measurement."""
    if fit.modes is None:
        return []
    tau_max = 1.0 / (2.0 * np.pi * signal.f_range[0])
    keep = [m for m in fit.modes.modes if m.tau <= tau_max]
    return sorted(keep, key=lambda m: -m.importance)


n = len(keys)
fig = plt.figure(figsize=(TEXT_WIDTH, min(TEXT_WIDTH * (1.3 * n + 1.2) / 8.4, 0.8 * TEXT_HEIGHT)))
gs = fig.add_gridspec(n, 4, hspace=0.55, wspace=0.55, width_ratios=[1, 1, 1, 0.6])
med = lambda a: np.median(np.vstack(a), axis=0)

for i, key in enumerate(keys):
    data = cache[key]
    sub = fits[fits['dataset'] == key]
    signal = D.prepare_trace(data['X'][0], data)
    mask = signal.fit_mask & signal.exclude_mask
    first, last = i == 0, i == n - 1

    # Every panel comes from the spectral AR(p) fit, which describes these
    # recordings best.  Panel d boxes the modes of every trace's fit.  Panels
    # a-c draw the median spectrum and autocorrelation across traces with ONE
    # fit to the median spectrum: its coefficients give the analytic AR(p)
    # autocorrelation in b and the modes that autocorrelation is made of in c.
    powers_all, corrs_all = [], []
    mode_taus = [[] for _ in range(N_MODES_SHOWN)]
    nlags = len(signal.corrs) - 1
    for trace in data['X']:
        sig = D.prepare_trace(trace, data)
        if len(sig.corrs) - 1 != nlags:
            continue
        powers_all.append(sig.powers)
        corrs_all.append(sig.corrs)
        fp_i = estimate(sig, 'arp_psd', robust=True)
        if fp_i.success and fp_i.phi is not None:
            for r, mode in enumerate(resolvable_modes(fp_i, sig)[:N_MODES_SHOWN]):
                mode_taus[r].append(mode.tau * 1e3)
    median_signal = copy.copy(signal)       # keeps the masks prepare() attaches
    median_signal.powers, median_signal.corrs = med(powers_all), med(corrs_all)
    psd_data, acf_data = median_signal.powers[mask], median_signal.corrs
    fp = estimate(median_signal, 'arp_psd', robust=True)
    psd_fit = ar_psd(fp.phi, signal.freqs, signal.fs, fp.offset)
    # The data's autocorrelation is the inverse transform of its spectrum over
    # the analysis band (prepare), so the fit is translated the same way: its
    # spectrum on the same grid, over the same band.  The model's analytic
    # full-band autocorrelation would instead carry its extrapolation below
    # the band's lower edge, where most of a 1/f-like model's variance sits,
    # and miss the data even where the fit is exact.  The components are the
    # partial fractions of the fitted spectrum, transformed the same way; all
    # of them together add up to the fit.
    acov_fit = spectrum_acov(median_signal, psd_fit, nlags)
    acf_fit = acov_fit / acov_fit[0]
    comps = spectrum_acov(median_signal, fp.modes.psd_components(signal.freqs),
                          nlags) / acov_fit[0]
    index = {id(m): j for j, m in enumerate(fp.modes.modes)}
    shown = resolvable_modes(fp, median_signal)[:N_MODES_SHOWN]
    mode_curves = [comps[index[id(m)]] for m in shown]
    # c and d show the same components: the ranks the fit to the median
    # spectrum resolves, boxed over the traces that resolve them.
    n_shown = len(shown)
    lags = np.arange(nlags + 1) / signal.fs
    taus = [t for t in (sub['tau'].median(), sub['tau_relaxation'].median())
            if np.isfinite(t) and t > 0]
    span = float(np.clip(max(6 * max(taus), 12 / signal.fs), 4 / signal.fs, lags[-1]))
    # Only the lags up to the span are drawn and the axes keep their default
    # limits, so lag zero sits at the same place in every row.
    keep = lags <= span

    # -- a: spectrum -------------------------------------------------------
    ax = fig.add_subplot(gs[i, 0])
    ax.loglog(signal.freqs[mask], psd_data, color=DATA_COLOR, lw=1.1, zorder=1)
    ax.loglog(signal.freqs[mask], psd_fit[mask], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, zorder=3)
    ax.set_ylabel(data['panel_name'].replace(' ', '\n'), labelpad=2, fontsize=6.5)
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if first:
        ax.set_title('Power spectrum')
        P.panel_label(ax, 'a', dx=-0.26, dy=1.45, fontsize=9)
    if last:
        ax.set_xlabel('Frequency (Hz)')

    # -- b: autocorrelation and the fit's analytic autocorrelation ----------
    axb = fig.add_subplot(gs[i, 1])
    axb.plot(lags[keep], acf_data[keep], color=DATA_COLOR, lw=1.1, zorder=1)
    axb.plot(lags[keep], acf_fit[keep], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, zorder=3)
    axb.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    if first:
        axb.set_title('Autocorrelation')
        P.panel_label(axb, 'b', dx=-0.24, dy=1.45, fontsize=9)
    if last:
        axb.set_xlabel('Lag (s)')

    # -- c: the modes that autocorrelation is made of, sharing b's axes ------
    axc = fig.add_subplot(gs[i, 2], sharex=axb, sharey=axb)
    for r, curve in enumerate(mode_curves):
        axc.plot(lags[keep], curve[keep], color=RANK_COLORS[r], lw=1.3, zorder=2)
    axc.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    axc.tick_params(labelleft=False)
    if first:
        axc.set_title('ACF\ndecomposition')
        P.panel_label(axc, 'c', dx=-0.12, dy=1.45, fontsize=9)
    if last:
        axc.set_xlabel('Lag (s)')

    # -- d: the timescale of each component, over all traces -----------------
    ax = fig.add_subplot(gs[i, 3])
    drawn = []
    for r in range(n_shown):
        vals = np.array([v for v in mode_taus[r] if np.isfinite(v) and v > 0])
        if not len(vals):
            continue
        drawn.append(vals)
        colour = RANK_COLORS[r]
        bp = ax.boxplot([vals], positions=[r], widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color='k', lw=1.2),
                        whiskerprops=dict(color=colour, lw=0.9),
                        capprops=dict(color=colour, lw=0.9))
        for box in bp['boxes']:
            box.set(facecolor=colour, alpha=0.55, lw=0.7, edgecolor=colour)
    ax.set_yscale('log')
    ax.set_xlim(-0.7, N_MODES_SHOWN - 0.3)
    ax.set_xticks(np.arange(N_MODES_SHOWN))
    ax.set_xticklabels([f'{r + 1}' for r in range(N_MODES_SHOWN)] if last else [])
    ax.set_ylabel('ms', labelpad=1)
    if drawn:
        decade_ticks(ax, np.concatenate(drawn))
    if first:
        ax.set_title('Timescales')
        P.panel_label(ax, 'd', dx=-0.55, dy=1.45, fontsize=9)
    if last:
        ax.set_xlabel('Component')

handles = ([proxy(color=DATA_COLOR, lw=1.2, label='data'),
            proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')]
           + [proxy(color=RANK_COLORS[r], lw=1.4, label=f'component {r + 1}')
              for r in range(N_MODES_SHOWN)])
fig.subplots_adjust(bottom=0.085)
fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0),
           handlelength=1.5, columnspacing=1.0)

save(fig, 'fig_realdata.svg')

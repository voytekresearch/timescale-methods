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
from timescales.decomposition.select import dominant_modes, variance_shares

# %%
fits = pd.read_csv(RES / 'real_data_fits.csv')
keys = D.available()
cache = {k: D.load(k) for k in keys}
N_MODES_SHOWN = 3
#: The noise floor is drawn only when an additive constant really does
#: reproduce the fitted high-pass factor (see ARModes.noise_floor).
FLOOR_MIN_R2 = 0.99
FLOOR_COLOR = '#937860'


def resolvable_modes(fit, signal):
    """The fit's modes the analysis band can measure, most important first.

    The rule lives in the library (ARModes.resolvable), so this figure and
    the supplementary tables cannot drift apart on what counts as resolvable:
    a mode is kept when the frequency it shapes the spectrum at -- a
    relaxation's knee, an oscillation's frequency, fs/2 for a Nyquist mode --
    falls inside the band the model was fit over.
    """
    if fit.modes is None:
        return []
    return fit.modes.resolvable(signal.f_range)


def reported_modes(fit, signal, nlags, n_max=None):
    """The modes this figure draws: resolvable, and holding real variance.

    Ranked and filtered by each mode's share of the variance, which is its
    additive autocorrelation component at lag zero.  Those components are the
    ones panel d draws, so panels b, d and e necessarily show the same set of
    modes in the same order.  See timescales.decomposition.select.

    Returns a list of (mode, share, component curve).
    """
    keep = resolvable_modes(fit, signal)
    if not keep:
        return []
    acov = spectrum_acov(signal, ar_psd(fit.phi, signal.freqs, signal.fs,
                                        fit.offset), nlags)
    parts = spectrum_acov(signal, fit.modes.psd_components(signal.freqs),
                          nlags) / acov[0]
    index = {id(m): j for j, m in enumerate(fit.modes.modes)}
    curves = np.array([parts[index[id(m)]] for m in keep])
    shares = variance_shares(keep, curves)
    chosen = dominant_modes(keep, shares, n_max=n_max)
    by_id = {id(m): c for m, c in zip(keep, curves)}
    return [(m, s, by_id[id(m)]) for m, s in chosen]


n = len(keys)

with plt.rc_context({'text.usetex': True}):
    fig = plt.figure(figsize=(TEXT_WIDTH, min(TEXT_WIDTH * (1.3 * n + 1.2) / 8.4, 0.8 * TEXT_HEIGHT)))
    gs = fig.add_gridspec(n, 5, hspace=0.55, wspace=0.5, width_ratios=[1, 1, 1, 1, 0.6])
    # Panel e sits closer to d than the grid's spacing allows, by this
    # fraction of the figure width, now that its unit is in its title rather
    # than in a y-label that collided with d.
    E_SHIFT = 0.025
    med = lambda a: np.median(np.vstack(a), axis=0)

    for i, key in enumerate(keys):
        data = cache[key]
        sub = fits[fits['dataset'] == key]
        signal = D.prepare_trace(data['X'][0], data)
        mask = signal.fit_mask & signal.exclude_mask
        first, last = i == 0, i == n - 1

        # Every panel comes from the spectral AR(p) fit, which describes these
        # recordings best.  Panel d boxes the modes of every trace's fit.  Panels
        # a-d draw the median spectrum and autocorrelation across traces with ONE
        # fit to the median spectrum: its factors in b, its band-limited
        # autocorrelation in c and the components that autocorrelation is made of in d.
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
                for r, (mode, _, _) in enumerate(
                        reported_modes(fp_i, sig, nlags, n_max=N_MODES_SHOWN)):
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
        chosen = reported_modes(fp, median_signal, nlags, n_max=N_MODES_SHOWN)
        shown = [m for m, _, _ in chosen]
        mode_curves = [c for _, _, c in chosen]
        # b, d and e show the same components: the ranks the fit to the median
        # spectrum resolves, boxed over the traces that resolve them.
        n_shown = len(shown)
        lags = np.arange(nlags + 1) / signal.fs
        taus = [t for t in (sub['tau'].median(), sub['tau_relaxation'].median())
                if np.isfinite(t) and t > 0]
        span = float(np.clip(max(6 * max(taus), 12 / signal.fs), 4 / signal.fs, lags[-1]))
        # Only the lags up to the span are drawn and the axes keep their default
        # limits, so lag zero sits at the same place in every row.
        keep = lags <= span

        # -- a: spectrum and its fit -------------------------------------------
        axa = fig.add_subplot(gs[i, 0])
        f_band = signal.freqs[mask]
        axa.loglog(f_band, psd_data, color=DATA_COLOR, lw=1.1, zorder=1)
        axa.loglog(f_band, psd_fit[mask], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, zorder=3)
        # The fit's noise floor, where it has one.  A negative real pole is a
        # high-pass factor, which is how an all-pole model represents a flat
        # floor added under a falling spectrum; it has a level, not a
        # timescale, so it is drawn here and excluded from b, d and e.
        floor = fp.modes.noise_floor(signal.freqs[mask], offset=fp.offset)
        if floor is not None and floor['r2'] >= FLOOR_MIN_R2:
            axa.axhline(floor['level'], color=FLOOR_COLOR, lw=0.8, ls=(0, (1, 1.5)),
                        zorder=2)
        axa.set_ylabel(data['panel_name'].replace(' ', '\n'), labelpad=2, fontsize=6.5)
        axa.set_yticks([])
        axa.yaxis.set_minor_locator(mticker.NullLocator())
        axa.xaxis.set_minor_locator(mticker.NullLocator())
        if first:
            axa.set_title('PSD fit')
            P.panel_label(axa, 'a', dx=-0.3, dy=1.45, fontsize=9)
        if last:
            axa.set_xlabel('Frequency (Hz)')

        # -- b: the factors the fitted spectrum is a product of ----------------
        # Each mode's own factor 1/|1 - lambda z^-1|^2 (a conjugate pair as one
        # factor), as in fig_modes b: modes multiply in the spectrum, so each is
        # drawn relative to its geometric mean over the band.  The same three
        # modes, in the same colours, are the additive components of d.
        axp = fig.add_subplot(gs[i, 1], sharex=axa)
        centre = lambda power: power / np.exp(np.mean(np.log(power)))
        for r, mode in enumerate(shown):
            axp.loglog(f_band, centre(mode.psd(f_band)), color=RANK_COLORS[r], lw=1.3, zorder=2)
        axp.set_yticks([])
        axp.yaxis.set_minor_locator(mticker.NullLocator())
        axp.xaxis.set_minor_locator(mticker.NullLocator())
        if first:
            axp.set_title('PSD\ncomponents')
            P.panel_label(axp, 'b', dx=-0.2, dy=1.45, fontsize=9)
        if last:
            axp.set_xlabel('Frequency (Hz)')

        # -- c: autocorrelation and the fit's autocorrelation --------------------
        axb = fig.add_subplot(gs[i, 2])
        axb.plot(lags[keep], acf_data[keep], color=DATA_COLOR, lw=1.1, zorder=1)
        axb.plot(lags[keep], acf_fit[keep], color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, zorder=3)
        axb.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
        axb.xaxis.set_major_locator(mticker.MaxNLocator(2))
        axb.set_yticks([0, 1])
        if first:
            axb.set_title('ACF')
            P.panel_label(axb, 'c', dx=-0.3, dy=1.45, fontsize=9)
        if last:
            axb.set_xlabel('Lag (s)')

        # -- d: the modes that autocorrelation is made of, sharing c's axes ------
        axc = fig.add_subplot(gs[i, 3], sharex=axb, sharey=axb)
        for r, curve in enumerate(mode_curves):
            axc.plot(lags[keep], curve[keep], color=RANK_COLORS[r], lw=1.3, zorder=2)
        axc.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
        axc.tick_params(labelleft=False)
        if first:
            axc.set_title('ACF\ncomponents')
            P.panel_label(axc, 'd', dx=-0.2, dy=1.45, fontsize=9)
        if last:
            axc.set_xlabel('Lag (s)')

        # -- e: the timescale of each component, over all traces -----------------
        ax = fig.add_subplot(gs[i, 4])
        pos = ax.get_position()
        ax.set_position([pos.x0 - E_SHIFT, pos.y0, pos.width, pos.height])
        # Whiskers span the 10th to 90th percentile, and the axis is set from
        # what is drawn -- a range set from every value, outliers included,
        # left the lower half of some rows empty.
        drawn = []
        for r in range(n_shown):
            vals = np.array([v for v in mode_taus[r] if np.isfinite(v) and v > 0])
            if not len(vals):
                continue
            drawn.append(np.percentile(vals, [10, 90]))
            colour = RANK_COLORS[r]
            bp = ax.boxplot([vals], positions=[r], widths=0.6, whis=(10, 90),
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='k', lw=1.2),
                            whiskerprops=dict(color=colour, lw=0.9),
                            capprops=dict(color=colour, lw=0.9))
            for box in bp['boxes']:
                box.set(facecolor=colour, alpha=0.55, lw=0.7, edgecolor=colour)
        ax.set_yscale('log')
        ax.set_xlim(-0.7, N_MODES_SHOWN - 0.3)
        ax.set_xticks(np.arange(N_MODES_SHOWN))
        ax.set_xticklabels([f'{r + 1}' for r in range(N_MODES_SHOWN)] if last else [])
        if drawn:
            decade_ticks(ax, np.concatenate(drawn), pad=1.3)
        if first:
            ax.set_title('Timescale\n(ms)')
            P.panel_label(ax, 'e', dx=-0.45, dy=1.45, fontsize=9)
        if last:
            ax.set_xlabel('Component')

    handles = ([proxy(color=DATA_COLOR, lw=1.2, label='data'),
                proxy(color=FIT_COLOR, lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit')]
            + [proxy(color=RANK_COLORS[r], lw=1.4, label=f'component {r + 1}')
                for r in range(N_MODES_SHOWN)]
            + [proxy(color=FLOOR_COLOR, lw=0.9, ls=(0, (1, 1.5)),
                     label='noise floor')])
    fig.subplots_adjust(bottom=0.085)
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.0),
            handlelength=1.5, columnspacing=1.0)

    save(fig, 'fig_realdata.svg')

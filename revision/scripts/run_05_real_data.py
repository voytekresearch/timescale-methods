"""Stage 3: apply the framework across species, modalities and oscillations.

Every dataset receives the identical pipeline: resample to 250 Hz, one Welch PSD
and one ACF per continuous trace, then ACF-exponential, AR(1)-direct, AR(1)-PSD
and order-selected AR(p) fits, followed by the mode decomposition.

Outputs
-------
results/real_data_fits.csv, results/real_data_summary.csv,
results/real_data_modes.csv
figures/fig_realdata.svg
"""

import os
import sys
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from tsmodes import plotting as P
from tsmodes import datasets as D
from tsmodes.benchmark import flatten_fit
from tsmodes.estimators import ESTIMATOR_LABELS, estimate_all
from tsmodes.modes import ar_acf, ar_acf_fft, ar_psd

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'
METHODS = ['ar1_direct', 'arp_direct', 'ar1_acf', 'arp_acf',
           'ar1_psd', 'arp_psd']
DOMINANT_TOL = 0.1   # keep modes until log-spectral R^2 >= 0.99


def resolvable_modes(fit, signal):
    """The fit's modes, most important first, dropping unresolvable ones.

    A mode slower than ``1 / (2 pi f_lo)`` has its knee below the fitted
    band, so its timescale is an extrapolation of the model past the data
    rather than something the recording measured.  Left in, these are the
    near-unit-root poles that report a 393 s timescale from a 300 s
    recording; they carry high importance precisely because deleting them
    changes the lowest frequencies a lot.
    """
    if fit.modes is None:
        return []
    tau_max = 1.0 / (2.0 * np.pi * signal.f_range[0])
    keep = [m for m in fit.modes.modes if m.tau <= tau_max]
    return sorted(keep, key=lambda m: -m.importance)


def in_band(freq, data, signal):
    """Is this mode a rhythm in the band the recording is expected to show?

    A mode whose frequency falls outside the fitted range is an
    extrapolation of the model past the data, so it cannot count as a
    detected rhythm however well it sits inside the nominal band.
    """
    lo, hi = data['band_hz']
    f_lo, f_hi = signal.f_range
    return bool(lo <= freq <= hi and f_lo <= freq <= f_hi)


def analyse_dataset(key):
    data = D.load(key)
    rows, mode_rows = [], []
    for i, trace in enumerate(data['X']):
        signal = D.prepare_trace(trace, data, name=f'{key}[{i}]')
        fits = estimate_all(signal, methods=METHODS, robust=True)
        for fit in fits.values():
            row = flatten_fit(fit, dominant_tol=DOMINANT_TOL)
            row.update({'dataset': key, 'label': data['label'], 'trace': i,
                        'fs': data['fs'], 'duration': signal.duration,
                        'short': data['short'],
                        'species': data['meta'].get('species'),
                        'modality': data['meta'].get('modality'),
                        'band_lo': data['band_hz'][0],
                        'band_hi': data['band_hz'][1]})
            rows.append(row)
            if fit.modes is not None and fit.method in ('arp_psd', 'arp_acf'):
                dom = fit.modes.dominant_subset(tol=DOMINANT_TOL)
                if fit.method == 'arp_psd':
                    curve = dom['r2_curve']
                    for level in (0.90, 0.95, 0.99):
                        hit = np.where(curve >= level)[0]
                        rows[-1][f'n_modes_r2_{int(level*100)}'] = (
                            int(hit[0]) if len(hit) else len(fit.modes.modes))
                    rows[-1]['stabilized'] = bool(
                        fit.params.get('stabilized', False))
                osc_rank = 0
                for rank, mode in enumerate(fit.modes.modes):
                    if mode.kind == 'oscillatory':
                        osc_rank += 1
                    mode_rows.append({
                        'dataset': key, 'label': data['label'], 'trace': i,
                        'method': fit.method, 'rank': rank,
                        'osc_rank': osc_rank if mode.kind == 'oscillatory' else 0,
                        'kind': mode.kind, 'tau_s': mode.tau,
                        'freq_hz': mode.freq, 'n_cycles': mode.n_cycles,
                        'radius': mode.radius,
                        'importance': mode.importance,
                        'rel_importance': mode.rel_importance,
                        'in_dominant_set': rank in dom['indices'],
                        'in_band': in_band(mode.freq, data, signal)})
    return data, pd.DataFrame(rows), pd.DataFrame(mode_rows)


def summarize_dataset(df, modes):
    out = []
    for key, sub in df.groupby('dataset', sort=False):
        rec = {'dataset': key, 'label': sub['label'].iloc[0],
               'species': sub['species'].iloc[0],
               'modality': sub['modality'].iloc[0],
               'n_traces': sub['trace'].nunique(),
               'duration_s': float(sub['duration'].iloc[0])}
        for method in METHODS:
            m = sub[sub['method'] == method]
            rec[f'tau_{method}'] = float(np.nanmedian(m['tau']))
            rec[f'r2_{method}'] = float(np.nanmedian(m['r2_logpsd']))
            if method.startswith('arp'):
                rec[f'order_{method}'] = float(np.nanmedian(m['order']))
        taus = [rec[f'tau_{m}'] for m in ('ar1_direct', 'ar1_acf', 'ar1_psd')]
        taus = [t for t in taus if np.isfinite(t) and t > 0]
        rec['ar1_tau_spread'] = float(max(taus) / min(taus)) if taus else np.nan
        rec['delta_r2_arp_vs_ar1'] = rec['r2_arp_psd'] - rec['r2_ar1_psd']
        rec['unexplained_ar1'] = 1 - rec['r2_ar1_psd']
        rec['unexplained_arp'] = 1 - rec['r2_arp_psd']
        sub_modes = modes[(modes['dataset'] == key)
                          & (modes['method'] == 'arp_psd')]
        arp = sub[sub['method'] == 'arp_psd']
        rec['n_dominant_modes'] = float(np.nanmedian(arp['n_dominant_modes']))
        for level in (90, 95, 99):
            col = f'n_modes_r2_{level}'
            if col in arp:
                rec[col] = float(np.nanmedian(arp[col]))
        rec['frac_stabilized'] = float(np.nanmean(arp.get('stabilized', np.nan)))
        dom = sub_modes[sub_modes['in_dominant_set']]
        acf_modes = modes[(modes['dataset'] == key)
                          & (modes['method'] == 'arp_acf')
                          & modes['in_band'] & (modes['osc_rank'] > 0)]
        rec['acf_band_mode_freq'] = (float(np.nanmedian(acf_modes['freq_hz']))
                                     if len(acf_modes) else np.nan)
        rec['acf_frac_traces_with_band_mode'] = float(
            acf_modes['trace'].nunique() / max(sub['trace'].nunique(), 1))
        rec['n_dominant_oscillatory'] = float(
            dom.groupby('trace')['kind'].apply(
                lambda s: (s == 'oscillatory').sum()).median())
        in_band = dom[dom['in_band']]
        rec['frac_traces_with_band_mode'] = float(
            in_band['trace'].nunique() / max(sub['trace'].nunique(), 1))
        rec['band_mode_freq'] = float(np.nanmedian(in_band['freq_hz'])) \
            if len(in_band) else np.nan
        rec['band_mode_tau'] = float(np.nanmedian(in_band['tau_s'])) \
            if len(in_band) else np.nan
        # Cycles the in-band mode rings before its envelope decays by 1/e.
        # Under about one, it is a broad shoulder rather than a rhythm.
        rec['band_mode_cycles'] = float(np.nanmedian(in_band['n_cycles'])) \
            if len(in_band) else np.nan
        rec['single_timescale_adequate'] = bool(
            (rec['n_dominant_modes'] <= 1) and (rec['unexplained_ar1'] < 0.02))
        out.append(rec)
    return pd.DataFrame(out)


def _cached(name):
    """Load a cached result table, or None when it has not been computed."""
    path = RES / name
    return pd.read_csv(path) if path.exists() else None


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    keys = D.available()
    print(f'Datasets: {keys}', flush=True)

    # --figures-only replots from the cached tables.  Fitting and plotting are
    # separate costs and only one of them is usually the thing being changed.
    fits = _cached('real_data_fits.csv') if '--figures-only' in sys.argv else None
    if fits is not None:
        print('Loading cached fits (omit --figures-only to refit)', flush=True)
        modes = _cached('real_data_modes.csv')
        cache = {k: D.load(k) for k in keys}
    else:
        frames, mode_frames, cache = [], [], {}
        for key in keys:
            print(f'  {key} ...', flush=True)
            data, df, mode_rows = analyse_dataset(key)
            frames.append(df)
            mode_frames.append(mode_rows)
            cache[key] = data
        fits = pd.concat(frames, ignore_index=True)
        modes = pd.concat(mode_frames, ignore_index=True)
        fits.to_csv(RES / 'real_data_fits.csv', index=False)
        modes.to_csv(RES / 'real_data_modes.csv', index=False)

    summary = summarize_dataset(fits, modes)
    summary.to_csv(RES / 'real_data_summary.csv', index=False)
    cols = ['dataset', 'tau_ar1_direct', 'tau_ar1_acf', 'tau_ar1_psd',
            'ar1_tau_spread', 'r2_ar1_psd', 'r2_arp_psd', 'order_arp_psd',
            'n_modes_r2_90', 'n_modes_r2_95', 'n_dominant_modes',
            'band_mode_freq', 'band_mode_tau', 'band_mode_cycles',
            'single_timescale_adequate']
    with pd.option_context('display.width', 200):
        print(summary[cols].round(4).to_string(index=False))

    make_figure(keys, cache, fits, modes, summary)


#: Two hues, one per model order, distinguishable to every kind of colour
#: vision and to a greyscale printer.  Domain is carried by the group labels
#: under panel c, not by colour: light and dark shades of one hue are hard to
#: separate when two curves are overlaid.
DATA_COLOR = '#000000'
FIT_COLOR = '#0072B2'          # the AR(p) fit, in whichever domain it is drawn
FIT_STYLE = (0, (4, 2))
FIT_LW = 1.5
# One colour per mode, by importance rank.  The same three colours carry the
# same three modes through panel b, where their autocorrelation components are
# drawn, and panel c, where their timescales are boxed -- so a curve in b can
# be read straight across to its box in c.
MODE_COLORS = ['#D55E00', '#009E73', '#CC79A7']
N_MODES_SHOWN = 3
#: Draw a mode's box only when this fraction of traces reached that rank.
#: A box built from one trace is not a distribution.
MIN_TRACE_FRACTION = 0.5


def decade_ticks(ax, vals, pad=1.7):
    """Put two or three labelled ticks on a log axis, whatever the span.

    Within one recording the estimators can agree to within a factor of two or
    disagree by two decades, so a fixed locator either crowds the axis or
    leaves it blank.  Step down through progressively finer 1-2-5 ladders and
    take the first that lands at least two ticks inside the data range, so the
    spacing between boxes is always readable off the axis.
    """
    lo = float(np.min(vals)) / pad
    hi = float(np.max(vals)) * pad
    ax.set_ylim(lo, hi)
    best = None
    for mant in ([1.0], [1.0, 3.0], [1.0, 2.0, 5.0],
                 [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]):
        ticks = np.array(sorted(m * 10.0 ** e
                                for e in range(-9, 10) for m in mant))
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        if len(ticks) > 3:
            ticks = ticks[::int(np.ceil(len(ticks) / 3.0))]
        # Three labelled ticks read better than two; keep looking through the
        # finer ladders until one gives three, and fall back on the best of
        # the coarser ones if none does.
        if best is None or len(ticks) > len(best):
            best = ticks
        if len(best) >= 3:
            break
    if best is not None and len(best) >= 2:
        ax.set_yticks(best)
        ax.set_yticklabels([f'{t:g}' for t in best])
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def make_figure(keys, cache, fits, modes, summary):
    n = len(keys)
    fig = plt.figure(figsize=(6.4, 1.15 * n + 0.9))
    gs = fig.add_gridspec(n, 3, hspace=0.45, wspace=0.42)
    from tsmodes.estimators import estimate

    for i, key in enumerate(keys):
        data = cache[key]
        sub = fits[fits['dataset'] == key]
        signal = D.prepare_trace(data['X'][0], data)
        mask = signal.fit_mask & signal.exclude_mask
        first, last = i == 0, i == n - 1

        # Median across traces, matching panel c, which pools them all.
        psd_data, acf_data = [], []
        psd_fit, acf_fit = [], []
        # Mode components and timescales, collected by importance rank so the
        # curve in b and the box in c describe the same mode.
        mode_curves = [[] for _ in range(N_MODES_SHOWN)]
        mode_taus = [[] for _ in range(N_MODES_SHOWN)]
        nlags = len(signal.corrs) - 1
        n_traces = 0
        for trace in data['X']:
            sig = D.prepare_trace(trace, data)
            if len(sig.corrs) - 1 != nlags:
                continue
            n_traces += 1
            psd_data.append(sig.powers[mask])
            acf_data.append(sig.corrs)

            fp = estimate(sig, 'arp_psd', robust=True)
            if fp.success and fp.phi is not None:
                psd_fit.append(ar_psd(fp.phi, sig.freqs[mask], sig.fs, fp.offset))

            fa = estimate(sig, 'arp_acf', robust=True)
            if not (fa.success and fa.phi is not None):
                continue
            height = fa.params['height']
            acf_fit.append(height * ar_acf_fft(fa.phi, nlags)
                           + fa.params['acf_offset'])
            comps = fa.modes.acf_components(nlags)
            index = {id(m): j for j, m in enumerate(fa.modes.modes)}
            for r, mode in enumerate(resolvable_modes(fa, sig)[:N_MODES_SHOWN]):
                mode_curves[r].append(height * comps[index[id(mode)]])
                mode_taus[r].append(mode.tau * 1e3)
        med = lambda a: np.median(np.vstack(a), axis=0)
        psd_data, acf_data = med(psd_data), med(acf_data)
        enough = lambda a: len(a) >= MIN_TRACE_FRACTION * max(n_traces, 1)

        # -- a: spectrum ---------------------------------------------------
        ax = fig.add_subplot(gs[i, 0])
        ax.loglog(signal.freqs[mask], psd_data, color=DATA_COLOR, lw=1.1,
                  label='data', zorder=1)
        if psd_fit:
            ax.loglog(signal.freqs[mask], med(psd_fit), color=FIT_COLOR,
                      lw=FIT_LW, ls=FIT_STYLE, label='AR($p$) fit', zorder=3)
        ax.set_ylabel(data['panel_name'], fontsize=7, labelpad=3)
        ax.set_yticks([])
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(labelsize=6.5)
        if first:
            ax.set_title('Power spectrum', fontsize=9)
            ax.legend(fontsize=5.8, loc='lower left', handlelength=1.8,
                      frameon=True, framealpha=1.0, edgecolor='none',
                      borderpad=0.2, labelspacing=0.2)
            P.panel_label(ax, 'a', dx=-0.24, dy=1.40)
        if last:
            ax.set_xlabel('Frequency (Hz)', fontsize=7.5)

        # -- b: autocorrelation --------------------------------------------
        ax = fig.add_subplot(gs[i, 1])
        lags = np.arange(nlags + 1) / signal.fs
        ax.plot(lags, acf_data, color=DATA_COLOR, lw=1.1, label='data', zorder=1)
        # The modes the fit is made of, drawn under it: each is an exponential
        # decay or a damped cosine, and together with the modes too small to
        # draw they add up to the dashed curve.
        for r in range(N_MODES_SHOWN):
            if not enough(mode_curves[r]):
                continue
            ax.plot(lags, med(mode_curves[r]), color=MODE_COLORS[r], lw=1.2,
                    zorder=2)
        if acf_fit:
            ax.plot(lags, med(acf_fit), color=FIT_COLOR, lw=FIT_LW,
                    ls=FIT_STYLE, zorder=3)
        ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
        taus = [t for t in (sub['tau'].median(), sub['tau_relaxation'].median())
                if np.isfinite(t) and t > 0]
        span = float(np.clip(max(6 * max(taus), 12 / signal.fs),
                             4 / signal.fs, lags[-1]))
        ax.set_xlim(0, span)
        lo = float(np.min(acf_data[lags <= span]))
        ax.set_ylim(min(-0.25, 1.15 * lo), 1.05)
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=6.5)
        if first:
            from matplotlib.lines import Line2D
            ax.set_title('Autocorrelation', fontsize=9)
            handles = [Line2D([0], [0], color=DATA_COLOR, lw=1.1, label='data')]
            handles += [Line2D([0], [0], color=MODE_COLORS[r], lw=1.2,
                               label=f'mode {r + 1}')
                        for r in range(N_MODES_SHOWN)]
            handles.append(Line2D([0], [0], color=FIT_COLOR, lw=FIT_LW,
                                  ls=FIT_STYLE, label='AR($p$) fit'))
            ax.legend(handles=handles, fontsize=5.5, loc='upper right',
                      handlelength=1.8, frameon=True, framealpha=1.0,
                      edgecolor='none', borderpad=0.2, labelspacing=0.18)
            P.panel_label(ax, 'b', dx=-0.22, dy=1.40)
        if last:
            ax.set_xlabel('Lag (s)', fontsize=7.5)

        # -- c: the timescale each estimator reports ------------------------
        # A log axis, because within one recording the estimators can differ
        # by two orders of magnitude and every box has to be visible.
        ax = fig.add_subplot(gs[i, 2])
        drawn = []
        for r in range(N_MODES_SHOWN):
            vals = np.array([v for v in mode_taus[r] if np.isfinite(v) and v > 0])
            if not enough(vals):
                continue
            drawn.append(vals)
            colour = MODE_COLORS[r]
            bp = ax.boxplot([vals], positions=[r], widths=0.6,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='k', lw=1.1),
                            whiskerprops=dict(color=colour, lw=0.8),
                            capprops=dict(color=colour, lw=0.8))
            for box in bp['boxes']:
                box.set(facecolor=colour, alpha=0.55, lw=0.7, edgecolor=colour)
        ax.set_yscale('log')
        ax.set_xlim(-0.7, N_MODES_SHOWN - 0.3)
        ax.set_xticks(np.arange(N_MODES_SHOWN))
        ax.set_xticklabels([f'mode {r + 1}' for r in range(N_MODES_SHOWN)]
                           if last else [], fontsize=5.6, rotation=90)
        ax.set_ylabel('ms', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6.5)
        if drawn:
            decade_ticks(ax, np.concatenate(drawn))
        if first:
            ax.set_title('Mode timescales', fontsize=9)
            P.panel_label(ax, 'c', dx=-0.30, dy=1.40)
        if last:
            ax.text(0.5, -0.92, 'by importance', ha='center', va='top',
                    fontsize=6.5, color=P.GREY,
                    transform=ax.transAxes)

    P.save(fig, FIG / 'fig_realdata.svg')


if __name__ == '__main__':
    main()

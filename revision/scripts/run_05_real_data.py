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
                        'in_band': (data['band_hz'][0] <= mode.freq
                                    <= data['band_hz'][1])})
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


def make_figure(keys, cache, fits, modes, summary):
    n = len(keys)
    fig = plt.figure(figsize=(7.4, 1.12 * n + 0.8))
    gs = fig.add_gridspec(n, 4, hspace=0.42, wspace=0.46,
                          width_ratios=[1.0, 1.0, 1.05, 1.0])
    from tsmodes.estimators import estimate

    for i, key in enumerate(keys):
        data = cache[key]
        sub = fits[fits['dataset'] == key]
        band = (sub['band_lo'].iloc[0], sub['band_hi'].iloc[0])
        signal = D.prepare_trace(data['X'][0], data)
        fit1 = estimate(signal, 'ar1_psd', robust=True)
        fitp = estimate(signal, 'arp_psd', robust=True)
        facf1 = estimate(signal, 'ar1_acf', robust=True)
        facfp = estimate(signal, 'arp_acf', robust=True)
        mask = signal.fit_mask & signal.exclude_mask
        first, last = i == 0, i == n - 1

        # -- a: spectrum ---------------------------------------------------
        ax = fig.add_subplot(gs[i, 0])
        ax.axvspan(band[0], band[1], color='#FFB000', alpha=0.16, lw=0)
        ax.loglog(signal.freqs[mask], signal.powers[mask], color=P.GREY,
                  lw=0.7, label='data')
        ax.loglog(signal.freqs[mask],
                  ar_psd(fit1.phi, signal.freqs[mask], signal.fs, fit1.offset),
                  color=P.METHOD_COLORS['ar1_psd'], lw=1.3, ls='--',
                  label='AR(1)')
        ax.loglog(signal.freqs[mask],
                  ar_psd(fitp.phi, signal.freqs[mask], signal.fs, fitp.offset),
                  color=P.METHOD_COLORS['arp_psd'], lw=1.3, ls='-.',
                  label='AR($p$)')
        import matplotlib.ticker as mticker
        ax.set_ylabel(data['short'], fontsize=7, labelpad=3)
        ax.set_yticks([])
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(labelsize=6.5)
        if first:
            ax.set_title('Power spectrum', fontsize=9)
            ax.legend(fontsize=6, loc='lower left', handlelength=1.4,
                      framealpha=0.85, borderpad=0.25)
            P.panel_label(ax, 'a', dx=-0.30, dy=1.42)
        if last:
            ax.set_xlabel('Frequency (Hz)', fontsize=7.5)

        # -- b: autocorrelation --------------------------------------------
        ax = fig.add_subplot(gs[i, 1])
        nlags = len(signal.corrs) - 1
        lags = np.arange(nlags + 1) / signal.fs
        ax.plot(lags, signal.corrs, color=P.GREY, lw=0.9, label='data')
        # Data solid, both models dashed, so the empirical curve stays
        # visible underneath the fits.
        for fit, method, ls in ((facf1, 'ar1_acf', '--'),
                                (facfp, 'arp_acf', '-.')):
            # The fitted curve itself, not a band-limited redrawing of it: an
            # AR(1) fit must appear on the figure as the decaying exponential
            # that it is.
            model = (fit.params['height'] * ar_acf_fft(fit.phi, nlags)
                     + fit.params['acf_offset'])
            ax.plot(lags, model, color=P.METHOD_COLORS[method], lw=1.3, ls=ls,
                    label='AR(1)' if method == 'ar1_acf' else 'AR($p$)')
        ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
        # Zoom on the decay.  Most of these autocorrelations reach zero within
        # a fraction of the lags computed, and plotting the full range would
        # compress the part of the curve the fit is actually about.
        taus = [t for t in (fit1.tau, facf1.tau, fitp.tau_relaxation,
                            facfp.tau_relaxation) if np.isfinite(t) and t > 0]
        # ... but never so tight that only a handful of lags remain.
        span = float(np.clip(max(6 * max(taus), 12 / signal.fs),
                             4 / signal.fs, lags[-1]))
        ax.set_xlim(0, span)
        lo = float(np.min(signal.corrs[lags <= span]))
        ax.set_ylim(min(-0.25, 1.15 * lo), 1.05)
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=6.5)
        if first:
            ax.set_title('Autocorrelation', fontsize=9)
            ax.legend(fontsize=6, loc='upper right', handlelength=1.4)
            P.panel_label(ax, 'b', dx=-0.24, dy=1.42)
        if last:
            ax.set_xlabel('Lag (s)', fontsize=7.5)

        # -- c: estimates, linear time -------------------------------------
        ax = fig.add_subplot(gs[i, 2])
        medians = {}
        for k, method in enumerate(METHODS):
            col = 'tau' if method.startswith('ar1_') else 'tau_relaxation'
            vals = sub.loc[sub['method'] == method, col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            medians[method] = np.median(vals) if len(vals) else np.nan
        scale, unit = ((1e3, 'ms') if np.nanmedian(list(medians.values())) < 1
                       else (1.0, 's'))
        for k, method in enumerate(METHODS):
            col = 'tau' if method.startswith('ar1_') else 'tau_relaxation'
            vals = sub.loc[sub['method'] == method, col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)] * scale
            x = k + 0.9 * (k // 2)
            if not len(vals):
                ax.text(x, 0.05, 'none', fontsize=5.5, rotation=90,
                        ha='center', va='bottom', color=P.GREY,
                        transform=ax.get_xaxis_transform())
                continue
            jitter = np.random.default_rng(k).uniform(-0.16, 0.16, len(vals))
            ax.scatter(x + jitter, vals, s=6,
                       color=P.METHOD_COLORS[method], alpha=0.75, lw=0)
            ax.plot([x - 0.32, x + 0.32], [np.median(vals)] * 2, color='k',
                    lw=1.3, solid_capstyle='butt')
        finite = [v * scale for v in medians.values() if np.isfinite(v)]
        # A linear axis is far easier to read than a log one, but a single
        # badly mismatched estimator would otherwise flatten it.  The axis is
        # scaled to the typical estimate and off-scale points are marked.
        top = 3.0 * float(np.median(finite)) if finite else 1.0
        ax.set_ylim(0, top)
        for k, method in enumerate(METHODS):
            col = 'tau' if method.startswith('ar1_') else 'tau_relaxation'
            vals = sub.loc[sub['method'] == method, col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)] * scale
            n_over = int(np.sum(vals > top))
            if n_over:
                ax.plot(k + 0.9 * (k // 2), top * 0.97, marker='^', ms=3.5,
                        color=P.METHOD_COLORS[method], clip_on=False)
        ax.locator_params(axis='y', nbins=3)
        ax.set_xlim(-0.8, 7.6)
        ax.set_xticks([0, 1, 2.9, 3.9, 5.8, 6.8])
        ax.set_xticklabels(['AR(1)', 'AR($p$)'] * 3 if last else [],
                           fontsize=5.6, rotation=90)
        ax.set_ylabel(unit, fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6.5)
        if first:
            ax.set_title('Estimated $\\tau$', fontsize=9)
            P.panel_label(ax, 'c', dx=-0.28, dy=1.42)
        if last:
            for x, (dom, name) in zip([0.5, 3.4, 6.3],
                                      [('time', 'time'), ('acf', 'ACF'),
                                       ('psd', 'PSD')]):
                ax.text(x, -1.05, name, ha='center', va='top', fontsize=6.5,
                        color=P.DOMAIN_COLORS[dom],
                        transform=ax.get_xaxis_transform())

        # -- d: oscillatory modes -------------------------------------------
        ax = fig.add_subplot(gs[i, 3])
        ax.axvspan(band[0], band[1], color='#FFB000', alpha=0.16, lw=0)
        # The three most important rhythmic modes per trace, from each of the
        # two AR(p) fits.  Showing every mode buries the rhythm in the
        # high-frequency modes that describe the recording's filter roll-off.
        for method, colour, off in (('arp_acf', P.METHOD_COLORS['arp_acf'], 0.97),
                                    ('arp_psd', P.METHOD_COLORS['arp_psd'], 1.03)):
            osc = modes[(modes['dataset'] == key)
                        & (modes['method'] == method)
                        & (modes['kind'] == 'oscillatory')
                        & (modes['osc_rank'] <= 3)]
            if not len(osc):
                continue
            ax.vlines(osc['freq_hz'] * off, 0, osc['importance'],
                      color=colour, lw=1.0, alpha=0.6)
            ax.scatter(osc['freq_hz'] * off, osc['importance'], s=6,
                       color=colour, lw=0, alpha=0.8,
                       label=('ACF' if method == 'arp_acf' else 'PSD')
                       if first else None)
        ax.set_xscale('log')
        ax.set_xlim(max(0.5 * band[0], data['analysis']['f_range'][0] * 0.8),
                    data['analysis']['f_range'][1])
        ax.set_ylim(bottom=0)
        ax.set_ylabel('importance', fontsize=7)
        ax.tick_params(labelsize=6.5)
        if first:
            ax.set_title('Rhythmic modes', fontsize=9)
            ax.legend(fontsize=6, ncol=2, loc='lower center',
                      bbox_to_anchor=(0.5, 1.12), handlelength=1.0,
                      columnspacing=0.8, frameon=False)
            P.panel_label(ax, 'd', dx=-0.26, dy=1.42)
        if last:
            ax.set_xlabel('Mode frequency (Hz)', fontsize=7.5)

    P.save(fig, FIG / 'fig_realdata.svg')


def _wrap(label, width=22):
    words, lines, cur = label.split(), [], ''
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f'{cur} {word}'.strip()
    lines.append(cur)
    return '\n'.join(lines)


if __name__ == '__main__':
    main()

"""Supplementary figures: the technical detail kept out of the main text.

Reads the cached result tables written by the stage scripts.
"""

import os
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmodes import plotting as P
from tsmodes.estimators import ESTIMATOR_LABELS
from tsmodes.modes import ARModes, poles_to_ar

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'
FS = 500.0
TAU = 0.05
SNRS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
FREQS = [4.0, 6.0, 10.0, 16.0, 25.0, 40.0]
SHORT = ['AR(1)', 'AR(1) + sine', 'AR(2) oscillator', 'AR(8), 5 modes',
         'Power law']
PROCESS_KEYS = ['ar1', 'ar1_sine', 'ar2_damped', 'arp_modes', 'power_law']


def build_true_model():
    spec = [(0.100, 0.0), (0.050, 10.0), (0.010, 40.0), (0.002, 0.0),
            (0.003, 180.0)]
    poles = []
    for tau, freq in spec:
        r = np.exp(-1.0 / (tau * FS))
        if freq == 0:
            poles.append(r + 0j)
        else:
            theta = 2 * np.pi * freq / FS
            poles += [r * np.exp(1j * theta), r * np.exp(-1j * theta)]
    return ARModes.from_phi(poles_to_ar(np.array(poles)), FS)


def fig_modes_technical():
    """Pole geometry, the additive ACF decomposition, and raw coefficients."""
    model = build_true_model()
    sweep = pd.read_csv(RES / 'modes_vs_coefficients_sweep.csv')
    dom = model.dominant_subset(tol=0.1)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    P.plot_pole_diagram(ax, model, annotate=False)
    for mode in model.modes[:3]:
        lam = mode.poles[0]
        txt = (f'{mode.tau*1e3:.0f} ms' if mode.kind == 'relaxation'
               else f'{mode.tau*1e3:.0f} ms, {mode.freq:.0f} Hz')
        ax.annotate(txt, (lam.real, lam.imag), fontsize=6,
                    color=P.MODE_COLORS[mode.kind], ha='right',
                    textcoords='offset points', xytext=(-8, 6))
    ax.set_title('Poles in the complex plane', fontsize=8.5)
    P.panel_label(ax, 'a', dx=-0.24)

    ax = fig.add_subplot(gs[0, 1])
    nlags = 300
    lags = np.arange(nlags + 1) / FS
    comps = model.acf_components(nlags)
    for mode, comp in zip(model.modes, comps):
        ax.plot(lags, comp, lw=0.9, ls='--', color=P.MODE_COLORS[mode.kind])
    ax.plot(lags, model.acf(nlags), color='k', lw=1.6, label='AR($p$) ACF')
    ax.plot(lags, comps.sum(axis=0), color='#FFB000', lw=1.0, ls=':',
            label='sum of modes')
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Modes add in the autocorrelation', fontsize=8.5)
    ax.legend(fontsize=6.5)
    P.panel_label(ax, 'b', dx=-0.22)

    ax = fig.add_subplot(gs[1, 0])
    names = [(f'{m.tau*1e3:.0f} ms\n{m.freq:.0f} Hz' if m.kind != 'relaxation'
              else f'{m.tau*1e3:.0f} ms\nno rhythm') for m in model.modes]
    ax.bar(np.arange(len(names)), [m.importance for m in model.modes],
           color=[P.MODE_COLORS[m.kind] for m in model.modes], width=0.62)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=6)
    ax.set_ylabel('RMS $\\Delta\\log_{10}$ power\non deletion')
    ax.set_title('Mode importance', fontsize=8.5)
    P.panel_label(ax, 'c', dx=-0.24)

    ax = fig.add_subplot(gs[1, 1])
    n_phi = sum(1 for c in sweep.columns if c.startswith('phi'))
    for i in range(1, n_phi + 1):
        ax.plot(sweep['f_osc'], sweep[f'phi{i}'], lw=0.9,
                label=(f'$\\varphi_{i}$' if i <= 3 else None))
    ax.axhline(1.0, color='k', ls=':', lw=0.8)
    ax.set_xlabel('Constructed oscillation frequency (Hz)')
    ax.set_ylabel('AR coefficient $\\varphi_k$')
    ax.set_title('Coefficients carry no direct meaning', fontsize=8.5)
    ax.legend(ncol=3, fontsize=6.5, loc='lower center')
    P.panel_label(ax, 'd', dx=-0.22)

    P.save(fig, FIG / 'figS_modes_technical.svg')


def fig_modes_two_domains():
    """How the modes of an AR(p) combine, and how differently in each domain.

    The same five modes multiply in the spectrum and add in the
    autocorrelation.  Both statements are exact and they are not the same
    decomposition, which is the thing that makes higher-order models
    confusing to read.
    """
    model = build_true_model()
    freqs = np.logspace(np.log10(0.5), np.log10(FS / 2), 600)
    nlags = 250
    lags = np.arange(nlags + 1) / FS
    colours = [P.MODE_COLORS[m.kind] for m in model.modes]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.34)

    # (a) the spectrum: a product of one factor per mode -------------------
    ax = fig.add_subplot(gs[0, 0])
    ref = np.exp(np.mean(np.log(model.psd(freqs))))
    for mode, c in zip(model.modes, colours):
        f = mode.psd(freqs)
        ax.loglog(freqs, f / np.exp(np.mean(np.log(f))), lw=0.9, ls='--',
                  color=c)
    ax.loglog(freqs, model.psd(freqs) / ref, color='k', lw=1.8,
              label='AR($p$) spectrum')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Relative power')
    ax.set_title('Spectrum: the modes multiply', fontsize=8.5)
    ax.text(0.03, 0.06, r'$S(f)=\sigma^2\prod_k|1-\lambda_k e^{-i\omega}|^{-2}$',
            transform=ax.transAxes, fontsize=7)
    ax.legend(fontsize=6.5, loc='upper right')
    P.panel_label(ax, 'a', dx=-0.24)

    # (b) the autocorrelation: a sum of one term per mode ------------------
    ax = fig.add_subplot(gs[0, 1])
    comps = model.acf_components(nlags)
    for comp, c in zip(comps, colours):
        ax.plot(lags, comp, lw=0.9, ls='--', color=c)
    ax.plot(lags, model.acf(nlags), color='k', lw=1.8, label='AR($p$) ACF')
    ax.plot(lags, comps.sum(axis=0), color='#FFB000', lw=1.0, ls=':',
            label='sum of modes')
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(0, 0.3)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation: the modes add', fontsize=8.5)
    ax.text(0.03, 0.06, r'$\gamma(k)=\sum_j A_j\lambda_j^{\,k}$',
            transform=ax.transAxes, fontsize=7)
    ax.legend(fontsize=6.5, loc='upper right')
    P.panel_label(ax, 'b', dx=-0.24)

    # (c) the two decompositions are not the same object -------------------
    ax = fig.add_subplot(gs[1, 0])
    mode = model.modes[0]
    n = 1 << 15
    fr = np.fft.rfftfreq(n, d=1.0 / FS)
    own = np.fft.irfft(mode.psd(fr))[: nlags + 1]
    own = own / own[0]
    add = comps[0] / comps[0][0]
    ax.plot(lags, add, color=P.MODE_COLORS[mode.kind], lw=1.6,
            label="its term in the sum")
    ax.plot(lags, own, color='k', lw=1.2, ls='--',
            label="transform of its factor")
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(0, 0.3)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Normalized')
    ax.set_title(f'One mode, two decompositions\n'
                 f'({mode.tau*1e3:.0f} ms, {mode.freq:.0f} Hz)', fontsize=8.5)
    ax.legend(fontsize=6.5, loc='upper right')
    P.panel_label(ax, 'c', dx=-0.24)

    # (d) what to report for each mode ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    dom = model.dominant_subset(tol=0.1)
    keep = set(dom['indices'])
    # Relaxation modes have no frequency, so they get their own column at the
    # left rather than being piled on top of each other at the axis edge.
    relax_x = [0.9, 1.6]
    n_rel = 0
    for i, (mode, c) in enumerate(zip(model.modes, colours)):
        if mode.kind == 'relaxation':
            x = relax_x[min(n_rel, len(relax_x) - 1)]
            n_rel += 1
        else:
            x = mode.freq
        ax.scatter([x], [mode.tau * 1e3], s=30 + 260 * mode.importance,
                   color=c, alpha=0.85 if i in keep else 0.3,
                   edgecolor='k' if i in keep else 'none', lw=0.7, zorder=3)
        txt = (f'{mode.tau*1e3:.0f} ms' if mode.kind == 'relaxation'
               else f'{mode.tau*1e3:.0f} ms, {mode.n_cycles:.1f} cyc')
        ax.annotate(txt, (x, mode.tau * 1e3), fontsize=5.8, ha='left',
                    va='center', textcoords='offset points', xytext=(11, 0))
    ax.axvline(2.6, color=P.LIGHT, lw=0.8, ls=':', zorder=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.6, 1200)
    ax.set_xlabel('Mode frequency (Hz)')
    ax.set_ylabel(r'Mode $\tau$ (ms)')
    ax.set_title(f'What to report: {len(model.modes)} modes,\n'
                 f'{dom["n_modes"]} of them dominant', fontsize=8.5)
    ax.text(0.02, 0.97, 'relaxations\n(no frequency)', transform=ax.transAxes,
            fontsize=5.6, color=P.GREY, va='top')
    ax.text(0.98, 0.97, 'area = importance\noutlined = dominant',
            transform=ax.transAxes, fontsize=6, color=P.GREY, ha='right',
            va='top')
    ax.set_ylim(1, 500)
    P.panel_label(ax, 'd', dx=-0.24)

    P.save(fig, FIG / 'figS_modes_domains.svg')


def fig_population():
    """What AR(p) converges to with infinite data, and how order is chosen."""
    pop = pd.read_csv(RES / 'population_ar.csv')
    df = pd.read_csv(RES / 'benchmark_main.csv.gz')
    orders = pd.read_csv(RES / 'oscillation_population_orders.csv')

    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 3, wspace=0.46)

    # (a) what order finite data actually buys ----------------------------
    ax = fig.add_subplot(gs[0, 0])
    methods = ['arp_direct', 'arp_psd']
    for i, method in enumerate(methods):
        vals, lo, hi = [], [], []
        for key in PROCESS_KEYS:
            o = df[(df['process'] == key) & (df['method'] == method)]['order']
            vals.append(o.median())
            lo.append(o.quantile(0.1))
            hi.append(o.quantile(0.9))
        ax.errorbar(np.arange(len(PROCESS_KEYS)) + (i - 0.5) * 0.24, vals,
                    yerr=[np.array(vals) - np.array(lo),
                          np.array(hi) - np.array(vals)],
                    fmt='o', ms=3, lw=1.0, color=P.METHOD_COLORS[method],
                    label=ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'))
    ax.set_xticks(np.arange(len(PROCESS_KEYS)))
    ax.set_xticklabels(SHORT, rotation=32, ha='right', fontsize=6)
    ax.set_ylabel('Order chosen by BIC')
    ax.set_title('Order selected from finite data', fontsize=8.5)
    ax.legend(fontsize=5.8, loc='center left', bbox_to_anchor=(0.02, 0.62))
    P.panel_label(ax, 'a', dx=-0.30)

    # (b) what order would be needed with unlimited data ------------------
    # AR(1) and the AR(2) oscillator are omitted: an AR(p) model contains
    # them exactly, so their divergence falls to numerical zero at p = 1 and
    # p = 2 and the curve is a vertical drop that carries no information.
    ax = fig.add_subplot(gs[0, 1])
    exact = {'ar1', 'ar2_damped', 'arp_modes'}
    keys = [k for k in PROCESS_KEYS if k not in exact]
    labels = [t for k, t in zip(PROCESS_KEYS, SHORT) if k not in exact]
    colours = plt.cm.viridis(np.linspace(0, 0.82, len(keys)))
    for key, label, colour in zip(keys, labels, colours):
        sub = pop[pop['process'] == key]
        ax.semilogy(sub['order'], sub['is_div'], color=colour, lw=1.1,
                    label=label)
    ax.set_xlabel('AR order $p$')
    ax.set_ylabel('Itakura\u2013Saito divergence\nfrom the true spectrum',
                  fontsize=7)
    ax.set_title('Order needed with infinite data', fontsize=8.5)
    ax.legend(fontsize=5.8, loc='upper right')
    P.panel_label(ax, 'b', dx=-0.34)

    # (c) --------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    firsts = orders.groupby('f_osc')['first_order_with_osc'].first()
    ax.plot(firsts.index, firsts.values, 'o-', color='k', lw=1.1, ms=4)
    ax.axvline(1 / (2 * np.pi * TAU), color=P.GREY, ls=':', lw=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('Oscillation frequency (Hz)')
    ax.set_ylabel('Smallest order with\nan oscillatory mode')
    ax.set_title('Cost of an added rhythm', fontsize=8.5)
    P.panel_label(ax, 'c', dx=-0.36)

    P.save(fig, FIG / 'figS_population.svg')


def fig_oscillation_grid():
    """Full frequency x amplitude grid for the additive-sinusoid simulation."""
    sweep = pd.read_csv(RES / 'oscillation_sweep.csv.gz')
    show = ['ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']

    fig, axes = plt.subplots(1, len(show), figsize=(7.4, 2.1))
    for ax, method in zip(axes, show):
        grid = np.full((len(SNRS), len(FREQS)), np.nan)
        for i, snr in enumerate(SNRS):
            for j, f_osc in enumerate(FREQS):
                sel = ((sweep['method'] == method)
                       & np.isclose(sweep['param_snr'], snr)
                       & np.isclose(sweep['param_f_osc'], f_osc))
                vals = sweep.loc[sel, 'tau_relaxation'].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    grid[i, j] = np.median(vals) / TAU
        # Colour and printed number are the same quantity in the same units:
        # the ratio of the estimate to the truth.  The colour scale is spaced
        # logarithmically so that a factor of two too big and a factor of two
        # too small are equally far from white.
        im = ax.imshow(np.log10(grid), origin='lower', cmap='RdBu_r',
                       vmin=-1, vmax=1, aspect='auto')
        for i in range(len(SNRS)):
            for j in range(len(FREQS)):
                value = grid[i, j]
                txt = f'{value:.2f}' if np.isfinite(value) else 'none'
                ax.text(j, i, txt, ha='center', va='center', fontsize=4.4,
                        color='k' if abs(np.log10(max(value, 1e-9))) < 0.6
                        else 'white')
        ax.set_xticks(range(len(FREQS)))
        ax.set_xticklabels([f'{f:g}' for f in FREQS], fontsize=6)
        ax.set_yticks(range(len(SNRS)))
        ax.set_yticklabels([f'{v:g}' for v in SNRS] if ax is axes[0] else [],
                           fontsize=6)
        ax.set_xlabel('Oscillation freq. (Hz)', fontsize=7)
        ax.set_title(ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'),
                     fontsize=8)
    axes[0].set_ylabel('Oscillation /\naperiodic power', fontsize=7)
    ticks = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                      ticks=np.log10(ticks))
    cb.ax.set_yticklabels([f'{t:g}' for t in ticks], fontsize=6)
    cb.set_label('estimated $\\tau$ / true $\\tau$', fontsize=7)
    P.save(fig, FIG / 'figS_oscillation_grid.svg')


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    fig_modes_technical()
    fig_modes_two_domains()
    fig_population()
    fig_oscillation_grid()


if __name__ == '__main__':
    main()

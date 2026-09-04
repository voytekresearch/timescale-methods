"""Stage 1: make AR(p) interpretable through its poles/modes.

Outputs
-------
figures/fig_modes.svg    mode gallery, spectrum with the fit, importance,
                         and recovery against reading the coefficients
results/modes_*.csv      the numbers quoted in the manuscript
"""
import sys

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmodes.modes import (ARModes, ar_impulse_response, ar_psd,
                           ar_to_poles, pole_residues, poles_to_ar,
                           psd_from_poles)
from tsmodes.estimators import prepare, estimate, population_ar
from tsmodes.sims import sim_ar_process
from tsmodes import plotting as P

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
RES = ROOT / 'results'
FS = 500.0

# A ground-truth AR(5): one slow relaxation plus two damped oscillators.
TRUE_MODES = [
    {'kind': 'relaxation', 'tau': 0.100, 'freq': 0.0},
    {'kind': 'oscillatory', 'tau': 0.050, 'freq': 10.0},
    {'kind': 'oscillatory', 'tau': 0.010, 'freq': 40.0},
    {'kind': 'relaxation', 'tau': 0.002, 'freq': 0.0},
    {'kind': 'oscillatory', 'tau': 0.003, 'freq': 180.0},
]


def build_poles(spec, fs):
    poles = []
    for m in spec:
        r = np.exp(-1.0 / (m['tau'] * fs))
        if m['kind'] == 'relaxation':
            poles.append(r + 0j)
        else:
            theta = 2 * np.pi * m['freq'] / fs
            poles += [r * np.exp(1j * theta), r * np.exp(-1j * theta)]
    return np.array(poles)


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    poles_true = build_poles(TRUE_MODES, FS)
    phi_true = poles_to_ar(poles_true)
    freqs = np.logspace(np.log10(0.5), np.log10(FS / 2), 600)
    model = ARModes.from_phi(phi_true, FS, offset=1.0,
                             freqs=np.linspace(0, FS / 2, 1025))

    # ---------------------------------------------------------------- checks
    direct = ar_psd(phi_true, freqs, FS, 1.0)
    factored = psd_from_poles(poles_true, freqs, FS, 1.0)
    max_rel = float(np.max(np.abs(direct - factored) / direct))
    phi_round = poles_to_ar(ar_to_poles(phi_true))
    checks = {
        'psd_reconstruction_max_rel_error': max_rel,
        'phi_roundtrip_max_abs_error': float(np.max(np.abs(phi_round - phi_true))),
        'order': int(len(phi_true)),
        'n_modes': int(len(model.modes)),
        'phi_true': [float(v) for v in phi_true],
    }

    # Recovered mode parameters vs the construction.
    rows = []
    for m in model.modes:
        rows.append({'kind': m.kind, 'tau_s': m.tau, 'freq_hz': m.freq,
                     'n_cycles': m.n_cycles,
                     'radius': m.radius, 'importance': m.importance,
                     'importance_is': m.importance_is,
                     'rel_importance': m.rel_importance})
    mode_table = pd.DataFrame(rows)
    mode_table.to_csv(RES / 'modes_ground_truth_table.csv', index=False)

    # Variance decomposition: why "percent variance" is not available.
    var_dec = model.variance_decomposition()
    checks['cross_mode_variance_fraction'] = var_dec['cross_fraction']
    checks['within_mode_variance_fractions'] = [
        float(v / var_dec['total_variance']) for v in var_dec['within_mode']]

    # Additive (impulse-response) decomposition.
    n_psi = 400
    psi = ar_impulse_response(phi_true, n_psi)
    poles_est = ar_to_poles(phi_true)
    residues = pole_residues(poles_est)
    psi_modes = np.real(np.sum(
        residues[:, None] * poles_est[:, None] ** np.arange(n_psi)[None, :], axis=0))
    checks['impulse_response_max_error'] = float(np.max(np.abs(psi - psi_modes)))
    acf_comps = model.acf_components(300)
    checks['acf_component_sum_max_error'] = float(np.max(np.abs(
        acf_comps.sum(axis=0) - model.acf(300))))

    dom = model.dominant_subset(tol=0.1)
    checks['dominant_subset'] = {
        'criterion': 'smallest set with log-spectral R^2 >= 0.99',
        'n_modes': dom['n_modes'],
        'kept_modes': [{'tau_s': model.modes[i].tau, 'freq_hz': model.modes[i].freq}
                       for i in dom['indices']],
        'error': dom['error'],
        'curve': [float(v) for v in dom['curve']],
        'r2_curve': [float(v) for v in dom['r2_curve']]}

    # ------------------------------------------- estimation from a realization
    sig = sim_ar_process(phi_true, int(FS * 300), rng)
    signal = prepare(sig, FS, welch_win_s=4.0)
    fit = estimate(signal, 'arp_psd', robust=True)  # the process has rhythms
    est_rows = [{'kind': m.kind, 'tau_s': m.tau, 'freq_hz': m.freq,
                 'n_cycles': m.n_cycles, 'importance': m.importance}
                for m in fit.modes.modes]
    pd.DataFrame(est_rows).to_csv(RES / 'modes_estimated_table.csv', index=False)
    checks['estimated_order'] = int(fit.order)
    checks['estimated_r2_logpsd'] = float(fit.gof['r2_logpsd'])

    # ------------------------------------ modes vs raw coefficients: stability
    # --figures-only replots from the cached tables; the 200-realization
    # repeatability run is the expensive part and rarely the thing changed.
    cached = RES / 'modes_vs_coefficients_repeatability.csv'
    if '--figures-only' in sys.argv and cached.exists():
        print('Loading cached mode results (omit --figures-only to recompute)',
              flush=True)
        sweep = pd.read_csv(RES / 'modes_vs_coefficients_sweep.csv')
        repeat = pd.read_csv(cached)
        make_figure(model, fit, signal, sweep, repeat, dom)
        return

    sweep = coefficient_vs_mode_sweep(FS)
    sweep.to_csv(RES / 'modes_vs_coefficients_sweep.csv', index=False)

    repeat = repeatability(phi_true, FS, n_reps=200, duration=60.0)
    repeat.to_csv(RES / 'modes_vs_coefficients_repeatability.csv', index=False)
    checks['repeatability'] = repeat.to_dict(orient='records')

    with open(RES / 'modes_checks.json', 'w') as handle:
        json.dump(checks, handle, indent=2)
    print(json.dumps({k: v for k, v in checks.items()
                      if k not in ('repeatability',)}, indent=2)[:1500])

    make_figure(model, fit, signal, sweep, repeat, dom)


def coefficient_vs_mode_sweep(fs, freqs_hz=None):
    """Sweep the oscillatory mode's frequency; track phi_k and (tau, f)."""
    freqs_hz = np.linspace(4, 60, 40) if freqs_hz is None else freqs_hz
    rows = []
    for f_osc in freqs_hz:
        spec = [dict(TRUE_MODES[0]), {'kind': 'oscillatory', 'tau': 0.05,
                                      'freq': float(f_osc)},
                dict(TRUE_MODES[2])]
        phi = poles_to_ar(build_poles(spec, fs))
        model = ARModes.from_phi(phi, fs)
        osc = [m for m in model.modes if m.kind == 'oscillatory']
        osc = sorted(osc, key=lambda m: abs(m.freq - f_osc))
        row = {'f_osc': f_osc,
               'mode_tau': osc[0].tau, 'mode_freq': osc[0].freq,
               'mode_importance': osc[0].importance}
        for i, val in enumerate(phi):
            row[f'phi{i+1}'] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def match_modes(fitted, truth_poles):
    """Greedily pair fitted modes to constructed ones by pole location.

    Matching on frequency alone cannot separate the two relaxation modes,
    which both sit at 0 Hz, and matching on tau alone cannot separate modes
    of similar decay at different frequencies.  The pole carries both
    coordinates, so pair on distance in the complex plane.
    """
    avail = list(fitted)
    out = []
    for lam in truth_poles:
        if not avail:
            out.append(None)
            continue
        best = min(avail, key=lambda m: abs(complex(m.poles[0]) - lam))
        avail.remove(best)
        out.append(best)
    return out


def repeatability(phi_true, fs, n_reps=200, duration=60.0):
    """Across-realization recovery of each constructed mode.

    Each of the five modes has a known ``tau``, so every one of them can be
    scored; there is no ground truth for a coefficient read as if it were an
    AR(1), which is why only modes appear here.
    """
    rng = np.random.default_rng(7)
    order = len(phi_true)
    # One representative pole per true mode (the one in the upper half plane).
    truth_poles = [complex(p) for p in build_poles(TRUE_MODES, fs)
                   if p.imag >= 0]
    phis, naive, ar1 = [], [], []
    taus = {i: [] for i in range(len(TRUE_MODES))}
    freqs_ = {i: [] for i in range(len(TRUE_MODES))}
    for _ in range(n_reps):
        sig = sim_ar_process(phi_true, int(fs * duration), rng)
        signal = prepare(sig, fs, welch_win_s=4.0)
        fit = estimate(signal, 'arp_psd', order=order, robust=True)
        phis.append(fit.phi)
        # Kept for the text, not for the figure: neither has a ground truth
        # to be scored against, which is why only modes are plotted.
        naive.append(-1.0 / (fs * np.log(fit.phi[0])) if 0 < fit.phi[0] < 1
                     else np.nan)
        ar1.append(estimate(signal, 'ar1_psd', robust=True).tau)
        paired = match_modes(fit.modes.modes, truth_poles)
        for i, mode in enumerate(paired):
            taus[i].append(mode.tau if mode is not None else np.nan)
            freqs_[i].append(mode.freq if mode is not None else np.nan)
    phis = np.asarray(phis)
    rows = []
    for i in range(order):
        rows.append({'quantity': f'phi_{i+1}', 'mean': float(np.mean(phis[:, i])),
                     'sd': float(np.std(phis[:, i])),
                     'cv': float(np.std(phis[:, i]) / abs(np.mean(phis[:, i])))})
    specs = [('tau_from_phi1_of_arp', np.asarray(naive, dtype=float), 0.10),
             ('tau_from_ar1_psd_fit', np.asarray(ar1, dtype=float), 0.10)]
    for i, spec in enumerate(TRUE_MODES):
        specs.append((f'mode{i}_tau', np.asarray(taus[i], dtype=float), spec['tau']))
        if spec['kind'] != 'relaxation':
            specs.append((f'mode{i}_freq', np.asarray(freqs_[i], dtype=float),
                          spec['freq']))
    for name, vals, truth in specs:
        finite = vals[np.isfinite(vals)]
        rec = {'quantity': name, 'truth': truth,
               'frac_defined': float(np.mean(np.isfinite(vals)))}
        if len(finite):
            rec.update({'mean': float(np.mean(finite)), 'sd': float(np.std(finite)),
                        'cv': float(np.std(finite) / abs(np.mean(finite))),
                        'rel_bias': float(np.mean(finite) / truth - 1),
                        'rel_error': float(np.median(np.abs(finite - truth) / truth))})
        else:
            rec.update({'mean': np.nan, 'sd': np.nan, 'cv': np.nan,
                        'rel_bias': np.nan, 'rel_error': np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


FIT_COLOR = '#009E73'      # AR(p) fit: far from both mode colours
SUM_COLOR = '#FFB000'      # sum-of-modes check in panel c
DOM_STYLE = (0, (2, 1.5))  # dominant mode
NEG_STYLE = ':'            # negligible mode


def _proxy(**kw):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], **kw)


def make_figure(model, fit, signal, sweep, repeat, dom):
    """Five panels: what the modes are, how they combine in each domain, which
    of them matter, and how well each one is recovered."""
    fig = plt.figure(figsize=(7.4, 5.0))
    # Row 1 spans are chosen so that every panel edge in the bottom row falls
    # on a panel edge in the top row: d under a, e under b and c.
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.05)
    freqs = np.logspace(np.log10(0.5), np.log10(250), 600)
    # Slowest mode at the top, fastest at the bottom.
    order = np.argsort([-m.tau for m in model.modes])
    shown = [model.modes[i] for i in order]
    colours = [P.MODE_COLORS[m.kind] for m in shown]
    nlags = 150
    lags = np.arange(nlags + 1) / model.fs
    keep = set(dom['indices'])

    style_handles = [
        _proxy(color=P.MODE_COLORS['relaxation'], lw=1.4, label='relaxation mode'),
        _proxy(color=P.MODE_COLORS['oscillatory'], lw=1.4, label='oscillatory mode'),
    ]
    weight_handles = [
        _proxy(color=P.GREY, lw=1.1, ls=DOM_STYLE, label='mode, dominant'),
        _proxy(color=P.GREY, lw=0.9, ls=NEG_STYLE, alpha=0.6,
               label='mode, negligible'),
    ]

    # (a) the modes themselves -------------------------------------------
    ax = fig.add_subplot(gs[0, 0:2])
    n, step = 125, 1.35
    t = np.arange(n) / model.fs
    comps = model.impulse_components(n)
    total = comps.sum(axis=0)
    top = len(shown) * step
    ax.plot(t, total / np.max(np.abs(total)) + top, color='k', lw=1.3)
    ax.text(t[-1], top + 0.42, 'response to an input', fontsize=6.5,
            ha='right', va='bottom')
    for i, mode in enumerate(shown):
        base = (len(shown) - 1 - i) * step
        y = comps[order[i]] / np.max(np.abs(comps[order[i]])) + base
        ax.axhline(base, color=P.LIGHT, lw=0.5, zorder=0)
        ax.plot(t, y, color=colours[i], lw=1.2)
        label = (f'$\\tau$ = {mode.tau*1e3:.0f} ms'
                 + ('' if mode.kind == 'relaxation'
                    else f', {mode.freq:.0f} Hz ({mode.n_cycles:.1f} cyc)'))
        ax.text(t[-1], base + 0.42, label, fontsize=6.2, ha='right',
                va='bottom', color=colours[i])
    ax.set_yticks([])
    ax.set_ylim(-1.5, top + 1.25)
    ax.set_xlabel('Time after input (s)')
    ax.set_title('Modes', fontsize=9)
    ax.spines['left'].set_visible(False)
    ax.legend(handles=style_handles, fontsize=5.8, loc='lower center', ncol=2,
              handlelength=1.5, borderpad=0.22, labelspacing=0.2,
              columnspacing=1.0, frameon=True, framealpha=1.0,
              edgecolor='none')
    P.panel_label(ax, 'a', dx=-0.09)

    # (b) they multiply in the spectrum ------------------------------------
    ax = fig.add_subplot(gs[0, 2:4])
    mask = signal.fit_mask & signal.exclude_mask
    fmask = signal.freqs[mask]

    def centred(power, ref=None):
        ref = power if ref is None else ref
        return power / np.exp(np.mean(np.log(ref)))

    ax.loglog(fmask, centred(signal.powers[mask]), color=P.LIGHT, lw=1.0,
              zorder=0)
    for j, (mode, c) in enumerate(zip(shown, colours)):
        f = mode.psd(freqs)
        dominant = order[j] in keep
        ax.loglog(freqs, f / np.exp(np.mean(np.log(f))), color=c, zorder=1,
                  lw=1.1 if dominant else 0.9,
                  ls=DOM_STYLE if dominant else NEG_STYLE,
                  alpha=1.0 if dominant else 0.6)
    ax.loglog(freqs, centred(model.psd(freqs), model.psd(fmask)), color='k',
              lw=1.6, zorder=3)
    fitted = ar_psd(fit.phi, freqs, signal.fs, fit.offset)
    ax.loglog(freqs, centred(fitted, ar_psd(fit.phi, fmask, signal.fs,
                                            fit.offset)),
              color=FIT_COLOR, lw=1.2, ls=(0, (4, 2.6)), zorder=4)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Relative power')
    ax.set_title('Spectrum: modes multiply', fontsize=9)
    ax.legend(handles=[
        _proxy(color=P.LIGHT, lw=1.0, label='one realization'),
        _proxy(color='k', lw=1.6, label='generating spectrum'),
        _proxy(color=FIT_COLOR, lw=1.2, ls=(0, (4, 2.6)),
               label=f'AR({fit.order}) fit'),
    ] + weight_handles,
        fontsize=5.5, loc='lower left', handlelength=1.5,
        borderpad=0.22, labelspacing=0.2, frameon=True,
        framealpha=1.0, edgecolor='none')
    P.panel_label(ax, 'b', dx=-0.26)

    # (c) they add in the autocorrelation ----------------------------------
    ax = fig.add_subplot(gs[0, 4:6])
    acf_comps = model.acf_components(nlags)
    for i, c in enumerate(colours):
        dominant = order[i] in keep
        ax.plot(lags, acf_comps[order[i]], color=c,
                lw=1.1 if dominant else 0.9,
                ls=DOM_STYLE if dominant else NEG_STYLE,
                alpha=1.0 if dominant else 0.6)
    ax.plot(lags, model.acf(nlags), color='k', lw=1.6)
    ax.plot(lags, acf_comps.sum(axis=0), color=SUM_COLOR, lw=1.1, ls='--')
    ax.axhline(0, color=P.LIGHT, lw=0.6, zorder=0)
    ax.set_xlim(0, 0.3)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation: modes add', fontsize=9)
    ax.legend(handles=[
        _proxy(color='k', lw=1.6, label='generating ACF'),
        _proxy(color=SUM_COLOR, lw=1.1, ls='--', label='sum of modes'),
    ] + weight_handles,
        fontsize=5.5, loc='upper right', handlelength=1.5,
        borderpad=0.22, labelspacing=0.2, frameon=True,
        framealpha=1.0, edgecolor='none')
    P.panel_label(ax, 'c', dx=-0.26)

    # (d) how many modes are needed ---------------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    curve = dom['r2_curve']
    ax.plot(np.arange(len(curve)), curve, 'o-', color='k', ms=4)
    ax.axhline(0.99, color=P.GREY, ls='--', lw=1.0)
    ax.text(0.1, 1.03, '$R^2 = 0.99$', fontsize=6.2, color=P.GREY, va='bottom')
    ax.axvline(dom['n_modes'], color=P.LIGHT, lw=1.0, zorder=0)
    ax.text(dom['n_modes'] + 0.15, 0.22,
            f"{dom['n_modes']} of {len(shown)}\nsuffice", fontsize=6.5)
    ax.set_xlabel('Modes kept')
    ax.set_ylabel('Spectrum reproduced ($R^2$)', fontsize=7.5)
    ax.set_xticks(np.arange(len(curve)))
    ax.set_ylim(-0.08, 1.18)
    ax.set_title('Mode ranking', fontsize=9)
    ax.text(0.97, 0.06, 'modes added in order\nof importance',
            transform=ax.transAxes, fontsize=5.8, color=P.GREY,
            ha='right', va='bottom')
    P.panel_label(ax, 'd', dx=-0.34)

    # (e) how well each mode's timescale is recovered ----------------------
    # Only quantities with a ground truth appear here, which means modes:
    # each constructed mode has a known tau, and every one is scored.
    ax = fig.add_subplot(gs[1, 2:6])
    idx = sorted(range(len(TRUE_MODES)), key=lambda i: -TRUE_MODES[i]['tau'])
    vals, labels, bar_colors = [], [], []
    for i in idx:
        spec = TRUE_MODES[i]
        row = repeat[repeat['quantity'] == f'mode{i}_tau'].iloc[0]
        vals.append(float(row['rel_error']))
        bar_colors.append(P.MODE_COLORS[spec['kind']])
        labels.append(f"{spec['tau']*1e3:g} ms"
                      + ('' if spec['kind'] == 'relaxation'
                         else f", {spec['freq']:g} Hz"))
    xs = np.arange(len(idx))
    ax.bar(xs, vals, color=bar_colors, width=0.62, edgecolor='white')
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.25, f'{v*100:.0f}%', ha='center', va='bottom',
                fontsize=6)
    ax.set_yscale('log')
    top_v = max(vals)
    ax.set_ylim(min(vals) / 5.0, top_v * 12.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.2, rotation=22, ha='right')
    ax.set_xlabel('Constructed mode', fontsize=7.5)
    ax.set_ylabel('Median error in $\\tau$', fontsize=7.5)
    ax.set_title('Timescale recovery, per mode', fontsize=9)
    ax.legend(handles=style_handles, fontsize=5.8, loc='upper left',
              handlelength=1.5, borderpad=0.22, labelspacing=0.2,
              frameon=True, framealpha=1.0, edgecolor='none')
    P.panel_label(ax, 'e', dx=-0.16)

    P.save(fig, FIG / 'fig_modes.svg')


if __name__ == '__main__':
    main()

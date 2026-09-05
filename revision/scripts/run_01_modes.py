"""Stage 1: make AR(p) interpretable through its poles/modes.

Outputs
-------
results/modes_*.csv      the numbers quoted in the manuscript
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsmodes.modes import (ARModes, ar_impulse_response, ar_psd,
                           ar_to_poles, pole_residues, poles_to_ar,
                           psd_from_poles)
from tsmodes.estimators import prepare, estimate
from tsmodes.sims import ARP_MODES, sim_ar_process

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
FS = 500.0

# A ground-truth AR(7): two relaxations that both matter, two rhythms that
# ring for more than a cycle, and one fast relaxation that does not matter.
#
# Two dominant relaxations are the point.  A single timescale cannot describe
# this process however it is estimated, so it is the case where reading the
# model as modes is not a refinement but the only way to get an answer.
#
# Both rhythms have f*tau > 1, so tau is unambiguously the envelope decay and
# cannot be misread as the period: the 10 Hz mode has a 100 ms period and a
# 150 ms envelope.  Both stay inside the range seen in the real recordings,
# where the largest in-band f*tau is 1.93.
#
# The relaxation knees sit at 1/(2 pi tau) = 0.80 and 8.0 Hz, an order of
# magnitude apart and both well inside the band a 4 s Welch window resolves.
# The same process as every other AR(p) simulation (tsmodes.sims.ARP_MODES).
TRUE_MODES = [{'kind': 'relaxation' if freq == 0 else 'oscillatory',
               'tau': tau, 'freq': freq} for tau, freq in ARP_MODES]


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
    sweep = coefficient_vs_mode_sweep(FS)
    sweep.to_csv(RES / 'modes_vs_coefficients_sweep.csv', index=False)

    repeat = repeatability(phi_true, FS, n_reps=200, duration=60.0)
    repeat.to_csv(RES / 'modes_vs_coefficients_repeatability.csv', index=False)
    checks['repeatability'] = repeat.to_dict(orient='records')

    with open(RES / 'modes_checks.json', 'w') as handle:
        json.dump(checks, handle, indent=2)
    print(json.dumps({k: v for k, v in checks.items()
                      if k not in ('repeatability',)}, indent=2)[:1500])



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


if __name__ == '__main__':
    main()

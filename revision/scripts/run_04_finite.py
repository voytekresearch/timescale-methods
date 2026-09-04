"""Stage 2c: finite recordings bias timescale estimates.

Everything is parameterized by the dimensionless ratio T/tau.  Monte-Carlo
estimates are compared against the closed-form AR(1) sampling distribution
(bias, standard error, confidence interval) from ``tsmodes.finite``, and the
same durations are applied to the non-AR(1) processes to show which of them
converge to a stable timescale and which remain window-dependent.

Outputs
-------
results/finite_sampling.csv.gz, results/finite_summary.csv,
results/finite_analytic.csv
figures/figS_finite.svg
"""

import os
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmodes import plotting as P
from tsmodes.benchmark import run_condition, summarize
from tsmodes.estimators import ESTIMATOR_LABELS
from tsmodes.finite import (ar1_tau_ci, ar1_tau_moments, phi_of_tau,
                            relative_bias_asymptotic, relative_sd_asymptotic,
                            sample_tau_finite, duration_for_relative_sd)
from tsmodes.sims import (AR1, AR1Mixture, AR1PlusSine, AR2Damped, PowerLaw,
                          SumAR1)

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'

FS = 500.0
TAU = 0.05
RATIOS = np.array([10, 20, 40, 100, 200, 400, 1000, 2000, 4000], dtype=float)
DURATIONS = RATIOS * TAU
N_REPS_AR1 = 150
N_REPS_OTHER = 60
N_JOBS = int(os.environ.get('TSMODES_N_JOBS', '12'))
METHODS = ['ar1_direct', 'arp_direct', 'ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']

OTHER = [
    AR1PlusSine(tau=TAU, f_osc=10.0, snr=1.0),
    AR2Damped(tau=TAU, f_osc=10.0),
    SumAR1(tau0=0.01, tau1=0.20, weight=0.5),
    AR1Mixture(tau_min=0.002, tau_max=2.0, n_components=12),
    PowerLaw(d=0.45),
]
OTHER_LABELS = ['AR(1) + sine', 'AR(2) damped', 'Sum of 2 AR(1)',
                'AR(1) mixture', 'Power law']


def prep_for(duration):
    """Analysis settings that scale with the recording.

    The Welch segment is a quarter of the recording and the ACF is taken out to
    a quarter of the recording, so that the lowest analysed frequency is
    ``4/T``.  This is what makes the analysis window, rather than a fixed
    segment length, the thing that changes as recordings are shortened -- the
    situation a shorter experiment actually puts an analyst in.
    """
    return {'welch_win_s': duration / 4, 'nlags_s': duration / 4}


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    rows = []
    print('AR(1) duration sweep ...', flush=True)
    for duration in DURATIONS:
        print(f'  T={duration:g}s (T/tau={duration/TAU:g})', flush=True)
        rows.extend(run_condition(AR1(tau=TAU), fs=FS, duration=float(duration),
                                  n_reps=N_REPS_AR1, seed=41, n_jobs=N_JOBS,
                                  methods=METHODS,
                                  prep_kwargs=prep_for(duration)))
    print('Other processes ...', flush=True)
    for proc in OTHER:
        for duration in DURATIONS:
            rows.extend(run_condition(proc, fs=FS, duration=float(duration),
                                      n_reps=N_REPS_OTHER, seed=51,
                                      n_jobs=N_JOBS, methods=METHODS,
                                      prep_kwargs=prep_for(duration)))
        print(f'  {proc.name} done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / 'finite_sampling.csv.gz', index=False, compression='gzip')

    summary = summarize(df, truth_col='tau_true', value_col='tau_relaxation')
    summary.to_csv(RES / 'finite_summary.csv', index=False)

    analytic = pd.DataFrame({
        'duration_over_tau': RATIOS,
        'duration_s': DURATIONS,
        'rel_bias': [ar1_tau_moments(TAU, d, FS)[0] / TAU for d in DURATIONS],
        'rel_sd': [ar1_tau_moments(TAU, d, FS)[1] / TAU for d in DURATIONS],
        'rel_bias_asymptotic': relative_bias_asymptotic(RATIOS),
        'rel_sd_asymptotic': relative_sd_asymptotic(RATIOS),
        'ci_lo': [ar1_tau_ci(TAU, d, FS)[0] for d in DURATIONS],
        'ci_hi': [ar1_tau_ci(TAU, d, FS)[1] for d in DURATIONS],
    })
    analytic.to_csv(RES / 'finite_analytic.csv', index=False)
    print(analytic.to_string(index=False))
    print('\nT/tau for 10% / 20% relative standard error: '
          f'{duration_for_relative_sd(1.0, 0.1):.0f} / '
          f'{duration_for_relative_sd(1.0, 0.2):.0f}')

    make_figure(df, analytic)


def method_curve(df, process, method, column='tau_relaxation'):
    med, lo, hi, sd = [], [], [], []
    for duration in DURATIONS:
        sub = df[(df['process'] == process) & (df['method'] == method)
                 & np.isclose(df['duration'], duration)]
        v = sub[column].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        med.append(np.median(v) if len(v) else np.nan)
        lo.append(np.percentile(v, 16) if len(v) else np.nan)
        hi.append(np.percentile(v, 84) if len(v) else np.nan)
        sd.append(np.std(v) / np.mean(v) if len(v) > 2 else np.nan)
    return np.array(med), np.array(lo), np.array(hi), np.array(sd)


def make_figure(df, analytic):
    fig = plt.figure(figsize=(7.6, 5.6))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.45)

    # (a) tau_hat / tau vs T/tau, AR(1) -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for method in METHODS:
        med, lo, hi, _ = method_curve(df, 'ar1', method)
        ax.plot(RATIOS, med / TAU, 'o-', color=P.METHOD_COLORS[method], lw=1.1,
                ms=2.5,
                label=ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'))
        if method == 'ar1_direct':
            ax.fill_between(RATIOS, lo / TAU, hi / TAU,
                            color=P.METHOD_COLORS[method], alpha=0.15, lw=0)
    ax.plot(RATIOS, 1 + analytic['rel_bias'], 'k--', lw=1.0,
            label='analytic mean')
    ax.plot(RATIOS, analytic['ci_lo'] / TAU, color='k', ls=':', lw=0.8)
    ax.plot(RATIOS, analytic['ci_hi'] / TAU, color='k', ls=':', lw=0.8,
            label='analytic 95% CI')
    ax.axhline(1.0, color=P.LIGHT, lw=0.8, zorder=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$T/\\tau$')
    ax.set_ylabel('$\\hat\\tau/\\tau$')
    ax.set_title('AR(1): finite-sample bias', fontsize=8)
    ax.legend(fontsize=5.2, loc='lower right')
    P.panel_label(ax, 'a', dx=-0.28)

    # (b) relative sd -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for method in METHODS:
        _, _, _, sd = method_curve(df, 'ar1', method)
        ax.plot(RATIOS, sd, 'o-', color=P.METHOD_COLORS[method], lw=1.1, ms=2.5)
    ax.plot(RATIOS, analytic['rel_sd_asymptotic'], 'k--', lw=1.1,
            label='$\\sqrt{2\\tau/T}$')
    ax.axhline(0.1, color=P.LIGHT, lw=0.8)
    ax.text(RATIOS[0], 0.105, '10% s.e.', fontsize=5.6, color=P.GREY)
    ax.axvline(200, color=P.LIGHT, lw=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$T/\\tau$')
    ax.set_ylabel('Relative s.d. of $\\hat\\tau$')
    ax.set_title('Precision scales as $\\sqrt{\\tau/T}$', fontsize=8)
    ax.legend(fontsize=6, loc='lower left')
    P.panel_label(ax, 'b', dx=-0.30)

    # (c) analytic sampling distribution vs Monte Carlo --------------------
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    for ratio, color in zip((20.0, 100.0, 1000.0),
                            ('#C44E52', '#DD8452', '#4C72B0')):
        duration = ratio * TAU
        sub = df[(df['process'] == 'ar1') & (df['method'] == 'ar1_direct')
                 & np.isclose(df['duration'], duration)]
        emp = sub['tau'].to_numpy(dtype=float)
        emp = emp[np.isfinite(emp)] / TAU
        draws = sample_tau_finite(TAU, duration, FS, 200000, rng) / TAU
        draws = draws[np.isfinite(draws)]
        bins = np.linspace(0, 3, 61)
        ax.hist(emp, bins=bins, density=True, histtype='stepfilled',
                color=color, alpha=0.3, lw=0)
        counts, edges = np.histogram(draws, bins=bins, density=True)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), counts, color=color, lw=1.2,
                label=f'$T/\\tau$={ratio:g}')
    ax.axvline(1.0, color='k', ls='--', lw=0.9)
    ax.set_xlabel('$\\hat\\tau/\\tau$')
    ax.set_ylabel('Density')
    ax.set_title('Analytic vs simulated\nsampling distribution', fontsize=8)
    ax.legend(fontsize=5.6)
    P.panel_label(ax, 'c', dx=-0.28)

    # (d) window dependence of the ACF estimator ---------------------------
    ax = fig.add_subplot(gs[1, 0])
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(OTHER)))
    med, _, _, _ = method_curve(df, 'ar1', 'ar1_acf')
    ax.plot(DURATIONS, med, 'o--', color='k', lw=1.1, ms=2.5, label='AR(1)')
    for proc, label, color in zip(OTHER, OTHER_LABELS, colors):
        med, _, _, _ = method_curve(df, proc.name, 'ar1_acf')
        ax.plot(DURATIONS, med, 'o-', color=color, lw=1.1, ms=2.5, label=label)
    ax.plot(DURATIONS, DURATIONS * 0.035, color=P.GREY, ls=':', lw=1.0)
    ax.text(DURATIONS[-3], DURATIONS[-3] * 0.05, '$\\hat\\tau \\propto T$',
            fontsize=5.6, color=P.GREY)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Recording duration $T$ (s)')
    ax.set_ylabel('ACF-fit $\\hat\\tau$ (s)')
    ax.set_title('Window dependence (ACF fit)', fontsize=8)
    ax.legend(fontsize=5.0, loc='upper left', ncol=2, columnspacing=0.7)
    P.panel_label(ax, 'd', dx=-0.28)

    # (e) is tau a property of the signal or the window? -------------------
    ax = fig.add_subplot(gs[1, 1])
    names = ['ar1'] + [p.name for p in OTHER]
    labels = ['AR(1)'] + OTHER_LABELS
    shown = ['ar1_acf', 'ar1_direct', 'ar1_psd']
    height = 0.24
    upper = DURATIONS >= 5.0
    for i, method in enumerate(shown):
        slopes = []
        for name in names:
            med, _, _, _ = method_curve(df, name, method)
            good = np.isfinite(med) & (med > 0) & upper
            slopes.append(np.polyfit(np.log2(DURATIONS[good]),
                                     np.log2(med[good]), 1)[0]
                          if good.sum() > 2 else np.nan)
        ax.barh(np.arange(len(names)) + (i - 1) * height, slopes, height=height,
                color=P.METHOD_COLORS[method],
                label=ESTIMATOR_LABELS[method])
    ax.axvline(0, color=P.LIGHT, lw=0.8)
    ax.axvline(1, color=P.GREY, ls=':', lw=0.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlabel('$d\\log_2\\hat\\tau / d\\log_2 T$  ($T\\geq 5$ s)')
    ax.set_title('Is $\\hat\\tau$ a property of the\nsignal or of the window?',
                 fontsize=8)
    ax.legend(fontsize=5.0, loc='lower right')
    P.panel_label(ax, 'e', dx=-0.50)

    # (f) order and dominant-mode stability --------------------------------
    ax = fig.add_subplot(gs[1, 2])
    for proc, label, color in zip([AR1(TAU)] + OTHER, ['AR(1)'] + OTHER_LABELS,
                                  ['k'] + list(plt.cm.viridis(
                                      np.linspace(0, 0.9, len(OTHER))))):
        orders = []
        for duration in DURATIONS:
            sub = df[(df['process'] == proc.name) & (df['method'] == 'arp_psd')
                     & np.isclose(df['duration'], duration)]
            orders.append(sub['order'].median())
        ax.plot(RATIOS, orders, 'o-', color=color, lw=1.0, ms=2.5, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('$T/\\tau$')
    ax.set_ylabel('Selected AR order')
    ax.set_title('Model order needs data', fontsize=8)
    ax.legend(fontsize=5.2, ncol=2, loc='upper left', columnspacing=0.7)
    P.panel_label(ax, 'f', dx=-0.28)

    P.save(fig, FIG / 'figS_finite.svg')


if __name__ == '__main__':
    main()

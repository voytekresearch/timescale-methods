"""Stage 2a: the core simulation benchmark.

Six generative processes with closed-form spectra are estimated with all five
timescale estimators at a long recording duration, so that everything seen here
is model mismatch rather than finite-sample noise.  A population (Yule-Walker
on the true ACF) analysis quantifies how much AR structure each process needs
in the limit of infinite data.

Outputs
-------
results/benchmark_main.csv.gz, results/benchmark_main_summary.csv
results/population_ar.csv
figures/fig_benchmark.svg
"""

import os
import sys
from pathlib import Path

# One BLAS thread per worker: the Monte Carlo is parallelised over
# repetitions, and nested threading oversubscribes the machine badly.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmodes import plotting as P
from tsmodes.benchmark import run_condition, summarize
from tsmodes.estimators import (ESTIMATOR_LABELS, estimate, population_ar,
                                prepare, is_divergence)
from tsmodes.modes import ARModes, ar_psd
from tsmodes.sims import (AR1, AR1Mixture, AR1PlusSine, AR2Damped, PowerLaw,
                          SumAR1)

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'
FS = 500.0
DURATION = 100.0
N_REPS = 200
WELCH_WIN_S = 4.0
N_JOBS = int(os.environ.get('TSMODES_N_JOBS', '12'))

PROCESSES = [
    AR1(tau=0.05),
    AR1PlusSine(tau=0.05, f_osc=10.0, snr=1.0),
    AR2Damped(tau=0.05, f_osc=10.0),
    SumAR1(tau0=0.01, tau1=0.20, weight=0.5),
    AR1Mixture(tau_min=0.002, tau_max=2.0, n_components=12),
    PowerLaw(d=0.45),
]
SHORT_LABELS = ['AR(1)', 'AR(1) + sine', 'AR(2) oscillator',
                'Two AR(1) summed', 'Many AR(1) summed', 'Power law']
# The main figure shows five of the six; the AR(1) mixture makes the same point
# as the power law and is kept for the supplement.
MAIN_PROCESSES = ['ar1', 'ar1_sine', 'ar2_damped', 'sum_ar1', 'power_law']
MAX_ORDER = 30

# Processes whose ground-truth timescale is the decay of an *intrinsic*
# oscillator: there the AR(p) answer is the oscillatory mode, not a relaxation
# mode.  AR(1)-family estimators can only ever report a single relaxation time.
OSCILLATORY_TRUTH = {'ar2_damped'}


def add_target_column(df):
    """The timescale each estimator is being asked for, per process."""
    target = np.where(
        df['method'].str.startswith('ar1_'), df['tau'],
        np.where(df['process'].isin(OSCILLATORY_TRUTH),
                 df.get('top_osc_tau', np.nan), df['tau_relaxation']))
    out = df.copy()
    out['tau_target'] = target.astype(float)
    return out


def enbw(fs, nperseg):
    w = np.hanning(nperseg)
    return fs * np.sum(w ** 2) / np.sum(w) ** 2


def population_analysis(processes, fs, orders=range(1, MAX_ORDER + 1),
                        max_lag_s=8.0, welch_win_s=WELCH_WIN_S):
    """How well can an AR(p) represent each process given infinite data?"""
    nperseg = int(welch_win_s * fs)
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    band = (freqs >= 1.0) & (freqs <= fs / 2)
    bw = enbw(fs, nperseg)
    nlags = int(max_lag_s * fs)

    rows = []
    for proc in processes:
        rho = proc.acf(np.arange(nlags + 1), fs)
        target = proc.psd_with_lines(freqs, fs, bw)[band]
        for p in orders:
            phi = population_ar(rho, p)
            model = ar_psd(phi, freqs[band], fs, 1.0)
            model = model * np.mean(target / model)  # IS-optimal scale
            modes = ARModes.from_phi(phi, fs, freqs=freqs[band])
            dom = modes.dominant_subset(tol=0.1)
            rows.append({
                'process': proc.name, 'label': proc.label, 'order': p,
                'is_div': is_divergence(target, model),
                'rmse_logpsd': float(np.sqrt(np.mean(
                    (np.log10(target) - np.log10(model)) ** 2))),
                'n_modes': len(modes.modes),
                'n_dominant_modes': dom['n_modes'],
                'tau_dominant': modes.tau_dominant,
                'tau_relaxation': modes.tau_relaxation,
                'top_osc_freq': next((m.freq for m in modes.modes
                                      if m.kind == 'oscillatory'), np.nan),
                'tau_integrated': proc.tau_integrated(fs),
                'tau_true': proc.tau_true,
                'is_stationary': modes.is_stationary,
            })
    return pd.DataFrame(rows)


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    print('Population AR analysis ...', flush=True)
    pop = population_analysis(PROCESSES, FS)
    pop.to_csv(RES / 'population_ar.csv', index=False)

    cache = RES / 'benchmark_main.csv.gz'
    if cache.exists() and '--rerun' not in sys.argv:
        print(f'Loading cached {cache.name} (pass --rerun to recompute)',
              flush=True)
        df = pd.read_csv(cache)
    else:
        print(f'Monte Carlo: {len(PROCESSES)} processes x {N_REPS} reps',
              flush=True)
        rows = []
        for proc in PROCESSES:
            print(f'  {proc.name}', flush=True)
            rows.extend(run_condition(proc, fs=FS, duration=DURATION,
                                      n_reps=N_REPS, seed=11, n_jobs=N_JOBS,
                                      prep_kwargs={'welch_win_s': WELCH_WIN_S}))
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False, compression='gzip')
    df = add_target_column(df)

    summary = summarize(df, truth_col='tau_true', value_col='tau_target')
    summary.to_csv(RES / 'benchmark_main_summary.csv', index=False)
    summary_int = summarize(df, truth_col='tau_integrated',
                            value_col='tau_target')
    summary_int.to_csv(RES / 'benchmark_main_summary_integrated.csv', index=False)
    cols = ['process', 'method', 'truth', 'median', 'rel_bias', 'rel_error',
            'rel_sd', 'r2_logpsd_median', 'order_median']
    print('\n--- vs the constructed tau ---')
    print(summary[cols].to_string(index=False))
    print('\n--- vs the integrated timescale ---')
    print(summary_int[cols].to_string(index=False))

    make_figure(df, pop)


def example_fits(proc, fs, duration, seed=3):
    rng = np.random.default_rng(seed)
    sig = proc.simulate(duration, fs, rng)
    signal = prepare(sig, fs, welch_win_s=WELCH_WIN_S)
    robust = proc.has_oscillation
    fit1 = estimate(signal, 'ar1_psd', robust=robust)
    fitp = estimate(signal, 'arp_psd', robust=robust)
    return signal, fit1, fitp


def make_figure(df, pop):
    """Three rows: what the signal looks like, what its spectrum looks like,
    and what each estimator reports."""
    idx = [i for i, proc in enumerate(PROCESSES) if proc.name in MAIN_PROCESSES]
    procs = [PROCESSES[i] for i in idx]
    labels = [SHORT_LABELS[i] for i in idx]
    n = len(procs)

    fig = plt.figure(figsize=(7.3, 5.2))
    gs = fig.add_gridspec(3, n, hspace=0.45, wspace=0.3,
                          height_ratios=[0.75, 1.0, 1.15])

    methods = list(P.METHOD_ORDER)
    groups = [('time', 'time'), ('acf', 'ACF'), ('psd', 'PSD')]
    rng = np.random.default_rng(7)

    for j, (proc, label) in enumerate(zip(procs, labels)):
        signal, fit1, fitp = example_fits(proc, FS, DURATION)

        # -- row 1: the signal itself -------------------------------------
        ax = fig.add_subplot(gs[0, j])
        seconds = 1.5
        t = np.arange(int(seconds * FS)) / FS
        y = signal.sig[:len(t)]
        ax.plot(t, y / np.std(y), color='k', lw=0.5)
        ax.set_ylim(-4, 4)
        ax.set_xticks([0, 1])
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ax.set_title(label, fontsize=8.5)
        ax.spines['left'].set_visible(False)
        if j == 0:
            ax.set_ylabel('signal', fontsize=8)
            P.panel_label(ax, 'a', dx=-0.30, dy=1.30)
        if j == n // 2:
            ax.set_xlabel('Time (s)', fontsize=8)

        # -- row 2: spectrum and fits --------------------------------------
        ax = fig.add_subplot(gs[1, j])
        mask = signal.fit_mask
        f, p_emp = signal.freqs[mask], signal.powers[mask]
        ax.loglog(f, p_emp, color=P.LIGHT, lw=0.7)
        ax.loglog(f, ar_psd(fit1.phi, f, FS, fit1.offset),
                  color=P.METHOD_COLORS['ar1_psd'], lw=1.4, ls='--',
                  label='AR(1) fit')
        ax.loglog(f, ar_psd(fitp.phi, f, FS, fitp.offset),
                  color=P.METHOD_COLORS['arp_psd'], lw=1.4,
                  label=f'AR($p$) fit')
        ax.set_xticks([1, 100])
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('Power', fontsize=8)
            ax.legend(fontsize=6.5, loc='lower left')
            P.panel_label(ax, 'b', dx=-0.30, dy=1.18)
        if j == n // 2:
            ax.set_xlabel('Frequency (Hz)', fontsize=8)

        # -- row 3: what each estimator reports ----------------------------
        ax = fig.add_subplot(gs[2, j])
        sub = df[df['process'] == proc.name]
        for i, method in enumerate(methods):
            vals = sub.loc[sub['method'] == method, 'tau_target'].to_numpy(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            x = i + 0.9 * (i // 2)
            if not len(vals):
                ax.text(x, -2.9, 'n/a', fontsize=5.5, rotation=90,
                        ha='center', color=P.GREY)
                continue
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(x + jitter, np.log10(vals), s=1.2,
                       color=P.METHOD_COLORS[method], alpha=0.4, lw=0)
            ax.plot([x - 0.34, x + 0.34], [np.log10(np.median(vals))] * 2,
                    color='k', lw=1.3, solid_capstyle='butt')
        if np.isfinite(proc.tau_true):
            # A damped oscillator has no relaxation timescale: it rings rather
            # than decaying monotonically.  What it does have is an envelope
            # that decays, and that is what the line marks.
            colour = (P.MODE_COLORS['oscillatory']
                      if proc.name in OSCILLATORY_TRUTH else 'k')
            ax.axhline(np.log10(proc.tau_true), color=colour, ls='--', lw=1.0)
        tint = proc.tau_integrated(FS)
        if np.isfinite(tint):
            ax.axhline(np.log10(tint), color='#FFB000', ls=':', lw=1.3)
        ax.set_xlim(-0.8, 7.6)
        ax.set_ylim(-3.1, 1.0)
        ax.set_xticks([0, 1, 2.9, 3.9, 5.8, 6.8])
        ax.set_xticklabels(['AR(1)', 'AR($p$)'] * 3, fontsize=5.6,
                           rotation=90)
        for x, (domain, name) in zip([0.5, 3.4, 6.3], groups):
            # Below the rotated tick labels, not on top of them.
            ax.text(x, -0.46, name, ha='center', va='top', fontsize=7.5,
                    color=P.DOMAIN_COLORS[domain],
                    transform=ax.get_xaxis_transform())
        P.log_tick_labels(ax)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel('Estimated $\\tau$ (s)', fontsize=8)
            P.panel_label(ax, 'c', dx=-0.30, dy=1.14)
        else:
            ax.set_yticklabels([])

    handles = [plt.Line2D([], [], color='k', ls='--', lw=1.0,
                          label='true $\\tau$'),
               plt.Line2D([], [], color=P.MODE_COLORS['oscillatory'], ls='--',
                          lw=1.0,
                          label='envelope decay of the oscillator'),
               plt.Line2D([], [], color='#FFB000', ls=':', lw=1.3,
                          label='integrated $\\tau$ (area under the ACF)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=7.0,
               bbox_to_anchor=(0.5, -0.12), frameon=False)

    P.save(fig, FIG / 'fig_benchmark.svg')


if __name__ == '__main__':
    main()

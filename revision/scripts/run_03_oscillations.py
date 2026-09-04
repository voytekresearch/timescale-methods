"""Stage 2b: oscillations and richer dynamics vs the single-timescale model.

Sweeps an additive sinusoid over amplitude and frequency, and a genuine damped
stochastic oscillator over decay time and frequency, to show (i) how each
single-timescale estimator fails, and (ii) that AR(p) modes separate decay time
from oscillation frequency when the oscillation is intrinsic.

Outputs
-------
results/oscillation_sweep.csv.gz, results/damped_grid.csv.gz
figures/fig_oscillations.svg
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
from tsmodes.estimators import ESTIMATOR_LABELS, estimate, population_ar, prepare
from tsmodes.modes import ARModes, ar_psd
from tsmodes.sims import AR1PlusSine, AR2Damped

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / 'figures', ROOT / 'results'

FS = 500.0
TAU = 0.05
DURATION = 100.0
N_REPS = 20
WELCH_WIN_S = 2.0
N_JOBS = int(os.environ.get('TSMODES_N_JOBS', '12'))

SNRS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
FREQS = [4.0, 6.0, 10.0, 16.0, 25.0, 40.0]
DAMPED_TAUS = [0.02, 0.05, 0.15]
DAMPED_FREQS = [5.0, 10.0, 20.0, 40.0]
METHODS = ['ar1_direct', 'arp_direct', 'ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']


def run_sweep():
    rows = []
    for f_osc in FREQS:
        for snr in SNRS:
            proc = AR1PlusSine(tau=TAU, f_osc=f_osc, snr=snr)
            print(f'  sine f={f_osc:g} Hz snr={snr:g}', flush=True)
            rows.extend(run_condition(proc, fs=FS, duration=DURATION,
                                      n_reps=N_REPS, seed=21, n_jobs=N_JOBS,
                                      methods=METHODS,
                                      prep_kwargs={'welch_win_s': WELCH_WIN_S}))
    return pd.DataFrame(rows)


def run_damped():
    rows = []
    for tau in DAMPED_TAUS:
        for f_osc in DAMPED_FREQS:
            proc = AR2Damped(tau=tau, f_osc=f_osc)
            print(f'  damped tau={tau:g} f={f_osc:g}', flush=True)
            rows.extend(run_condition(proc, fs=FS, duration=DURATION,
                                      n_reps=N_REPS, seed=31, n_jobs=N_JOBS,
                                      methods=METHODS,
                                      prep_kwargs={'welch_win_s': WELCH_WIN_S}))
    return pd.DataFrame(rows)


def population_order_for_oscillation(fs=FS, tau=TAU, freqs=FREQS, snr=1.0,
                                     orders=range(1, 31), max_lag_s=8.0):
    """Smallest population AR order whose modes include the oscillation."""
    rows = []
    nlags = int(max_lag_s * fs)
    for f_osc in freqs:
        proc = AR1PlusSine(tau=tau, f_osc=f_osc, snr=snr)
        rho = proc.acf(np.arange(nlags + 1), fs)
        first = np.nan
        for p in orders:
            phi = population_ar(rho, p)
            modes = ARModes.from_phi(phi, fs)
            osc = [m for m in modes.modes
                   if m.kind == 'oscillatory' and abs(m.freq - f_osc) < 0.2 * f_osc]
            relax = [m for m in modes.modes if m.kind == 'relaxation']
            rows.append({'f_osc': f_osc, 'order': p,
                         'has_osc_mode': bool(osc),
                         'osc_freq': osc[0].freq if osc else np.nan,
                         'osc_tau': osc[0].tau if osc else np.nan,
                         'tau_relaxation': max((m.tau for m in relax),
                                               default=np.nan)})
            if osc and not np.isfinite(first):
                first = p
        rows[-1]['first_order_with_osc'] = first
    df = pd.DataFrame(rows)
    firsts = (df[df['has_osc_mode']].groupby('f_osc')['order'].min()
              .rename('first_order_with_osc').reset_index())
    return df.merge(firsts, on='f_osc', how='left', suffixes=('', '_min'))


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    rerun = '--rerun' in sys.argv
    sweep_path = RES / 'oscillation_sweep.csv.gz'
    damped_path = RES / 'damped_grid.csv.gz'
    pop_path = RES / 'oscillation_population_orders.csv'

    if sweep_path.exists() and not rerun:
        print('Loading cached sweep (pass --rerun to recompute)', flush=True)
        sweep = pd.read_csv(sweep_path)
    else:
        print('Additive-oscillation sweep ...', flush=True)
        sweep = run_sweep()
        sweep.to_csv(sweep_path, index=False, compression='gzip')

    if damped_path.exists() and not rerun:
        damped = pd.read_csv(damped_path)
    else:
        print('Damped-oscillator grid ...', flush=True)
        damped = run_damped()
        damped.to_csv(damped_path, index=False, compression='gzip')

    if pop_path.exists() and not rerun:
        pop = pd.read_csv(pop_path)
    else:
        print('Population order analysis ...', flush=True)
        pop = population_order_for_oscillation()
        pop.to_csv(pop_path, index=False)

    make_figure(sweep, damped, pop)


def _median_curve(df, method, column, xs, xcol, fixed=None):
    med = []
    for x in xs:
        sel = np.isclose(df[xcol], x) & (df['method'] == method)
        if fixed:
            for key, val in fixed.items():
                sel &= np.isclose(df[key], val)
        vals = df.loc[sel, column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        med.append(np.median(vals) if len(vals) else np.nan)
    return np.asarray(med)


def make_figure(sweep, damped, pop):
    """Two rows: an oscillation added to the signal, and an oscillation that is
    part of the signal."""
    fig = plt.figure(figsize=(7.6, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.70, wspace=0.42)
    rng = np.random.default_rng(3)
    show = ['ar1_acf', 'arp_acf', 'ar1_psd', 'arp_psd']

    rows = [
        (AR1PlusSine(tau=TAU, f_osc=10.0, snr=1.0), 'AR(1) + Oscillation',
         'ab c'),
        (AR2Damped(tau=TAU, f_osc=10.0), 'AR(2) oscillator', 'def'),
    ]

    for r, (proc, title, _) in enumerate(rows):
        signal = prepare(proc.simulate(200.0, FS, rng), FS, welch_win_s=4.0)
        fit1 = estimate(signal, 'ar1_psd', robust=True)
        # Let BIC choose, as the benchmark does.  Forcing order 2 here used
        # to give the AR(2) oscillator exactly the model it was built from,
        # but the relaxation requirement now spends one of those two poles,
        # leaving a single pole for the rhythm -- so a forced AR(2) came back
        # as two real poles and no oscillation at all.
        fitp = estimate(signal, 'arp_psd', robust=True)

        # -- the signal ----------------------------------------------------
        ax = fig.add_subplot(gs[r, 0])
        t = np.arange(int(1.5 * FS)) / FS
        y = signal.sig[:len(t)]
        ax.plot(t, y / np.std(y), color='k', lw=0.6)
        ax.set_ylim(-4, 4)
        ax.set_yticks([])
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.spines['left'].set_visible(False)
        ax.set_title(title, fontsize=10, loc='left')
        ax.tick_params(labelsize=8)
        P.panel_label(ax, 'ad'[r], dx=-0.16)

        # -- the spectrum --------------------------------------------------
        ax = fig.add_subplot(gs[r, 1])
        mask = signal.fit_mask
        f = signal.freqs[mask]
        # Data solid, every model dashed, so the data stays visible
        # underneath the curves that claim to describe it.
        ax.loglog(f, signal.powers[mask], color='k', lw=1.0, label='data')
        ax.loglog(f, ar_psd(fit1.phi, f, FS, fit1.offset),
                  color='#E69F00', lw=1.6, ls=(0, (5, 2)), label='AR(1)')
        if fitp.order > 1:
            # In the first row the order search returns p = 1, so an AR(p)
            # curve there would just be the AR(1) curve drawn twice.
            ax.loglog(f, ar_psd(fitp.phi, f, FS, fitp.offset),
                      color='#0072B2', lw=1.6, ls=(0, (1, 1.4)),
                      label=f'AR({fitp.order})')
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('Power', fontsize=9)
        ax.set_yticks([])
        ax.legend(fontsize=7, loc='lower left')
        ax.tick_params(labelsize=8)
        P.panel_label(ax, 'be'[r], dx=-0.20)

        # -- what the estimators report -------------------------------------
        ax = fig.add_subplot(gs[r, 2])
        if r == 0:
            for method in show:
                med = _median_curve(sweep, method, 'tau_relaxation', SNRS,
                                    'param_snr', {'param_f_osc': 10.0})
                ax.plot(SNRS, med, 'o-', ms=3, lw=1.2,
                        color=P.METHOD_COLORS[method],
                        label=ESTIMATOR_LABELS[method].replace('AR(p)', 'AR($p$)'))
            ax.axhline(TAU, color='k', ls='--', lw=1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Oscillation / aperiodic power', fontsize=9)
            ax.set_ylabel('Aperiodic $\\hat\\tau$ (s)', fontsize=9)
            ax.set_title('Bias',
                         fontsize=10)
            ax.set_ylim(8e-3, 2e-1)
            ax.legend(fontsize=5.6, loc='upper left', ncol=2,
                      columnspacing=0.6, handlelength=1.4, frameon=True,
                      framealpha=1.0, edgecolor='none')
        else:
            for method, col, lab in (('arp_acf', 'top_osc_tau', 'AR($p$) ACF'),
                                     ('arp_psd', 'top_osc_tau', 'AR($p$) PSD')):
                med = _median_curve(damped, method, col, DAMPED_TAUS,
                                    'param_tau', {'param_f_osc': 10.0})
                ax.plot(DAMPED_TAUS, med, 'o-', ms=4, lw=1.2,
                        color=P.METHOD_COLORS[method], label=lab)
            for method in ('ar1_acf', 'ar1_psd'):
                med = _median_curve(damped, method, 'tau', DAMPED_TAUS,
                                    'param_tau', {'param_f_osc': 10.0})
                ax.plot(DAMPED_TAUS, med, 's--', ms=3, lw=1.0,
                        color=P.METHOD_COLORS[method],
                        label=ESTIMATOR_LABELS[method])
            lims = [0.012, 0.45]
            ax.plot(lims, lims, color='k', ls=':', lw=1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            # Not the relaxation time of panel c: an oscillator has none.
            # This is the decay of the envelope multiplying its cosine.
            ax.set_xlabel('True envelope decay $\\tau_{env}$ (s)', fontsize=9)
            ax.set_ylabel('Estimated $\\tau_{env}$ (s)', fontsize=9)
            ax.set_title('Envelope decay',
                         fontsize=10)
            ax.set_ylim(8e-3, 1.1)
            ax.legend(fontsize=5.6, loc='upper left', ncol=2,
                      columnspacing=0.6, handlelength=1.4, frameon=True,
                      framealpha=1.0, edgecolor='none')
        ax.tick_params(labelsize=8)
        P.panel_label(ax, 'cf'[r], dx=-0.26, dy=1.16)

    P.save(fig, FIG / 'fig_oscillations.svg')


if __name__ == '__main__':
    main()

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

from tsmodes.benchmark import run_condition
from tsmodes.estimators import population_ar
from tsmodes.modes import ARModes
from tsmodes.sims import AR1PlusSine, AR2Damped

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'

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



if __name__ == '__main__':
    main()

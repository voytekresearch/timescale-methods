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
(no figure; results feed fig_windows)
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
from tsmodes.benchmark import run_condition, summarize
from tsmodes.estimators import ESTIMATOR_LABELS
from tsmodes.finite import (ar1_tau_ci, ar1_tau_moments, phi_of_tau,
                            relative_bias_asymptotic, relative_sd_asymptotic,
                            sample_tau_finite, duration_for_relative_sd)
from tsmodes.sims import AR1, AR1PlusSine, AR2Damped, ARpModes, PowerLaw

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
    ARpModes(),
    PowerLaw(d=0.45),
]
OTHER_LABELS = ['AR(1) + sine', 'AR(2) damped', 'AR(8), 5 modes',
                'Power law']


def prep_for(duration):
    """Analysis settings that scale with the recording.

    The Welch segment is a quarter of the recording and the ACF is taken out to
    a quarter of the recording, so that the lowest analysed frequency is
    ``4/T``.  This is what makes the analysis window, rather than a fixed
    segment length, the thing that changes as recordings are shortened -- the
    situation a shorter experiment actually puts an analyst in.
    """
    return {'welch_win_s': duration / 4, 'nlags_s': duration / 4}


def _cached(name):
    """Load a cached result table, or None when it has not been computed."""
    path = RES / name
    return pd.read_csv(path) if path.exists() else None


def main():
    P.use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    # --figures-only replots from the cached table; see run_05_real_data.
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

    # No figure of its own: finite-sample bias and windowing bias are the
    # same question asked twice, so they share the main windows figure.


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


if __name__ == '__main__':
    main()

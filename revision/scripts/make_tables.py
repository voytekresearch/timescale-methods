"""Supplementary tables, written as LaTeX from the cached result tables.

Every number the Methods and Supplement quote lives in one of these tables,
so the text can point at a table instead of listing percentages.  Tables S1,
S2, S5 and S6 come straight from the result tables of run_02, run_03 and
run_06.  Tables S3 and S4 need per-trace spectral fits of the recordings (the
same fits fig_realdata draws), which take a few minutes; they are cached in
``results/tables_realdata.json`` and recomputed with ``--rerun``.

Output: ``tables/tab_s*.tex``, one file per table, and the same text on
stdout for pasting into the manuscript.
"""

import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = ROOT / 'tables'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_02_benchmark import add_target_column, FINITE_DURATION, DURATION  # noqa: E402

METHODS = [('ar1_direct', 'AR(1), time'), ('ar1_acf', 'AR(1), ACF'), ('ar1_psd', 'AR(1), PSD'),
           ('arp_direct', 'AR($p$), time'), ('arp_acf', 'AR($p$), ACF'), ('arp_psd', 'AR($p$), PSD')]
PROCESSES = [('ar1', DURATION, 'AR(1), 100 s'), ('ar2_damped', DURATION, 'AR(2) oscillator'),
             ('arp_modes', DURATION, 'AR(7)'), ('power_law', DURATION, 'Power law'),
             ('ar1', FINITE_DURATION, 'AR(1), 4 s')]
RECORDINGS = [('human_fmri_rest', 'Human fMRI'), ('human_lemon_alpha', 'Human EEG'),
              ('human_meg_beta', 'Human MEG'), ('human_cole_m1_beta', 'Human ECoG'),
              ('macaque_visual_grating_ecog', 'Macaque ECoG'),
              ('mouse_neuropixels_natural_movie', 'Mouse LFP'),
              ('mouse_neuropixels_spikes', 'Mouse spikes'), ('rat_hc3_ca1_ca3', 'Rat LFP')]
FIELD = {'human_lemon_alpha', 'human_meg_beta', 'human_cole_m1_beta',
         'macaque_visual_grating_ecog', 'mouse_neuropixels_natural_movie', 'rat_hc3_ca1_ca3'}


def ms(x):
    """Milliseconds with the precision the magnitude deserves."""
    if not np.isfinite(x):
        return '--'
    if x >= 19000:
        return '20000$^\\dagger$'
    if x >= 20:
        return f'{x:.0f}'
    if x >= 10:
        return f'{x:.1f}'
    return f'{x:.2f}'


def ms_range(lo, hi):
    if lo >= 19000:
        return '20000$^\\dagger$'
    return f'{ms(lo)}--{ms(hi)}'


def pct(x, digits=1):
    if not np.isfinite(x):
        return '--'
    if x >= 100:
        return f'{x:.0f}'
    if x < 0.1:
        return f'{x:.2f}'
    return f'{x:.{digits}f}'


def table(label, caption, header, rows, align, note=None, sideways=False):
    """One table in the form the Springer class expects.

    sn-jnl wraps every ``table`` in ``center`` and ``threeparttable`` and sets
    the body font itself, so the content must be exactly a caption, one
    tabular, and optional ``\\footnotetext`` notes: no ``\\centering``, no size
    commands, no ``\\resizebox``, and no ``[H]`` (float's H placement breaks
    the class's nesting).  A wide table goes in ``sidewaystable``.
    """
    env = 'sidewaystable' if sideways else 'table'
    lines = [f'\\begin{{{env}}}[h]', f'\\caption{{{caption}}}\\label{{{label}}}',
             f'\\begin{{tabular}}{{{align}}}', '\\toprule',
             ' & '.join(header) + ' \\\\', '\\midrule']
    for row in rows:
        lines.append(' & '.join(row) + ' \\\\' if row else '\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    if note:
        mark, text = note
        lines.append(f'\\footnotetext[{mark}]{{{text}}}' if mark else f'\\footnotetext{{{text}}}')
    lines.append(f'\\end{{{env}}}')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# S1: the benchmark, every estimator on every process
# ---------------------------------------------------------------------------

def table_s1():
    df = add_target_column(pd.read_csv(RES / 'benchmark_main.csv.gz'))
    df['tau_use'] = np.where(np.isfinite(df['tau_target']), df['tau_target'], df['tau'])
    rows = []
    for i, (proc, dur, plabel) in enumerate(PROCESSES):
        if i:
            rows.append(None)
        sub = df[(df['process'] == proc) & np.isclose(df['duration'], dur)]
        for j, (m, mlabel) in enumerate(METHODS):
            s = sub[sub['method'] == m]
            v = s['tau_use'].to_numpy(float) * 1e3
            lo, hi = np.percentile(v, [10, 90])
            rows.append([plabel if j == 0 else '', mlabel, ms(np.median(v)), ms_range(lo, hi),
                         f'{s["order"].median():.0f}',
                         pct(100 * (1 - s['r2_logpsd'].median())),
                         pct(100 * (1 - s['r2_acf'].median()))])
    caption = ('\\textbf{Simulation benchmark.} Six estimators on the four benchmark processes and on '
               'four seconds of the AR(1) process, 200 repetitions each. The estimate is the timescale '
               'each estimator reports: $\\tau$ for the AR(1) estimators, the envelope decay of the '
               'oscillator and the most important non-rhythmic mode of the AR(7) for the AR($p$) '
               'estimators. The true value is 200 ms for every autoregressive process; the power law '
               'has none. Order is the median BIC selection. Unexplained variance is the fraction of '
               'the $\\log_{10}$ spectrum, or of the autocorrelation, that the fitted model leaves '
               'unexplained, median across repetitions.')
    note = ('\\dagger', 'Upper bound of the search range.')
    return table('tab:s_benchmark', caption,
                 ['Process', 'Estimator', '$\\hat\\tau$ (ms)', '10--90\\% (ms)', 'Order',
                  'Unexpl.\\ PSD (\\%)', 'Unexpl.\\ ACF (\\%)'],
                 rows, 'llrrrrr', note)


# ---------------------------------------------------------------------------
# S2: the damped-oscillator sweep
# ---------------------------------------------------------------------------

def table_s2():
    d = pd.read_csv(RES / 'damped_grid.csv.gz')
    rows = []
    for (tau, f), g in d.groupby(['param_tau', 'param_f_osc']):
        row = [f'{tau*1e3:.0f}', f'{f:.0f}']
        for m, _ in METHODS:
            s = g[g['method'] == m]
            v = s['top_osc_tau'] if m.startswith('arp') else s['tau']
            row.append(f'{v.median() / tau:.2f}')
        rows.append(row)
    caption = ('\\textbf{Damped-oscillator sweep.} Median estimate divided by the true envelope decay '
               'for AR(2) oscillators with envelope decays of 20, 50 and 150 ms and frequencies of 5 to '
               '40 Hz, 20 repetitions of 100 s per condition. The AR($p$) estimators report the envelope '
               'decay of the fitted oscillatory mode; the AR(1) estimators report their single timescale.')
    return table('tab:s_damped', caption,
                 ['$\\tau$ (ms)', '$f$ (Hz)'] + [m for _, m in METHODS], rows, 'rrrrrrrr')


# ---------------------------------------------------------------------------
# S3 and S4: the recordings, fits and modes
# ---------------------------------------------------------------------------

def _resolvable(fit, sig):
    tau_max = 1.0 / (2.0 * np.pi * sig.f_range[0])
    return sorted([m for m in fit.modes.modes if m.tau <= tau_max], key=lambda m: -m.importance)


def _realdata_one(key):
    from tsmodes import datasets as D
    from tsmodes.estimators import estimate
    data = D.load(key)
    w0 = data['analysis']['welch_win_s']
    out = {'n_traces': int(data['X'].shape[0]), 'duration_s': float(data['X'].shape[1] / data['fs']),
           'welch_win_s': float(w0), 'n_dom': {'0.90': [], '0.95': [], '0.99': []},
           'n_res': [], 'dropped': 0, 'total': 0, 'reported_pair': 0, 'reported_tau': [],
           'reported_freq': [], 'tau_by_window': {'0.5': [], '1': [], '2': []}}
    for trace in data['X']:
        for mult, key_w in ((0.5, '0.5'), (1.0, '1'), (2.0, '2')):
            sig = D.prepare_trace(trace, data, welch_win_s=w0 * mult)
            fit = estimate(sig, 'arp_psd', robust=True)
            out['tau_by_window'][key_w].append(float(fit.tau_relaxation))
            if mult != 1.0:
                continue
            out['band'] = [float(sig.f_range[0]), float(sig.f_range[1])]
            for r2 in out['n_dom']:
                out['n_dom'][r2].append(int(fit.modes.dominant_subset(tol=np.sqrt(1 - float(r2)))['n_modes']))
            keep = _resolvable(fit, sig)
            out['n_res'].append(len(keep))
            out['dropped'] += len(fit.modes.modes) - len(keep)
            out['total'] += len(fit.modes.modes)
            # The reported mode is the one the library's rule picks
            # (ARModes.tau_relaxation), identified by its timescale.
            tau_rep = fit.tau_relaxation
            rep = [m for m in fit.modes.modes if np.isclose(m.tau, tau_rep)]
            if rep:
                out['reported_pair'] += int(rep[0].kind == 'oscillatory')
                out['reported_tau'].append(float(rep[0].tau))
                out['reported_freq'].append(float(rep[0].freq))
    return key, out


def realdata_stats(rerun=False):
    cache = RES / 'tables_realdata.json'
    if cache.exists() and not rerun:
        return json.loads(cache.read_text())
    with Pool(8) as pool:
        stats = dict(pool.map(_realdata_one, [k for k, _ in RECORDINGS]))
    cache.write_text(json.dumps(stats))
    return stats


def spikes_full_band_unexplained():
    """The AR(1) spectral fit to the spike counts over the full 0.05-8 Hz band."""
    from tsmodes import datasets as D
    from tsmodes.estimators import estimate, spectral_gof
    from tsmodes.modes import ar_psd
    data = D.load('mouse_neuropixels_spikes')
    out = []
    for trace in data['X']:
        sig = D.prepare_trace(trace, data, f_range=(0.05, 8.0))
        fit = estimate(sig, 'ar1_psd', robust=True)
        out.append(1 - spectral_gof(sig, ar_psd(fit.phi, sig.freqs, sig.fs, fit.offset))['r2_logpsd'])
    return float(np.median(out))


def table_s3(stats):
    s = pd.read_csv(RES / 'real_data_summary.csv').set_index('dataset')
    full = spikes_full_band_unexplained()
    rows = []
    for key, label in RECORDINGS:
        r, st = s.loc[key], stats[key]
        w = st['tau_by_window']
        med = {k: np.nanmedian(v) for k, v in w.items()}
        rows.append([label, f'{st["n_traces"]}', f'{st["duration_s"]:.0f}',
                     f'{st["band"][0]:g}--{st["band"][1]:g}', f'{st["welch_win_s"]:g}',
                     ms(r['tau_ar1_psd'] * 1e3), ms(r['tau_arp_psd'] * 1e3), f'{r["order_arp_psd"]:.0f}',
                     pct(100 * (1 - r['r2_ar1_psd'])), pct(100 * (1 - r['r2_arp_psd'])),
                     f'{med["0.5"] / med["1"]:.2f}', f'{med["2"] / med["1"]:.2f}'])
    caption = ('\\textbf{Recordings: analysis settings and spectral fits.} Traces per recording and '
               'their duration after cropping, the fit band, the Welch window, and the median across '
               'traces of the AR(1) spectral timescale, the timescale of the most important mode of the '
               'AR($p$) spectral fit, the BIC order, and the fraction of the $\\log_{10}$ spectrum each '
               'model leaves unexplained. The last two columns give the median AR($p$) timescale with '
               'the Welch window halved or doubled, relative to the value at the window used.')
    note = ('', f'Over the full 0.05--8 Hz band available for the spike counts, the AR(1) fit leaves '
                f'{100 * full:.0f}\\% of the log spectrum unexplained.')
    return table('tab:s_recordings', caption,
                 ['Recording', 'Traces', 'T (s)', 'Band (Hz)', 'Welch (s)', 'AR(1) $\\hat\\tau$ (ms)',
                  'AR($p$) mode $\\hat\\tau$ (ms)', 'Order', 'Unexpl.\\ AR(1) (\\%)', 'Unexpl.\\ AR($p$) (\\%)',
                  'Window $\\times\\tfrac12$', 'Window $\\times 2$'],
                 rows, 'lrrrrrrrrrrr', note, sideways=True)


def table_s4(stats):
    rows = []
    for key, label in RECORDINGS:
        st = stats[key]
        n_res = st['n_res']
        rows.append([label, f'{int(np.median(n_res))} ({min(n_res)}--{max(n_res)})']
                    + [f'{int(np.median(st["n_dom"][r2]))}' for r2 in ('0.90', '0.95', '0.99')]
                    + [pct(100 * st['dropped'] / st['total'], 0),
                       f'{st["reported_pair"]}/{st["n_traces"]}',
                       ms(1e3 * np.median(st['reported_tau'])),
                       f'{np.median(st["reported_freq"]):.1f}'])
    caption = ('\\textbf{Recordings: modes of the AR($p$) spectral fit.} Resolvable modes per trace '
               '(median and range), the number of modes needed to reproduce the fitted log spectrum '
               'at three cutoffs (median across traces), the fraction of fitted modes dropped because '
               'the fit band cannot resolve their timescale, and the aperiodic mode selected by the '
               'rule in Methods (the most important mode with $f\\tau < 1$): the number of traces in '
               'which it is a conjugate pair rather than a real pole, and its median timescale and '
               'frequency.')
    return table('tab:s_modes', caption,
                 ['Recording', 'Modes/trace', '$R^2 \\ge 0.90$', '$R^2 \\ge 0.95$', '$R^2 \\ge 0.99$',
                  'Dropped (\\%)', 'Aperiodic mode a pair', 'Aperiodic $\\hat\\tau$ (ms)', 'Aperiodic $f$ (Hz)'],
                 rows, 'lrrrrrrrr')


# ---------------------------------------------------------------------------
# S5: windowing
# ---------------------------------------------------------------------------

def table_s5():
    s = pd.read_csv(RES / 'empirical_stability_requirements.csv')
    rows = []
    for key, label in RECORDINGS:
        g = s[s['dataset'] == key].set_index('method')
        p = g.loc['arp_psd']
        rows.append([label, f'{p["T_over_tau_bias"]:.0f}', f'{p["shortest_over_longest"]:.2f}']
                    + [f'{g.loc[m, "excess_sd_longest"]:.0f}' if m in g.index else '--'
                       for m, _ in METHODS if m != 'arp_direct'])
    caption = ('\\textbf{Windowing.} For the AR($p$) spectral estimate: the window duration, in units '
               'of the estimated timescale, beyond which the median estimate stays within 20\\% of its '
               'value in the longest window, and the median estimate in the shortest window relative to '
               'the longest. The remaining columns give, for each estimator, the spread of estimates '
               'across the longest windows divided by the AR(1) sampling bound of eq. \\ref{eq:finite} '
               'at that duration.')
    return table('tab:s_windows', caption,
                 ['Recording', '$T/\\hat\\tau$ to 20\\%', 'Shortest/longest',
                  'AR(1), time', 'AR(1), ACF', 'AR(1), PSD', 'AR($p$), ACF', 'AR($p$), PSD'],
                 rows, 'lrrrrrrr')


# ---------------------------------------------------------------------------
# S6: disagreement between the two fit domains from sampling alone
# ---------------------------------------------------------------------------

def table_s6():
    df = add_target_column(pd.read_csv(RES / 'benchmark_main.csv.gz'))
    rows = []
    for dur, label in ((DURATION, f'{DURATION:.0f} s ($T = 500\\tau$)'),
                       (FINITE_DURATION, f'{FINITE_DURATION:.0f} s ($T = 20\\tau$)')):
        d = df[(df['process'] == 'ar1') & np.isclose(df['duration'], dur)]
        row = [label]
        for a, b in (('ar1_acf', 'ar1_psd'), ('arp_acf', 'arp_psd')):
            x = d[d['method'] == a].set_index('rep')['tau_target']
            y = d[d['method'] == b].set_index('rep')['tau_target'].reindex(x.index)
            r = np.maximum(x / y, y / x)
            row.append(f'{100 * np.mean(r > 1.5):.1f}')
        rows.append(row)
    caption = ('\\textbf{Disagreement between fit domains from sampling alone.} Percentage of 200 '
               'repetitions of the 200 ms AR(1) process in which the ACF and PSD fits of the same model '
               'differ by more than 1.5-fold.')
    return table('tab:s_disagree', caption,
                 ['Duration', 'AR(1) fits (\\%)', 'AR($p$) fits (\\%)'], rows, 'lrr')


def main():
    OUT.mkdir(exist_ok=True)
    stats = realdata_stats(rerun='--rerun' in sys.argv)
    tables = {'tab_s1_benchmark': table_s1(), 'tab_s2_damped': table_s2(),
              'tab_s3_recordings': table_s3(stats), 'tab_s4_modes': table_s4(stats),
              'tab_s5_windows': table_s5(), 'tab_s6_disagree': table_s6()}
    for name, text in tables.items():
        (OUT / f'{name}.tex').write_text(text)
        print(f'% ---- {name}\n{text}')


if __name__ == '__main__':
    main()

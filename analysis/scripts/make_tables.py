"""Supplementary tables, written as LaTeX from the cached result tables and
placed into the manuscript.

Every number the Methods and Supplement quote lives in one of these tables,
so the text points at a table instead of listing values.  Tables S1, S2, S5
and S6 come straight from the result tables of run_02, run_03 and run_06.
Tables S3 and S4 need per-trace spectral fits of the recordings (the same
fits fig_realdata draws), which take a few minutes; they are cached in
``results/tables_realdata.json`` and recomputed with ``--rerun``.

Output: ``tables/tab_s*.tex``, one file per table.  With ``--write PATH ...``
each table replaces the block between its markers in the given .tex files::

    % AUTO-TABLE BEGIN tab:s_benchmark
    ...
    % AUTO-TABLE END tab:s_benchmark

A file is left untouched, and the run fails, if a label's markers are
missing or appear more than once.  Nothing outside the markers is changed.

Every table follows the form the Springer class (sn-jnl) expects: a caption,
one tabular, and class footnotes.  The class wraps the table in ``center``
and ``threeparttable`` and sets the body font itself, so ``\\centering``,
size commands, ``\\resizebox`` and ``[H]`` placement all break it.  Column
padding is reduced and long headers are stacked so every table fits the
text width.
"""
import sys
from pathlib import Path

# `tsmodes` (this paper's own code) lives beside `scripts/`, so the
# analysis directory has to be importable however the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = ROOT / 'tables'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_02_benchmark import (add_target_column, FINITE_DURATION, DURATION,  # noqa: E402
                              N_REPS, N_REPS_FINITE)
from tsmodes.constants import BENCH_TAU  # noqa: E402

METHODS = [('ar1_direct', 'AR(1), time'), ('ar1_acf', 'AR(1), ACF'), ('ar1_psd', 'AR(1), PSD'),
           ('arp_direct', 'AR($p$), time'), ('arp_acf', 'AR($p$), ACF'), ('arp_psd', 'AR($p$), PSD')]
PROCESSES = [('ar1', DURATION, 'AR(1)'), ('ar2_damped', DURATION, 'AR(2) oscillator'),
             ('arp_modes', DURATION, 'AR(7)'),
             ('ar1', FINITE_DURATION, f'AR(1), {FINITE_DURATION:g} s')]
RECORDINGS = [('human_fmri_rest', 'Human fMRI'), ('human_lemon_alpha', 'Human EEG'),
              ('human_meg_beta', 'Human MEG'), ('human_cole_m1_beta', 'Human ECoG'),
              ('macaque_visual_grating_ecog', 'Macaque ECoG'),
              ('mouse_neuropixels_natural_movie', 'Mouse LFP'),
              ('mouse_neuropixels_spikes', 'Mouse spikes'), ('rat_hc3_ca1_ca3', 'Rat LFP')]
KIND = {'relaxation': 'real', 'oscillatory': 'pair', 'floor': 'floor'}


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

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


def stack(*lines):
    """A two-line column header."""
    return '\\shortstack[c]{' + ' \\\\ '.join(lines) + '}'


def mode_summary(kinds, freqs):
    """Majority kind of a set of reported modes, with the median frequency of
    the pairs and, when kinds are mixed, the share of the majority."""
    kinds = [KIND.get(k, k) for k in kinds if isinstance(k, str)]
    if not kinds:
        return '--'
    top = max(set(kinds), key=kinds.count)
    share = kinds.count(top) / len(kinds)
    if top == 'pair':
        f = np.nanmedian([f for f, k in zip(freqs, kinds) if k == 'pair'])
        text = f'pair, {f:.1f} Hz'
    else:
        text = top
    return text if share == 1.0 else f'{text} ({100 * share:.0f}\\%)'


def table(label, caption, header, rows, align, note=None):
    lines = ['\\begin{table}[h]', f'\\caption{{{caption}}}\\label{{{label}}}',
             '\\setlength{\\tabcolsep}{3pt}',
             f'\\begin{{tabular}}{{{align}}}', '\\toprule',
             ' & '.join(header) + ' \\\\', '\\midrule']
    for row in rows:
        lines.append(' & '.join(row) + ' \\\\' if row else '\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    if note:
        mark, text = note
        lines.append(f'\\footnotetext[{mark}]{{{text}}}' if mark else f'\\footnotetext{{{text}}}')
    lines.append('\\end{table}')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# S1: the benchmark, every estimator on every process
# ---------------------------------------------------------------------------

def reported_mode(row):
    """Kind and frequency of the mode a benchmark row reports: the AR(1)
    estimators report a real pole; an AR(p) estimator's reported timescale is
    matched to the fit's three most important modes."""
    if row['method'].startswith('ar1_'):
        return 'relaxation', 0.0
    tau = row['tau_target']
    for k in range(3):
        if np.isfinite(row[f'mode{k}_tau']) and np.isclose(row[f'mode{k}_tau'], tau, rtol=1e-6):
            return row[f'mode{k}_kind'], row[f'mode{k}_freq']
    return 'relaxation', 0.0


def table_s1():
    df = add_target_column(pd.read_csv(RES / 'benchmark_main.csv.gz'))
    rows = []
    for i, (proc, dur, plabel) in enumerate(PROCESSES):
        if i:
            rows.append(None)
        sub = df[(df['process'] == proc) & np.isclose(df['duration'], dur)]
        for j, (m, mlabel) in enumerate(METHODS):
            s = sub[sub['method'] == m]
            v = s['tau_target'].to_numpy(float) * 1e3
            lo, hi = np.percentile(v, [10, 90])
            modes = [reported_mode(r) for _, r in s.iterrows()]
            rows.append([plabel if j == 0 else '', mlabel, ms(np.median(v)), ms_range(lo, hi),
                         mode_summary([k for k, _ in modes], [f for _, f in modes]),
                         f'{s["order"].median():.0f}',
                         pct(100 * (1 - s['r2_logpsd'].median())),
                         pct(100 * (1 - s['r2_acf'].median()))])
    caption = ('\\textbf{Simulation benchmark.} Six estimators on the three benchmark processes, '
               f'{N_REPS} repetitions of {DURATION:g} s each, and on {FINITE_DURATION:g} s records of the '
               f'AR(1) process, {N_REPS_FINITE} repetitions. The estimate is the timescale each '
               'estimator reports: $\\tau$ for the AR(1) estimators, the envelope decay of the '
               'oscillator and the most important non-rhythmic mode of the AR(7) for the AR($p$) '
               'estimators; the true value is 200 ms in every case. Mode gives the kind of the '
               'reported mode, a real pole or a conjugate pair with its median frequency. $p$ is '
               'the median BIC order. Unexplained variance is the fraction of the $\\log_{10}$ '
               'spectrum, or of the autocorrelation, that the fitted model leaves unexplained, '
               'median across repetitions.')
    note = ('\\dagger', 'Upper bound of the search range.')
    return table('tab:s_benchmark', caption,
                 ['Process', 'Estimator', stack('$\\hat\\tau$', '(ms)'), stack('10--90\\%', '(ms)'),
                  'Mode', '$p$', stack('Unexpl.', 'PSD (\\%)'), stack('Unexpl.', 'ACF (\\%)')],
                 rows, 'llrrlrrr', note)


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
               'decay of the fitted oscillatory mode, a conjugate pair at the frequency given in the '
               'second column; the AR(1) estimators report their single real pole.')
    return table('tab:s_damped', caption,
                 [stack('$\\tau$', '(ms)'), stack('$f$', '(Hz)')]
                 + [stack(*m.split(', ')) for _, m in METHODS], rows, 'rrrrrrrr')


# ---------------------------------------------------------------------------
# S3 and S4: the recordings, fits and modes
# ---------------------------------------------------------------------------

def _resolvable(fit, sig):
    """The same rule the figure uses, from the library (ARModes.resolvable)."""
    return fit.modes.resolvable(sig.f_range)


def _reported(fit, sig, nlags):
    """The modes fig_realdata draws: resolvable, and holding real variance.

    Same rule and same numbers as scripts/figures/fig_06.py: rank by each
    mode's share of the variance, its additive autocorrelation component at
    lag zero, and keep those above the library's threshold.
    """
    from timescales.decomposition.modes import ar_psd
    from timescales.decomposition.select import dominant_modes, variance_shares
    from tsmodes.prep import spectrum_acov
    keep = _resolvable(fit, sig)
    if not keep:
        return []
    acov = spectrum_acov(sig, ar_psd(fit.phi, sig.freqs, sig.fs, fit.offset), nlags)
    parts = spectrum_acov(sig, fit.modes.psd_components(sig.freqs), nlags) / acov[0]
    index = {id(m): j for j, m in enumerate(fit.modes.modes)}
    curves = np.array([parts[index[id(m)]] for m in keep])
    return dominant_modes(keep, variance_shares(keep, curves))


def _realdata_one(key):
    from tsmodes import datasets as D
    from tsmodes.estimators import estimate
    data = D.load(key)
    w0 = data['analysis']['welch_win_s']
    out = {'n_traces': int(data['X'].shape[0]), 'duration_s': float(data['X'].shape[1] / data['fs']),
           'welch_win_s': float(w0), 'n_dom': {'0.90': [], '0.95': [], '0.99': []},
           'n_res': [], 'dropped': 0, 'total': 0, 'reported_pair': 0, 'reported_tau': [],
           'reported_freq': [], 'tau_by_window': {'0.5': [], '1': [], '2': []},
           'floor_traces': 0, 'floor_relative': [], 'n_rep': []}
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
            # The noise floor, where the fit has one: a high-pass factor
            # standing in for a flat floor.  Recorded as a level relative to
            # the power at the top of the band, not as a timescale.
            nf = fit.modes.noise_floor(sig.freqs[sig.fit_mask & sig.exclude_mask],
                                       offset=fit.offset)
            if nf is not None and nf['r2'] >= 0.99:
                out['floor_traces'] += 1
                out['floor_relative'].append(float(nf['relative']))
            keep = _resolvable(fit, sig)
            out['n_res'].append(len(keep))
            out['dropped'] += len(fit.modes.modes) - len(keep)
            out['total'] += len(fit.modes.modes)
            # Of the resolvable modes, the ones actually reported: those
            # holding at least the library's minimum share of the variance.
            out['n_rep'].append(len(_reported(fit, sig, len(sig.corrs) - 1)))
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
    from timescales.ar.metrics import spectral_gof
    from timescales.decomposition.modes import ar_psd
    from tsmodes.estimators import estimate
    data = D.load('mouse_neuropixels_spikes')
    out = []
    for trace in data['X']:
        sig = D.prepare_trace(trace, data, f_range=(0.05, 8.0))
        fit = estimate(sig, 'ar1_psd', robust=True)
        out.append(1 - spectral_gof(sig, ar_psd(fit.phi, sig.freqs, sig.fs, fit.offset))['r2_logpsd'])
    return float(np.median(out))


def table_s3(stats):
    s = pd.read_csv(RES / 'real_data_summary.csv').set_index('dataset')
    fits = pd.read_csv(RES / 'real_data_fits.csv')
    fits = fits[fits['method'] == 'arp_psd']
    full = spikes_full_band_unexplained()
    rows = []
    for key, label in RECORDINGS:
        r, st = s.loc[key], stats[key]
        g = fits[fits['dataset'] == key]
        rows.append([label, f'{st["n_traces"]}', f'{st["duration_s"]:.0f}',
                     ms(r['tau_ar1_psd'] * 1e3), ms(r['tau_arp_psd'] * 1e3),
                     mode_summary(g['mode0_kind'].tolist(), g['mode0_freq'].tolist()),
                     f'{r["order_arp_psd"]:.0f}',
                     pct(100 * (1 - r['r2_ar1_psd'])), pct(100 * (1 - r['r2_arp_psd']))])
    caption = ('\\textbf{Recordings: spectral fits.} Traces per recording and their duration after '
               'cropping, and the median across traces of the AR(1) spectral timescale (a real pole), '
               'the timescale of the most important mode of the AR($p$) spectral fit with the kind '
               'of that mode (a real pole or a conjugate pair with its median frequency), the BIC '
               'order, and the fraction of the $\\log_{10}$ spectrum each model leaves unexplained.')
    note = ('', f'Over the full 0.05--8 Hz band available for the spike counts, the AR(1) fit leaves '
                f'{100 * full:.0f}\\% of the log spectrum unexplained.')
    return table('tab:s_recordings', caption,
                 ['Recording', 'Traces', '$T$ (s)', stack('AR(1)', '$\\hat\\tau$ (ms)'),
                  stack('AR($p$) top', 'mode $\\hat\\tau$ (ms)'), stack('AR($p$) top', 'mode kind'),
                  '$p$', stack('Unexpl.', 'AR(1) (\\%)'), stack('Unexpl.', 'AR($p$) (\\%)')],
                 rows, 'lrrrrlrrr', note)


def table_s4(stats):
    rows = []
    for key, label in RECORDINGS:
        st = stats[key]
        n_res = st['n_res']
        n_rep = st['n_rep']
        rows.append([label, f'{int(np.median(n_res))} ({min(n_res)}--{max(n_res)})',
                     f'{int(np.median(n_rep))} ({min(n_rep)}--{max(n_rep)})']
                    + [f'{int(np.median(st["n_dom"][r2]))}' for r2 in ('0.90', '0.95', '0.99')]
                    + [pct(100 * st['dropped'] / st['total'], 0),
                       f'{st["floor_traces"]}/{st["n_traces"]}',
                       (f'{np.median(st["floor_relative"]):.2f}'
                        if st['floor_relative'] else '--'),
                       f'{st["reported_pair"]}/{st["n_traces"]}',
                       ms(1e3 * np.median(st['reported_tau'])),
                       f'{np.median(st["reported_freq"]):.1f}'])
    caption = ('\\textbf{Recordings: modes of the AR($p$) spectral fit.} Resolvable modes per trace '
               '(median and range), how many of those are reported: the modes holding at least 1\\% '
               'of the variance, which is the set \\figpanel{fig:realdata}{b,d,e} draws. The number '
               'of modes needed to reproduce the fitted log spectrum '
               'at three cutoffs (median across traces), the fraction of fitted modes dropped because '
               'their characteristic frequency lies outside the fit band, the noise floor, and the aperiodic '
               'mode selected by the rule in Methods (the most important mode with $f\\tau < 1$): the '
               'number of traces in which it is a conjugate pair rather than a real pole, and its '
               'median timescale and frequency (0 for a real pole). The noise floor is the negative '
               'real pole a fit uses when the spectrum flattens onto a floor inside the band: the '
               'number of traces with one, and its median level as a fraction of the fitted power at '
               'the top of the band. It has a level rather than a timescale and is not counted as a '
               'mode.')
    return table('tab:s_modes', caption,
                 ['Recording', stack('Modes', 'per trace'), stack('Reported', 'per trace'),
                  '$R^2 \\ge 0.90$', '$R^2 \\ge 0.95$',
                  '$R^2 \\ge 0.99$', stack('Dropped', '(\\%)'), stack('Noise', 'floor'),
                  stack('Floor', 'level'), stack('Aperiodic', 'mode a pair'),
                  stack('Aperiodic', '$\\hat\\tau$ (ms)'), stack('Aperiodic', '$f$ (Hz)')],
                 rows, 'lrrrrrrrrrrr')


# ---------------------------------------------------------------------------
# S5: windowing, and the Welch-window sensitivity
# ---------------------------------------------------------------------------

def table_s5(stats):
    s = pd.read_csv(RES / 'empirical_stability_requirements.csv')
    rows = []
    for key, label in RECORDINGS:
        g = s[s['dataset'] == key].set_index('method')
        p = g.loc['arp_psd']
        w = stats[key]['tau_by_window']
        med = {k: np.nanmedian(v) for k, v in w.items()}
        excess = g['excess_sd_longest']
        rows.append([label, f'{med["0.5"] / med["1"]:.2f}', f'{med["2"] / med["1"]:.2f}',
                     f'{p["T_over_tau_bias"]:.0f}', f'{p["shortest_over_longest"]:.2f}',
                     f'{excess.median():.0f}', f'{excess.max():.0f}'])
    caption = ('\\textbf{Analysis window and recording window.} The first two columns give the '
               'median AR($p$) spectral timescale with the Welch window halved or doubled, relative to '
               'its value at the window used. The rest concern the windowing analysis: the window '
               'duration, in units of the estimated timescale, beyond which the median AR($p$) spectral '
               'estimate stays within 20\\% of its value in the longest window; the median estimate in '
               'the shortest window relative to the longest; and the spread of estimates across the '
               'longest windows divided by the AR(1) sampling bound of eq. \\ref{eq:finite} at that '
               'duration, its median and maximum across the five estimators.')
    return table('tab:s_windows', caption,
                 ['Recording', stack('Welch', '$\\times\\tfrac12$'), stack('Welch', '$\\times 2$'),
                  stack('$T/\\hat\\tau$', 'to 20\\%'), stack('Shortest/', 'longest'),
                  stack('Excess', 'median'), stack('Excess', 'max')],
                 rows, 'lrrrrrr')


# ---------------------------------------------------------------------------
# S6: disagreement between the two fit domains from sampling alone
# ---------------------------------------------------------------------------

def table_s6():
    df = add_target_column(pd.read_csv(RES / 'benchmark_main.csv.gz'))
    rows = []
    for dur, n in ((DURATION, N_REPS), (FINITE_DURATION, N_REPS_FINITE)):
        d = df[(df['process'] == 'ar1') & np.isclose(df['duration'], dur)]
        row = [f'{dur:g} s ($T = {dur / BENCH_TAU:.0f}\\tau$, {n} repetitions)']
        for a, b in (('ar1_acf', 'ar1_psd'), ('arp_acf', 'arp_psd')):
            x = d[d['method'] == a].set_index('rep')['tau_target']
            y = d[d['method'] == b].set_index('rep')['tau_target'].reindex(x.index)
            r = np.maximum(x / y, y / x)
            row.append(f'{100 * np.mean(r > 1.5):.1f}')
        rows.append(row)
    caption = ('\\textbf{Disagreement between fit domains from sampling alone.} Percentage of '
               'repetitions of the 200 ms AR(1) process in which the ACF and PSD fits of the same '
               'model differ by more than 1.5-fold.')
    return table('tab:s_disagree', caption,
                 ['Duration', 'AR(1) fits (\\%)', 'AR($p$) fits (\\%)'], rows, 'lrr')


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def place(path, tables):
    """Replace each table between its markers in ``path``; fail loudly if a
    label's markers are missing or repeated, without touching the file."""
    text = Path(path).read_text()
    new = text
    for label, body in tables.items():
        pattern = re.compile(r'(% AUTO-TABLE BEGIN ' + re.escape(label) + r'\n)(.*?)(% AUTO-TABLE END '
                             + re.escape(label) + r')', re.DOTALL)
        found = pattern.findall(new)
        if len(found) != 1:
            raise SystemExit(f'{path}: expected one marker pair for {label}, found {len(found)}')
        new = pattern.sub(lambda m: m.group(1) + body + m.group(3), new, count=1)
    changed = new != text
    if changed:
        Path(path).write_text(new)
    return changed


def main():
    OUT.mkdir(exist_ok=True)
    stats = realdata_stats(rerun='--rerun' in sys.argv)
    tables = {'tab:s_benchmark': table_s1(), 'tab:s_damped': table_s2(),
              'tab:s_recordings': table_s3(stats), 'tab:s_modes': table_s4(stats),
              'tab:s_windows': table_s5(stats), 'tab:s_disagree': table_s6()}
    names = {'tab:s_benchmark': 'tab_s1_benchmark', 'tab:s_damped': 'tab_s2_damped',
             'tab:s_recordings': 'tab_s3_recordings', 'tab:s_modes': 'tab_s4_modes',
             'tab:s_windows': 'tab_s5_windows', 'tab:s_disagree': 'tab_s6_disagree'}
    for label, text in tables.items():
        (OUT / f'{names[label]}.tex').write_text(text)
    args = sys.argv[1:]
    if '--write' in args:
        for path in args[args.index('--write') + 1:]:
            print(f'{path}: {"updated" if place(path, tables) else "unchanged"}')
    else:
        for label, text in tables.items():
            print(f'% ---- {label}\n{text}')


if __name__ == '__main__':
    main()

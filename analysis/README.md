# Analysis: interpretable AR(p) timescales

The analysis behind the neural-timescales methods manuscript.

This directory lives on the `analysis` branch.  The `main` branch carries only
the `timescales` package, so that installing the tool does not bring a
particular application of it along.

Every number, table and figure in the manuscript is produced by the pipeline
here.  Nothing is computed by hand and nothing is copied between files: the
manuscript reads the tables this pipeline writes.

The generalizable machinery lives in the parent `timescales` package and is
imported from there: the estimators (`timescales.ar`, `ARPSD` and `ARACF`),
the pole/mode decomposition (`timescales.decomposition`), the generative
processes (`timescales.sim`) and the finite-sample theory
(`timescales.finite`).

The split is deliberate.  `timescales` fits an AR(p) model to whatever
spectrum or autocorrelation it is given and carries no default chosen for a
recording.  Everything that *is* such a choice lives here: the 250-lag budget,
the 4 s Welch window, the 1-100 Hz band, the order grid, the early-stopping
patience, the robust-loss margins, and the constraint that every fit must hold
a real pole so that a timescale can always be reported.


## Setup

From the repository root:

```
pip install -e .[all]
cd analysis
dvc pull                 # the extracted recordings, ~46 MB
```

`dvc pull` needs access to the DVC remote.  Without it,
`scripts/extract_datasets.py` rebuilds `data/` from the public sources, which
takes far longer and needs the download credentials each source requires.

Pin BLAS to one thread whenever anything here runs: the Monte Carlo
parallelizes over repetitions, and nested threading oversubscribes the machine
badly.

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```


## Reproducing everything

```
dvc repro
```

This runs the six fitting stages, the table stage and the figure stage in
dependency order, skipping any stage whose inputs have not changed.  A full run
from cold takes roughly twenty minutes; the fitting stages are most of it and
the table and figure stages take about twenty seconds between them.

```
dvc repro --dry            # what would rerun, without running it
dvc repro fit_benchmark    # one stage and everything downstream of it
dvc repro -f plot_figures  # force, ignoring the staleness check
dvc dag                    # the dependency graph
```

Each stage lists the scripts and library modules it imports, so a change is
scoped to the work it can affect: editing a figure's layout reruns seconds of
plotting, editing an estimator reruns the fits that use it.  A module that only
affects *reporting* is listed on the table and figure stages alone, so changing
how results are presented never invalidates the Monte Carlo.

Raw recordings (`data/`, ~46 MB), result tables and figures are tracked by DVC
rather than git; `dvc checkout` restores them from the cache.


## Checking that the repository is synchronized

```
dvc status
```

Silence means every result and figure on disk was produced by the code
currently in the repository.  Any other output lists, per stage, which
dependencies have changed since the outputs were last written, and those
stages need `dvc repro`.

Run this before trusting a figure.


## Checking that the code is self-consistent

```
python scripts/check_sync.py
```

`dvc status` and `check_sync.py` catch different failures, and both have to be
silent.

DVC tracks **staleness**: it knows which stage wrote which output, and reruns a
stage when a file it depends on changes.

DVC does not track **consistency between two constants that have to agree**.
If a value is written down in two files, DVC sees both files change, reruns
both stages, and produces a wrong result without complaint.  A stage decides
what goes into a result table; a figure has to read that table back on the same
terms.  When the figure restates the stage's constant instead of importing it,
the two drift apart silently.

That is what `check_sync.py` reads for.  It parses the module-level constants
of every stage script and every figure script without importing them, and fails
if a figure defines a name that a stage also defines.  Two stages sharing a
name is not a defect: each stage owns its settings, and `N_REPS` is
legitimately 200 in one and 20 in another.

**A constant a figure and a stage both need belongs to the stage, and the
figure imports it.**  The figure scripts do this for:

| constant | owner | read by |
|---|---|---|
| `FINITE_DURATION`, `FINITE_NLAGS`, `FINITE_SEED`, `N_REPS_FINITE` | `run_02_benchmark.py` | `fig_04.py` |
| `DURATION`, `WELCH_WIN_S`, `OSCILLATORY_TRUTH` | `run_02_benchmark.py` | `fig_04.py` |
| `SNRS`, `FREQS` | `run_03_oscillations.py` | `fig_s1.py` |
| `SIM_FS`, `AR1_TAU`, `BENCH_TAU`, `ARP_MODES`, `AR2_RHYTHM` | `tsmodes/constants.py` | every stage and figure |

`dvc.yaml` lists `run_02_benchmark.py` and `run_03_oscillations.py` as
dependencies of `plot_figures` for this reason: changing a condition redraws
the figure that reads it.

### The failure this prevents

The finite-sample condition of the benchmark figure was defined as
`FINITE_DURATION` in `scripts/run_02_benchmark.py` and again as `T_FINITE` in
`scripts/figures/fig_04.py`.  Changing the duration in the stage left the
figure filtering the result table for a duration that was no longer in it, and
the panel came out empty.  Nothing errored.  `check_sync.py` reports that
pairing before it can happen again.


## Layout

```
dvc.yaml            the pipeline: six fitting stages, tables, figures
dvc.lock            the hash of every dependency and output, written by dvc repro
data/               extracted recordings (dvc-tracked, not in git)
results/            result tables and raw Monte-Carlo output (dvc-tracked)
figures/            manuscript figures (dvc-tracked)
tables/             supplementary tables, written by scripts/make_tables.py
debug/              scratch diagnostics, git-ignored and outside the pipeline
scripts/            one script per stage, plus check_sync.py and make_tables.py
  figures/          one module per figure, run by plot_figures.py
tsmodes/            code specific to this paper
  estimators.py     the six estimators compared here, and their settings
  prep.py           the Welch PSD and band-limited ACF they all share
  constants.py      the AR(7), AR(5) and AR(2) the figures simulate
  datasets.py       loading + common preprocessing for the real recordings
  benchmark.py      Monte-Carlo harness
  plotting.py       shared style and reusable panels
tests/              tests of the analysis layer: pytest tests  (74 tests)
```

The tests for the estimators, modes, processes and finite-sample theory live in
the package with the code and run from the repository root with `pytest`.


## Estimators

Six estimators on a 3 x 2 grid of domain and model order.  All consume the same
prepared signal (one Welch PSD, one ACF) so that differences between them are
properties of the model, not of the preprocessing.

|            | AR(1)        | AR(p)        |
|------------|--------------|--------------|
| time series| `ar1_direct` | `arp_direct` |
| ACF        | `ar1_acf`    | `arp_acf`    |
| spectrum   | `ar1_psd`    | `arp_psd`    |

The ACF and spectral fits are deliberately symmetric: the same AR(p) model, the
same robust (soft-L1) loss, the same reflection-coefficient parameterization,
and BIC order selection on the same grid.  Every fitted model is scored in
*both* domains, because the two objectives are not equivalent.


## Real data

Eight recordings across fMRI, MEG, EEG, ECoG, LFP and spike counts in four
species.  Electrophysiology is resampled to 250 Hz and fit over 1-100 Hz with
line noise excluded; fMRI and spike counts carry their own analysis bands.

`scripts/extract_datasets.py` rebuilds `data/` from the public sources and must
be run with the `oscillatory_eigenvectors` environment, which has the readers
(mne, h5py, allensdk) and the cached downloads:

```bash
/home/rph/oscillatory_eigenvectors/.venv/bin/python scripts/extract_datasets.py
```

`dvc pull` is the fast path and should be preferred.


## Supplementary tables

`dvc repro` rebuilds `tables/tab_s*.tex` from the result tables, in the
`make_tables` stage, so `dvc status` reports them stale like anything else.
The tables are derived from the results, never typed.

Putting them into the manuscript is a separate step, on purpose, because it
edits a file the pipeline does not own:

```
python scripts/make_tables.py --write <the current manuscript>
```

**Ask Ryan for the current manuscript before running this.**  The live document
is in Overleaf; the copy in this repository is whatever was last exported from
it, and is not authoritative.  Writing tables into a stale export produces a
file that looks updated and is not.  Older exports (`paper_revised.tex`,
`paper_overleaf.tex`) are not the manuscript.

Each table sits between `% AUTO-TABLE BEGIN <label>` and
`% AUTO-TABLE END <label>`.  The script refuses to touch a file whose markers
are missing or duplicated, and changes nothing outside them.


## Conventions

AR(p) is written `x_t = sum_k phi_k x_{t-k} + eps_t`, so the characteristic
polynomial is `z^p - phi_1 z^{p-1} - ... - phi_p` and stationarity requires all
poles inside the unit circle.  The one-sided spectral density convention is
`S(f) = (2 sigma^2 / fs) / |1 - sum_k phi_k e^{-i 2 pi f k / fs}|^2`, which
integrates to the process variance over `[0, fs/2]` and matches
`scipy.signal.welch`.


## After changing an estimator

The estimators live in the `timescales` package, not here.  A change there
affects both the paper and everyone using the package, so:

1. `pytest` from the repository root, including the slow tests
   (`pytest -m slow`), which fit every example recording.
2. `python scripts/check_sync.py` here.
3. `dvc repro` here, which refits and rebuilds `tables/`.
4. Ask Ryan for the current manuscript exported from Overleaf, then
   `python scripts/make_tables.py --write <that file>`.
5. Re-read the manuscript numbers that are quoted in prose rather than
   generated.  `grep -n '[0-9]\+\\%' <that file>` finds most of them.


## Before submitting

```
python scripts/check_sync.py     # no figure restates a stage constant
dvc status                       # nothing on disk is stale
pytest -m slow                   # from the repository root
```

All three silent means every figure and table on disk was produced by the code
currently in the repository, and no two files hold a constant that has to
agree.  A stale result is not detectable by looking at it.

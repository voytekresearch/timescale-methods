# Revision: interpretable AR(p) timescales

Self-contained sub-module for the revised neural-timescales manuscript
(`../overleaf/main_claude.tex`).  Nothing in the parent `timescales/` package is
modified; `tsmodes` reimplements the few primitives it needs so it runs against
any build of the parent package.

```
revision/
  tsmodes/          the library
    modes.py        AR(p) poles -> modes: decay time, frequency, importance
    estimators.py   one fitting/evaluation API for all six estimators
    sims.py         generative processes with closed-form PSD and ACF
    finite.py       analytic finite-sample bias / variance / CIs for AR(1)
    benchmark.py    Monte-Carlo harness
    datasets.py     loading + common preprocessing for the real recordings
    plotting.py     shared style and reusable panels
  scripts/          one script per stage (see below)
  tests/            pytest suite (133 tests)
  data/             extracted continuous traces (.npz, git-ignored)
  results/          tables and raw Monte-Carlo output
  figures/          manuscript figures (.svg + .png)
```

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

## Running

Always pin BLAS to one thread; the Monte Carlo parallelizes over repetitions and
nested threading oversubscribes the machine badly.

```bash
cd revision
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m pytest tests -q                     # 133 tests

python scripts/run_01_modes.py                # Fig 6  fig_modes
python scripts/run_02_benchmark.py --rerun    # Fig 4  fig_benchmark
python scripts/run_03_oscillations.py --rerun # Fig 5  fig_oscillations
python scripts/run_04_finite.py               # Fig S4 figS_finite
python scripts/run_05_real_data.py            # Fig 7  fig_realdata
python scripts/run_06_empirical_truncation.py --rerun   # Fig S5 figS_truncation
python scripts/run_07_windows.py              # Fig 8  fig_windows
python scripts/make_supplement.py             # Figs S1-S3
python scripts/sync_overleaf.py               # copy figures to ../overleaf
```

Scripts with a `--rerun` flag reuse cached `results/*.csv.gz` otherwise, so
figures can be re-styled without repeating the Monte Carlo.  `run_05`, `run_06`
and `run_07` require `scripts/extract_datasets.py` to have been run first.
Total Monte-Carlo runtime is roughly 45 minutes on 12 cores.

## Real data

`scripts/extract_datasets.py` must be run with the `oscillatory_eigenvectors`
environment, which has the readers (mne, h5py, allensdk) and the cached
downloads:

```bash
/home/rph/oscillatory_eigenvectors/.venv/bin/python scripts/extract_datasets.py
```

Nine recordings across fMRI, MEG, EEG, ECoG and LFP in four species.  Seven come
from the dataset registry of
`oscillatory_eigenvectors/notebooks/06_multimodal_eigenspectra_2s.ipynb`, but as
*continuous* traces rather than 2 s windows, because the timescale and windowing
analyses need long uninterrupted segments.  The MEG (FieldTrip cortico-muscular
coherence) and fMRI (ABIDE, C-PAC `nofilt_noglobal` ROI time series) datasets are
added here.  Electrophysiology is resampled to 250 Hz and fit over 1-100 Hz with
line noise excluded; fMRI carries its own analysis band.

## Conventions

AR(p) is written `x_t = sum_k phi_k x_{t-k} + eps_t`, so the characteristic
polynomial is `z^p - phi_1 z^{p-1} - ... - phi_p` and stationarity requires all
poles inside the unit circle.  The one-sided spectral density convention is
`S(f) = (2 sigma^2 / fs) / |1 - sum_k phi_k e^{-i 2 pi f k / fs}|^2`, which
integrates to the process variance over `[0, fs/2]` and matches
`scipy.signal.welch`.

## Reproducing

The analysis is a DVC pipeline (`dvc.yaml`) with fitting and plotting as
separate stages, so the cost of a change matches its size:

    dvc repro                 # rebuild whatever is stale
    dvc status                # what would rebuild, and why
    dvc dag                   # the dependency graph

Editing `tsmodes/plotting.py` or a figure's layout rebuilds only the seven
plotting stages, which take seconds. Editing an estimator rebuilds the fits
that use it, and the figures downstream of those. Raw recordings
(`revision/data`, ~46 MB) and every result table are tracked by DVC rather
than by git; `dvc checkout` restores them from the cache.

Individual scripts still run standalone. Each takes `--figures-only` to
replot from cached tables, and the fitting scripts take `--rerun` to force
recomputation.

"""PSD and ACF fitting optimization."""

from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm.notebook import tqdm

import numpy as np

from neurodsp.spectral import compute_spectrum

from fooof import FOOOF

from timescales.fit import convert_knee_val, ACF


def fit_grid(sig, fs, grid, mode='psd', max_n_params=None, rsq_thresh=0,
             n_jobs=-1, chunksize=1):

    # Create an array of param grid indices
    param_inds = np.array(list(product(*[list(range(len(i))) for i in list(grid.values())])))

    if max_n_params is not None and max_n_params < len(param_inds):
        sub_inds = np.random.choice(np.arange(len(param_inds)), max_n_params, replace=False)
        param_inds = param_inds[sub_inds]

    # Fit grid
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    with Pool(processes=n_jobs) as pool:

        if mode == 'psd':
            mapping = pool.imap(partial(_fit_psd, sig=sig, fs=fs, grid=grid,
                                        rsq_thresh=rsq_thresh),
                                param_inds, chunksize=chunksize)
        elif mode == 'acf':
            mapping = pool.imap(partial(_fit_acf, sig=sig, fs=fs, grid=grid,
                                        rsq_thresh=rsq_thresh),
                                param_inds, chunksize=chunksize)

        params = list(tqdm(mapping, desc='Fitting Spectra or ACF', total=len(param_inds)))

    return params


def _fit_psd(index, sig=None, fs=None, grid=None, rsq_thresh=0):
    """Parallel PSD wrapper."""

    init_params = ['peak_width_limits', 'max_n_peaks', 'peak_threshold']
    self_params = ['knee_freq_bounds', 'exp_bounds']
    spec_params = ['nperseg', 'noverlap']

    init_kwargs = {}
    self_kwargs = {}
    spec_kwargs = {}

    for (k, v), ind in zip(grid.items(), index):

        if k in init_params:
            init_kwargs[k] = v[ind]
        elif k in spec_params:
            spec_kwargs[k] = v[ind]
        elif k in self_params:
            self_kwargs[k] = v[ind]
        elif k == 'freq_range':
            f_range = tuple(v[ind])

    # Compute spectrum
    freqs, powers = compute_spectrum(sig, fs, f_range=f_range, **spec_kwargs)

    # Fit
    peak_width_limits = tuple(init_kwargs.pop('peak_width_limits'))

    fm = FOOOF(aperiodic_mode='knee', verbose=False, peak_width_limits=peak_width_limits,
               **init_kwargs)

    # Bounds for aperiodic fitting, as: ((offset_low_bound, knee_low_bound, exp_low_bound),
    #                                    (offset_high_bound, knee_high_bound, exp_high_bound))
    knee_low_bound, knee_high_bound = self_kwargs['knee_freq_bounds']
    exp_low_bound, exp_high_bound = self_kwargs['exp_bounds']

    fm._ap_bounds = ((-np.inf, knee_low_bound, exp_low_bound),
                     (np.inf, knee_high_bound,exp_high_bound))

    fm._ap_guess = (None, (knee_low_bound+knee_high_bound)/2, (exp_low_bound+exp_low_bound)/2)

    try:
        fm.fit(freqs, powers, freq_range=f_range)
        knee_freq = fm.get_params('aperiodic', 'knee')
        rsq = fm.r_squared_
    except:
        knee_freq = np.nan
        rsq = np.nan
        fm = np.nan

    if rsq < rsq_thresh:
        knee_freq = np.nan
        rsq = np.nan
        fm = np.nan

    return [index, knee_freq, rsq, fm]


def _fit_acf(index, sig=None, fs=None, grid=None, rsq_thresh=0):

    bounds = [[], []]
    guess = []
    for (k, v), ind in zip(grid.items(), index):

        if k == 'nlags':
            nlags = v[ind]
        else:
            bounds[0].append(v[ind][0])
            bounds[1].append(v[ind][1])

            if v[ind][0] is None or v[ind][1] is None:
                guess.append(None)
            else:
                guess.append((v[ind][0] + v[ind][1])/2)

    # Compute acf
    acf = ACF(low_mem=True)
    acf.compute_acf(sig, fs, nlags=nlags)

    acf.fit_cos(bounds=bounds, guess=guess, maxfev=1000)

    if np.isnan(acf.params).any() or acf.rsq < rsq_thresh:
        return [index, np.nan, np.nan, np.nan]

    knee_freq = convert_knee_val(acf.params[0])

    return[index, knee_freq, acf.rsq, acf]

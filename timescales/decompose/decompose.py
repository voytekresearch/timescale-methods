"""Autoregressive deomposition."""

from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.fft import fftfreq

from statsmodels.tsa.arima.estimators.burg import burg

from spectrum import eigen


def decompose_ar(sig, fs, osc_order, ar_order, bounds=None, guess=None, maxfev=1000):
    """Decompose a signal in periodic and aperiodic components.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    osc_order : int
        Oscillatory model order.
    ar_order : int
        Autoregressive order.
    bounds : list, optional, default: None
        Parameter bounds for curve_fit.
    guess : list, optional, default: None
        Parameter guess for curve_fit.
    maxfev : int, optional, default: 1000
        Maximum number of fitting iterations.

    Returns
    -------
    params : 1d array
        Oscillatory and/or autoregressive parameters.
    """

    xs = np.arange(len(sig))

    if bounds is None:
        bounds = [
            [.2, -1, 0, -1] * osc_order,
            [ 2,  1, 100, 1] * osc_order
        ]

    if guess is None:
        guess = [1, 0, 1, 0] * osc_order


    # Inital oscillatory fit
    if ar_order > 0:

        guess, _ = curve_fit(
            lambda xs, *args : _fit_decompose_ar(xs, sig, fs, osc_order, 0, *args),
            xs, sig, bounds=bounds, p0=guess, maxfev=maxfev
        )

    # Final Fit
    params, _ = curve_fit(
        lambda xs, *args : _fit_decompose_ar(xs, sig, fs, osc_order, ar_order, *args),
        xs, sig, bounds=bounds, p0=guess, maxfev=maxfev
    )

    return params


def _fit_decompose_ar(xs, sig, fs, osc_order, ar_order, *fit_args, return_params=False):
    """Fit a signal decomposition model.

    Parameters
    ----------
    xs : 1d array
        Sample definition.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    osc_order : int
        Oscillatory model order.
    ar_order : int
        Autoregressive order.
    fit_args : list
        Fit parameters as [heights, phis, freqs, rdsyms].
    return_params : bool, optional, default: False
        Return fit parameters when True. Returns only signal fit when False.
        If true,

    Returns
    -------
    full_fit : 1d array, optional
        Full model fit. Returned if return_params is False.
    osc_fit : 1d array, optional
        Oscillatory fit. Returned if osc_order is > 0 and return_paras is True.
    ar_fit : 1d array, opttional
        Autoregressive fit. Returned if ar_order is > 0 and return_params is True.
    params : 1d array, optional
        Oscillatory and/or autoregressive parameters Returned if return_params is True.
    """

    heights, phis, freqs, rdsyms = np.array(fit_args).reshape(osc_order, -1).T
    params = {}

    # Construct (asym) oscillation
    if osc_order > 0:

        osc_fit = np.zeros(len(xs))

        for i in range(osc_order):

            osc_fit += sim_asine(xs, fs, freqs[i], rdsyms[i],
                                 phis[i], heights[i])

        params['osc_height'] = heights
        params['osc_phis'] = phis
        params['osc_freqs'] = freqs
        params['osc_asym'] = phis

    if ar_order == 0 and not return_params:
        return osc_fit
    elif ar_order == 0:
        return osc_fit, params

    # Fit AR Burg
    b_params, _ = burg(sig-osc_fit, ar_order=ar_order, demean=True)
    ar_params = b_params.ar_params
    params['ar_coefs'] = ar_params

    ar_fit = gen_ar_fit(sig-osc_fit, ar_params)

    if not return_params:
        return osc_fit + ar_fit

    if ar_order > 0 and osc_order > 0 :
        return osc_fit, ar_fit, params
    elif ar_order > 0:
        return ar_fit, params
    elif osc_order > 0:
        return osc_fit, params


def gen_ar_fit(sig, ar_params):
    """Generate autoregressive fit.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    ar_params : 1d array
        Autoregressive parameters.

    Returns
    -------
    ar_fit : 1d array
        Autoregressive model fit.
    """
    ar_params = ar_params[::-1]
    ar_order = len(ar_params)

    _sig = np.pad(sig, ar_order)

    ar_fit = np.zeros(len(_sig))
    for i in range(ar_order, len(_sig)):
        ar_fit[i] = sum(_sig[i-ar_order:i] * ar_params)

    ar_fit = ar_fit[ar_order:-ar_order]

    return ar_fit


def sim_asine(xs, fs, freq, rdsym, phi, height):
    """Simulate an asymmetrical sinusoidal wave.

    Parmeters
    ---------
    xs : 1d array
        Sample indices.
    fs : float
        Sampling rate, in Hz.
    freq : float
        Oscillatory frequency, in Hz.
    rdsym : float
        Rise-decay symmetry of oscillation.
    phi : float
        Phase of oscillation.
    height : float
        Height of oscillations.

    Returns
    -------
    sig : 1d array
        Oscillatory signal.
    """

    pha = np.exp(1.j * np.pi * phi)

    sig = height * pha * np.exp(2.j * np.pi * (freq/fs) * xs)

    sig = (sig * np.exp(1.j * rdsym * sig)).real

    return sig


def iter_estimate_freq(sig, fs, osc_order, n_eig, freq_pad=10):
    """Iterative oscillatory frequency search.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    osc_order : int
        Oscillatory model order.
    n_eig : int
        Number of eigenvalues to compute the pseudo spectrum.
    freq_pad : int
        Frequency search range.

    Returns
    -------
    params : 1d array
        Model parameters with highest r-squared.
    bounds : 1d array
        Model bounds with highest r-squared.
    """

    xs = np.arange(len(sig))

    # Get initial estimate
    freq_guess = estimate_freq(sig, fs, n_eig).round()

    lower_freq = freq_guess - freq_pad
    lower_freq = 1 if lower_freq < 0 else lower_freq
    upper_freq = freq_guess + freq_pad

    bounds = [
        [0, -1, lower_freq, -1] * osc_order,
        [2,  1, upper_freq, 1] * osc_order
    ]

    guess = [1, 0, freq_guess, 0] * osc_order

    f_range = np.arange(lower_freq, upper_freq+1)

    rsq = -1
    params = None
    freq = None

    # Step through range of frequencies
    for f in f_range:

        guess = np.array(guess).reshape(osc_order, -1)
        guess[:, 2] = f
        guess = guess.flatten()

        osc_params = decompose_ar(sig, fs, osc_order, 0,
                                  bounds=bounds, guess=guess, maxfev=1000)

        osc_fit = _fit_decompose_ar(xs, sig, fs, osc_order, 0, *osc_params, return_params=False)

        _rsq = np.corrcoef(osc_fit, sig)[0][1]

        if _rsq > rsq:
            rsq = _rsq
            params = osc_params
            bounds = bounds

    return params, bounds


def estimate_freq(sig, fs, n_eig):
    """Estimate oscillatory frequency using the eigenvalue pseudo spectrum.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    n_eig : int
        Number of eigenvalues to compute the pseudo spectrum.

    Returns
    -------
    freq_guess : float
        Peak frequency in the pseudo spectrum.
    """

    psd, _ = eigen(sig, n_eig, method='ev')

    freqs = fftfreq(4096, 1/fs)
    freqs = freqs[:4096//2]

    psd = psd[4096//2:]

    peaks, _ = find_peaks(psd)

    if len(peaks) == 0:
        raise ValueError('No peaks detected.')

    freq_guess = freqs[peaks[0]]

    return freq_guess


def decompose_ar_windows(sig, fs, osc_order, ar_order, nperseg, noverlap,
                         bounds=None, guess=None, maxfev=1000, n_jobs=-1, progress=None):
    """Windowed autoregressive decomposition.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    osc_order : int
        Oscillatory model order.
    ar_order : int
        Autoregressive order.
    nperseg : int
        Samples per window.
    noverlap : int
        Overlapping samples between windows.
    bounds : list, optional, default: None
        Parameter bounds for curve_fit.
    guess : list, optional, default: None
        Parameter guess for curve_fit.
    maxfev : int, optional, default: 1000
        Maximum number of fitting iterations.
    n_jobs : int, optional, default: -1
        Number of jobs to run in parallel. Defaults to all cores.
    progress : func, {tqdm.notebook.tqdm or tqdm.tqdm}
        Progress bar.

    Returns
    -------
    osc_fit : 1d array
        Oscillatory fit.
    ar_fit : 1d array, optional
        Autoregressive fit. Returned if ar_order is > 0.
    params : 1d array
        Oscillatory and/or autoregressive parameters.
    t_def : 2d array
        Time definition of windows as [start, end].
    """

    # Window signal
    inds = np.arange(0, len(sig)-nperseg+1, nperseg-noverlap)

    starts = inds
    ends = inds + nperseg

    sig_win = np.array([sig[s:e] for s, e in zip(starts, ends)])
    xs = np.arange(len(sig_win[0]))

    # Parallel
    decompose_ar(sig, fs, osc_order, ar_order, bounds=None, guess=None, maxfev=1000)

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    with Pool(n_jobs) as pool:

        mapping = pool.imap(partial(decompose_ar, fs=fs, osc_order=osc_order, ar_order=ar_order,
                            bounds=bounds, guess=guess, maxfev=maxfev), sig_win)

        if progress is None:
            osc_params = np.array(list(mapping, total=len(sig_win)))
        else:
            osc_params = np.array(list(progress(mapping, total=len(sig_win))))

    # Regenerate fits
    t_def = np.column_stack((starts, ends))

    osc_fit = np.zeros_like(sig_win)

    params = []

    # Only fit oscillatory model
    print(osc_params[-2])
    print(osc_params[-1])
    if ar_order == 0:

        for ind, (_sig, _params) in enumerate(zip(sig_win, osc_params)):

            _osc_fit, _full_params = _fit_decompose_ar(
                xs, _sig, fs, osc_order, ar_order, *_params, return_params=True)

            osc_fit[ind] = _osc_fit
            params.append(_full_params)

        return osc_fit, params, t_def

    # Fit oscillatory and ar models
    ar_fit = np.zeros_like(sig_win)

    for ind, (_sig, _params) in enumerate(zip(sig_win, osc_params)):

        _osc_fit, _ar_fit, _full_params = _fit_decompose_ar(
            xs, _sig, fs, osc_order, ar_order, *_params, return_params=True)

        osc_fit[ind] =_osc_fit
        ar_fit[ind] = _ar_fit

        params.append(_full_params)

    return osc_fit, ar_fit, params, t_def



"""Preparation of the PSD and ACF this paper's estimators share.

One :func:`prepare` call turns a time series into a :class:`Signal`: a Welch
(or multitaper) spectrum, the autocorrelation implied by that same spectrum
over the same analysis band, and the masks describing which frequencies are
fit.  Every estimator in :mod:`tsmodes.estimators` reads this one object, so
differences between estimators cannot come from preprocessing.

This is analysis code, not library code: the lag budget, the segment length
and the band are choices made for these recordings.  The general estimators
take ``(freqs, powers)`` or ``(lags, corrs)`` directly and impose none of
them; :mod:`timescales.utils` has plain helpers for computing either.

Deriving the ACF from the band-limited spectrum rather than from the raw
signal is what makes the lag and frequency domains comparable: the raw
autocorrelation carries drift below the fit band and line noise within it,
which the spectral fit never sees.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, welch

from timescales.conversions import phi_to_tau, tau_to_phi  # noqa: F401

# ---------------------------------------------------------------------------
# Shared signal container
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A time series plus the PSD and ACF every estimator shares."""

    sig: np.ndarray
    fs: float
    freqs: np.ndarray = None
    powers: np.ndarray = None
    lags: np.ndarray = None
    corrs: np.ndarray = None
    f_range: tuple = None
    nperseg: int = None
    n_segments: int = None
    name: str = ''

    @property
    def n_samples(self):
        return len(self.sig)

    @property
    def duration(self):
        return len(self.sig) / self.fs

    @property
    def fit_mask(self):
        lo, hi = self.f_range
        return (self.freqs >= lo) & (self.freqs <= hi) & (self.freqs > 0)


N_LAGS = 250          # lag budget for ACF-domain fitting
def multitaper_psd(sig, fs, nw=4.0, n_tapers=None):
    """Multitaper spectrum of a whole record, and its implied lag window.

    Welch trades resolution for variance by cutting the record up, which a
    short record cannot afford: 296 fMRI samples leave either too few
    frequency bins to fit or too few segments to average.  Multitapering
    averages orthogonal tapers of the *whole* record instead, so it buys the
    same variance reduction without spending record length.
    """
    from scipy.signal.windows import dpss
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    # A short record cannot support a large time-bandwidth product: DPSS
    # requires more samples than tapers, and a resolution bandwidth of 2NW/n
    # wider than the band is meaningless anyway.  Windowed analyses cut the
    # same recording into pieces a few samples long, so clamp rather than
    # fail.
    nw = float(np.clip(nw, 1.25, max(1.25, n / 8.0)))
    n_tapers = int(2 * nw - 1) if n_tapers is None else int(n_tapers)
    n_tapers = int(np.clip(n_tapers, 1, max(1, n - 1)))
    tapers = dpss(n, nw, n_tapers)
    power = np.zeros(n // 2 + 1)
    for taper in tapers:
        power += np.abs(np.fft.rfft(sig * taper)) ** 2
    power *= 2.0 / (n_tapers * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # The estimate is the average of tapered periodograms, so the lag window
    # it implies is the average of the tapers' autocorrelations.
    lagwin = np.mean([np.correlate(t, t, 'full')[n - 1:] for t in tapers],
                     axis=0)
    return freqs, power, lagwin / lagwin[0], n_tapers


def prepare(sig, fs, welch_win_s=2.0, nlags_s=None, nlags=None, f_range=None,
            name='', detrend=False, exclude_bands=(), welch_overlap=0.5,
            method='welch', nw=4.0):
    """Compute the PSD and ACF used by every estimator.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, Hz.
    welch_win_s : float
        Welch segment length in seconds (capped at the signal duration).
    nlags_s : float, optional
        ACF lag range in seconds.  By default a fixed budget of ``N_LAGS``
        lags is used, which keeps the cost of ACF-domain fitting constant and
        spans many timescales at any sampling rate.  A longer range was
        tried and rejected: on the AR(7) benchmark one second of lags at
        500 Hz widened the lag-domain estimate (taper-corrected tail lags are
        noisy) without moving its median.
    f_range : tuple, optional
        Frequency range used for spectral fitting.  Defaults to
        ``(1 / welch_segment_seconds, fs / 2)``.
    detrend : False or str
        Per-segment detrending passed to Welch's method.  The signal is
        mean-centered as a whole, and removing the mean of every segment on
        top of that acts as a high-pass at the segment's resolution: it takes
        power out of the lowest bins, where a slow relaxation's knee sits.
        On the AR(7) benchmark (200 ms relaxation, 4 s segments, 100 s) the
        spectral AR(p) fit returned a median of 183 ms with per-segment
        detrending and 197 ms without, the same as Burg's method, so the
        default is no per-segment detrending.
    exclude_bands : sequence of (lo, hi)
        Frequency bands (e.g. line noise) removed before spectral fitting.
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    n = len(sig)

    if method == 'multitaper':
        freqs, powers, lagwin, n_segments = multitaper_psd(sig, fs, nw=nw)
        nperseg = n
    else:
        nperseg = int(min(n, round(welch_win_s * fs)))
        nperseg = max(nperseg, 8)
        noverlap = int(round(welch_overlap * nperseg))
        freqs, powers = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                              window='hann', detrend=detrend, scaling='density')
        n_segments = max(1, 1 + (n - nperseg) // (nperseg - noverlap))
        lagwin = None

    if f_range is None:
        f_range = (fs / nperseg, fs / 2)

    if nlags is None:
        nlags = N_LAGS if nlags_s is None else int(round(nlags_s * fs))
    nlags = int(max(4, min(n // 2, n - 2, nlags)))

    keep = np.ones(len(freqs), dtype=bool)
    for lo, hi in exclude_bands:
        keep &= ~((freqs >= lo) & (freqs <= hi))

    # The autocorrelation is derived from the same band-limited spectrum the
    # spectral estimators see, rather than from the raw signal.  Otherwise the
    # two domains are fit to different data: the raw autocorrelation carries
    # drift below the fit band and everything above it, so a lag-domain fit
    # would report a timescale set by frequencies the spectral fit excluded.
    # Cut only where content is deliberately excluded.  The lower edge of
    # f_range is usually just the Welch resolution floor, and rejecting
    # everything below it would put a brick wall in the spectrum that the lag
    # domain sees as ringing, costing poles to describe an artefact of the
    # analysis.  A low edge the caller actually asked for is a different
    # matter and is applied.
    f_lo = f_range[0] if f_range[0] > 1.5 * fs / nperseg else 0.0
    band = (freqs >= f_lo) & (freqs <= f_range[1])
    corrs = band_limited_acf(freqs, powers, nlags, band & keep, band,
                             nperseg=nperseg, window='hann', lagwin=lagwin)
    lags = np.arange(len(corrs))          # the taper correction may shorten it

    out = Signal(sig=sig, fs=fs, freqs=freqs, powers=powers, lags=lags,
                 corrs=corrs, f_range=f_range, nperseg=nperseg,
                 n_segments=n_segments, name=name)
    out.exclude_mask = keep
    out.band_mask = band
    out.raw_corrs = _acf(sig, len(corrs) - 1)
    # Adjacent multitaper bins inside the resolution bandwidth 2*NW/T are
    # correlated; the spectral fits thin to that spacing.
    out.fit_decimate = int(round(2 * nw)) if method == 'multitaper' else 1
    return out


def band_limited_acf(freqs, powers, nlags, keep, band, nperseg=None,
                     window='hann', lagwin=None):
    """Autocorrelation of a spectrum restricted to the analysis band.

    Bins inside ``band`` but excluded by ``keep`` (line noise and its
    harmonics) are filled by log-linear interpolation from their neighbours
    rather than zeroed, so that notching does not ring in the lag domain.

    Welch's estimate is the periodogram of tapered segments, so the
    autocovariance it implies is the true one multiplied by the taper's own
    autocorrelation -- 0.66 at a lag a quarter of the segment length, for a
    Hann window.  Left in, that taper masquerades as a faster decay and biases
    every timescale read off the result, so it is divided out.
    """
    powers = np.asarray(powers, dtype=float).copy()
    if np.any(band & ~keep) and np.any(band & keep):
        good = band & keep
        powers[band & ~keep] = np.exp(np.interp(
            freqs[band & ~keep], freqs[good], np.log(np.maximum(powers[good],
                                                                1e-300))))
    powers[~band] = 0.0
    acov = np.fft.irfft(powers)[: nlags + 1]
    if lagwin is not None or nperseg:
        if lagwin is None:
            taper = get_window(window, int(nperseg))
            lagwin = np.correlate(taper, taper, 'full')[int(nperseg) - 1:]
        lagwin = np.asarray(lagwin, dtype=float)
        lagwin = lagwin[: nlags + 1] / lagwin[0]
        # Below about a fifth of the taper's energy the correction amplifies
        # noise faster than it removes bias, so those lags are dropped.
        acov = acov[: max(2, int(np.sum(lagwin > 0.2)))]
        acov = acov / lagwin[: len(acov)]
    return acov / acov[0]


def spectrum_acov(signal, powers, nlags=None):
    """Autocovariance implied by a spectrum on ``signal``'s frequency grid,
    restricted to the band ``signal.corrs`` was derived from.

    :func:`prepare` derives the autocorrelation from the spectrum inside the
    analysis band, so a model fit to that spectrum is translated to the lag
    domain the same way: its spectrum on the same grid, zero outside the same
    band, inverse transformed.  The model's full-band autocorrelation would
    instead carry its extrapolation outside the band, mostly the plateau below
    the lower edge, and disagree with the data even where the fit is exact.
    A model spectrum carries no Welch taper, so none is divided out.  The
    result is left unnormalized so that additive parts of a spectrum (the mode
    components) can be scaled by their total; ``powers`` may be 2d, one
    spectrum per row.
    """
    nlags = len(signal.corrs) - 1 if nlags is None else int(nlags)
    powers = np.array(powers, dtype=float)
    powers[..., ~signal.band_mask] = 0.0
    return np.fft.irfft(powers, axis=-1)[..., : nlags + 1]


def _acf(sig, nlags):
    """Biased (FFT) sample autocorrelation, lags 0..nlags."""
    x = sig - sig.mean()
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1)))
    spec = np.fft.rfft(x, nfft)
    corrs = np.fft.irfft(spec * np.conj(spec), nfft)[: nlags + 1]
    return corrs / corrs[0]

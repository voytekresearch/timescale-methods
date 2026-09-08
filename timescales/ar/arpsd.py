"""AR(p) fit to a power spectrum."""

import numpy as np

from timescales.decomposition.modes import ar_psd, stabilize
from .base import ARBase
from .core import fit_ar_psd, select_order_psd
from .metrics import r2_logpsd, rmse_logpsd, is_divergence


class ARPSD(ARBase):
    """Fit an AR(p) model to a power spectrum.

    The AR(p) spectrum

        ``S(f) = sigma2 / |1 - sum_k phi_k exp(-i 2 pi f k / fs)|^2``

    is fit to ``log10`` power, so every octave is weighted by how well its
    shape is described rather than by how much power it carries.  At ``p = 1``
    this is a Lorentzian and the fit reduces to the familiar knee model; above
    it, the same model describes peaks and multiple bends without any of them
    being added by hand.

    The estimator does not compute the spectrum.  Pass whatever ``(freqs,
    powers)`` you trust -- Welch, multitaper, wavelet, an average over trials.

    Parameters
    ----------
    See :class:`~timescales.ar.base.ARBase`.

    Examples
    --------
    >>> from timescales.ar import ARPSD                        # doctest: +SKIP
    >>> ar = ARPSD(order=range(1, 11), fs=500.)                # doctest: +SKIP
    >>> ar.fit(freqs, powers)                                  # doctest: +SKIP
    >>> ar.decompose()                                         # doctest: +SKIP
    >>> ar.tau                                                 # doctest: +SKIP
    """

    domain = 'psd'

    def __init__(self, order=1, fs=None, criterion='bic', loss='linear',
                 f_scale=None, k_bound=None, require_relaxation=False,
                 patience=None, maxfev=4000, guess=None):
        from .core import K_BOUND
        super().__init__(order=order, fs=fs, criterion=criterion, loss=loss,
                         f_scale=f_scale,
                         k_bound=K_BOUND if k_bound is None else k_bound,
                         require_relaxation=require_relaxation,
                         patience=patience, maxfev=maxfev, guess=guess)
        self.freqs = None
        self.powers = None
        self.offset = np.nan
        self.stabilized_ = False

    @property
    def _offset(self):
        return 1.0 if not np.isfinite(self.offset) else self.offset

    def decompose(self, freqs=None, n_freqs=513, f_range=None):
        """Decompose the fitted model into modes.

        Mode importance defaults to being measured over the frequencies that
        were fit, rather than the full band: a model is only evidence about
        the band it saw, and ranking its modes by their effect outside that
        band would rank them on an extrapolation.
        """
        if freqs is None and f_range is None:
            freqs = self.freqs
        return super().decompose(freqs=freqs, n_freqs=n_freqs, f_range=f_range)

    def fit(self, freqs, powers):
        """Fit the model to a spectrum.

        Parameters
        ----------
        freqs : 1d array
            Frequencies, Hz.  Only the points passed are fit, so restrict them
            to the band of interest before calling.
        powers : 1d array
            Power at each frequency, strictly positive.

        Returns
        -------
        self
        """
        freqs = np.asarray(freqs, dtype=float)
        powers = np.asarray(powers, dtype=float)
        if freqs.shape != powers.shape:
            raise ValueError('`freqs` and `powers` must have the same shape.')
        if np.any(powers <= 0):
            raise ValueError('`powers` must be positive; the fit is in log power.')
        self.freqs, self.powers = freqs, powers

        kwargs = self._fit_kwargs()
        orders = self._orders
        if orders is None:
            phi, offset, ok = fit_ar_psd(freqs, powers, self.fs, int(self.order),
                                         guess=self.guess, **kwargs)
            self._record_order(int(self.order), None, None)
        else:
            order, scores, (phi, offset, ok) = select_order_psd(
                freqs, powers, self.fs, orders, criterion=self.criterion,
                patience=self.patience, **kwargs)
            self._record_order(order, scores, orders)

        # A spectral fit can converge on a non-stationary model.  Reflecting
        # those poles inside the unit circle leaves the spectrum exactly
        # unchanged and makes the timescales finite.
        self.phi, self.offset, self.stabilized_ = stabilize(phi, offset)
        self.success_ = bool(ok)
        self.modes = None
        return self

    def predict(self, freqs=None):
        """Model spectrum at ``freqs`` (the fitted frequencies by default)."""
        self._check_fitted()
        freqs = self.freqs if freqs is None else np.asarray(freqs, dtype=float)
        return ar_psd(self.phi, freqs, self.fs, self.offset)

    @property
    def rsq(self):
        """Coefficient of determination of the fit, in log10 power."""
        self._check_fitted()
        return r2_logpsd(self.powers, self.predict())

    @property
    def rmse(self):
        """Root-mean-square error of the fit, in log10 power."""
        self._check_fitted()
        return rmse_logpsd(self.powers, self.predict())

    @property
    def divergence(self):
        """Itakura-Saito divergence between the spectrum and the fit."""
        self._check_fitted()
        return is_divergence(self.powers, self.predict())

    def error_by_band(self, n_bands=4):
        """Median absolute error in log10 power within each of ``n_bands``.

        The bands are equal width in log frequency.  A spectrum sampled on a
        linear frequency grid puts most of its points in the top decade, so
        :attr:`rsq` is dominated by the fastest frequencies and can look
        excellent while the model misses the low-frequency structure the
        timescale comes from.  A per-band error exposes that.

        The unit is dex, an order of magnitude in power: 0.05 dex is a 12%
        error, 0.15 dex is 41%, 0.30 dex is a factor of two.  Median absolute
        error is used rather than a per-band :math:`R^2` because a band may
        hold few points and little variance, which makes :math:`R^2` there
        unstable even for a good fit.

        Returns
        -------
        edges : 1d array
            Band edges in Hz, length ``n_bands + 1``.
        error : 1d array
            Median absolute error in each band, in log10 power.  NaN for a
            band holding fewer than three points.
        """
        self._check_fitted()
        edges = np.logspace(np.log10(self.freqs[0]), np.log10(self.freqs[-1]),
                            int(n_bands) + 1)
        resid = np.abs(np.log10(self.powers) - np.log10(self.predict()))
        out = np.full(int(n_bands), np.nan)
        for i in range(int(n_bands)):
            keep = (self.freqs >= edges[i]) & (self.freqs <= edges[i + 1])
            if keep.sum() >= 3:
                out[i] = float(np.median(resid[keep]))
        return edges, out

    def plot(self, ax=None, plot_modes=False, title=None):
        """Plot the spectrum and the fit, optionally with the per-mode factors.

        ``plot_modes`` may be ``True`` for every mode or an integer for the
        that many most important, which is usually what is readable.

        The AR(p) spectrum is a *product* of one factor per mode,
        ``1 / |1 - lambda z^-1|^2`` for a real pole and one such factor per
        conjugate pair, so on a logarithmic axis the factors add up to the
        fit.  Each is drawn with its own geometric mean over the band divided
        out and the fit's put back, which removes the arbitrary split of the
        overall scale between factors and keeps them beside the data.

        The partial-fraction components returned by
        :meth:`~timescales.decomposition.ARModes.psd_components` add rather
        than multiply, and are negative over much of the band for an
        oscillatory mode, so they are not what is drawn here.  They are the
        right decomposition in the lag domain, where they are the per-mode
        autocorrelations (see :class:`~timescales.ar.ARACF`).
        """
        import matplotlib.pyplot as plt
        self._check_fitted()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(self.freqs, self.powers, color='k', lw=1., label='Data')
        ax.loglog(self.freqs, self.predict(), color='C0', lw=2., ls='--',
                  label=f'AR({self.order_}) fit')
        if plot_modes:
            level = np.exp(np.mean(np.log(self.predict())))
            shown = self._modes().modes
            if not isinstance(plot_modes, bool):
                shown = shown[:int(plot_modes)]
            for mode in shown:
                factor = mode.psd(self.freqs)
                factor = level * factor / np.exp(np.mean(np.log(factor)))
                label = (f'{mode.tau * 1e3:.0f} ms' if mode.kind == 'relaxation'
                         else f'{mode.tau * 1e3:.0f} ms, {mode.freq:.1f} Hz')
                ax.loglog(self.freqs, factor, lw=1., alpha=.85, label=label)
        ax.set(xlabel='Frequency (Hz)', ylabel='Power',
               title=title if title is not None else self._title())
        ax.legend(fontsize=8)
        return ax

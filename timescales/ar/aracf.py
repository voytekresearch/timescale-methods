"""AR(p) fit to an autocorrelation."""

import numpy as np

from timescales.decomposition.modes import ar_acf_fft, stabilize
from .base import ARBase
from .core import fit_ar_acf, select_order_acf
from .metrics import acf_gof


class ARACF(ARBase):
    """Fit an AR(p) model to an autocorrelation.

    The model is ``A rho_p(k) + B``, where ``rho_p`` is the exact AR(p)
    autocorrelation, ``A`` absorbs a white-noise floor at lag zero and ``B`` a
    constant offset.  At ``p = 1`` this is the exponential-plus-offset fit used
    throughout the timescale literature; above it, the same model describes a
    decay with ringing on top, and the two are separated by the fit rather than
    by eye.

    The estimator does not compute the autocorrelation.  Pass whatever
    ``(lags, corrs)`` you trust; :mod:`timescales.utils` has helpers.

    Parameters
    ----------
    with_offset : bool, optional, default: True
        Fit the constant offset ``B``.  Fix it at zero for an autocorrelation
        already known to decay to zero.
    drop_zero_lag : bool, optional, default: True
        Exclude lag zero, which is one by construction and carries the
        white-noise floor rather than the decay.
    tol : float, optional
        Optimizer tolerance.

    Other parameters are documented in :class:`~timescales.ar.base.ARBase`.

    Examples
    --------
    >>> from timescales.ar import ARACF                        # doctest: +SKIP
    >>> ar = ARACF(order=range(1, 11), fs=500.)                # doctest: +SKIP
    >>> ar.fit(lags, corrs)                                    # doctest: +SKIP
    >>> ar.decompose()                                         # doctest: +SKIP
    >>> ar.tau                                                 # doctest: +SKIP
    """

    domain = 'acf'

    def __init__(self, order=1, fs=None, criterion='bic', loss='linear',
                 f_scale=None, k_bound=None, require_relaxation=False,
                 patience=None, maxfev=600, guess=None, with_offset=True,
                 drop_zero_lag=True, tol=1e-6, n_eff=None):
        from .core import K_BOUND
        super().__init__(order=order, fs=fs, criterion=criterion, loss=loss,
                         f_scale=f_scale,
                         k_bound=K_BOUND if k_bound is None else k_bound,
                         require_relaxation=require_relaxation,
                         patience=patience, maxfev=maxfev, guess=guess)
        self.with_offset = bool(with_offset)
        self.drop_zero_lag = bool(drop_zero_lag)
        self.tol = float(tol)
        self.n_eff = n_eff
        self.lags = None
        self.corrs = None
        self.height = np.nan
        self.acf_offset = np.nan
        self.stabilized_ = False

    def _fit_kwargs(self):
        kwargs = super()._fit_kwargs()
        kwargs.update(with_offset=self.with_offset,
                      drop_zero_lag=self.drop_zero_lag, tol=self.tol)
        return kwargs

    def fit(self, lags, corrs):
        """Fit the model to an autocorrelation.

        Parameters
        ----------
        lags : 1d array
            Lags, in samples, starting at zero and integer-valued.
        corrs : 1d array
            Autocorrelation at each lag.

        Returns
        -------
        self
        """
        lags = np.asarray(lags, dtype=float)
        corrs = np.asarray(corrs, dtype=float)
        if lags.shape != corrs.shape:
            raise ValueError('`lags` and `corrs` must have the same shape.')
        if len(lags) < 3:
            raise ValueError('Need at least three lags to fit.')
        self.lags, self.corrs = lags, corrs

        kwargs = self._fit_kwargs()
        orders = self._orders
        if orders is None:
            phi, height, offset, ok = fit_ar_acf(lags, corrs, self.fs,
                                                 int(self.order),
                                                 guess=self.guess, **kwargs)
            self._record_order(int(self.order), None, None)
        else:
            order, scores, (phi, height, offset, ok) = select_order_acf(
                lags, corrs, self.fs, orders, criterion=self.criterion,
                patience=self.patience, n_eff=self.n_eff, **kwargs)
            self._record_order(order, scores, orders)

        self.phi, _, self.stabilized_ = stabilize(phi)
        self.height, self.acf_offset = float(height), float(offset)
        self.success_ = bool(ok)
        self.modes = None
        return self

    def predict(self, lags=None):
        """Model autocorrelation at ``lags`` (the fitted lags by default)."""
        self._check_fitted()
        lags = self.lags if lags is None else np.asarray(lags, dtype=float)
        nlags = int(lags[-1])
        return self.height * ar_acf_fft(self.phi, nlags)[lags.astype(int)] \
            + self.acf_offset

    @property
    def rsq(self):
        """Coefficient of determination of the fit over the fitted lags."""
        self._check_fitted()
        return acf_gof(self.corrs, self.predict())['r2_acf']

    @property
    def rmse(self):
        self._check_fitted()
        return acf_gof(self.corrs, self.predict())['rmse_acf']

    def plot(self, ax=None, plot_modes=False, title=None):
        """Plot the autocorrelation, the fit, and optionally its components.

        The mode components add up to the fitted autocorrelation exactly: one
        exponential decay per relaxation and one damped cosine per rhythm.
        ``plot_modes`` may be ``True`` for every mode or an integer for that
        many most important.
        """
        import matplotlib.pyplot as plt
        self._check_fitted()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        t = self.lags / self.fs
        ax.plot(t, self.corrs, color='k', lw=1., label='Data')
        ax.plot(t, self.predict(), color='C0', lw=2., ls='--',
                label=f'AR({self.order_}) fit')
        if plot_modes:
            modes = self._modes()
            comps = modes.acf_components(int(self.lags[-1]))
            n = len(modes.modes) if isinstance(plot_modes, bool) else int(plot_modes)
            for i, comp in enumerate(comps[:n]):
                mode = modes.modes[i]
                label = (f'{mode.tau * 1e3:.0f} ms' if mode.kind == 'relaxation'
                         else f'{mode.tau * 1e3:.0f} ms, {mode.freq:.1f} Hz')
                ax.plot(t, self.height * comp[self.lags.astype(int)], lw=1.,
                        alpha=.8, label=label)
        ax.axhline(0, color='.7', lw=.5, zorder=0)
        ax.set(xlabel='Lag (s)', ylabel='Autocorrelation',
               title=title if title is not None else self._title())
        ax.legend(fontsize=8)
        return ax

"""Shared behaviour of the AR(p) estimators.

:class:`ARPSD` and :class:`ARACF` fit the same model -- the AR coefficients
``phi`` -- to two different representations of a process.  Everything that does
not depend on which representation was used lives here: how an order is chosen,
how the fitted coefficients are turned into modes, and what is reported.

The estimators are deliberately agnostic about where their input came from.
They take ``(freqs, powers)`` or ``(lags, corrs)`` and never compute them, so
any spectral or autocorrelation estimator can be used, including one the caller
wrote.  :mod:`timescales.utils` has convenience functions for the common cases.
"""

import warnings

import numpy as np

from timescales.conversions import phi_to_tau
from timescales.decomposition import ARModes
from .core import K_BOUND


class ARBase:
    """Settings and results shared by the AR(p) estimators.

    Parameters
    ----------
    order : int or iterable of int, optional, default: 1
        Model order.  An integer fits that order.  An iterable fits each order
        in turn and keeps the one minimizing ``criterion``, e.g.
        ``order=range(1, 21)``.
    fs : float
        Sampling rate, Hz.  Needed to express poles as times and frequencies.
    criterion : {'bic', 'aic'}, optional, default: 'bic'
        Information criterion used when ``order`` is an iterable.
    loss : str, optional, default: 'linear'
        Loss passed to ``scipy.optimize.curve_fit``.  ``'soft_l1'`` with an
        ``f_scale`` down-weights outlying points, which in the frequency domain
        means treating a narrowband rhythm as an outlier rather than as
        evidence about the slow decay.
    f_scale : float, optional
        Soft-loss margin, in the units of the fitted residual.  Required for
        any ``loss`` other than ``'linear'``.
    k_bound : float, optional
        Largest reflection-coefficient magnitude allowed, which bounds the
        slowest representable relaxation at ``-1 / (fs log k_bound)``.
    require_relaxation : bool, optional, default: False
        Factor one real pole out of the characteristic polynomial, so the
        fitted model always has a relaxation time to report.  This costs one
        order: an AR(2) so constrained is two real poles and cannot represent
        an oscillation at all.  Turn it on when a timescale must be reported
        for every fit, and give the model the extra order.
    patience : int, optional
        Stop the order search this many orders after the best so far.  ``None``
        searches the whole grid.
    maxfev : int, optional
        Maximum objective evaluations per fit.
    guess : 1d array, optional
        Starting parameters.  Defaults to a Yule-Walker initialization.

    Attributes
    ----------
    phi : 1d array
        Fitted AR coefficients, ``x_t = sum_k phi_k x_{t-k} + eps_t``.
    order_ : int
        The order actually fitted, which differs from ``order`` when an
        iterable was given.
    order_scores_ : 1d array or None
        Criterion value per candidate order.
    success_ : bool
        Whether the optimizer reported convergence.
    modes : ARModes or None
        Set by :meth:`decompose`.
    """

    #: Set by subclasses: the name of the domain, used in messages and plots.
    domain = None

    def __init__(self, order=1, fs=None, criterion='bic', loss='linear',
                 f_scale=None, k_bound=K_BOUND, require_relaxation=False,
                 patience=None, maxfev=None, guess=None):
        if fs is None:
            raise ValueError('`fs` is required to express poles as times.')
        if loss != 'linear' and f_scale is None:
            raise ValueError(f'loss={loss!r} needs an `f_scale` margin.')
        self.order = order
        self.fs = float(fs)
        self.criterion = criterion
        self.loss = loss
        self.f_scale = f_scale
        self.k_bound = float(k_bound)
        self.require_relaxation = bool(require_relaxation)
        self.patience = patience
        self.maxfev = maxfev
        self.guess = guess

        self.phi = None
        self.order_ = None
        self.order_scores_ = None
        self.orders_ = None
        self.success_ = False
        self.order_at_ceiling_ = False
        self.tau_near_boundary_ = False
        self.modes = None

    # -- order -------------------------------------------------------------
    @property
    def _orders(self):
        """The orders to try: ``None`` for a fixed order, else the grid."""
        if isinstance(self.order, (int, np.integer)):
            return None
        orders = [int(p) for p in self.order]
        if not orders:
            raise ValueError('`order` is empty.')
        return orders

    def _record_order(self, order, scores, orders):
        """Store the selected order and warn if the search stopped at its top.

        BIC over AR order is not unimodal: a rhythm needs several poles before
        any of them pays for itself, so the criterion can sit on a local
        minimum for many orders and then drop sharply.  A search that ends at
        its largest order has not seen where the curve turns, and the fit it
        returns may be far from the best available.
        """
        self.order_ = int(order)
        self.order_scores_ = scores
        self.orders_ = orders
        self.order_at_ceiling_ = bool(orders is not None and len(orders) > 1
                                      and self.order_ == max(orders))
        if self.order_at_ceiling_:
            warnings.warn(
                f'{type(self).__name__}: the selected order ({self.order_}) is '
                f'the largest offered, so the search may have stopped before '
                f'the criterion turned. Extend `order` and refit.',
                RuntimeWarning, stacklevel=3)

    def _fit_kwargs(self):
        kwargs = dict(k_bound=self.k_bound, loss=self.loss,
                      f_scale=self.f_scale,
                      require_relaxation=self.require_relaxation)
        if self.maxfev is not None:
            kwargs['maxfev'] = self.maxfev
        return kwargs

    # -- reporting ---------------------------------------------------------
    def _check_fitted(self):
        if self.phi is None:
            raise RuntimeError('Call fit() first.')

    def decompose(self, freqs=None, n_freqs=513, f_range=None):
        """Decompose the fitted model into modes, ranked by importance.

        Each mode is a real pole (a relaxation) or a complex-conjugate pair (a
        damped oscillation), carrying a decay time, a frequency and an
        importance: how much the model spectrum changes when it is removed.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequencies over which importance is measured.  Defaults to a
            uniform grid over ``f_range``, or the full band.

        Returns
        -------
        modes : ARModes
        """
        self._check_fitted()
        self.modes = ARModes.from_phi(self.phi, self.fs, offset=self._offset,
                                      freqs=freqs, n_freqs=n_freqs,
                                      f_range=f_range)
        # `tau` keeps modes ringing less than a cycle and discards the rest.
        # When the leading mode sits on that cut, the reported timescale is
        # decided by which side of it the fit landed, and a small change in
        # order or band moves it by a large factor.  Say so once, here, rather
        # than returning the number as if it were settled.
        self.tau_near_boundary_ = bool(self.modes.tau_is_near_boundary)
        if self.tau_near_boundary_:
            top = max((m for m in self.modes.modes if np.isfinite(m.importance)),
                      key=lambda m: m.importance)
            warnings.warn(
                f'{type(self).__name__}: the leading mode rings {top.n_cycles:.2f} '
                f'cycles, close to the f*tau = 1 cut that separates a relaxation '
                f'from a rhythm, so `tau` is decided by a near tie. A small change '
                f'in order or band can move it by a large factor. Inspect `modes` '
                f'and consider `tau_dominant`.',
                RuntimeWarning, stacklevel=2)
        return self.modes

    @property
    def _offset(self):
        """Scale of the model spectrum; subclasses that fit one override it."""
        return 1.0

    def _modes(self):
        return self.modes if self.modes is not None else self.decompose()

    @property
    def tau(self):
        """The single timescale reported for a fit, seconds: its most
        important short-lived mode.

        For an AR(1) this is the single relaxation time.  At higher orders it
        is the most important mode that decays before completing one cycle.
        That mode is often a relaxation, a real pole, but it need not be: a
        heavily damped oscillatory mode can dominate, and on field-potential
        recordings it usually does.

        It is NaN when every mode of the fitted model is a genuine rhythm, a
        pure AR(2) oscillator being the simplest case: such a model has an
        envelope decay but no timescale to report, and reporting the envelope
        here would misname it.  Use :attr:`tau_dominant` for the leading mode
        whatever its kind.
        """
        self._check_fitted()
        if len(self.phi) == 1:
            return phi_to_tau(float(self.phi[0]), self.fs)
        return self._modes().tau_relaxation

    @property
    def tau_dominant(self):
        """Decay time of the most important mode, whatever kind it is.

        Unlike :attr:`tau` this is always defined: for a model whose every
        mode is a rhythm it is that rhythm's envelope decay, which is a decay
        time but not one this rule would report.
        """
        return self._modes().tau_dominant

    @property
    def taus(self):
        """Decay time of every mode, seconds, ranked by importance."""
        return self._modes().taus

    @property
    def freqs_hz(self):
        """Frequency of every mode, Hz; zero for a relaxation."""
        return self._modes().freqs_hz

    @property
    def is_stationary(self):
        self._check_fitted()
        return bool(np.all(np.abs(np.roots(np.r_[1.0, -self.phi])) < 1.0))

    def _title(self):
        """Default plot title: the reported timescale, or the leading mode."""
        tau = self.tau
        if np.isfinite(tau):
            return f'AR({self.order_}), tau = {tau * 1e3:.1f} ms'
        return f'AR({self.order_}), leading mode = {self.tau_dominant * 1e3:.1f} ms'

    def simulate(self, n_seconds, fs=None, rng=None, sigma=1.0):
        """Simulate a realization of the fitted model."""
        from timescales.sim import sim_ar_process
        self._check_fitted()
        fs = self.fs if fs is None else fs
        rng = np.random.default_rng() if rng is None else rng
        return sim_ar_process(self.phi, int(round(n_seconds * fs)), rng,
                              sigma=sigma)

    def __repr__(self):
        if self.phi is None:
            return f'{type(self).__name__}(order={self.order}, fs={self.fs:g}) [unfitted]'
        tau = self.tau
        shown = f'tau={tau:.4g}s' if np.isfinite(tau) else \
            f'tau=none (all modes rhythmic), tau_dominant={self.tau_dominant:.4g}s'
        return (f'{type(self).__name__}(order={self.order_}, fs={self.fs:g}, '
                f'{shown}, stationary={self.is_stationary})')

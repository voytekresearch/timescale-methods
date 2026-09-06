"""Pole/mode decomposition and interpretation of AR(p) models.

An AR(p) process

    x_t = sum_{k=1}^p phi_k x_{t-k} + eps_t

is fully characterized by the p roots (poles) of its characteristic polynomial

    z^p - phi_1 z^{p-1} - ... - phi_p = 0.

Each pole lam_k contributes a relaxation time and, when complex, an oscillation
frequency:

    tau_k = -(1 / fs) / log|lam_k|
    f_k   = fs |arg(lam_k)| / (2 pi).

Two facts organize the interpretation implemented here:

1.  The AR(p) spectral denominator *factorizes* into first-order pole factors,

        1 - sum_k phi_k z^{-k} = prod_k (1 - lam_k z^{-1}),

    so the model PSD is a *product* of first-order factor spectra.  Complex
    poles occur in conjugate pairs; each pair multiplies into one real second
    order (damped oscillator) factor.  This is a factorization of the transfer
    function, not a statement that the signal is a sum of independent AR(1)
    processes.

2.  In the time domain the same poles appear *additively*: the impulse response
    is psi_n = sum_k c_k lam_k^n, so x_t = sum_k c_k u_{k,t} where each u_k is an
    AR(1) process with coefficient lam_k -- but all u_k are driven by the *same*
    innovation sequence.  Because the mode processes are perfectly dependent,
    the process variance does not decompose into per-mode shares, and mode
    importance must not be reported as "percent variance explained".

Mode importance is therefore defined spectrally, as the degradation of the
fitted model spectrum when a mode is deleted and the overall scale refit.
Because log P(f) = log sigma^2 - sum_k g_k(f) with
g_k(f) = sum_{lam in mode k} log|1 - lam e^{-i omega}|^2, the deletion residual
has a closed form (see ``mode_importance``).
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Polynomial <-> pole conversions
# ---------------------------------------------------------------------------

def ar_to_poles(phi):
    """Poles (characteristic roots) of an AR(p) model.

    Parameters
    ----------
    phi : 1d array
        AR coefficients [phi_1, ..., phi_p] in the convention
        x_t = sum_k phi_k x_{t-k} + eps_t.

    Returns
    -------
    poles : 1d complex array
        Roots of z^p - phi_1 z^{p-1} - ... - phi_p, i.e. the eigenvalues of the
        companion matrix.  Stationarity requires |lam| < 1 for all poles.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    coeffs = np.concatenate(([1.0], -phi))
    return np.roots(coeffs)


def poles_to_ar(poles):
    """AR coefficients implied by a set of poles.

    Inverse of :func:`ar_to_poles` up to pole ordering.  Poles must be closed
    under conjugation for the coefficients to be real.
    """
    poles = np.atleast_1d(np.asarray(poles, dtype=complex))
    coeffs = np.poly(poles)  # [1, -phi_1, ..., -phi_p]
    phi = -coeffs[1:]
    if np.max(np.abs(phi.imag)) > 1e-8 * max(1.0, np.max(np.abs(phi.real))):
        raise ValueError('Poles are not closed under conjugation; phi is complex.')
    return np.real(phi)


def group_poles(poles, tol=1e-6):
    """Group poles into real poles and complex-conjugate pairs.

    A pole whose imaginary part is below ``tol`` (relative to its modulus) is
    real.  The tolerance is loose on purpose: a fitted model can put two real
    poles at the same radius, and root finding splits a repeated root into a
    conjugate pair with an imaginary part of order the square root of the
    coefficient round-off (about 5e-7 for a sixth-order model with poles near
    the unit circle).  An angle of 1e-6 rad is a period of hours at any
    sampling rate used here, so nothing that could be a rhythm is lost.

    Returns
    -------
    groups : list of tuple of complex
        Length-1 tuples for real poles, length-2 tuples for conjugate pairs.
    """
    remaining = [complex(p) for p in np.atleast_1d(poles)]
    # Deterministic ordering: by decreasing modulus, then by |angle|.
    remaining.sort(key=lambda p: (-abs(p), abs(np.angle(p)), -p.imag))

    groups = []
    while remaining:
        lam = remaining.pop(0)
        if abs(lam.imag) <= tol * max(1.0, abs(lam)):
            groups.append((complex(lam.real, 0.0),))
            continue
        if not remaining:
            raise ValueError('Unpaired complex pole; coefficients are not real.')
        idx = int(np.argmin([abs(p - np.conj(lam)) for p in remaining]))
        partner = remaining.pop(idx)
        if abs(partner - np.conj(lam)) > 1e-6 * max(1.0, abs(lam)):
            raise ValueError('Unpaired complex pole; coefficients are not real.')
        # Canonical order: positive imaginary part first.
        pair = (lam, partner) if lam.imag > 0 else (partner, lam)
        groups.append(pair)
    return groups


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def ar_psd(phi, freqs, fs, offset=1.0):
    """Theoretical AR(p) power spectrum, ``offset / |1 - sum_k phi_k z^-k|^2``.

    ``offset`` plays the role of the innovation variance (up to the constant
    convention used by :mod:`timescales`).
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    k = np.arange(1, len(phi) + 1)
    z_inv = np.exp(-2j * np.pi * np.outer(freqs, k) / fs)
    denom = 1.0 - z_inv @ phi
    return offset / np.abs(denom) ** 2


def pole_log_factor(poles, freqs, fs):
    """``g(f) = sum_lam log|1 - lam e^{-i 2 pi f / fs}|^2`` (natural log)."""
    freqs = np.asarray(freqs, dtype=float)
    z_inv = np.exp(-2j * np.pi * freqs / fs)
    g = np.zeros(freqs.shape, dtype=float)
    for lam in np.atleast_1d(poles):
        g = g + 2.0 * np.log(np.abs(1.0 - complex(lam) * z_inv))
    return g


def psd_from_poles(poles, freqs, fs, offset=1.0):
    """PSD reconstructed from the pole factorization.

    Exactly equal to :func:`ar_psd` evaluated at ``poles_to_ar(poles)``.
    """
    return offset * np.exp(-pole_log_factor(poles, freqs, fs))


def stabilize(phi, offset=1.0, eps=1e-6):
    """Reflect non-stationary poles inside the unit circle, preserving the PSD.

    On the unit circle ``|1 - lam z^-1|^2 = |lam|^2 |1 - z^-1 / conj(lam)|^2``,
    so replacing a pole ``lam`` with ``1 / conj(lam)`` and dividing the scale by
    ``|lam|^2`` leaves the model spectrum *exactly* unchanged while making the
    process stationary.  This is the minimum-phase spectral factorization, and
    it is what makes mode timescales well defined for a spectral AR fit whose
    optimizer wandered outside the stationary region.

    Returns
    -------
    phi, offset, changed
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    poles = ar_to_poles(phi)
    outside = np.abs(poles) >= 1.0
    if not np.any(outside):
        return phi, offset, False
    scale = 1.0
    new_poles = poles.copy()
    for i in np.where(outside)[0]:
        lam = poles[i]
        radius = abs(lam)
        if radius < 1.0 + eps:
            lam = lam * (1.0 - eps) / radius   # sitting on the circle
            new_poles[i] = lam
            scale *= 1.0
        else:
            new_poles[i] = 1.0 / np.conj(lam)
            scale *= radius ** 2
    return poles_to_ar(new_poles), offset / scale, True


def real_pole_reflection_to_ar(a, k):
    """AR coefficients of ``(1 - a z^-1) * A_rest(z^-1)``.

    Factoring one real pole out of the characteristic polynomial and
    parameterizing the remainder by reflection coefficients guarantees, by
    construction, that the model has a relaxation mode -- exactly as the
    reflection parameterization alone guarantees stationarity.  ``a`` in
    ``(0, 1)`` is that pole, so the model always has a decay time
    ``-1 / (fs log a)`` to report, whatever the optimizer does with the rest.

    Without this a fitted model can put every pole into a conjugate pair, and
    a timescale model with no relaxation mode has no timescale to report.
    """
    a = float(a)
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if k.size == 0:
        return np.array([a])
    rest = reflection_to_ar(k)
    poly = np.convolve([1.0, -a], np.concatenate(([1.0], -np.atleast_1d(rest))))
    return -poly[1:]


def ar_to_real_pole_reflection(phi, default_a=0.9, k_bound=0.9999):
    """Split an AR(p) into a real pole and reflection coefficients.

    The inverse of :func:`real_pole_reflection_to_ar`, used to convert a
    starting guess.  When the guess already has a positive real pole the
    slowest one is factored out and the remainder converted exactly.  When it
    has none -- which is the case this parameterization exists for -- the
    remainder is the order-(p-1) prefix of the guess's own reflection
    sequence, which is always a valid stationary model.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = len(phi)
    if p == 1:
        return float(np.clip(phi[0], 1e-3, k_bound)), np.zeros(0)

    poles = ar_to_poles(phi)
    real = [i for i, lam in enumerate(poles)
            if abs(lam.imag) < 1e-8 and lam.real > 0]
    if real:
        j = max(real, key=lambda i: poles[i].real)
        try:
            k = ar_to_reflection(poles_to_ar(np.delete(poles, j)))
            if np.all(np.abs(k) < 1.0):
                return (float(np.clip(poles[j].real, 1e-3, k_bound)),
                        np.clip(k, -k_bound, k_bound))
        except ValueError:
            pass
    k = np.clip(ar_to_reflection(phi)[:p - 1], -k_bound, k_bound)
    return float(np.clip(default_a, 1e-3, k_bound)), k


def pair_to_ar2(pair):
    """Real AR(2) coefficients of one complex-conjugate pole pair."""
    lam = complex(pair[0])
    return np.array([2.0 * lam.real, -(abs(lam) ** 2)])


# ---------------------------------------------------------------------------
# Time-domain quantities
# ---------------------------------------------------------------------------

def ar_impulse_response(phi, n):
    """Impulse (Green's function) response psi_0..psi_{n-1} of an AR(p)."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = len(phi)
    psi = np.zeros(n)
    psi[0] = 1.0
    for i in range(1, n):
        acc = 0.0
        for k in range(1, min(p, i) + 1):
            acc += phi[k - 1] * psi[i - k]
        psi[i] = acc
    return psi


def pole_residues(poles):
    """Residues c_k with psi_n = sum_k c_k lam_k^n (distinct poles).

    ``c_k = lam_k^{p-1} / prod_{i != k} (lam_k - lam_i)``.
    """
    poles = np.atleast_1d(np.asarray(poles, dtype=complex))
    p = len(poles)
    c = np.zeros(p, dtype=complex)
    for k in range(p):
        denom = np.prod([poles[k] - poles[i] for i in range(p) if i != k]) if p > 1 else 1.0
        c[k] = poles[k] ** (p - 1) / denom
    return c


def ar_acf(phi, nlags, sigma2=1.0, normalize=True):
    """Theoretical autocovariance/autocorrelation of an AR(p) process."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = len(phi)

    # Solve the Yule-Walker system for gamma(0..p).
    a = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)
    a[0, 0] = 1.0
    for k in range(1, p + 1):
        a[0, k] -= phi[k - 1]
    b[0] = sigma2
    for row in range(1, p + 1):
        a[row, row] = 1.0
        for k in range(1, p + 1):
            a[row, abs(row - k)] -= phi[k - 1]
    gamma = np.linalg.solve(a, b)

    out = np.zeros(nlags + 1)
    out[: min(p + 1, nlags + 1)] = gamma[: min(p + 1, nlags + 1)]
    for k in range(p + 1, nlags + 1):
        out[k] = sum(phi[i] * out[k - i - 1] for i in range(p))
    return out / out[0] if normalize else out


def impulse_mode_components(phi, n, groups=None):
    """Per-mode contributions to the impulse response, which sum to it exactly.

    ``psi_n = sum_k c_k lam_k^n``: the response of the process to a single input
    is a sum of decaying wiggles, one per mode, each with its own decay time and
    (for a conjugate pair) its own rhythm.  This is the most direct statement of
    what an AR(p) model *is*, and unlike the spectral factorization it is
    additive.

    Returns
    -------
    components : 2d array, (n_modes, n)
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    poles = ar_to_poles(phi)
    groups = group_poles(poles) if groups is None else groups
    c = pole_residues(poles)
    k = np.arange(n)
    terms = c[:, None] * poles[:, None] ** k[None, :]
    out = np.zeros((len(groups), n))
    for g, grp in enumerate(groups):
        for lam in grp:
            out[g] += np.real(terms[int(np.argmin(np.abs(poles - lam)))])
    return out


def _mode_amplitudes(poles, sigma2=1.0):
    """``A_j`` of :func:`acf_mode_components`: the weight of each pole's
    geometric term in the autocovariance."""
    c = pole_residues(poles)
    return np.array([sigma2 * c[j] * np.sum(c / (1.0 - poles * poles[j]))
                     for j in range(len(poles))])


def acf_mode_components(phi, nlags, sigma2=1.0, groups=None, normalize=True):
    """Exact additive decomposition of the AR(p) ACF into per-mode terms.

    With ``psi_n = sum_i c_i lam_i^n`` the autocovariance is

        gamma(k) = sigma2 sum_{i,j} c_i c_j lam_j^k / (1 - lam_i lam_j)
                 = sum_j A_j lam_j^k,   A_j = sigma2 c_j sum_i c_i/(1 - lam_i lam_j)

    so each pole contributes a geometric term and each conjugate pair a damped
    cosine.  The returned components sum exactly to the ACF: modes are additive
    in the time domain even though their spectral factors multiply.

    Returns
    -------
    components : 2d array, (n_modes, nlags + 1)
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    poles = ar_to_poles(phi)
    groups = group_poles(poles) if groups is None else groups
    amp = _mode_amplitudes(poles, sigma2)
    k = np.arange(nlags + 1)
    terms = amp[:, None] * poles[:, None] ** k[None, :]

    out = np.zeros((len(groups), nlags + 1))
    for g, grp in enumerate(groups):
        for lam in grp:
            idx = int(np.argmin(np.abs(poles - lam)))
            out[g] += np.real(terms[idx])
    if normalize:
        out = out / out.sum(axis=0)[0]
    return out


def psd_mode_components(phi, freqs, fs, sigma2=1.0, groups=None):
    """Exact additive decomposition of the AR(p) spectrum into per-mode terms.

    The autocovariance is ``sum_j A_j lam_j^|k|`` (:func:`acf_mode_components`)
    and each term transforms to

        S_j(w) = A_j (1 - lam_j^2) / (1 - 2 lam_j cos w + lam_j^2),

    so the modes that add in the lag domain add here too: over all poles the
    terms sum to ``sigma2 / |1 - sum_k phi_k e^{-iwk}|^2`` exactly, and a
    conjugate pair sums to one real spectral bump.  This is the partial-fraction
    form of the spectrum, as opposed to its factorization into poles, whose
    factors multiply.

    Returns
    -------
    components : 2d array, (n_modes, len(freqs))
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    poles = ar_to_poles(phi)
    groups = group_poles(poles) if groups is None else groups
    amp = _mode_amplitudes(poles, sigma2)
    w = 2 * np.pi * np.asarray(freqs, dtype=float) / fs
    out = np.zeros((len(groups), len(w)))
    for g, grp in enumerate(groups):
        for lam in grp:
            j = int(np.argmin(np.abs(poles - lam)))
            out[g] += np.real(amp[j] * (1.0 - poles[j] ** 2)
                              / (1.0 - 2.0 * poles[j] * np.cos(w) + poles[j] ** 2))
    return out


def reflection_to_ar(k):
    """Levinson step-up: reflection coefficients -> AR coefficients.

    ``|k_i| < 1`` for all ``i`` if and only if the resulting AR model is
    stationary, so this parameterization confines an optimizer to the
    stationary region and is far better scaled than bounding the coefficients
    directly (whose natural bounds grow like binomial coefficients).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    phi = np.zeros(0)
    for ki in k:
        phi = np.concatenate([phi - ki * phi[::-1], [ki]]) if len(phi) else np.array([ki])
    return phi


def ar_to_reflection(phi, eps=1e-9):
    """Levinson step-down: AR coefficients -> reflection coefficients."""
    a = np.atleast_1d(np.asarray(phi, dtype=float)).copy()
    p = len(a)
    k = np.zeros(p)
    for m in range(p, 0, -1):
        km = float(np.clip(a[m - 1], -1 + eps, 1 - eps))
        k[m - 1] = km
        if m == 1:
            break
        a = (a[:m - 1] + km * a[:m - 1][::-1]) / (1 - km ** 2)
    return k


_EXP_CACHE = {}


def _acf_exp_matrix(nfft, order):
    key = (nfft, order)
    mat = _EXP_CACHE.get(key)
    if mat is None:
        k = np.arange(1, order + 1)
        freqs = np.arange(nfft // 2 + 1) / nfft
        mat = np.exp(-2j * np.pi * np.outer(freqs, k))
        if len(_EXP_CACHE) > 64:
            _EXP_CACHE.clear()
        _EXP_CACHE[key] = mat
    return mat


def ar_acf_fft(phi, nlags, fs=None, nfft=None, normalize=True):
    """AR(p) autocorrelation via the spectrum, vectorized for repeated fitting.

    The autocovariance is the inverse Fourier transform of the model spectrum,
    so evaluating it this way costs one FFT rather than a Python-level
    recursion.  This matters because the ACF-domain fit evaluates the model
    thousands of times.  ``nfft`` is chosen large enough that circular
    wrap-around is negligible; :func:`ar_acf` remains the exact reference and
    the two agree to machine precision for stationary models.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    if nfft is None:
        # Pad past the model's own decay length so that circular wrap-around is
        # negligible.  1 - sum(phi) is the characteristic polynomial at z = 1,
        # whose inverse tracks the slowest relaxation.
        n_decay = 1.0 / max(abs(1.0 - float(np.sum(phi))), 1e-5)
        needed = max(4 * (nlags + 1), nlags + 16 * n_decay, 1024)
        nfft = int(2 ** np.ceil(np.log2(min(needed, 2 ** 19))))
    z_inv = _acf_exp_matrix(nfft, len(phi))
    denom = np.abs(1.0 - z_inv @ phi) ** 2
    denom = np.maximum(denom, 1e-300)
    acov = np.fft.irfft(1.0 / denom, n=nfft)[: nlags + 1]
    return acov / acov[0] if normalize else acov


def variance_decomposition(poles, sigma2=1.0, groups=None):
    """Diagnostic showing that per-mode variance shares are not well defined.

    ``Var(x) = sigma2 * sum_{i,j} c_i c_j / (1 - lam_i lam_j)``.  The diagonal
    (within-mode) blocks and the cross terms are returned separately; the cross
    terms are generally non-zero and can be negative, which is why mode
    importance is defined spectrally rather than as a variance share.
    """
    poles = np.atleast_1d(np.asarray(poles, dtype=complex))
    groups = group_poles(poles) if groups is None else groups
    c = pole_residues(poles)
    index = {complex(lam): i for i, lam in enumerate(poles)}

    def idx_of(lam):
        return int(np.argmin(np.abs(poles - lam)))

    m = np.zeros((len(poles), len(poles)), dtype=complex)
    for i in range(len(poles)):
        for j in range(len(poles)):
            m[i, j] = c[i] * c[j] / (1.0 - poles[i] * poles[j])
    m = sigma2 * m

    total = float(np.real(m.sum()))
    within = []
    used = np.zeros(len(poles), dtype=bool)
    blocks = []
    for grp in groups:
        ids = [idx_of(lam) for lam in grp]
        blocks.append(ids)
        used[ids] = True
        within.append(float(np.real(m[np.ix_(ids, ids)].sum())))
    within = np.asarray(within)
    cross = total - within.sum()
    return {
        'total_variance': total,
        'within_mode': within,
        'cross_mode': float(cross),
        'cross_fraction': float(cross / total) if total != 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

@dataclass
class Mode:
    """One AR(p) mode: a real pole or a complex-conjugate pole pair."""

    poles: tuple
    fs: float
    kind: str                      # 'relaxation' | 'oscillatory' | 'nyquist'
    tau: float                     # seconds
    freq: float                    # Hz
    radius: float                  # |lambda|
    angle: float                   # arg(lambda), radians
    phi: np.ndarray                # equivalent AR(1) or AR(2) coefficients
    importance: float = np.nan     # RMS log10-power degradation on deletion
    importance_is: float = np.nan  # Itakura-Saito (Whittle) degradation, nats
    rel_importance: float = np.nan

    @property
    def is_stationary(self):
        return self.radius < 1.0

    @property
    def n_cycles(self):
        """Cycles completed before the envelope decays by ``1/e``: ``f * tau``.

        A conjugate pair is called oscillatory because its poles are complex,
        which says nothing about whether it would be recognised as a rhythm.
        That is what this says.  Below about one cycle the mode rings down
        before completing a period and is better described as a broad
        shoulder in the spectrum than as an oscillation.

        It is a property of the mode alone, in the time domain.  Whether the
        *model's* spectrum shows a peak at this frequency is a different
        question, because modes multiply in the spectrum: a mildly resonant
        factor leaves no peak if the remaining factors fall steeply through
        the same frequencies.
        """
        # Only a pure relaxation has no frequency.  A Nyquist mode -- a
        # negative real pole -- alternates every sample, which is the fastest
        # oscillation the sampling rate allows, so it must not be counted as
        # non-rhythmic just because its poles are real.
        return 0.0 if self.kind == 'relaxation' else self.freq * self.tau

    def as_tuple(self):
        """The interpretable summary: ``(tau, frequency, importance)``."""
        return (self.tau, self.freq, self.importance)

    def psd(self, freqs, offset=1.0):
        """This mode's own factor spectrum (unit-numerator)."""
        return psd_from_poles(self.poles, freqs, self.fs, offset=offset)

    def __repr__(self):
        cyc = ('' if self.kind == 'relaxation'
               else f', {self.n_cycles:.2f} cyc')
        return (f'Mode({self.kind}, tau={self.tau:.4g}s, f={self.freq:.4g}Hz'
                f'{cyc}, |lam|={self.radius:.4f}, '
                f'importance={self.importance:.4g})')


def _mode_from_group(grp, fs):
    lam = complex(grp[0])
    radius = abs(lam)
    angle = float(np.angle(lam))
    with np.errstate(divide='ignore'):
        tau = float(-(1.0 / fs) / np.log(radius)) if radius > 0 else 0.0
    if radius >= 1.0:
        tau = np.inf
    if len(grp) == 2:
        kind = 'oscillatory'
        freq = fs * abs(angle) / (2 * np.pi)
        phi = pair_to_ar2(grp)
    elif lam.real < 0:
        kind = 'nyquist'
        freq = fs / 2.0
        phi = np.array([lam.real])
    else:
        kind = 'relaxation'
        freq = 0.0
        phi = np.array([lam.real])
    return Mode(poles=tuple(grp), fs=fs, kind=kind, tau=tau, freq=freq,
                radius=radius, angle=angle, phi=phi)


def mode_importance(modes, freqs, fs, kind='logpsd'):
    """Degradation of the model spectrum when each mode is deleted.

    Deleting mode k and refitting the overall scale leaves the log-spectral
    residual ``-(g_k(f) - mean_f g_k)``, so the RMS log10-power degradation is
    ``std_f(g_k) / ln(10)``.  The Itakura-Saito variant is the asymptotic
    Whittle log-likelihood degradation, ``log mean_f(e^{-g_k}) + mean_f(g_k)``,
    i.e. the log ratio of the arithmetic to geometric mean of the mode's own
    factor spectrum.

    Both are exact closed forms of "delete the mode, refit the scale, measure
    the mismatch"; they are properties of the individual mode because mode
    contributions are additive in log-power.
    """
    out = np.zeros(len(modes))
    for i, mode in enumerate(modes):
        g = pole_log_factor(mode.poles, freqs, fs)
        if kind == 'logpsd':
            out[i] = np.std(g) / LN10
        elif kind == 'is':
            out[i] = float(np.log(np.mean(np.exp(-g))) + np.mean(g))
        else:
            raise ValueError(f'Unknown importance kind {kind!r}.')
    return out


def subset_error(modes, keep, freqs, fs, kind='logpsd'):
    """Exact spectral error of the model that keeps only ``keep`` modes.

    The dropped modes' log-factors add, so the residual after refitting the
    overall scale is ``sum_{k dropped} g_k`` (demeaned).
    """
    keep = set(int(i) for i in keep)
    g = np.zeros(len(freqs))
    for i, mode in enumerate(modes):
        if i not in keep:
            g = g + pole_log_factor(mode.poles, freqs, fs)
    if kind == 'logpsd':
        return float(np.std(g) / LN10)
    if kind == 'is':
        return float(np.log(np.mean(np.exp(-g))) + np.mean(g))
    raise ValueError(f'Unknown error kind {kind!r}.')


@dataclass
class ARModes:
    """Mode decomposition of a fitted AR(p) model."""

    phi: np.ndarray
    fs: float
    offset: float = 1.0
    freqs: np.ndarray = None
    modes: list = field(default_factory=list)
    poles: np.ndarray = None

    @classmethod
    def from_phi(cls, phi, fs, offset=1.0, freqs=None, n_freqs=513, f_range=None):
        """Decompose AR coefficients into ranked modes.

        Parameters
        ----------
        phi : 1d array
            AR coefficients.
        fs : float
            Sampling rate, Hz.
        offset : float
            Innovation-variance scale of the model spectrum.
        freqs : 1d array, optional
            Frequency grid used to define mode importance.  Defaults to a
            uniform grid over the full band (or ``f_range``).
        """
        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        if freqs is None:
            lo, hi = (0.0, fs / 2.0) if f_range is None else f_range
            freqs = np.linspace(lo, hi, n_freqs)
        poles = ar_to_poles(phi)
        modes = [_mode_from_group(g, fs) for g in group_poles(poles)]

        imp = mode_importance(modes, freqs, fs, kind='logpsd')
        imp_is = mode_importance(modes, freqs, fs, kind='is')
        total = imp.sum()
        for mode, a, b in zip(modes, imp, imp_is):
            mode.importance = float(a)
            mode.importance_is = float(b)
            mode.rel_importance = float(a / total) if total > 0 else np.nan

        order = np.argsort(-imp)
        modes = [modes[i] for i in order]
        return cls(phi=phi, fs=fs, offset=offset, freqs=np.asarray(freqs),
                   modes=modes, poles=poles)

    # -- basic properties -------------------------------------------------
    @property
    def order(self):
        return len(self.phi)

    @property
    def is_stationary(self):
        return bool(np.all(np.abs(self.poles) < 1.0))

    @property
    def taus(self):
        return np.array([m.tau for m in self.modes])

    @property
    def freqs_hz(self):
        return np.array([m.freq for m in self.modes])

    @property
    def importances(self):
        return np.array([m.importance for m in self.modes])

    def summary(self):
        """``[(tau, frequency, importance), ...]`` ranked by importance."""
        return [m.as_tuple() for m in self.modes]

    def table(self):
        return [
            {'rank': i, 'kind': m.kind, 'tau_s': m.tau, 'freq_hz': m.freq,
             'radius': m.radius, 'importance': m.importance,
             'importance_is': m.importance_is, 'rel_importance': m.rel_importance}
            for i, m in enumerate(self.modes)
        ]

    # -- dominant / aperiodic modes ---------------------------------------
    def dominant_mode(self, kinds=None):
        """Highest-importance mode, optionally restricted to ``kinds``."""
        for m in self.modes:
            if kinds is None or m.kind in kinds:
                return m
        return None

    @property
    def tau_dominant(self):
        m = self.dominant_mode()
        return np.nan if m is None else m.tau

    @property
    def tau_relaxation(self):
        """The model's aperiodic timescale: its slowest non-rhythmic mode.

        A real pole is non-rhythmic by definition.  So, in every practical
        sense, is a conjugate pair that rings for less than one cycle before
        its envelope decays: it is a damped relaxation that the optimizer
        happened to give a small angle rather than exactly zero.  Requiring a
        *real* pole here is a statement about the algebra rather than about
        the process, and it left a third of real recordings with no timescale
        to report even though the fits were good.

        Fitted models are constrained to contain a real pole (see
        ``tsmodes.estimators.REQUIRE_RELAXATION``), so this normally returns
        one.  The closed-form estimators, Burg and lagged regression, cannot
        be constrained that way, and this is what covers them.
        """
        finite = [m for m in self.modes if np.isfinite(m.tau) and m.tau > 0]
        # Every non-rhythmic mode is a candidate: a real pole, and equally a
        # conjugate pair that rings for less than a cycle, which is a damped
        # relaxation in all but the algebraic classification.
        #
        # Importance decides between them, and it must.  Requiring a real pole
        # guarantees one exists but not that it means anything: on mouse LFP
        # the constrained spectral fit parks its required pole at 0.58 ms with
        # negligible importance while the aperiodic structure sits in an 88 ms
        # pair ringing 0.57 of a cycle.  Preferring the real pole reported the
        # throwaway.
        cand = [m for m in finite if m.n_cycles < 1.0]
        # A conjugate pair ringing less than a cycle at the frequency of a
        # rhythmic pair is the other half of that rhythm, not a relaxation.
        # A lag-domain fit to the AR(7) benchmark splits its 10 Hz rhythm into
        # pairs at 10.0 Hz (1.25 cycles) and 10.2 Hz (0.57 cycles), and the
        # second, being more important than the 200 ms real pole, was reported
        # as a 56 ms relaxation in a quarter of the repetitions.  A pair within
        # half an octave of a rhythmic pair is therefore not a candidate.
        rhythmic = [m.freq for m in finite if m.kind == 'oscillatory' and m.n_cycles >= 1.0]
        cand = [m for m in cand if m.kind != 'oscillatory'
                or not any(abs(np.log2(m.freq / f)) < 0.5 for f in rhythmic if f > 0)]
        if not cand:
            # Every mode is a genuine rhythm, so the model has no aperiodic
            # timescale and we say so.  Substituting a rhythm's envelope here
            # was actively misleading: an AR(1) with a strong 40 Hz sinusoid
            # is fit at order 2 by a pair pressed against the unit circle,
            # whose envelope decays over 40 s while ringing 1600 times, and
            # that 40 s was being reported as the aperiodic timescale.
            return np.nan
        # The most important, not the slowest.  A long-memory process is fit
        # with a pole pushed against the unit circle, whose timescale is
        # thousands of seconds and whose contribution to the spectrum is
        # negligible; taking the slowest mode would report that number.
        best = max(cand, key=lambda m: (m.importance if np.isfinite(m.importance)
                                        else -np.inf))
        return best.tau

    @property
    def tau_slowest(self):
        finite = [m.tau for m in self.modes if np.isfinite(m.tau)]
        return max(finite) if finite else np.nan

    def mode_log_factors(self, freqs=None):
        """``G[j, f] = g_j(f)`` for every mode (natural log)."""
        freqs = self.freqs if freqs is None else freqs
        return np.vstack([pole_log_factor(m.poles, freqs, self.fs) for m in self.modes])

    def dominant_subset(self, tol=0.1, kind='logpsd', relative=True,
                        max_exhaustive=14):
        """Smallest subset of modes reproducing the fitted PSD within ``tol``.

        With ``relative=True`` (the default) the error of each candidate subset
        is divided by the error of the empty model, which is exactly
        ``std_f(log10 P_full)``.  The criterion is then a coefficient of
        determination *for the fitted log-spectrum*,
        ``R^2 = 1 - (err / err_empty)^2``, so ``tol=0.1`` keeps the smallest set
        of modes reproducing 99% of the model's log-spectral variation.  This is
        a statement about the fitted spectrum, not about the variance of the
        signal, which does not decompose by mode (see
        :meth:`variance_decomposition`).

        With ``relative=False``, ``tol`` is in RMS log10-power units for
        ``kind='logpsd'`` and in nats for ``kind='is'``.

        Note that the best-achievable error is not monotone in subset size:
        modes whose log-spectral contributions partly cancel can be dropped
        together with less damage than either causes alone.

        Returns
        -------
        dict with ``indices`` (kept modes), ``error``, ``n_modes``, ``curve``
        (best achievable error per subset size), ``rel_curve`` and ``r2_curve``.
        """
        n = len(self.modes)
        best_per_size = {}
        if kind == 'logpsd' and n <= max_exhaustive:
            # err(S)^2 = v' C v with v the indicator of the *dropped* modes and
            # C the frequency covariance of the mode log-factors, so every
            # subset is one quadratic form.
            g = self.mode_log_factors()
            gc = g - g.mean(axis=1, keepdims=True)
            cov = gc @ gc.T / gc.shape[1]
            masks = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
            drop = 1.0 - masks  # 1 where the mode is dropped
            errs = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', drop, cov, drop), 0.0)) / LN10
            sizes = masks.sum(axis=1).astype(int)
            for size in range(n + 1):
                where = np.where(sizes == size)[0]
                best_i = where[int(np.argmin(errs[where]))]
                combo = tuple(np.where(masks[best_i] > 0)[0].tolist())
                best_per_size[size] = (combo, float(errs[best_i]))
        elif n <= max_exhaustive:
            for size in range(n + 1):
                best = None
                for combo in combinations(range(n), size):
                    err = subset_error(self.modes, combo, self.freqs, self.fs, kind)
                    if best is None or err < best[1]:
                        best = (combo, err)
                best_per_size[size] = best
        else:
            keep = []
            best_per_size[0] = ((), subset_error(self.modes, (), self.freqs, self.fs, kind))
            for size in range(1, n + 1):
                cands = [(k, subset_error(self.modes, keep + [k], self.freqs, self.fs, kind))
                         for k in range(n) if k not in keep]
                k, err = min(cands, key=lambda t: t[1])
                keep.append(k)
                best_per_size[size] = (tuple(sorted(keep)), err)

        curve = np.array([best_per_size[s][1] for s in range(n + 1)])
        scale = curve[0] if curve[0] > 0 else 1.0
        rel_curve = curve / scale
        score = rel_curve if relative else curve
        chosen = n
        for size in range(n + 1):
            if score[size] <= tol:
                chosen = size
                break
        return {'indices': list(best_per_size[chosen][0]),
                'error': float(curve[chosen]),
                'rel_error': float(rel_curve[chosen]),
                'n_modes': chosen,
                'curve': curve,
                'rel_curve': rel_curve,
                'r2_curve': 1.0 - rel_curve ** 2}

    # -- spectra ----------------------------------------------------------
    def psd(self, freqs=None):
        freqs = self.freqs if freqs is None else freqs
        return ar_psd(self.phi, freqs, self.fs, self.offset)

    def psd_from_modes(self, freqs=None, indices=None, refit_scale=True):
        """PSD of the (possibly reduced) pole factorization."""
        freqs = self.freqs if freqs is None else freqs
        indices = range(len(self.modes)) if indices is None else indices
        poles = [lam for i in indices for lam in self.modes[i].poles]
        if len(poles) == 0:
            reduced = np.ones(len(np.atleast_1d(freqs)))
        else:
            reduced = psd_from_poles(poles, freqs, self.fs, offset=1.0)
        if not refit_scale:
            return self.offset * reduced
        full = self.psd(freqs)
        scale = np.exp(np.mean(np.log(full) - np.log(reduced)))
        return scale * reduced

    def acf(self, nlags, normalize=True):
        return ar_acf(self.phi, nlags, normalize=normalize)

    def impulse_components(self, n):
        """Per-mode contributions to the impulse response (they sum to it)."""
        return impulse_mode_components(self.phi, n,
                                       groups=[m.poles for m in self.modes])

    def acf_components(self, nlags, normalize=True):
        """Per-mode additive contributions to the ACF (they sum to it exactly)."""
        return acf_mode_components(self.phi, nlags,
                                   groups=[m.poles for m in self.modes],
                                   normalize=normalize)

    def psd_components(self, freqs=None):
        """Per-mode additive contributions to the spectrum (they sum to it)."""
        freqs = self.freqs if freqs is None else freqs
        return psd_mode_components(self.phi, freqs, self.fs, sigma2=self.offset,
                                   groups=[m.poles for m in self.modes])

    def variance_decomposition(self):
        return variance_decomposition(self.poles, groups=[m.poles for m in self.modes])

    def __repr__(self):
        head = f'ARModes(order={self.order}, fs={self.fs:g}, stationary={self.is_stationary})'
        rows = '\n'.join(f'  {i}. {m!r}' for i, m in enumerate(self.modes))
        return head + '\n' + rows

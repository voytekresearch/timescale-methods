"""Guardrails for the requirement that a timescale model has a timescale.

An AR(p) model of a neural recording is fit in order to read a relaxation
time off it.  A fitted model whose poles are all complex has no relaxation
mode and therefore no timescale to report, which is a failure of the fit as a
measurement instrument even when it is an excellent description of the data.
These tests pin down that a relaxation mode is always available.
"""

import numpy as np
import pytest

from timescales.ar.core import fit_ar_acf, fit_ar_psd

from tsmodes.estimators import estimate, psd_fit_points
from tsmodes.prep import prepare
from timescales.decomposition.modes import ARModes, ar_to_poles, group_poles, poles_to_ar
from timescales.sim import AR1, AR1PlusSine, AR2Damped, sim_ar_process

FS = 500.0
TAU = 0.05

PROCESSES = [
    (AR1(tau=TAU), 'ar1'),
    (AR1PlusSine(tau=TAU, f_osc=10.0, snr=1.0), 'ar1_sine'),
    (AR1PlusSine(tau=TAU, f_osc=4.0, snr=4.0), 'ar1_sine_strong'),
    (AR2Damped(tau=TAU, f_osc=10.0), 'ar2_damped'),
]


def _signal(proc, seed=0, duration=200.0):
    return prepare(proc.simulate(duration, FS, np.random.default_rng(seed)),
                   FS, welch_win_s=4.0)


def _n_relaxation(phi):
    # Use the library's own grouping: a repeated real pole comes back from
    # root finding as a conjugate pair with a tiny imaginary part, which is
    # still a relaxation mode to report.
    return sum(1 for grp in group_poles(ar_to_poles(phi))
               if len(grp) == 1 and grp[0].real > 0)


# ---------------------------------------------------------------------------
# The parameterization must guarantee the pole, not merely find it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('proc,name', PROCESSES)
@pytest.mark.parametrize('order', [4, 6, 9])
def test_constrained_acf_fit_always_has_a_relaxation_pole(proc, name, order):
    signal = _signal(proc)
    phi, _, _, _ = fit_ar_acf(signal.lags, signal.corrs, FS, order,
                              require_relaxation=True)
    assert _n_relaxation(phi) >= 1, f'{name}, order {order}: no relaxation pole'
    assert np.all(np.abs(ar_to_poles(phi)) < 1.0), 'not stationary'


@pytest.mark.parametrize('proc,name', PROCESSES)
@pytest.mark.parametrize('order', [4, 6, 9])
def test_constrained_psd_fit_always_has_a_relaxation_pole(proc, name, order):
    signal = _signal(proc)
    freqs, powers = psd_fit_points(signal)
    phi, _, _ = fit_ar_psd(freqs, powers, FS, order, require_relaxation=True)
    assert _n_relaxation(phi) >= 1, f'{name}, order {order}: no relaxation pole'
    assert np.all(np.abs(ar_to_poles(phi)) < 1.0), 'not stationary'


# ---------------------------------------------------------------------------
# ... and every estimator must report a finite timescale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('proc,name', PROCESSES)
@pytest.mark.parametrize('method', ['arp_direct', 'arp_acf', 'arp_psd'])
def test_every_arp_estimator_reports_a_timescale(proc, name, method):
    fit = estimate(_signal(proc), method, robust=proc.has_oscillation)
    assert np.isfinite(fit.tau_relaxation), f'{name}/{method}: no timescale'
    assert 1e-4 < fit.tau_relaxation < 100.0


# ---------------------------------------------------------------------------
# The constraint must not cost accuracy where the old fit was already right
# ---------------------------------------------------------------------------

def test_constraint_does_not_degrade_the_ar1_recovery():
    for seed in range(4):
        fit = estimate(_signal(AR1(tau=TAU), seed), 'arp_acf')
        assert abs(fit.tau_relaxation - TAU) / TAU < 0.25, f'seed {seed}'
        fit = estimate(_signal(AR1(tau=TAU), seed), 'arp_psd')
        assert abs(fit.tau_relaxation - TAU) / TAU < 0.25, f'seed {seed}'


def test_constraint_does_not_degrade_the_oscillator_recovery():
    """The AR(2) oscillator's rhythm must survive the extra real pole."""
    for seed in range(4):
        fit = estimate(_signal(AR2Damped(tau=TAU, f_osc=10.0), seed), 'arp_psd',
                       robust=True)
        osc = [m for m in fit.modes.modes if m.kind == 'oscillatory']
        assert osc, f'seed {seed}: rhythm lost'
        best = max(osc, key=lambda m: m.importance)
        assert abs(best.freq - 10.0) < 1.0
        assert abs(best.tau - TAU) / TAU < 0.35


def test_constrained_fit_matches_the_unconstrained_one_when_a_pole_exists():
    """Where the free fit already has a real pole, constraining changes little."""
    signal = _signal(AR1(tau=TAU))
    free = fit_ar_acf(signal.lags, signal.corrs, FS, 4,
                      require_relaxation=False)
    held = fit_ar_acf(signal.lags, signal.corrs, FS, 4,
                      require_relaxation=True)
    assert _n_relaxation(free[0]) >= 1, 'precondition: free fit has a real pole'
    t_free = max(m.tau for m in ARModes.from_phi(free[0], FS).modes
                 if m.kind == 'relaxation')
    t_held = max(m.tau for m in ARModes.from_phi(held[0], FS).modes
                 if m.kind == 'relaxation')
    assert abs(np.log(t_held / t_free)) < np.log(1.5)


def test_the_added_pole_is_where_the_slow_component_is():
    """On AR(1)+sine the constrained fit should find the 50 ms decay."""
    signal = _signal(AR1PlusSine(tau=TAU, f_osc=10.0, snr=1.0))
    fit = estimate(signal, 'arp_acf')
    assert np.isfinite(fit.tau_relaxation)
    assert 0.015 < fit.tau_relaxation < 0.30
    rhythm = [m for m in fit.modes.modes if m.n_cycles >= 1.0]
    assert rhythm, 'the rhythm was lost'
    assert abs(max(rhythm, key=lambda m: m.n_cycles).freq - 10.0) < 0.5


def test_order_selection_buys_the_order_the_constraint_costs():
    """Reserving a pole costs one order, and order selection pays it.

    At order 2 the constraint leaves a single pole for everything else, so an
    oscillator is unrepresentable there.  Rather than exempting order 2 --
    which let a purely rhythmic model through and had its 1600-cycle envelope
    reported as its timescale -- the constraint applies everywhere
    and BIC simply selects a higher order.
    """
    for seed in range(3):
        fit = estimate(_signal(AR2Damped(tau=TAU, f_osc=10.0), seed), 'arp_acf')
        assert fit.order >= 3, 'order selection did not pay for the pole'
        osc = [m for m in fit.modes.modes if m.kind == 'oscillatory']
        assert osc, f'seed {seed}: the rhythm was lost'
        best = max(osc, key=lambda m: m.importance)
        assert abs(best.freq - 10.0) < 1.5
        assert abs(best.tau - TAU) / TAU < 0.35
        assert _n_relaxation(fit.phi) >= 1


def test_a_purely_rhythmic_model_reports_no_timescale():
    """An undamped sinusoid has no relaxation, and we must not invent one.

    A strong 40 Hz sinusoid is fit by a pair pressed against the unit circle:
    its envelope decays over 40 s while the mode rings 1600 times.  Reporting
    that envelope as the reported timescale gave errors of 800x.
    """
    fs = 500.0
    r = np.exp(-1.0 / (40.0 * fs))
    theta = 2 * np.pi * 40.0 / fs
    phi = poles_to_ar(np.array([r * np.exp(1j * theta), r * np.exp(-1j * theta)]))
    model = ARModes.from_phi(phi, fs)
    assert model.modes[0].n_cycles > 100
    assert not np.isfinite(model.tau_relaxation)


def test_a_floor_mode_is_not_mistaken_for_a_relaxation():
    """A negative real pole alternates every sample; it is not a decay.

    Its poles are real, so a kind-based test called it non-rhythmic and the
    reported timescale was read off a mode with tau = 125 s.  n_cycles must
    count it as the fastest rhythm the sampling rate allows.
    """
    fs = 500.0
    model = ARModes.from_phi(np.array([-0.9]), fs)
    nyq = model.modes[0]
    assert nyq.kind == 'floor'
    assert nyq.n_cycles > 1.0
    assert not np.isfinite(model.tau_relaxation) or model.tau_relaxation > 0

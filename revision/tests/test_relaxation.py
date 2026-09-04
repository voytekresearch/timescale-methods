"""Guardrails for the requirement that a timescale model has a timescale.

An AR(p) model of a neural recording is fit in order to read a relaxation
time off it.  A fitted model whose poles are all complex has no relaxation
mode and therefore no timescale to report, which is a failure of the fit as a
measurement instrument even when it is an excellent description of the data.
These tests pin down that a relaxation mode is always available.
"""

import numpy as np
import pytest

from tsmodes.estimators import (estimate, fit_ar_acf, fit_ar_psd, prepare,
                                psd_fit_points)
from tsmodes.modes import ARModes, ar_to_poles
from tsmodes.sims import (AR1, AR1PlusSine, AR2Damped, PowerLaw,
                          sim_ar_process)

FS = 500.0
TAU = 0.05

PROCESSES = [
    (AR1(tau=TAU), 'ar1'),
    (AR1PlusSine(tau=TAU, f_osc=10.0, snr=1.0), 'ar1_sine'),
    (AR1PlusSine(tau=TAU, f_osc=4.0, snr=4.0), 'ar1_sine_strong'),
    (AR2Damped(tau=TAU, f_osc=10.0), 'ar2_damped'),
    (PowerLaw(d=0.45), 'power_law'),
]


def _signal(proc, seed=0, duration=200.0):
    return prepare(proc.simulate(duration, FS, np.random.default_rng(seed)),
                   FS, welch_win_s=4.0)


def _n_relaxation(phi):
    poles = ar_to_poles(phi)
    return sum(1 for p in poles if abs(p.imag) < 1e-8 and p.real > 0)


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
# ... and every estimator must report a finite aperiodic timescale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('proc,name', PROCESSES)
@pytest.mark.parametrize('method', ['arp_direct', 'arp_acf', 'arp_psd'])
def test_every_arp_estimator_reports_an_aperiodic_timescale(proc, name, method):
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


def test_the_added_pole_is_where_the_aperiodic_component_is():
    """On AR(1)+sine the constrained fit should find the 50 ms decay."""
    signal = _signal(AR1PlusSine(tau=TAU, f_osc=10.0, snr=1.0))
    fit = estimate(signal, 'arp_acf')
    assert np.isfinite(fit.tau_relaxation)
    assert 0.015 < fit.tau_relaxation < 0.30
    rhythm = [m for m in fit.modes.modes if m.n_cycles >= 1.0]
    assert rhythm, 'the rhythm was lost'
    assert abs(max(rhythm, key=lambda m: m.n_cycles).freq - 10.0) < 0.5


def test_order_two_is_exempt_so_an_oscillator_stays_representable():
    """Reserving a pole costs one order, and AR(2) has none to spare.

    An AR(2) oscillator is a conjugate pair.  Forcing a real pole at order 2
    leaves one pole for everything else, so the model could only be two
    exponentials and the rhythm would be unrepresentable.  From order 3 the
    requirement is free.
    """
    signal = _signal(AR2Damped(tau=TAU, f_osc=10.0))
    phi, _, _, _ = fit_ar_acf(signal.lags, signal.corrs, FS, 2,
                              require_relaxation=True)
    modes = ARModes.from_phi(phi, FS).modes
    assert any(m.kind == 'oscillatory' for m in modes)
    assert abs(max(modes, key=lambda m: m.importance).freq - 10.0) < 2.0

    phi3, _, _, _ = fit_ar_acf(signal.lags, signal.corrs, FS, 3,
                               require_relaxation=True)
    assert _n_relaxation(phi3) >= 1
    assert any(m.kind == 'oscillatory' for m in ARModes.from_phi(phi3, FS).modes)


def test_a_nyquist_mode_is_not_mistaken_for_a_relaxation():
    """A negative real pole alternates every sample; it is not a decay.

    Its poles are real, so a kind-based test called it non-rhythmic and the
    aperiodic timescale was read off a mode with tau = 125 s.  n_cycles must
    count it as the fastest rhythm the sampling rate allows.
    """
    fs = 500.0
    model = ARModes.from_phi(np.array([-0.9]), fs)
    nyq = model.modes[0]
    assert nyq.kind == 'nyquist'
    assert nyq.n_cycles > 1.0
    assert not np.isfinite(model.tau_relaxation) or model.tau_relaxation > 0

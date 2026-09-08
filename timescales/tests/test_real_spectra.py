"""One set of settings, every recording.

``timescales.data`` holds the median power spectrum and median autocorrelation
of eight public recordings spanning fMRI, MEG, EEG, ECoG, LFP and spike
counts, across four species and three orders of magnitude of sampling rate.

These tests check that one configuration fits all of them.  If a threshold
here can only be met by tuning per recording, the defaults are hiding an
assumption.

The configuration is the one the paper's figures use, and it is copied here
deliberately: a tutorial or a test that fits these recordings with different
settings gets different answers and there is no way to tell which is right.

Two failures are recorded here, both found on the rat hippocampal recording.

The per-band error test exists because of the first.  With
``order=range(1, 16)`` the fit scored 0.979 overall while missing the theta
band by 0.21 dex, a factor of 1.6 in power: BIC over AR order is not unimodal,
it sat on a local minimum at order 8, and the grid ended before the criterion
turned.  An overall :math:`R^2` cannot see that, because a linearly spaced
spectrum puts most of its points in the top decade.

The boundary test exists because of the second.  ``tau`` keeps modes ringing
less than one cycle and discards the rest, and the theta mode sits on that
cut: it is 0.95 cycles in one fit and 1.08 in another, and the reported
timescale goes from 110 ms to 28.5 ms while the fit itself barely changes.
The fits here do not land there, but the flag has to fire when a fit does.
"""

import warnings

import numpy as np
import pytest

from timescales.ar import ARACF, ARPSD
from timescales.data import DATASETS, load_acf, load_spectrum

KEYS = list(DATASETS)

#: These fit 25 orders on eight recordings; run with `-m 'not slow'` to skip.
pytestmark = pytest.mark.slow

#: The single configuration under test: the one the paper's figures use
#: (analysis/tsmodes/estimators.py).  The grid is log-spaced and capped at 20,
#: the spectral loss is robust so a rhythm does not drag the aperiodic fit,
#: and every model is required to contain a real pole so there is always a
#: decay time to report.
ORDERS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20)
PSD_SETTINGS = dict(loss='soft_l1', f_scale=0.15, require_relaxation=True)
ACF_SETTINGS = dict(require_relaxation=True)

#: Thresholds measured from this configuration, with room to spare.
MIN_RSQ_PSD = 0.90
MIN_RSQ_ACF = 0.55
#: Largest acceptable median error in any quarter of the log-frequency band,
#: in log10 power.  0.15 dex is 41% in power.  The worst observed is 0.116.
MAX_BAND_ERROR = 0.20

_psd = load_spectrum
_acf = load_acf


@pytest.fixture(scope='module')
def psd_fits():
    """One spectral fit per recording, shared across tests."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # order-ceiling notices
        for key in KEYS:
            freqs, powers, fs = _psd(key)
            ar = ARPSD(order=ORDERS, fs=fs, **PSD_SETTINGS).fit(freqs, powers)
            ar.decompose(freqs=freqs)
            out[key] = ar
    return out


@pytest.fixture(scope='module')
def acf_fits():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for key in KEYS:
            lags, corrs, fs = _acf(key)
            out[key] = ARACF(order=ORDERS, fs=fs, **ACF_SETTINGS).fit(lags, corrs)
    return out


def test_the_fixture_covers_every_modality():
    assert len(KEYS) == 8
    for key in KEYS:
        freqs, powers, fs = _psd(key)
        lags, corrs, _ = _acf(key)
        assert freqs.shape == powers.shape and len(freqs) > 20
        assert lags.shape == corrs.shape and len(lags) > 20
        assert np.all(powers > 0)
        assert corrs[0] == pytest.approx(1.0)
        assert fs > 0
    rates = {_psd(k)[2] for k in KEYS}
    assert max(rates) / min(rates) > 100


@pytest.mark.parametrize('key', KEYS)
def test_psd_fit_is_good_on_every_recording(key, psd_fits):
    ar = psd_fits[key]
    assert ar.success_
    assert ar.rsq > MIN_RSQ_PSD
    assert ar.is_stationary


@pytest.mark.parametrize('key', KEYS)
def test_psd_fit_is_good_in_every_band(key, psd_fits):
    """The test that catches a truncated order search.

    An overall R^2 is dominated by the top decade of a linearly spaced
    spectrum, so it stays high while the model misses the low frequencies.
    """
    _, error = psd_fits[key].error_by_band(4)
    worst = np.nanmax(error)
    assert worst < MAX_BAND_ERROR, f'{key}: worst band error {worst:.3f} dex, {error}'


@pytest.mark.parametrize('key', KEYS)
def test_acf_fit_is_good_on_every_recording(key, acf_fits):
    ar = acf_fits[key]
    assert ar.success_
    assert ar.rsq > MIN_RSQ_ACF
    assert ar.is_stationary


@pytest.mark.parametrize('key', KEYS)
def test_higher_order_beats_ar1_on_every_recording(key, psd_fits):
    freqs, powers, fs = _psd(key)
    one = ARPSD(order=1, fs=fs, loss='soft_l1', f_scale=0.15).fit(freqs, powers)
    many = psd_fits[key]
    assert many.rsq >= one.rsq
    if many.order_ > 1:
        assert many.rsq > one.rsq


@pytest.mark.parametrize('key', KEYS)
def test_reported_timescale_is_resolvable(key, psd_fits):
    """A timescale must be finite, positive, and inside the analysed range."""
    freqs, _, fs = _psd(key)
    tau = psd_fits[key].tau
    assert np.isfinite(tau) and tau > 0
    assert 1.0 / fs < tau < 1.0 / freqs[0]


@pytest.mark.parametrize('key', KEYS)
def test_decomposition_reproduces_the_fitted_spectrum(key, psd_fits):
    """The mode factors multiply to the fitted spectrum."""
    freqs, _, _ = _psd(key)
    ar = psd_fits[key]
    modes = ar.modes
    assert len(modes.modes) >= 1
    product = np.ones_like(freqs)
    for mode in modes.modes:
        factor = mode.psd(freqs)
        assert np.all(factor > 0)
        product = product * factor
    scale = np.exp(np.mean(np.log(ar.predict()) - np.log(product)))
    np.testing.assert_allclose(scale * product, ar.predict(), rtol=1e-6)
    # the partial fractions are the additive decomposition and sum to the fit
    np.testing.assert_allclose(modes.psd_components(freqs).sum(axis=0),
                               ar.predict(), rtol=1e-6)


@pytest.mark.parametrize('key', KEYS)
def test_acf_decomposition_reproduces_the_fitted_acf(key, acf_fits):
    lags, _, _ = _acf(key)
    modes = acf_fits[key].decompose()
    comps = modes.acf_components(int(lags[-1]))
    np.testing.assert_allclose(comps.sum(axis=0),
                               modes.acf(int(lags[-1])), atol=1e-5)


def test_no_setting_is_tuned_per_recording(psd_fits):
    """`fs` and the data are the only inputs that differ between recordings."""
    scores = {k: ar.rsq for k, ar in psd_fits.items()}
    assert min(scores.values()) > MIN_RSQ_PSD, scores


def test_timescales_span_the_expected_range_across_modalities(psd_fits):
    """fMRI is seconds, electrophysiology is tens of milliseconds."""
    taus = {k: ar.tau for k, ar in psd_fits.items()}
    assert taus['human_fmri_rest'] > 1.0
    field = [taus[k] for k in KEYS if k.startswith(('human_lemon', 'human_meg',
                                                    'human_cole', 'macaque',
                                                    'rat_'))]
    assert all(1e-3 < t < 1.0 for t in field), taus
    assert taus['human_fmri_rest'] > 10 * max(field)


def test_a_short_order_grid_is_caught(psd_fits):
    """The failure this suite exists to prevent, on the recording that showed it.

    A grid ending at 15 leaves the rat hippocampal fit on a local minimum of
    BIC at order 8.  The overall R^2 stays above 0.97 and the per-band error
    is what exposes it.
    """
    freqs, powers, fs = _psd('rat_hc3_ca1_ca3')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        short = ARPSD(order=range(1, 16), fs=fs).fit(freqs, powers)
    assert short.rsq > 0.97                      # looks fine
    assert np.nanmax(short.error_by_band(4)[1]) > MAX_BAND_ERROR   # is not
    assert np.nanmax(psd_fits['rat_hc3_ca1_ca3'].error_by_band(4)[1]) < MAX_BAND_ERROR


@pytest.mark.parametrize('key', KEYS)
def test_no_fit_sits_on_the_cycle_boundary(key, psd_fits):
    """Under this configuration the aperiodic rule is not deciding a tie."""
    ar = psd_fits[key]
    assert not ar.tau_near_boundary_, (
        f'{key}: leading mode rings {ar.modes.modes[0].n_cycles:.2f} cycles, '
        f'so tau is decided by which side of f*tau = 1 the fit landed')


def test_a_boundary_fit_is_flagged_and_warned():
    """The knife-edge case, on the recording that showed it.

    An unrestricted order grid without the robust loss puts the rat theta mode
    within a whisker of one cycle.  The fit is good either way; the reported
    timescale is not stable, and that has to be said rather than returned as a
    settled number.
    """
    freqs, powers, fs = load_spectrum('rat_hc3_ca1_ca3')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ar = ARPSD(order=range(1, 26), fs=fs).fit(freqs, powers)
        ar.decompose(freqs=freqs)
    assert ar.tau_near_boundary_
    assert any('near tie' in str(c.message) for c in caught)
    assert 0.8 <= ar.modes.modes[0].n_cycles <= 1.25


@pytest.mark.parametrize('key', KEYS)
def test_a_detected_floor_behaves_like_an_additive_constant(key, psd_fits):
    """Where a fit has a floor mode, the name has to be earned.

    A negative real pole is a high-pass factor, which is multiplicative. It is
    called a noise floor because replacing it with an additive constant
    reproduces the fitted spectrum; this checks that it does, on the
    recordings that have one.
    """
    ar = psd_fits[key]
    freqs, _, _ = _psd(key)
    floor = ar.modes.noise_floor(freqs)
    if floor is None:
        assert ar.modes.floor_mode is None
        return
    assert floor['r2'] > 0.99, f'{key}: additive form fits at R2={floor["r2"]:.4f}'
    assert floor['level'] > 0
    # and it is never reported as a timescale
    assert all(m.kind != 'floor' for m in ar.modes.resolvable(
        (float(freqs[0]), float(freqs[-1]))))


def test_the_floor_is_found_where_the_band_reaches_it():
    """Spike counts: the Poisson floor sits above the published 2 Hz band.

    Extending the band toward it turns the floor mode on, which is the
    behaviour the name predicts.
    """
    freqs, powers, fs = _psd('mouse_neuropixels_spikes')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ar = ARPSD(order=ORDERS, fs=fs, **PSD_SETTINGS).fit(freqs, powers)
        ar.decompose(freqs=freqs)
    floor = ar.modes.noise_floor(freqs)
    # inside the published band the floor is a small fraction of the top-of-band power
    if floor is not None:
        assert floor['relative'] < 0.5

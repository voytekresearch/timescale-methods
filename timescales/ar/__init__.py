"""Autoregressive timescale models.

``ARPSD`` fits an AR(p) model to a power spectrum and ``ARACF`` fits the same
model to an autocorrelation.  Both default to AR(1), the familiar single
timescale, and accept any order or a grid of orders to select from.
"""

from .arpsd import ARPSD
from .aracf import ARACF
from .spectral import compute_ar_spectrum, burg, ar_to_psd
from .time import ar_ols, ar_burg, population_ar, select_order_time
from .metrics import (r2_logpsd, rmse_logpsd, is_divergence, spectral_gof,
                      acf_fit_gof, acf_gof)

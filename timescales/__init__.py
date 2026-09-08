"""Timescale simulation, estimation and interpretation.

The public entry points are :class:`~timescales.ar.ARPSD` and
:class:`~timescales.ar.ARACF`, which fit an AR(p) model to a power spectrum or
an autocorrelation, and :class:`~timescales.decomposition.ARModes`, which reads
the fitted model as a set of relaxation and oscillatory modes.
"""

from .version import __version__

from .conversions import (convert_knee, convert_tau, knee_to_tau, tau_to_knee,
                          tau_to_phi, phi_to_tau, psd_to_acf, acf_to_psd)
from .ar import ARPSD, ARACF
from .decomposition import ARModes, Mode
from .utils import compute_psd, compute_acf
from .data import load_spectrum, load_acf, DATASETS
from . import ar, conversions, data, decomposition, finite, sim, utils

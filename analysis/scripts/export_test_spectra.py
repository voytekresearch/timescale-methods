"""Export one PSD and one ACF per recording as a test fixture for the package.

The `timescales` package ships general estimators; the recordings here are the
hardest inputs we have for them.  This writes the median spectrum and median
autocorrelation of each recording, which is enough for the package's tests to
check that one set of default settings fits every modality, without shipping
any of the recordings themselves.

Output: timescales/tests/data/real_spectra.npz  (a few hundred kB)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tsmodes import datasets as D

OUT = Path(__file__).resolve().parents[2] / 'timescales' / 'tests' / 'data' / 'real_spectra.npz'


def main():
    arrays = {}
    for key in D.available():
        data = D.load(key)
        sigs = [D.prepare_trace(trace, data) for trace in data['X']]
        sig = sigs[0]
        mask = sig.fit_mask & sig.exclude_mask
        nlag = min(len(s.corrs) for s in sigs)
        arrays[f'{key}/freqs'] = sig.freqs[mask]
        arrays[f'{key}/powers'] = np.median([s.powers[mask] for s in sigs], axis=0)
        arrays[f'{key}/lags'] = np.arange(nlag, dtype=float)
        arrays[f'{key}/corrs'] = np.median([s.corrs[:nlag] for s in sigs], axis=0)
        arrays[f'{key}/fs'] = np.array(float(data['fs']))
        print(f'{key}: {mask.sum()} bins {sig.f_range[0]:g}-{sig.f_range[1]:g} Hz, '
              f'{nlag} lags, fs={data["fs"]:g}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()

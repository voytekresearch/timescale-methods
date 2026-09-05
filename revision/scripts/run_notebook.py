"""Execute every code cell of a notebook, in order, without Jupyter.

    python scripts/run_notebook.py notebooks/figures.ipynb

The cells run in one shared namespace, exactly as they would in a kernel, with
the working directory set to the notebook's own directory.  This is what the
DVC ``plot_figures`` stage runs, so the figures in ``figures/`` are always the
ones the notebook draws and there is no second copy of the plotting code.

The notebook is the source of truth for every figure; edit it there.
"""
import json
import os
import sys
import time
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(path):
    path = Path(path).resolve()
    nb = json.loads(path.read_text())
    os.chdir(path.parent)
    namespace = {'__name__': '__main__'}
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        first = next((l for l in source.splitlines() if l.strip()), '')
        print(f'--- cell {i}: {first[:70]}', flush=True)
        t0 = time.time()
        exec(compile(source, f'{path.name}:cell{i}', 'exec'), namespace)
        plt.close('all')
        print(f'    {time.time() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'notebooks/figures.ipynb')

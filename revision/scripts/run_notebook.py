"""Execute every code cell of a notebook, in order, without Jupyter.

    python scripts/run_notebook.py notebooks/figures.ipynb --export scripts/plot_figures.py

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


def export_script(nb, path, out):
    """Write the notebook's code cells, in order, as one plain Python script.

    The notebook stays the place to edit; this is the same code as a script,
    regenerated on every run so the two cannot drift.
    """
    out = Path(out)
    lines = ['"""Every figure of the paper, as a script.',
             '',
             f'GENERATED from {path.name} by scripts/run_notebook.py; edit the notebook,',
             'not this file.  Run from revision/ with PYTHONPATH set to it.',
             '"""', '']
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        lines += [f'# {"=" * 74}', f'# notebook cell {i}', f'# {"=" * 74}',
                  ''.join(cell['source']).rstrip('\n'), '', '']
    out.write_text('\n'.join(lines))
    print(f'exported {out}', flush=True)


def main(path, export=None):
    path = Path(path).resolve()
    export = Path(export).resolve() if export else None     # before the chdir
    nb = json.loads(path.read_text())
    os.chdir(path.parent)
    if export:
        export_script(nb, path, export)
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
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('notebook', nargs='?', default='notebooks/figures.ipynb')
    ap.add_argument('--export', help='also write the code cells as one script')
    ap.add_argument('--export-only', action='store_true', help='export without running')
    args = ap.parse_args()
    if args.export_only:
        nb = json.loads(Path(args.notebook).read_text())
        export_script(nb, Path(args.notebook), args.export or 'scripts/plot_figures.py')
    else:
        main(args.notebook, export=args.export)

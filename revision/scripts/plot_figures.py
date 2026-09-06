"""Draw every figure of the paper: run the scripts in scripts/figures/ in order.

    python scripts/plot_figures.py            # all figures
    python scripts/plot_figures.py 3 s1       # only fig_03 and fig_s1

Each figure is its own script (fig_01.py ... fig_06.py, fig_s1.py, fig_s2.py)
that can be run or edited on its own; the shared style, colours and helpers
live in scripts/figures/fig_common.py.  TSMODES_ARP=5 in the environment
draws the alternative AR(5) in place of the paper's AR(7) (fig_common).  This runner is what the DVC
``plot_figures`` stage executes, so figures/ always holds what the scripts
draw.
"""
import runpy
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent / 'figures'
sys.path.insert(0, str(HERE))                    # so `from fig_common import ...` resolves

ORDER = ['01', '02', '03', '04', '05', '06', 's1', 's2']


def main(which=None):
    which = [w.lstrip('fig_').zfill(2) if w.isdigit() else w for w in (which or ORDER)]
    for tag in which:
        script = HERE / f'fig_{tag}.py'
        print(f'--- {script.name}', flush=True)
        t0 = time.time()
        runpy.run_path(str(script), run_name='__main__')
        plt.close('all')
        print(f'    {time.time() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or None)

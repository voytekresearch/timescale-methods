"""Matplotlib settings."""

from cycler import cycler
import matplotlib.pyplot as plt

#: Seaborn's "colorblind" palette, inlined.  It was the only thing seaborn was
#: used for, and it is a fixed list of ten colours.
COLORBLIND = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc',
              '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']


SMALL_SIZE = 12
MEDIUM_SIZE = 18
LARGE_SIZE = 24

def set_default_rc():
    """Default plot settings."""

    plt.rc('lines', linewidth=2)
    plt.rc('font', size=SMALL_SIZE, family='sans-serif')
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)

    plt.rc('ytick.major', size=5)
    plt.rc('ytick.minor', size=2.5)
    plt.rc('xtick.major', size=5)
    plt.rc('xtick.minor', size=2.5)

    plt.rc('axes', prop_cycle=cycler(color=COLORBLIND))

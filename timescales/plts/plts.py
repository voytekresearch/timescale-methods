"""Plotting functions."""

import numpy as np


def plot_connected_scatter(dist0, dist1, ax0, paired=True, twin=False, fill_nans=False,
                           scatter_jit=.05, scatter_alpha=.5, violin_locs=(1, 2),
                           line_color=None, line_alpha=.1, xticklabels=None, colors=None, **kwargs):
    """Plot connected violin scattter plots.

    Parameters
    ----------
    dist0 : 1d or 2d array
        First parameter to plot.
    dist1 : 1d or 2d array
        Second parameter to plot.
    ax0 : AxesSubplot
        Subplot to plot onto.
    paired : bool, optional, default: True
        Plot lines connected pairs of points from dist0 and dist1.
    twin : bool, optional, default: True
        Twin the two axes to equalize different y-axis scalings.
    fill_nan : bool, optional, default: False
        Set nans to this value if not True.
    scatter_jit : float, optional, default: .05
        Scatter random jit standard deviation.
    scatter_alpha : float, optional, default: .05
        Transparency of scatter points.
    violin_locs : tuple, optional, default: (1, 2)
        Location of violin and scatter lots on the x-axis.
    line_color : str, optional, default: None
        Color of paired lines
    line_alpha : float, optional, default: .1
        Transparency of paired lines.
    xticklabels: list, optional, default: None
        X-axis labels.
    colors : list, optional, default: None
        Colors of the two violin and scatter plots.
    **kwargs : optional
        Additional plotting arguments.

    Returns
    -------
    ax0 : AxesSubplot
        Drawn subplot.
    ax1 : AxesSubplot, optional
        Drawn twin subplot.
    """

    # Pop kwargs
    colors = ['C0', 'C0'] if colors is None else colors
    title = kwargs.pop('title', '')
    ylabel = kwargs.pop('ylabel', '')
    xticks = kwargs.pop('xticks', violin_locs)
    xticklabels = ['Dist0', 'Dist1'] if xticklabels is None else xticklabels
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)

    # Fill nans with zeros
    if fill_nans is not False:
        dist0[np.isnan(dist0)] = fill_nans
        dist1[np.isnan(dist1)] = fill_nans

    # Violinplots
    if not twin:
        vp = ax0.violinplot([dist0, dist1], positions=violin_locs)
        vp = _set_vp_colors(vp, *colors)
    else:
        ax1 = ax0.twinx()
        vp0 = ax0.violinplot(dist0, positions=[violin_locs[0]])
        vp1 = ax1.violinplot(dist1, positions=[violin_locs[1]])

        vp0 = _set_vp_colors(vp0, c0=colors[0], ind=0)
        vp1 = _set_vp_colors(vp1, c1=colors[1], ind=0)

    # Scatterplots
    if twin:
        # Rescale y-axis of first plot to connect from second
        h1 = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        p1 = (dist1 - ax1.get_ylim()[0]) / h1

        min2, max1 = ax0.get_ylim()
        dist1_rescale = ((max1-min2) * p1) + min2
    else:
        dist1_rescale = dist1
        ax1 = ax0

    if isinstance(scatter_jit, (tuple, list)):
        scatter_jit0, scatter_jit1 = scatter_jit
    else:
        scatter_jit0, scatter_jit1 = scatter_jit, scatter_jit

    xs_a = _weight_scatter_points(dist0, violin_locs[0], scatter_jit0)
    xs_a = np.abs(xs_a - violin_locs[0]) + violin_locs[0]

    xs_b = _weight_scatter_points(dist1, violin_locs[1], scatter_jit1)
    xs_b = np.abs(xs_b - violin_locs[1]) + violin_locs[1]
    xs_b = -(xs_b - violin_locs[1]) + violin_locs[1]

    if paired:

        line_color = 'C0' if line_color is None else line_color

        for i, (d0, d1) in enumerate(zip(dist0, dist1_rescale)):
            ax0.plot([xs_a[i], xs_b[i]], [d0, d1], color=line_color, alpha=line_alpha)

    ax0.scatter(xs_a, dist0, alpha=scatter_alpha, color=colors[0], s=10)
    ax1.scatter(xs_b, dist1, alpha=scatter_alpha, color=colors[1], s=10)

    # Axis settings
    ax0.set_title(title)
    ax0.set_ylabel(ylabel)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticklabels, fontsize=24)
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)

    if twin:
        return ax0, ax1

    return ax0


def _weight_scatter_points(dist, loc, std, bins=500):
    """Weight scatter points by density."""

    weights, edges = np.histogram(dist, bins=bins)
    weights = weights / weights.max()

    bins = np.array([edges[:-1], edges[1:]]).T
    bins[-1][1] += 1

    inds = np.zeros(len(dist), dtype=int)
    for i, r in enumerate(dist):
        inds[i] = np.where((r >= bins[:, 0]) & (r < bins[:, 1]))[0][0]

    weights = weights[inds]

    xs = np.zeros_like(dist)

    for i in range(len(dist)):
        xs[i] = np.random.normal(loc, weights[i]*std)

    return xs


def _set_vp_colors(vp, c0=None, c1=None, ind=None):
    """Update violin plot colors."""

    if ind is None:
        i0, i1 = 0, 1
    else:
        i0, i1 = ind, ind

    if c0 is not None and c1 is not None:
        colors = [c0, c1]
    elif c0 is not None:
        colors = c0
    else:
        colors = c1

    if c0 is not None:

        vp['cmins'].set_color(colors)
        vp['cmaxes'].set_color(colors)
        vp['cbars'].set_color(colors)

        vp['bodies'][i0].set_color(c0)
        vp['bodies'][i0].set_facecolor(c0)
        vp['bodies'][i0].set_edgecolor(c0)

        b = vp['bodies'][i0]
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf,
                                                  np.mean(b.get_paths()[0].vertices[:, 0]))

    if c1 is not None:

        vp['cmins'].set_color(colors)
        vp['cmaxes'].set_color(colors)
        vp['cbars'].set_color(colors)

        vp['bodies'][i1].set_color(c1)
        vp['bodies'][i1].set_facecolor(c1)
        vp['bodies'][i1].set_edgecolor(c1)

        b = vp['bodies'][i1]
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0],
                                                  np.mean(b.get_paths()[0].vertices[:, 0]),
                                                  np.inf)
    return vp

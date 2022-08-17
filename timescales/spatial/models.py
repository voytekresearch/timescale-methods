"""Spatial models."""

import numpy as np
import matplotlib.pyplot as plt



class Ring:
    """Ring (1-D) network.

    Attributes
    ----------
    radius : int
        Radius defining the inputs to each active unit.
        (e.g. how many neighboring nodes to the left / right to consider as inputs)
    N : int
        Number of active units.
    L : float
        Perimeter of ring.
    units : 1d array
        Positions of units, as linear indices. Has shape (N,).
    neighbors : 2d array
        Connections, as indices, for for each active unit's neighbors.
    weights : 1d array
        Connection strengths.
    """

    def __init__(self, radius: int):

        self.radius = radius

        self.N = (2 * radius) + 1

        # Perimeter
        self.L = 2 * np.pi * self.radius

        # Shift indices to centered around zero
        self.shift = (self.N-1) // 2

        # Positions of units
        self.units = np.arange(self.N) - self.shift

        self.neighbors = np.array([[j for j in range(i-self.radius, i+self.radius+1)]
                                   for i in range(self.N)])

        self.weights = 1 / np.arange(1, self.N//2+1)

        ## Spatial position
        self.a = self.L / self.N
        self.x = np.array([self.a*i for i in range(self.N)])


    def plot(self, active_unit: int, theta: float, ax=None):
        """Plot the network configuration.

        Parameters
        ----------
        active_unit : int
            Unit to consider active. Will be plotted in orange.
        theta : float
            Distance, in degrees, to plot between each pair of units.
        ax : matplotlib.axes.Axes, optional, default: None
            Axis to plot on.
        """
        # Create plot
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

        # Plot concentric rings
        _x = np.linspace(0, 2*np.pi, 100)
        ax.plot(_x, np.ones(100) * 90, color='k')
        ax.plot(_x, np.ones(100) * 100, color='k')

        # Units in range
        rad = np.radians(theta)

        for i in self.units * rad:
            ax.plot([i, i], [90, 100], color='k')
            ax.plot(i, 100, marker='o', ls='', color='dimgray')

        # Units outside range
        for i in np.arange(self.units[0]-self.radius, self.units[0]) * rad:
            ax.plot([i, i], [90, 100], color='k')
            ax.plot(i, 100, marker='o', ls='', color='lightgray')

        for i in np.arange(self.units[-1]+1, self.units[-1]+self.radius+1) * rad:
            ax.plot([i, i], [90, 100], color='k')
            ax.plot(i, 100, marker='o', ls='', color='lightgray')

        # Active unit's neighbors
        for i in np.arange(active_unit-self.radius, active_unit+self.radius+1) * rad:
            ax.plot(i, 100, marker='o', ls='', color='C0', alpha=.5)

        # Active unit
        ax.plot(active_unit * rad, 100, marker='o', ls='', color='C1')

        # Settings
        ax.grid(False)
        ax.set_ylim(0, 110)
        ax.axis('off')


class Lattice:
    """Lattice (2-D) network.

    Attributes
    ----------
    radius : int
        Radius defining the inputs to each active unit in each direction of the lattice.
    N : int
        Number of active units. Determined from radius.
    L : float
        Side length of lattice (square).
    units : 2d array
        Positions of units, as (x, y) indices. Has shape (N**2, 2).
    neighbors : 2d array
        Connections, as indices, for for each active unit's neighbors.
    weights : 1d array
        Connection strengths.
    """

    def __init__(self, radius: int):

        self.radius = radius

        # Number of nodes
        self.N = (2*radius + 1) ** 2

        # Side length
        self.L = (2*self.radius) + 1

        # Shift indices if centered around zero
        self.shift = (self.L-1) // 2

        # Positions of units
        self.units = np.array([[i-self.shift, j-self.shift] for i in range(self.L)
                               for j in range(self.L)])

        # Position of neighbors
        self.neighbors = np.zeros((self.L, self.L, (self.L**2)-1, 2))

        for ui in self.units[:, 0]:
            for uj in self.units[:, 1]:
                i = 0
                for _i in range(ui-self.radius, ui+self.radius+1):
                    for _j in range(uj-self.radius, uj+self.radius+1):
                        if (ui, uj) != (_i, _j):
                            self.neighbors[ui, uj, i] = [_i, _j]
                            i += 1

        self.weights = np.array([(8/(((2*r + 1)**2)-1))
                                  for r in range(1, self.radius+1)])

        # Lattice constant
        self.a = self.L / self.N

        # Spatial position
        self.x = np.array([[self.a*i, self.a*j] for i in range(self.N)
                           for j in range(self.N)])


    def plot(self, active_unit: tuple, ax=None):
        """Plot the network configuration.

        Parameters
        ----------
        active_unit : tuple
            Unit to consider active. Will be plotted in orange.
        ax : matplotlib.axes.Axes, optional, default: None
            Axis to plot on.
        """
        if active_unit[0] >= self.L or active_unit[1] >= self.L:
            raise ValueError('active_unit out of range.')

        min_coord = self.units[0][0] - self.radius

        max_coord = self.units[-1][-1] + self.radius

        pos = np.array([[i, j] for i in range(min_coord, max_coord+1)
                        for j in range(min_coord, max_coord+1)])

        # Create plot
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        # Grid lines
        for i in range(min_coord, max_coord+1):
            ax.axvline(i, color='k', zorder=-8, alpha=.1)
            ax.axhline(i, color='k', zorder=-8, alpha=.1)

        # All possible units
        for (i, j) in pos:
            ax.plot(i, j, marker='o', ls='', color='lightgray')

        # All units in range
        for (i, j) in self.units:
            ax.plot(i, j, marker='o', ls='', color='dimgray')

        # Neighbors to active unit
        for (i, j) in self.neighbors[active_unit[0], active_unit[1]]:
            ax.plot(i, j, marker='o', ls='', color='C0', alpha=.5)

        # Active unit
        ax.plot(*active_unit, marker='o', ls='', color='C1', alpha=1)

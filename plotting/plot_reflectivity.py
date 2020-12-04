# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: plot_reflectivity.py
#  Purpose: Plotting reflectivity as a function of incidence angle
#   Author: Erik Marten Blixt
#   Email: marten.blixt@gmail.com
#
# --------------------------------------------------------------------
"""
"""
import sys
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import logging

import rp.rp_core as rp
from utils.templates import handle_template

log = logging.getLogger(__name__)

cnames = list(np.roll([str(u) for u in colors.cnames.keys()], 10))

msymbols = np.array(['o','s','v','^','<','>','p','*','h','H','+','x','D','d','|','_','.','1','2','3','4','8'])


def main():
    # Set up a test plot
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

    ref_functions = [rp.reflectivity(2800, 3000, 1350, 1500, 2.5, 2.4),
                     rp.reflectivity(2900, 2800, 1400, 1300, 2.5, 2.4)]

    for i, ax in enumerate([ax1,ax2]):
        xdata = np.linspace(0, 40, 50)
        ydata = ref_functions[i](xdata)
        yerror = np.full(ydata.shape, 0.002)

        plot(
                    xdata,
                    ydata,
                    yerror=yerror,
                    c=cnames[i],
                    fig=fig,
                    ax=ax,
                    xtempl={'full_name': 'X label',
                            'unit': '-',
                            'min': 0.,
                            'max': 1.},
                    ytempl={'full_name': 'Y label',
                            'unit': '-',
                            'min': 0.,
                            'max': 1.}
                )

    # Handle the legends
    legends = []
    this_legend = ax1.legend(
        legends,
        prop=FontProperties(size='smaller'),
        scatterpoints = 1,
        markerscale=0.5,
        loc=1
    )
    plt.show()

def plot(
        xdata,
        ydata,
        c='b',      # color the data points
        xtempl=None,
        ytempl=None,
        yerror=None,
        yerr_style='fill',  # or None
        mask=None,
        fig=None,
        ax=None,
        show_masked=False,
        **kwargs
):
    """

    :param xdata:
    :param ydata:
    :param title:
    :param xtempl:
        dict
        dictionary with the following keys:
            All are optional
            'bounds': list of limits for a discrete colorscale, Optional
            'center': float, midpoint value of a centered colorscale, Optional
            'colormap': str, name of the colormap (Default 'jet') OR
                list of strings with color names if scale == 'discrete'. e.g.:
                ['b', 'saddlebrown', 'greenyellow', 'mediumpurple', 'red', 'lightgreen', 'yellow']
            'description': str, data description. Not used
            'full_name': str, name of variable. Used in the label
            'id': str, short name of the variable
            'max': max value to be used in the plot
            'min': min value to be used in the plot
            'scale': str, 'lin', 'log', or 'discrete'
            'type':  str, datatype 'str', 'float', 'int'
            'unit': str,
    :param ytempl:
        same as xtempl, but for the y axis
    :param yerror:
        numpy array of same length as ydata
    :param mask:
        boolean numpy array of same length as xdata
    :param fig:
    :return:
    """

    if 'l_fonts' in kwargs:
        l_fonts = kwargs.pop('l_fonts')
    else:
        l_fonts=16
    if 't_fonts' in kwargs:
        t_fonts = kwargs.pop('t_fonts')
    else:
        t_fonts=13

    grid = kwargs.get('grid', True)

    # handle templates
    xlabel, xlim, xcmap, xcnt, xbnds, xscale = handle_template(xtempl)
    ylabel, ylim, ycmap, ycnt, ybnds, yscale = handle_template(ytempl)

    # Handle mask
    if mask is None:
        mask = np.ones((len(xdata)), dtype=bool)
    else:
        if len(mask) != len(xdata):
            raise OSError('Length of mask must match input data')

    # set up plotting environment
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = fig.subplots()

    #
    # start plotting
    #

    # handle
    # errorbars
    if yerror is not None:
        if yscale == 'log':
            raise NotImplementedError('Error bars on logarithmic scale needs testing before relase')
        if yerr_style == 'fill':
            ax.fill_between(
                xdata[mask],
                ydata[mask]-yerror[mask],
                ydata[mask]+yerror[mask],
                facecolor=c,
                alpha=0.2
            )
        else:
            ax.errorbar(
                xdata[mask],
                ydata[mask],
                yerr=yerror[mask],
                capthick=1,
                elinewidth=0.5,
                ecolor=c,
                linestyle='none',
                zorder=-100.
            )

    #  plot masked data in gray
    if show_masked:
        im = ax.plot(
            xdata[~mask],
            ydata[~mask],
            c='0.4',
            s=npdata,
            edgecolors='none',
            alpha=0.5
        )
    # Then plot the remaining points
    im = ax.plot(
        xdata[mask],
        ydata[mask],
        c=c,
        lw=2,
        ls='-',
        picker=5,
        **kwargs
    )

    ax.grid(grid)

    #if not(None in xlim):
    #    ax.set_xlim(*xlim)
    #if not(None in ylim):
    #   ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel, fontsize=l_fonts)
    ax.set_ylabel(ylabel, fontsize=l_fonts)

    ax.tick_params(axis='both', labelsize=t_fonts)

    if xscale == 'log':
        ax.set_xscale('log')
    if yscale == 'log':
        ax.set_yscale('log')

if __name__ == '__main__':
    main()

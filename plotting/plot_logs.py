import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from copy import deepcopy
import logging
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from plotting import crossplot as xp
import core.well as cw

logger = logging.getLogger(__name__)

def plot_logs(well, logname_dict, wis, wi_name, templates, block_name=None, savefig=None, **kwargs):
    """
    Attempts to draw a plot similar to the "CPI plots", for one working interval with some buffer.
    :param well:
    :param logname_dict:
        dict
        Dictionary of log type: log name key: value pairs to plot
        The Vp, Vs, Rho and Phi logs are necessary for output to RokDoc compatible Sums & Average excel file
        E.G.
            logname_dict = {
               'P velocity': 'vp',
               'S velocity': 'vs',
               'Density': 'rhob',
               'Porosity': 'phie',
               'Volume': 'vcl'}
    :return:
    """
    if block_name is None:
        block_name = cw.def_lb_name

    fig = plt.figure(figsize=(20, 10))
    n_cols = 22  # subdivide plot in this number of equally wide columns
    l = 0.05; w = (1-l)/float(n_cols+1); b = 0.05; h = 0.8
    #l = 0.05; w = 0; b = 0.1; h = 0.8
    rel_pos = [1, 4, 5, 6, 9, 12, 15, 18]  # Column number (starting with one) of subplot
    rel_widths = [_x - _y for _x, _y in zip(np.roll(rel_pos + [n_cols], -1)[:-1], rel_pos)]
    ax_names = ['gr_ax', 'md_ax', 'tvd_ax', 'res_ax', 'rho_ax', 'cpi_ax', 'ai_ax', 'synt_ax']
    header_axes = {}
    for i in range(len(ax_names)):
        header_axes[ax_names[i]] = plt.subplot(2, n_cols, rel_pos[i],
                                               position=[l+(rel_pos[i]-1)*w, h+0.05, rel_widths[i]*w, 1-h-0.1],
                                               figure=fig)
    axes = {}
    for i in range(len(ax_names)):
        axes[ax_names[i]] = plt.subplot(2, n_cols, n_cols + rel_pos[i],
                                        position=[l+(rel_pos[i]-1)*w, b, rel_widths[i]*w, h], figure=fig)

    #fig.tight_layout()
    plt.show()


def axis_plot(ax, y, data, limits, styles, yticks=True, nxt=4, **kwargs):
    """
    Plot data in one subplot
    :param ax:
        matplotlib axes object
    :param y:
        numpy ndarray
        The depth data of length N
    :param data:
        list
        list of ndarrays, each of length N, which should be plotted in the same subplot
    :param limits:
        list
        list of lists, each with min, max value of respective curve, e.g.
        [[0, 150], [6, 16], ...]
    :param styles:
        list
        list of dictionaries that defines the plotting styles of the data
        E.G. [{'lw':1, 'color':'k', 'ls':'-'}, {'lw':2, 'color':'r', 'ls':'-'}, ... ]
    :param yticks:
        bool
        if False the yticklabels are not shown
    :param nxt:
        int
        Number of gridlines in x direction
    :param kwargs:
    :return:
    """
    if not (len(data) == len(limits) == len(styles)):
        raise IOError('Must be same number of items in data, limits and styles')

    # set up multiple twin axes to allow for different scaling of each plot
    axes = []
    for i in range(len(data)):
        if i == 0:
            axes.append(ax)
        else:
            axes.append(axes[-1].twiny())

    # start plotting
    for i in range(len(data)):
        axes[i].plot(data[i], y, **styles[i])

    # set up the x range differently for each plot
    for i in range(len(data)):
        axes[i].set_xlim(*limits[i])
        # Change major ticks to set up the grid as desired
        if nxt > 0:
            x_int = np.abs(limits[i][1] - limits[i][0]) / (nxt + 1)
            axes[i].xaxis.set_major_locator(MultipleLocator(x_int))
            axes[i].get_xaxis().set_ticklabels([])
        else:
            axes[i].get_xaxis().set_ticks([])
        if i == 0:
            axes[i].set_ylim(ax.get_ylim()[::-1])
            if not yticks:
                axes[i].get_yaxis().set_ticklabels([])
        else:
            axes[i].tick_params(axis='x', length=0.)

    ax.grid(which='major', alpha=0.5)


def axis_log_plot(ax, y, data, limits, styles, yticks=True,  **kwargs):
    """
    Plot data in one subplot
    Similar to axis_plot, but uses the same (logarithmic) x axis for all data
    :param ax:
        matplotlib axes object
    :param y:
        numpy ndarray
        The depth data of length N
    :param data:
        list
        list of ndarrays, each of length N, which should be plotted in the same subplot
    :param limits:
        list
        min, max value of axis
        E.G. [0.2, 200]
    :param styles:
        list
        list of dictionaries that defines the plotting styles of the data
        E.G. [{'lw':1, 'color':'k', 'ls':'-'}, {'lw':2, 'color':'r', 'ls':'-'}, ... ]
    :param yticks:
        bool
        if False the yticklabels are not shown
    :param nxt:
        int
        Number of gridlines in x direction
    :param kwargs:
    :return:
    """
    if not (len(data) == len(styles)):
        raise IOError('Must be same number of items in data and styles')

    # start plotting
    for i in range(len(data)):
        ax.plot(data[i], y, **styles[i])

    ax.set_xscale('log')
    ax.set_xlim(*limits)
    ax.get_xaxis().set_ticklabels([])
    ax.tick_params(axis='x', which='both', length=0)

    ax.set_ylim(ax.get_ylim()[::-1])
    if not yticks:
        ax.get_yaxis().set_ticklabels([])
        ax.tick_params(axis='y', length=0)

    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)


def annotate_plot(ax, y, pad=-30):
    ax.plot(np.ones(len(y)), y, lw=0)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.get_xaxis().set_ticks([])
    ax.tick_params(axis='y', direction='in', length=0., labelsize='smaller')
    yticks = [*ax.yaxis.get_major_ticks()]
    for tick in yticks:
        tick.set_pad(pad)


def header_plot(ax, limits, legends, styles):
    """
    Tries to create a "header" to a plot, similar to what is used in RokDoc and many CPI plots
    :param ax:
        matplotlib axes object
    :param limits:
        list
        list of lists, each with min, max value of respective curve, e.g.
        [[0, 150], [6, 16], ...]
        Should not be more than 4 items in this list
    :param legends:
        list
        list of strings, that should annotate the respective limits
    :param styles:
        list
        list of dicts which describes the line styles
        E.G. [{'lw':1, 'color':'k', 'ls':'-'}, {'lw':2, 'color':'r', 'ls':'-'}, ... ]
    :return:
    """

    if not (len(limits) == len(legends) == len(styles)):
        raise IOError('Must be same number of items in limits, legends and styles')

    # Sub divide plot in this number of horizontal parts
    n = 8
    for i in range(len(limits)):
        ax.plot([1, 2],  [n-1-2*i, n-1-2*i], **styles[i])
        ax.text(1-0.03, n-1-2*i, str(limits[i][0]), ha='right', va='center', fontsize='smaller')
        ax.text(2+0.03, n-1-2*i, str(limits[i][1]), ha='left', va='center', fontsize='smaller')
        ax.text(1.5, n-1-2*i+0.05, legends[i], ha='center', fontsize='smaller')

    ax.set_xlim(0.8, 2.2)
    ax.get_xaxis().set_ticks([])
    ax.set_ylim(0.5, 8)
    ax.get_yaxis().set_ticks([])


def wiggle_plot(ax, y, wiggle, zero_at=0., scaling=1., fill='pos',
                fill_style=None, zero_style=None, **kwargs):
    """
    Draws a (seismic) wiggle plot centered at 'zero_at'
    :param ax:
        matplotlib Axis object
    :param y:
        numpy ndarray
        Depth variable
    :param wiggle:
        numpy ndarray
        seismic trace
    :param zero_at:
        float
        x value at which the wiggle should be centered
    :param scaling:
        float
        scale the seismic data
    :param fill:
        str
        'neg': Fills the negative side of the wiggle
        'pos': Fills the positive side of the wiggle
        None : No fill
    :param zero_style:
        dict
        style keywords of the line marking zero
    :param kwargs:

    :return:
    """
    lw = kwargs.pop('lw', 0.5)
    c = kwargs.pop('c', 'k')

    if fill_style is None:
        fill_style = {'color': 'k', 'alpha': 1}
    if zero_style is None:
        zero_style = {'lw': 0.5, 'color': 'k', 'alpha': 0.2}
    # shift and scale the data so that it is centered at 'zero_at'
    wig = zero_at + wiggle*scaling
    ax.plot(wig, y, lw=lw, color=c, **kwargs)
    if fill == 'pos':
        ax.fill_betweenx(y, wig, 0, wig > zero_at, **fill_style)
    elif fill == 'neg':
        ax.fill_betweenx(y, wig, 0, wig < zero_at, **fill_style)

    ax.axvline(zero_at, **zero_style)
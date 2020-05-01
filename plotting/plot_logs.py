import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from copy import deepcopy
import logging

from plotting import crossplot as xp
import core.well as cw

logger = logging.getLogger(__name__)

def plot_logs(well, log_name_dict, wis, wi_name, templates, block_name=None, savefig=None, **kwargs):
    """
    Attempts to draw a plot similar to the "CPI plots", for one working interval with some buffer.
    :param well:
    :return:
    """
    if block_name is None:
        block_name = cw.def_lb_name

    fig = plt.figure(figsize=(20, 10))
    l = 0.05; w = 1/22.; b = 0.1; h = 0.8
    rel_widths = [3, 1, 1]
    rel_pos = [1, 4, 5]
    ax_names = ['gr_ax', 'md_ax', 'tvd_ax']
    axes = {}
    for i in range(3):
        axes[ax_names[i]] = plt.subplot(1, 22, rel_pos[i], position=[l+(rel_pos[i]-1)*w, b, rel_widths[i]*w, h], figure=fig)

    #fig.tight_layout()


def axis_plot(ax, y, data, limits, styles, yticks=True, **kwargs):
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
    :param yticks:
        bool
        if False the yticks and ticklabels are not shown
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
        axes[i].get_xaxis().set_ticks([])
        if i == 0:
            axes[i].set_ylim(ax.get_ylim()[::-1])
            if not yticks:
                axes[i].get_yaxis().set_ticks([])


def axis_header(ax, limits, legends, styles):
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

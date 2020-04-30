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
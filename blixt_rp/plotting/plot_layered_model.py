import matplotlib.pyplot as plt
import numpy as np
import logging

# TODO remove this two lines, only used in testing
import sys
sys.path.append('C:\\Users\\marten\\PycharmProjects\\blixt_utils')

import blixt_rp.core.well as cw
import blixt_utils.utils as uu
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot
import blixt_rp.rp.rp_core as rp
from bruges.filters import ricker

logger = logging.getLogger(__name__)


def plot_model(ax, header_ax, vps, vss, rhos, boundaries, buffer=None):
    """
    Plot a 1D model in TWT with N layers
    :param ax:
        matplotlib axes object
    :param vps:
        list
        List of length N of Vp velocities in m/s
    :param vss:
        list
        List of length N of Vs velocities in m/s
    :param rhos:
        list
        List of length N of densities in g/cm3
    :param boundaries:
        list
        List of length N-1 of TWT values in ms who define the boundaries between the different layers
        From top to bottom (increasing TWT)
    :param buffer:
        float
        buffer in ms TWT that determines how much to show of the top and bottom layers

    return
        Acoustic impedance [m/s g/cm3] and the TWT data [ms] used to define the model
    """
    # check data
    if not len(boundaries) == len(vps) - 1:
        raise ValueError(vps)

    my_linestyle = {'lw': 1, 'color': 'k', 'ls': '-'}
    text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}

    if buffer is None:
        buffer = 200.
    bwb = [boundaries[0] - buffer] + boundaries +  [boundaries[-1] + buffer]  #  boundaries with buffer
    y = np.linspace(bwb[0], bwb[-1],  100)
    ais = [_vp * _rho for _vp, _rho in zip(vps, rhos)]  # Acoustic impedance in m/s g/cm3
    max_ai = max(ais)
    data = np.ones(100)  # fake data
    # normalize the data according to AI
    for i in range(len(bwb)):
        if i == 0:
            continue
        data[(y >= bwb[i-1]) & (y <= bwb[i])] = ais[i-1]/max_ai

    axis_plot(ax, y, [data], [[0., 1.1]], [my_linestyle], nxt=0)
    for i, _y in enumerate(boundaries):
        ax.plot([0., ais[i+1]/max_ai], [_y, _y], **my_linestyle)

    i = 0
    for _vp, _vs, _rho in zip(vps, vss, rhos):
        info_txt = '{}.\nVp: {:.1f}\nVs: {:.1f}\nRho: {:.1f}'.format(i+1, _vp/1000., _vs/1000., _rho)
        ax.text(0.5*ais[i]/max_ai, 0.5*(bwb[i] + bwb[i+1]), info_txt, ha='center', va='center', **text_style)
        i += 1

    if header_ax is not None:
        header_plot(header_ax, None, None, None, title='Relative AI' )

    return ais, y

if __name__ == '__main__':

    vps = [3500., 3600., 3500.]  # Vp [m/s] in the different layers
    vss = [1800., 2100., 1800.]
    rhos = [2.6, 2.3, 2.6]
    boundaries = [4000., 4060.]
    time_step = 0.001  # s
    center_f = 12.
    duration = 0.128
    scaling = 50.0
    wiggle_fill_style = 'opposite'
    my_linestyle = {'lw': 1, 'color': 'k', 'ls': '-'}

    fig = plt.figure(figsize=(5, 10))
    fig.suptitle('Simple model test')
    n_cols = 10  # subdivide plot in this number of equally wide columns
    l = 0.05; w = (1-l)/float(n_cols+1); b = 0.05; h = 0.8
    rel_pos = [1, 4, 5]  # Column number (starting with one) of subplot
    rel_widths = [_x - _y for _x, _y in zip(np.roll(rel_pos + [n_cols], -1)[:-1], rel_pos)]
    ax_names = ['model_ax', 'twt_ax', 'synt_ax']
    header_axes = {}
    for i in range(len(ax_names)):
        header_axes[ax_names[i]] = fig.add_subplot(2, n_cols, rel_pos[i],
                                                   position=[l+(rel_pos[i]-1)*w, h+0.05, rel_widths[i]*w, 1-h-0.1])
    axes = {}
    for i in range(len(ax_names)):
        axes[ax_names[i]] = fig.add_subplot(2, n_cols, n_cols + rel_pos[i],
                                            position=[l+(rel_pos[i]-1)*w, b, rel_widths[i]*w, h])

    ais, twt = plot_model(axes['model_ax'], header_axes['model_ax'], vps, vss, rhos, boundaries)

    annotate_plot(axes['twt_ax'], twt)
    header_plot(header_axes['twt_ax'], None, None, None, title='TWT\n[ms]')

    # Start calculating wiggles
    # time
    t = np.arange(twt[0], twt[-1], time_step * 1000.) # NB, time in ms
    # arrays of Vp, Vs and Rho
    vp_arr = np.zeros(len(t))
    vs_arr= np.zeros(len(t))
    rho_arr= np.zeros(len(t))
    i = 0
    for _vp, _vs, _rho in zip(vps, vss, rhos):
        if i == 0:
            vp_arr[t <= boundaries[i]] = _vp
            vs_arr[t <= boundaries[i]] = _vs
            rho_arr[t <= boundaries[i]] = _rho
        elif i < len(vps) - 1:
            vp_arr[(t > boundaries[i - 1]) & (t <= boundaries[i])] = _vp
            vs_arr[(t > boundaries[i - 1]) & (t <= boundaries[i])] = _vs
            rho_arr[(t > boundaries[i - 1]) & (t <= boundaries[i])] = _rho
        else:
            vp_arr[t > boundaries[i-1]] = _vp
            vs_arr[t > boundaries[i-1]] = _vs
            rho_arr[t > boundaries[i-1]] = _rho
        i += 1
    reff = rp.reflectivity(vp_arr, None, vs_arr, None, rho_arr, None, along_wiggle=True)
    w = ricker(duration, time_step, center_f)

    header_plot(header_axes['synt_ax'], None, None, None,
                title='Incidence angle\nRicker f={:.0f} Hz, l={:.3f} s'.format(center_f, duration))
    for inc_a in range(0, 35, 5):
        wig = np.convolve(w, np.nan_to_num(reff(inc_a)), mode='same')
        wig = np.append(wig, np.ones(1)*wig[-1])  # extend with one item
        wiggle_plot(axes['synt_ax'], t, wig, inc_a, scaling=scaling,
                    fill_pos_style='pos-blue', fill_neg_style='neg-red')

    axes["synt_ax"].get_yaxis().set_ticklabels([])
    axes["synt_ax"].axhline(y=boundaries[0], color='k', ls='--')
    axes["synt_ax"].axhline(y=boundaries[1], color='k', ls='--')

    plt.show()

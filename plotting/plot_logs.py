import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import core.well as cw
import utils.io as uio
import utils.utils as uu
import rp.rp_core as rp
import tmp.tmp_avo_model as tta

logger = logging.getLogger(__name__)


def find_nearest(data, value):
    return np.nanargmin(np.abs(data - value))

def plot_logs(well, well_table, wis, wi_name, templates, buffer=None, block_name=None, savefig=None, **kwargs):
    """
    Attempts to draw a plot similar to the "CPI plots", for one working interval with some buffer.
    :param well:
    :param well_table:
        dict
        The resulting well_table from utils.io.project_wells(project_table_file)
    :param buffer:
        float
        distance in meters
        Log is plotted from top of working interval - buffer to base of working interval + buffer
    :return:
    """
    if buffer is None:
        buffer = 50.
    if block_name is None:
        block_name = cw.def_lb_name

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('{} interval in well {}'.format(wi_name, well.well))
    n_cols = 22  # subdivide plot in this number of equally wide columns
    l = 0.05; w = (1-l)/float(n_cols+1); b = 0.05; h = 0.8
    #l = 0.05; w = 0; b = 0.1; h = 0.8
    rel_pos = [1, 4, 5, 6, 9, 12, 15, 18]  # Column number (starting with one) of subplot
    rel_widths = [_x - _y for _x, _y in zip(np.roll(rel_pos + [n_cols], -1)[:-1], rel_pos)]
    ax_names = ['gr_ax', 'md_ax', 'twt_ax', 'res_ax', 'rho_ax', 'cpi_ax', 'ai_ax', 'synt_ax']
    header_axes = {}
    for i in range(len(ax_names)):
        header_axes[ax_names[i]] = fig.add_subplot(2, n_cols, rel_pos[i],
                                               position=[l+(rel_pos[i]-1)*w, h+0.05, rel_widths[i]*w, 1-h-0.1]) #,
                                               #figure=fig)
    axes = {}
    for i in range(len(ax_names)):
        axes[ax_names[i]] = fig.add_subplot(2, n_cols, n_cols + rel_pos[i],
                                            position=[l+(rel_pos[i]-1)*w, b, rel_widths[i]*w, h]) #,
                                            #figure=fig)

    #
    # Start plotting data
    #
    tb = well.block[block_name]  # this log block
    lognames = uio.invert_well_table(well_table, well_name=well.well)
    # TODO
    # We should probably apply the "rename" functionality on lognames!
    #well.calc_mask({}, 'wi_mask', wis=wis, wi_name=wi_name)
    depth = tb.logs['depth'].data
    mask = np.ma.masked_inside(depth, wis[well.well][wi_name][0]-buffer, wis[well.well][wi_name][1]+buffer).mask

    #
    # Gamma ray and Caliper
    try_these_log_types = ['Gamma ray', 'Caliper', 'Inclination']
    log_types = [x for x in try_these_log_types if x in list(lognames.keys())]
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x][0], templates[x]['unit']) for x in log_types]

    xlims = axis_plot(axes['gr_ax'], depth[mask],
              [tb.logs[lognames[xx][0]].data[mask] for xx in log_types],
              limits, styles)
    header_plot(header_axes['gr_ax'], xlims, legends, styles)

    #for ax in [axes[x] for x in ax_names if x not in ['twt_ax', 'synt_ax']]:
    for ax in [axes[x] for x in ax_names if x not in ['twt_ax']]:
        ax.axhline(y=wis[well.well][wi_name][0], color='k', ls='--')
        ax.axhline(y=wis[well.well][wi_name][1], color='k', ls='--')

    #
    # MD
    header_plot(header_axes['md_ax'], None, None, None, title='MD [m]')
    annotate_plot(axes['md_ax'], depth[mask])

    #
    # TWT
    # Get the time-depth relation (time as a function of md)
    tdr = well.time_to_depth(lognames['Sonic'][0])
    header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]')
    annotate_plot(axes['twt_ax'], tdr[mask])

    tops_twt = [tdr[find_nearest(depth, y)] for y in wis[well.well][wi_name]]
    #print(tops_twt)
    #for ax in [axes[x] for x in ['twt_ax', 'synt_ax']]:
    for ax in [axes[x] for x in ['twt_ax']]:
        ax.axhline(y=tops_twt[0], color='k', ls='--')
        ax.axhline(y=tops_twt[1], color='k', ls='--')

    #
    # Resistivity
    log_types = ['Resistivity']
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    cls = ['r', 'b', 'k', 'g', 'c']  # should not plot more than 5 lines in this plot!
    lws = [2, 1, 1, 1, 1]
    lss = ['-', '--', ':', '.-', '-']
    styles = [{'lw': lws[i], 'color': cls[i], 'ls': lss[i]} for i in range(len(lognames['Resistivity']))]
    legends = ['{} [{}]'.format(x, templates['Resistivity']['unit']) for x in lognames['Resistivity']]

    xlims = axis_log_plot(axes['res_ax'], depth[mask], [tb.logs[x].data[mask] for x in lognames['Resistivity']],
                  limits, styles, yticks=False)
    header_plot(header_axes['res_ax'], xlims*len(legends), legends, styles)

    #
    # Rho
    try_these_log_types = ['Density', 'Neutron density']
    log_types = [x for x in try_these_log_types if x in list(lognames.keys())]
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x][0], templates[x]['unit']) for x in log_types]

    xlims = axis_plot(axes['rho_ax'], depth[mask],
              [tb.logs[lognames[xx][0]].data[mask] for xx in log_types],
              limits, styles, yticks=False)
    header_plot(header_axes['rho_ax'], xlims, legends, styles)

    #
    # CPI
    try_these_log_types = ['Saturation', 'Porosity', 'Volume']
    log_types = [x for x in try_these_log_types if x in list(lognames.keys())]
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x][0], templates[x]['unit']) for x in log_types]

    if len(log_types) == 0:
        header_plot(header_axes['cpi_ax'], None, None, None)
        axis_plot(axes['cpi_ax'], None, None, None, None)
    xlims = axis_plot(axes['cpi_ax'], depth[mask],
              [tb.logs[lognames[xx][0]].data[mask] for xx in log_types],
              limits, styles, yticks=False)
    header_plot(header_axes['cpi_ax'], xlims, legends, styles)

    #
    # AI
    ai = tb.logs[lognames['Density'][0]].data / tb.logs[lognames['Sonic'][0]].data
    styles = [{'lw': 1, 'color': 'k', 'ls': '--'}]
    xlims = axis_plot(axes['ai_ax'], depth[mask], [ai[mask]], [[None, None]], styles,
              yticks=False)
    header_plot(header_axes['ai_ax'], xlims, ['AI'], styles)

    #
    # Wiggles
    #t = np.arange(tdr[mask][0], tdr[mask][-1], 0.004)  # A uniformly sampled array of time steps, from A to B
    t = np.arange(0., 3., 0.0001)  # A uniformly sampled array of time steps, from 0 to 3
    #print(len(t))
    vp_t = np.interp(x=t, xp=tdr, fp=1./tb.logs[lognames['Sonic'][0]].data)
    #print(len(vp_t))
    vs_t = np.interp(x=t, xp=tdr, fp=1./tb.logs[lognames['Shear sonic'][0]].data)
    rho_t = np.interp(x=t, xp=tdr, fp=tb.logs[lognames['Density'][0]].data)
    reff = rp.reflectivity(vp_t, None, vs_t, None, rho_t, None, along_wiggle=True)
    #print(len(reff(10)))
    #tw, w = ricker(_f=25, _length=0.512, _dt=0.001)
    w = tta.ricker(0.512, 0.001, 25.)
    #print(len(w))

    # Compute the depth-time relation
    dtr = np.array([depth[find_nearest(tdr, tt)] for tt in t])
    #print(np.nanmin(dtr), np.nanmax(dtr))
    # Translate the mask to the time variable
    t_mask = np.ma.masked_inside(t[:-1], np.nanmin(tdr[mask]), np.nanmax(tdr[mask])).mask
    #wiggle_plot(axes['synt_ax'], t[:-1][t_mask], wig[t_mask], 10)

    header_plot(header_axes['synt_ax'], None, None, None, title='Incidence angle')
    for inc_a in range(0, 35, 5):
        wig = np.convolve(w, np.nan_to_num(reff(inc_a)), mode='same')
        wiggle_plot(axes['synt_ax'], dtr[:-1][t_mask], wig[t_mask], inc_a, scaling=30.)


    if savefig is not None:
        fig.savefig(savefig)
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
        list
        xlim, list of list, each with the limits (min, max) of the x axis
    """
    if data is None:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        return

    if not (len(data) == len(limits) == len(styles)):
        raise IOError('Must be same number of items in data, limits and styles')

    # store the x axis limits that has been used in each axis
    xlims = []

    # set up multiple twin axes to allow for different scaling of each plot
    axes = []
    for i in range(len(data)):
        if i == 0:
            axes.append(ax)
        else:
            axes.append(axes[-1].twiny())
    # Adjust the positions according to the original axes
    for i, ax in enumerate(axes):
        if i == 0:
            pos = ax.get_position()
        else:
            ax.set_position(pos)

    # start plotting
    for i in range(len(data)):
        axes[i].plot(data[i], y, **styles[i])

    # set up the x range differently for each plot
    for i in range(len(data)):
        #axes[i].set_xlim(*limits[i])
        #set_lim(axes[i], limits[i], 'x')
        print(limits[i])
        xlims.append(axes[i].get_xlim())
        # Change major ticks to set up the grid as desired
        if nxt > 0:
            xlim = axes[i].get_xlim()
            x_int = np.abs(xlim[1] - xlim[0]) / (nxt + 1)
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

    return xlims


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
        list
        xlims, list of list, each with the limits (min, max) of the x axis
    """
    if not (len(data) == len(styles)):
        raise IOError('Must be same number of items in data and styles')

    # store the x axis limits that has been used in each axis
    xlims = []

    # start plotting
    for i in range(len(data)):
        ax.plot(data[i], y, **styles[i])

    ax.set_xscale('log')
    #ax.set_xlim(*limits)
    set_lim(ax, limits, 'x')
    xlims.append(ax.get_xlim())
    ax.get_xaxis().set_ticklabels([])
    ax.tick_params(axis='x', which='both', length=0)

    ax.set_ylim(ax.get_ylim()[::-1])
    if not yticks:
        ax.get_yaxis().set_ticklabels([])
        ax.tick_params(axis='y', length=0)

    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)

    return xlims


def annotate_plot(ax, y, pad=-30):
    ax.plot(np.ones(len(y)), y, lw=0)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.get_xaxis().set_ticks([])
    ax.tick_params(axis='y', direction='in', length=5., labelsize='smaller', right=True)
    yticks = [*ax.yaxis.get_major_ticks()]
    for tick in yticks:
        tick.set_pad(pad)


def header_plot(ax, limits, legends, styles, title=None):
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
    if limits is None:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if title is not None:
            ax.text(0.5, 0.1, title, ha='center')
        return

    if not (len(limits) == len(legends) == len(styles)):
        raise IOError('Must be same number of items in limits, legends and styles')

    # Sub divide plot in this number of horizontal parts
    n = 8
    for i in range(len(limits)):
        ax.plot([1, 2],  [n-1-2*i, n-1-2*i], **styles[i])
        ax.text(1-0.03, n-1-2*i, '{:.1f}'.format(limits[i][0]), ha='right', va='center', fontsize='smaller')
        ax.text(2+0.03, n-1-2*i, '{:.1f}'.format(limits[i][1]), ha='left', va='center', fontsize='smaller')
        ax.text(1.5, n-1-2*i+0.05, legends[i], ha='center', fontsize='smaller')

    ax.set_xlim(0.8, 2.3)
    ax.get_xaxis().set_ticks([])
    ax.set_ylim(0.5, 8)
    ax.get_yaxis().set_ticks([])


def wiggle_plot(ax, y, wiggle, zero_at=0., scaling=1., fill_pos_style='default',
                fill_neg_style='default', zero_style=None, **kwargs):
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
    #print(len(y), len(wiggle))
    lw = kwargs.pop('lw', 0.5)
    c = kwargs.pop('c', 'k')

    if fill_pos_style == 'default':
        fill_pos_style = {'color': 'r', 'alpha': 0.2, 'lw': 0.}
    if fill_neg_style == 'default':
        fill_neg_style = {'color': 'b', 'alpha': 0.2, 'lw': 0.}
    if zero_style is None:
        zero_style = {'lw': 0.5, 'color': 'k', 'alpha': 0.2}
    # shift and scale the data so that it is centered at 'zero_at'
    wig = zero_at + wiggle*scaling
    ax.plot(wig, y, lw=lw, color=c, **kwargs)
    if fill_pos_style is not None:
        ax.fill_betweenx(y, wig, zero_at, wig > zero_at, **fill_pos_style)
    if fill_neg_style is not None:
        ax.fill_betweenx(y, zero_at, wig, wig < zero_at, **fill_neg_style)

    ax.axvline(zero_at, **zero_style)

    ax.set_ylim(ax.get_ylim()[::-1])


def overview_plot(wells, log_table, wis, wi_name, templates, log_types=None, block_name=None, savefig=None):
    """
    Overview plot designed to show data coverage in given working interval together with sea water depth
    :param wells:
    :param log_table:
        dict
        Dictionary of log type: log name key: value pairs which decides which log, under each log type, to plot
        E.G.
            log_table = {
               'P velocity': 'vp',
               'S velocity': 'vs',
               'Density': 'rhob',
               'Porosity': 'phie',
               'Volume': 'vcl'}
    :param wis:
    :param wi_name:
    :param templates:
    :param block_name:
    :param savefig
        str
        full pathname of file to save the plot to
    :return:
    """
    if block_name is None:
        block_name = cw.def_lb_name
    if log_types is None:
        log_types = ['Sonic', 'Shear sonic', 'Density']
    _savefig = False
    if savefig is not None:
        _savefig = True

    sea_style = {'lw': 10, 'color': 'b', 'alpha': 0.5}
    kb_style = {'lw': 10, 'color': 'k', 'alpha': 0.5}
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('{} interval'.format(wi_name), pad=10.)

    # create fake x axes and plotting ranges
    x = np.arange(len(wells))
    pw = 0.7 * len(wells) / (len(wells) + 1)  # approximate plotting width for each well

    # create common TVD depth axis for all wells
    y_max = -1E6

    wnames = []
    c_styles = {}  # style of line that defines the center of each well.
                   # Thin dashed if TVD present, thicker solid if not
    for i, well in enumerate(wells.values()):
        wnames.append(well.well)
        # extract the relevant log block
        tb = well.block[block_name]

        # Try finding the depth interval of the desired working interval 'wi_name'
        well.calc_mask({}, name='XXX', wis=wis, wi_name=wi_name)
        try:
            mask = tb.masks['XXX'].data
            int_exists = True
        except TypeError:
            print('{} not present in well {}. Continue'.format(wi_name, well.well))
            mask = None
            int_exists = False

        if int_exists:
            if 'tvd' in well.log_names():
                c_styles[well.well] = {'color': 'k', 'ls': '--', 'lw': 0.5}
                depth_key = 'tvd'
            else:
                c_styles[well.well] = {'color': 'k', 'ls': '-', 'lw': 1}
                depth_key = 'depth'
            if np.nanmax(tb.logs[depth_key].data[mask]) > y_max:
                y_max = np.nanmax(tb.logs[depth_key].data[mask])

            # plot the desired logs
            for ltype in log_types:
                lname = log_table[ltype]
                if lname not in well.log_names():  # skip this log
                    continue
                x = uu.norm(tb.logs[lname].data[mask], method='median')
                styles = {'lw': templates[ltype]['line width'],
                          'color': templates[ltype]['line color'], 'ls': templates[ltype]['line style']}
                ax.plot(i + x*pw, tb.logs[depth_key].data[mask], **styles)

        ax.plot([i, i], [0.,  # this is not exact, because the linewidth makes the lines look longer than what they are
                        templates[well.well]['water depth']+templates[well.well]['kb']], label='_nolegend_', **sea_style)
        ax.plot([i, i], [0., templates[well.well]['kb']], label='_nolegend_', **kb_style)
        ax.axvline(i, label='_nolegend_', **c_styles[well.well])

    ax.set_ylim(y_max, 0)
    ax.set_ylabel('TVD [m]')
    ax.get_xaxis().set_ticks(range(len(wnames)))
    ax.get_xaxis().set_ticklabels(wnames)
    ax.legend(log_types)
    fig.tight_layout()

    if _savefig:
        fig.savefig(savefig)

def set_lim(ax, limits, axis=None):
    """
    Convinience function to set the x axis limits. Interprets None as autoscale
    :param ax:
        matplotlib.axes
    :param limits:
        list
        list of  min, max value.
        If any is None, axis is autoscaled
    :param axis
        str
        'x' or 'y'
    :return:
    """
    if axis is None:
        return

    if None in limits:
        # first autoscale
        ax.autoscale(True, axis=axis)
        if axis == 'x':
            _lims = ax.get_xlim()
            ax.set_xlim(
                limits[0] if limits[0] is not None else _lims[0],
                limits[1] if limits[1] is not None else _lims[1]
            )
        else:
            _lims = ax.get_ylim()
            ax.set_ylim(
                limits[0] if limits[0] is not None else _lims[0],
                limits[1] if limits[1] is not None else _lims[1]
            )
    else:
        if axis == 'x':
            ax.set_xlim(*limits)
        else:
            ax.set_ylim(*limits)


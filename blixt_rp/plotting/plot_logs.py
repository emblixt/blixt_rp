import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import numpy as np
import logging
from copy import deepcopy

import blixt_rp.core.well as cw
import blixt_utils.utils as uu
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot
import blixt_rp.rp.rp_core as rp
from blixt_utils.plotting import crossplot as xp
from bruges.filters import ricker
import bruges.rockphysics.anisotropy as bra

logger = logging.getLogger(__name__)


def find_nearest(data, value):
    return np.nanargmin(np.abs(data - value))


def plot_logs(well, log_table, wis, wi_name, templates, buffer=None, block_name=None, savefig=None, **kwargs):
    """
    Attempts to draw a plot similar to the "CPI plots", for one working interval with some buffer.
    :param well:
        well object
    :param log_table:
        dict
        Dictionary of log type: log name key: value pairs which decides which log to use when selecting which
        velocity / sonic and density logs. Other logs (e.g. 'Resistivity') are selected based on their presence
        E.G.
            log_table = {
               'P velocity': 'vp',
               'S velocity': 'vs',
               'Density': 'rhob'}
    :param wis:
        dict
        Dictionary of working intervals, keys are working interval names, and the values are a two
        items list with top & bottom
    :param wi_name:
        str
        Name of working interval to use in this plot
    :param templates:
        dict
        Dictionary of different templates as returned from Project().load_all_templates()
    :param buffer:
        float
        distance in meters
        Log is plotted from top of working interval - buffer to base of working interval + buffer
        Default is 50 m
    :param savefig:
        str
        Full path name to file (.png or .pdf) to which the plot is exported
        if None, the plot is displayed instead
    :kwargs
        keyword arguments
        backus_length: The Backus averaging length in m
        suffix: string. Added to the default title
    :return:
    """
    log_table = small_log_table(log_table)
    if buffer is None:
        buffer = 50.
    if block_name is None:
        block_name = cw.def_lb_name

    time_step = kwargs.pop('time_step', 0.001)
    c_f = kwargs.pop('center_frequency', 30.)
    duration = kwargs.pop('duration', 0.512)
    scaling = kwargs.pop('scaling', 30.0)
    suffix = kwargs.pop('suffix', '')
    wiggle_fill_style = kwargs.pop('wiggle_fill_style', 'default')
    backus_length = kwargs.pop('backus_length', 5.0)

    if wiggle_fill_style == 'opposite':
        fill_pos_style = 'pos-blue'
        fill_neg_style = 'neg-red'
    else:
        fill_pos_style = 'default'
        fill_neg_style = 'default'

    # Set up plot window
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('{} interval in well {} {}'.format(wi_name, well.well, suffix))
    ax_names = ['gr_ax', 'md_ax', 'twt_ax', 'res_ax', 'rho_ax', 'cpi_ax', 'ai_ax', 'synt_ax']
    n_cols = 8
    n_rows = 2
    width_ratios = [1, 0.3, 0.3, 1, 1, 1, 1, 2]
    height_ratios = [1, 5]
    spec = fig.add_gridspec(nrows=n_rows, ncols=n_cols,
                            height_ratios=height_ratios, width_ratios=width_ratios,
                            hspace=0., wspace=0.,
                            left=0.05, bottom=0.03, right=0.98, top=0.96)
    header_axes = {}
    axes = {}
    for i in range(len(ax_names)):
        header_axes[ax_names[i]] = fig.add_subplot(spec[0, i])
        axes[ax_names[i]] = fig.add_subplot(spec[1, i])

    #
    # Start plotting data
    #
    tb = well.block[block_name]  # this log block
    depth = tb.logs['depth'].data
    mask = np.ma.masked_inside(depth, wis[well.well][wi_name][0]-buffer, wis[well.well][wi_name][1]+buffer).mask
    md_min = np.min(depth[mask])
    md_max = np.max(depth[mask])

    if 'twt' in tb.log_names():
        twt = tb.logs['twt'].data
        twt_min = np.min(twt[mask])
        twt_max = np.max(twt[mask])
        print('Using real twt data')
    else:
        twt = None
        twt_min = None
        twt_max = None

    #
    # Gamma ray and Caliper
    try_these_log_types = ['Gamma ray', 'Caliper', 'Bit size', 'Inclination']
    log_types = [x for x in try_these_log_types if (len(well.get_logs_of_type(x)) > 0)]
    lognames = {ltype: well.get_logs_of_type(ltype)[0].name for ltype in log_types}
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]

    xlims = axis_plot(axes['gr_ax'], depth[mask],
              [tb.logs[lognames[xx]].data[mask] for xx in log_types],
              limits, styles, ylim=[md_min, md_max])
    header_plot(header_axes['gr_ax'], xlims, legends, styles)

    #for ax in [axes[x] for x in ax_names if x not in ['twt_ax', 'synt_ax']]:
    for ax in [axes[x] for x in ax_names if x not in ['twt_ax']]:
        ax.axhline(y=wis[well.well][wi_name][0], color='k', ls='--')
        ax.axhline(y=wis[well.well][wi_name][1], color='k', ls='--')

    #
    # MD
    header_plot(header_axes['md_ax'], None, None, None, title='MD [m]')
    annotate_plot(axes['md_ax'], depth[mask], ylim=[md_min, md_max])

    #
    # TWT
    if twt is None:
        # Get the time-depth relation (two way time as a function of md) by calculating it from vp or sonic
        _x = None
        if 'P velocity' in list(log_table.keys()):
            _x = log_table['P velocity']
        elif 'Sonic' in list(log_table.keys()):
            _x = log_table['Sonic']
        if _x is not None:
            twt = well.time_to_depth(_x, templates=templates)
            if twt is None:
                header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]\nLacking info')
                annotate_plot(axes['twt_ax'], None)
            else:
                twt_min = np.min(twt[mask])
                twt_max = np.max(twt[mask])
                header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]')
                annotate_plot(axes['twt_ax'], twt[mask], ylim=[twt_min, twt_max])
        else:
            header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]\nLacking info')
            annotate_plot(axes['twt_ax'], None)
    else:
        header_plot(header_axes['twt_ax'], None, None, None, title='TWT [s]')
        annotate_plot(axes['twt_ax'], twt[mask], ylim=[twt_min, twt_max])

    if twt is not None:
        tops_twt = [twt[find_nearest(depth, y)] for y in wis[well.well][wi_name]]
        for ax in [axes[x] for x in ['twt_ax', 'synt_ax']]:
            ax.axhline(y=tops_twt[0], color='k', ls='--')
            ax.axhline(y=tops_twt[1], color='k', ls='--')

    #
    # Resistivity
    log_types = ['Resistivity']
    lognames = {ltype: [x.name for x in well.get_logs_of_type(ltype)] for ltype in log_types}
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    cls = ['r', 'b', 'k', 'g', 'c']  # should not plot more than 5 lines in this plot!
    lws = [2, 1, 1, 1, 1]
    lss = ['-', '--', ':', '.-', '-']
    styles = [{'lw': lws[i], 'color': cls[i], 'ls': lss[i]} for i in range(len(lognames['Resistivity']))]
    legends = ['{} [{}]'.format(x, templates['Resistivity']['unit']) for x in lognames['Resistivity']]

    xlims = axis_log_plot(axes['res_ax'], depth[mask], [tb.logs[x].data[mask] for x in lognames['Resistivity']],
                  limits, styles, yticks=False, ylim=[md_min, md_max])
    header_plot(header_axes['res_ax'], xlims*len(legends), legends, styles)

    #
    # Rho
    try_these_log_types = ['Density', 'Neutron density']
    log_types = [x for x in try_these_log_types if (len(well.get_logs_of_type(x)) > 0)]
    lognames = {ltype: well.get_logs_of_type(ltype)[0].name for ltype in log_types}
    # Replace the density with the one selected by log_table
    lognames['Density'] = log_table['Density']
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]

    for xx in log_types:
        print(lognames[xx])

    xlims = axis_plot(axes['rho_ax'], depth[mask],
              [tb.logs[lognames[xx]].data[mask] for xx in log_types],
              limits, styles, yticks=False, ylim=[md_min, md_max])
    header_plot(header_axes['rho_ax'], xlims, legends, styles)

    #
    # CPI
    try_these_log_types = ['Saturation', 'Porosity', 'Volume']
    log_types = [x for x in try_these_log_types if (len(well.get_logs_of_type(x)) > 0)]
    lognames = {ltype: well.get_logs_of_type(ltype)[0].name for ltype in log_types}
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    styles = [{'lw': templates[x]['line width'],
               'color': templates[x]['line color'],
               'ls': templates[x]['line style']} for x in log_types]
    legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]

    if len(log_types) == 0:
        header_plot(header_axes['cpi_ax'], None, None, None, title='CPI is lacking')
        axis_plot(axes['cpi_ax'], None, None, None, None, ylim=[md_min, md_max])
    else:
        xlims = axis_plot(axes['cpi_ax'], depth[mask],
                  [tb.logs[lognames[xx]].data[mask] for xx in log_types],
                  limits, styles, yticks=False, ylim=[md_min, md_max])
    header_plot(header_axes['cpi_ax'], xlims, legends, styles)

    #
    # AI
    tt = ''
    ba = None
    if 'Density' in list(log_table.keys()):
        tt += log_table['Density']
        if 'P velocity' in list(log_table.keys()):
            try:
                _vp = tb.logs[log_table['P velocity']].data
                _vs = tb.logs[log_table['S velocity']].data
                _rho = tb.logs[log_table['Density']].data
                if backus_length is not None:
                    ba = bra.backus(_vp, _vs, _rho, backus_length, tb.step)
            except:
                ba = None
            ai = tb.logs[log_table['Density']].data * tb.logs[log_table['P velocity']].data
            tt += '*{}'.format(log_table['P velocity'])
        elif 'Sonic' in list(log_table.keys()):
            ai = tb.logs[log_table['Density']].data / tb.logs[log_table['Sonic']].data
            tt += '/{}'.format(log_table['Sonic'])
        else:
            ai = None
    else:
        ai = None
    if ai is not None:
        #styles = [{'lw': 1, 'color': 'k', 'ls': '--'}]
        log_types = ['AI']
        lognames = {'AI': 'AI'}
        limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
        styles = [{'lw': templates[x]['line width'],
                   'color': templates[x]['line color'],
                   'ls': templates[x]['line style']} for x in log_types]
        legends = ['{} [{}]'.format(lognames[x], templates[x]['unit']) for x in log_types]
        # TODO
        # Now we blindly assumes AI is in m/s g/cm3 Make this more robust
        # print(tb.logs[log_table['P velocity']].header['unit'])
        xlims = axis_plot(axes['ai_ax'], depth[mask], [ai[mask]], limits, styles,
                  yticks=False, ylim=[md_min, md_max])
        #header_plot(header_axes['ai_ax'], xlims, ['AI ({})'.format(tt)], styles)
        if ba is not None: # Backus averaged data exists
            styles_ba = deepcopy(styles)
            styles_ba[0]['lw'] = styles[0]['lw'] * 3
            ai = ba[2] * ba[0]
            _ = axis_plot(axes['ai_ax'], depth[mask], [ai[mask]], limits, styles_ba,
                          yticks=False, ylim=[md_min, md_max])
            # modify the settings to accommodate the backus averaged data
            limits.append(limits[0])
            styles.append(styles_ba[0])
            legends.append('AI backus {:.1f}m'.format(backus_length))
            xlims.append(xlims[0])
        header_plot(header_axes['ai_ax'], xlims, legends, styles)
    else:
        header_plot(header_axes['ai_ax'], None, None, None, title='AI is lacking')
        axis_plot(axes['ai_ax'], None, None, None, None, ylim=[md_min, md_max])


    #
    # Wiggles
    t = np.arange(np.nanmin(twt), np.nanmax(twt), time_step)  # A uniformly sampled array of time steps, from A to B
    if 'P velocity' in list(log_table.keys()):
        vp_t = np.interp(x=t, xp=twt, fp=tb.logs[log_table['P velocity']].data)
    elif 'Sonic' in list(log_table.keys()):
        vp_t = np.interp(x=t, xp=twt, fp=1./tb.logs[log_table['Sonic']].data)
    else:
        vp_t = None
    if 'S velocity' in list(log_table.keys()):
        vs_t = np.interp(x=t, xp=twt, fp=tb.logs[log_table['S velocity']].data)
    elif 'Shear sonic' in list(log_table.keys()):
        vs_t = np.interp(x=t, xp=twt, fp=1./tb.logs[log_table['Shear sonic']].data)
    else:
        vs_t = None
    if 'Density' in list(log_table.keys()):
        rho_t = np.interp(x=t, xp=twt, fp=tb.logs[log_table['Density']].data)
    else:
        rho_t = None

    if (vp_t is not None) and (vs_t is not None) and (rho_t is not None):
        reff = rp.reflectivity(vp_t, None, vs_t, None, rho_t, None, along_wiggle=True)
    else:
        reff = None

    if reff is not None:
        w = ricker(duration, time_step, c_f)

        # Translate the mask to the time variable
        t_mask = np.ma.masked_inside(t[:-1], np.nanmin(twt[mask]), np.nanmax(twt[mask])).mask

        header_plot(header_axes['synt_ax'], None, None, None,
                    title='Incidence angle\nRicker f={:.0f} Hz, l={:.3f} s'.format(c_f, duration))
        for inc_a in range(0, 35, 5):
            wig = np.convolve(w, np.nan_to_num(reff(inc_a)), mode='same')
            wiggle_plot(axes['synt_ax'], t[:-1][t_mask], wig[t_mask], inc_a, scaling=scaling,
                        fill_pos_style=fill_pos_style, fill_neg_style=fill_neg_style, ylim=[twt_min, twt_max])
    else:
        header_plot(header_axes['synt_ax'], None, None, None, title='Refl. coeff. lacking')
        wiggle_plot(axes['synt_ax'], None, None, None)

    if savefig is not None:
        fig.savefig(savefig)
    else:
        plt.show()


def overview_plot(wells, log_table, wis, wi_name, templates, log_types=None, block_name=None, savefig=None):
    """
    Overview plot designed to show data coverage in given working interval together with sea water depth
    Wells with no TVD data are plotted as dashed lines, with TVD the well is drawn with a solid line
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
        log_types = list(log_table.keys())
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
    for i, well in enumerate(wells.values()):
        wnames.append(well.well)
        c_styles[well.well] = {'color': 'k', 'ls': '--', 'lw': 1}
        # extract the relevant log block
        tb = well.block[block_name]

        # Try finding the depth interval of the desired working interval 'wi_name'
        # TODO missing working intervals are tricky to capture, sometimes they cause a TypeError, sometimes not
        # So I need to handle both cases
        try:
            well.calc_mask({}, name='XXX', wis=wis, wi_name=wi_name)
            mask = tb.masks['XXX'].data
            int_exists = True
        except TypeError:
            print('{} not present in well {}. Continue'.format(wi_name, well.well))
            mask = None
            int_exists = False
        if int_exists:
            if mask.all():
                # All values in mask is True, meaning that all data in well is masked out
                print('{} not present in well {}. Continue'.format(wi_name, well.well))
                mask = None
                int_exists = False

        if int_exists:
            if 'tvd' in well.log_names():
                c_styles[well.well] = {'color': 'k', 'ls': '-', 'lw': 1}
                depth_key = 'tvd'
            else:
                depth_key = 'depth'
            if wis[well.well][wi_name][1] > y_max:
                y_max = wis[well.well][wi_name][1]
            #if np.nanmax(tb.logs[depth_key].data[mask]) > y_max:
            #    y_max = np.nanmax(tb.logs[depth_key].data[mask])

            # plot the desired logs
            missing_logs_txt = ''
            for ltype in log_types:
                lname = log_table[ltype]
                if lname not in well.log_names():  # skip this log
                    missing_logs_txt += '{}\n'.format(lname)
                    continue
                if np.isnan(tb.logs[lname].data[mask]).all(): # all nan's
                    missing_logs_txt += '{}\n'.format(lname)
                    continue
                x = uu.norm(tb.logs[lname].data[mask], method='median')
                styles = {'lw': templates[ltype]['line width'],
                          'color': templates[ltype]['line color'], 'ls': templates[ltype]['line style']}
                ax.plot(i + x*pw, tb.logs[depth_key].data[mask], **styles)
            if len(missing_logs_txt) > 1:
                ax.text(i, min(tb.logs[depth_key].data[mask]), 'Missing logs: \n'+missing_logs_txt[:-1],
                    bbox = {'boxstyle': 'round', 'facecolor': 'orangered', 'alpha': 0.5},
                    verticalalignment = 'bottom',
                    horizontalalignment = 'center')
            else:
                ax.text(i, min(tb.logs[depth_key].data[mask]), 'No missing logs',
                        bbox={'boxstyle': 'round', 'facecolor': 'lightgreen', 'alpha': 0.5},
                        verticalalignment='bottom',
                        horizontalalignment='center')

        water_depth = well.get_from_well_info('water depth', templates=templates)
        kb = well.get_from_well_info('kb', templates=templates)
        ax.plot([i, i],
                [0.,  # this is not exact, because the linewidth makes the lines look longer than what they are
                np.abs(water_depth) + kb], label='_nolegend_',
                **sea_style)
        ax.plot([i, i], [0., templates[well.well]['kb']], label='_nolegend_', **kb_style)
        ax.axvline(i, label='_nolegend_', **c_styles[well.well])

    ax.set_ylim(y_max, 0)
    ax.set_ylabel('TVD [m]')
    ax.get_xaxis().set_ticks(range(len(wnames)))
    ax.get_xaxis().set_ticklabels(wnames)
    ax.legend(['{}: {}'.format(xx, log_table[xx]) for xx in list(log_table.keys())])
    fig.tight_layout()

    if _savefig:
        fig.savefig(savefig)
    else:
        plt.show()


def plot_depth_trends(wells, log_table, wis, wi_name, templates, cutoffs,
                      block_name=None, results_folder=None, verbose=True, suffix=None, **kwargs):
    """
    Plots the depth trends (TVD) for each individual log within the given working interval, for all wells

    :param wells:
    :param log_table:
    :param wis:
    :param wi_name:
    :param templates:
    :param cutoffs:
        dict
       E.G. {'depth': ['><', [2100, 2200]], 'phie': ['>', 0.1]}
    :param block_name:
    :param results_folder:
        str
        full pathname of folder where results plots should be saved.
        If verbose is False, no plots are saved
    :param verbose:
        bool
        If True plots are generated, and least_square optimisation shows progress
    :param suffix:
        str
        String used in title and in saved figure names, to distinguish different cases
    :param kwargs:
    :return:
    """
    from scipy.optimize import least_squares
    from blixt_utils.misc.curve_fitting import residuals
    from blixt_utils.utils import mask_string

    if suffix is None:
        suffix = ''
    buffer = 0.
    y_log_name = kwargs.pop('y_log_name', 'tvd')
    if block_name is None:
        block_name = cw.def_lb_name

    # The linear target function we would like to fit the data:
    def linear_function(_t, _a, _b):
        return _a*_t + _b

    depth_trends = {}
    # Start looping over the different log types
    for log_type in log_table:
        log_name = log_table[log_type]
        if verbose:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = None, None

        # Start looping over wells and plot the data in TVD domain
        data_container = np.zeros(0)  # empty container
        tvd_container = np.zeros(0)
        legend_items = []
        tvd_min = 1E6
        tvd_max = -1E6
        for well in wells:
            wells[well].calc_mask(cutoffs, 'my_mask', log_table=log_table, wis=wis, wi_name=wi_name)
            mask = wells[well].block[block_name].masks['my_mask'].data
            legend_items.append(well)
            xdata = wells[well].block[block_name].logs[log_name].data[mask]
            data_container = np.append(data_container, xdata)
            ydata = wells[well].block[block_name].logs['tvd'].data[mask]
            tvd_container = np.append(tvd_container, ydata)
            if np.min(ydata) < tvd_min:
                tvd_min = np.min(ydata)
            if np.max(ydata) > tvd_max:
                tvd_max = np.max(ydata)
            if verbose:
                xp.plot(
                    xdata,
                    ydata,
                    cdata=templates[well]['color'],
                    mdata=templates[well]['symbol'],
                    xtempl=templates[log_type],
                    ytempl=templates['TVD'],
                    edge_color=False,
                    ax=ax
                )
        # Calculate depth trend
        verbosity_level = 0
        if verbose:
            verbosity_level = 2
        res = least_squares(residuals, [1., 1.], args=(tvd_container, data_container),
                            kwargs={'target_function': linear_function}, verbose=verbosity_level)

        depth_trends[log_name] = res.x
        new_tvd = np.linspace(tvd_min, tvd_max)
        if verbose:
            ax.plot(linear_function(new_tvd, *res.x), new_tvd)
            legend_items.append('{} = {:.3}xTVD + {:.3}'.format(log_name, res.x[0], res.x[1]))

            ax.set_title('{}: {}. {} {}'.format(log_type, log_name, mask_string(cutoffs, wi_name), suffix))
            ax.set_ylim(tvd_max, tvd_min)
            this_legend = ax.legend(
                legend_items,
                prop=FontProperties(size='smaller'),
                scatterpoints=1,
                markerscale=2,
                loc=1
            )

            if results_folder:
                if suffix != '' and suffix[0] != '_':
                    suffix = '_{}'.format(suffix)
                fig.savefig(os.path.join(results_folder, 'Depth trend {} in {}{}.png'.format(
                    log_name, wi_name, suffix
                )))
            else:
                plt.show()
    return depth_trends


def chi_rotation(well, log_tables, wis, wi_name, templates, buffer=None, chi_angles=None,
                 common_limit=None, ref_logtable=None,
                 plot_tops=None, annotate_tops=None,
                 block_name=None, results_folder=None, verbose=True, suffix=None, **kwargs):
    """
    For the given set of brine / oil / gas elastic logs, it plots extended elastic impedance at different chi angles
    in the selected working interval
    Args:
        well:
            well object
        log_tables:
            list
            List of (three) logtables which determines which elastic logs to use in the calculation of  the
            extended elastic impedance. E.G.
                [ {'P velocity': 'vp_brine', 'S velocity': 'vs_brine', ...},
                  {'P velocity': 'vp_oil', 'S velocity': 'vs_oil', ...}, ... ]
            The order is assumed to be brine, oil, gas, and they will be plotted in blue, green, red
        wis:
        wi_name:
        templates:
        buffer:
        chi_angles:
            list
            List of floats specifying the Chi angles (deg) to plot
            Defaults to [0., 5., 10., 15., 20., 30., 40., 50.]
        common_limit:
            list, tuple
            Common min, max values for the EEI at all Chi angles
        ref_logtable:
            dict
            A log table that is used as reference to make it easier for the user to orient themself
            on where we are looking.
            Preferably a gamma ray log, e.g. {'Gamma ray': 'gr'}
        plot_tops:
            list
            List of MD [m] values to plot horizontal lines across all plots
        annotate_tops:
            list
            List of strings, with same length as plot_tops, which are used to annotate the tops
        block_name:
        results_folder:
        verbose:
        suffix:
        **kwargs:

    Returns:

    """
    if buffer is None:
        buffer = 50.
    if chi_angles is None:
        chi_angles = [0., 5., 10., 15., 20., 30., 40., 50.]
    if block_name is None:
        block_name = cw.def_lb_name
    if suffix is None:
        suffix = ''
    if common_limit is not None:
        if len(common_limit) != 2:
            raise IOError('Common limits must have length 2, a min and max value')

    required_log_types = ['P velocity', 'S velocity', 'Density']
    # Check that the given log tables contain the required log types
    for log_table in log_tables:
        for log_type in required_log_types:
            if log_type not in log_table:
                raise IOError('Log type {} not included in log table'.format(log_type))

    # sety up plot window
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('EEI in {} interval in well {}, {}'.format(wi_name, well.well, suffix))
    n_cols = len(chi_angles)
    if ref_logtable is not None:
        n_cols += 1  # add an extra column for the reference log
    n_rows = 2
    height_ratios = [1, 5]
    spec = fig.add_gridspec(nrows=n_rows, ncols=n_cols,
                            height_ratios=height_ratios,
                            hspace=0., wspace=0.,
                            left=0.05, bottom=0.03, right=0.98, top=0.96)
    header_axes = []
    axes = []
    for i in range(n_cols):
        header_axes.append(fig.add_subplot(spec[0, i]))
        axes.append(fig.add_subplot(spec[1, i]))

    # default colors used in the plotting assuming we plot brine, oil and gas logs
    def_colors = ['b', 'g', 'r']
    legends = ['brine', 'oil', 'gas']
    # if we plot more logs per chi angle, we need to extend the lists
    if len(log_tables) > 3:
        def_colors += xp.cnames
        for i in range(len(log_tables) - 3):
            legends.append('xxx')

    styles = [{'lw': 1.,
               'color': def_colors[i],
               'ls': '-'} for i in range(len(log_tables))]

    tb = well.block[block_name]  # this log block
    depth = tb.logs['depth'].data
    mask = np.ma.masked_inside(depth, wis[well.well][wi_name][0] - buffer, wis[well.well][wi_name][1] + buffer).mask
    md_min = np.min(depth[mask])
    md_max = np.max(depth[mask])

    # Calculate the EEI for the different elastic logs specified in the log tables
    # First calculate common average values used in the normalization
    vp0 = np.nanmean(tb.logs[log_tables[0]['P velocity']].data[mask])
    vs0 = np.nanmean(tb.logs[log_tables[0]['S velocity']].data[mask])
    rho0 = np.nanmean(tb.logs[log_tables[0]['Density']].data[mask])
    eeis = []
    for log_table in log_tables:
        vp = tb.logs[log_table['P velocity']].data[mask]
        vs = tb.logs[log_table['S velocity']].data[mask]
        rho = tb.logs[log_table['Density']].data[mask]
        eeis.append(rp.eei(vp, vs, rho, vp0=vp0, vs0=vs0, rho0=rho0))

    # Find the common limits for all log types at each chi angle
    limits = []
    if common_limit is None:
        for i, chi in enumerate(chi_angles):
            x_min = 1.E6
            x_max = -1.E6
            this_min, this_max = None, None
            for j in range(len(log_tables)):
                this_eei = eeis[j](chi)
                this_min = np.nanmin(this_eei)
                this_max = np.nanmax(this_eei)
                if this_max > x_max:
                    x_max = this_max
                if this_min < x_min:
                    x_min = this_min
            these_limits = [this_min, this_max]
            limits.append([these_limits] * len(log_tables))
    else:
        for i, chi in enumerate(chi_angles):
            limits.append([common_limit] * len(log_tables))

    # Start plotting data
    if ref_logtable is not None:
        ref_log_type = list(ref_logtable.keys())[0]
        ref_log_name = ref_logtable[ref_log_type]
        this_data = tb.logs[ref_log_name].data[mask]
        this_template = templates[ref_log_type]
        #normalize = mpl.colors.Normalize(vmin=this_template['min'], vmax=this_template['max'])
        this_style = [{'lw': this_template['line width'],
                       'color': this_template['line color'],
                        'ls': this_template['line style']}]
        this_legend = ['{} [{}]'.format(ref_log_name, this_template['unit'])]
        xlims = axis_plot(axes[0], depth[mask],
                          [this_data],
                          [[this_template['min'], this_template['max']]],
                          this_style,
                          ylim=[md_min, md_max]
                          )
        header_plot(header_axes[0], xlims, this_legend, this_style)
        #axes[0].fill_betweenx(depth[mask], 0, tb.logs[ref_log_name].data[mask],
        #                      facecolor=plt.get_cmap(this_template['colormap'])(normalize(this_data)),
        #                      #cmap=this_template['colormap'],
        #                      clim=(this_template['min'], this_template['max'])
        #                      )

    for i, chi in enumerate(chi_angles):
        if ref_logtable is not None:
            axes_i = i + 1
        else:
            axes_i = i
        xlims = axis_plot(axes[axes_i], depth[mask],
                          [eei(chi) for eei in eeis],
                          limits[i], styles, ylim=[md_min, md_max],
                          yticks=axes_i==0)
        header_plot(header_axes[axes_i], xlims, legends, styles, title='Chi {:.0f}$^\circ$'.format(chi))

    if plot_tops is not None:
        for ax in axes:
            for this_top in plot_tops:
                ax.axhline(y=this_top, color='k', ls='--')
        if annotate_tops is not None:
            for i, this_name in enumerate(annotate_tops):
                axes[0].text(axes[0].get_xlim()[0], plot_tops[i], this_name, ha='left', va='bottom')

    axes[0].set_ylabel('Measured Depth [m]')
    if results_folder:
        if suffix != '' and suffix[0] != '_':
            suffix = '_{}'.format(suffix)
        fig.savefig(os.path.join(results_folder, 'EEI Chi {:.0f}-{:.0f} in {} in {}{}.png'.format(
            chi_angles[0], chi_angles[-1], wi_name, well.well, suffix
        )))
    else:
        plt.show()





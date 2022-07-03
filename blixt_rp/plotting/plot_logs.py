import matplotlib.pyplot as plt
import numpy as np
import logging
from copy import deepcopy

import blixt_rp.core.well as cw
import blixt_utils.utils as uu
from blixt_utils.utils import log_table_in_smallcaps as small_log_table
from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot
import blixt_rp.rp.rp_core as rp
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

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('{} interval in well {} {}'.format(wi_name, well.well, suffix))
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
    print(log_types)
    lognames = {ltype: well.get_logs_of_type(ltype)[0].name for ltype in log_types}
    print(lognames)
    limits = [[templates[x]['min'], templates[x]['max']] for x in log_types]
    print(limits)
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
        #print(tops_twt)
        #for ax in [axes[x] for x in ['twt_ax', 'synt_ax']]:
        for ax in [axes[x] for x in ['twt_ax']]:
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
        xlims = axis_plot(axes['ai_ax'], depth[mask], [ai[mask]/1000.], limits, styles,
                  yticks=False, ylim=[md_min, md_max])
        #header_plot(header_axes['ai_ax'], xlims, ['AI ({})'.format(tt)], styles)
        if ba is not None: # Backus averaged data exists
            styles_ba = deepcopy(styles)
            styles_ba[0]['lw'] = styles[0]['lw'] * 3
            ai = ba[2] * ba[0]
            _ = axis_plot(axes['ai_ax'], depth[mask], [ai[mask]/1000.], limits, styles_ba,
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
    #t = np.arange(twt[mask][0], twt[mask][-1], 0.004)  # A uniformly sampled array of time steps, from A to B
    t = np.arange(0., np.nanmax(twt), time_step)  # A uniformly sampled array of time steps, from 0 to 3
    #print(len(t))
    if 'P velocity' in list(log_table.keys()):
        vp_t = np.interp(x=t, xp=twt, fp=tb.logs[log_table['P velocity']].data)
    elif 'Sonic' in list(log_table.keys()):
        vp_t = np.interp(x=t, xp=twt, fp=1./tb.logs[log_table['Sonic']].data)
    else:
        vp_t = None
    #print(len(vp_t))
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
        # TODO
        # TODO XXX
        # The dtr I have introduced here is confusing. Data is plotted in MD after calculating the twt from check shots

        #print(len(reff(10)))
        #tw, w = ricker(_f=c_f, _length=duration, _dt=time_step)
        w = ricker(duration, time_step, c_f)
        #print(len(w))

        # Compute the depth-time relation
        dtr = np.array([depth[find_nearest(twt, tt)] for tt in t])
        #print(np.nanmin(dtr), np.nanmax(dtr))
        # Translate the mask to the time variable
        t_mask = np.ma.masked_inside(t[:-1], np.nanmin(twt[mask]), np.nanmax(twt[mask])).mask
        #wiggle_plot(axes['synt_ax'], t[:-1][t_mask], wig[t_mask], 10)

        header_plot(header_axes['synt_ax'], None, None, None,
                    title='Incidence angle\nRicker f={:.0f} Hz, l={:.3f} s'.format(c_f, duration))
        for inc_a in range(0, 35, 5):
            wig = np.convolve(w, np.nan_to_num(reff(inc_a)), mode='same')
            wiggle_plot(axes['synt_ax'], dtr[:-1][t_mask], wig[t_mask], inc_a, scaling=scaling,
                        fill_pos_style=fill_pos_style, fill_neg_style=fill_neg_style)  #, ylim=[twt_min, twt_max])
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



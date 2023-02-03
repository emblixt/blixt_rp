import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from copy import deepcopy
import bruges.rockphysics.anisotropy as bra
import logging

from blixt_utils.plotting import crossplot as xp
import blixt_rp.rp.rp_core as rp
import blixt_rp.core.well as cw
from blixt_utils.misc.convert_data import convert as cnvrt
from blixt_utils.utils import log_table_in_smallcaps as small_log_table

logger = logging.getLogger(__name__)
opt1 = {'bbox': {'facecolor': '0.9', 'alpha': 0.5, 'edgecolor': 'none'}}
opt2 = {'ha': 'right', 'bbox': {'facecolor': '0.9', 'alpha': 0.5, 'edgecolor': 'none'}}


def plot_rp(wells, log_table, wis, wi_name, cutoffs=None, templates=None, legend_items=None,
            plot_type=None, ref_val=None, fig=None, ax=None, block_name=None, color_by=None, savefig=None,
            backus=False, **kwargs):

    """
    Plots some standard rock physics crossplots for the given wells

    :param wells:
        dict
        dictionary with well names as keys, and core.well.Well object as values
        As returned from core.well.Project.load_all_wells()
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
        dict
        working intervals, as defined in the "Working intervals" sheet of the project table, and
        loaded through:
        wp = Project()
        wis = rp_utils.io.project_working_intervals(wp.project_table)
    :param wi_name:
        str
        name of working interval to plot
    :param cutoffs:
        dict
        Dictionary with log types as keys, and list with mask definition as value
        E.G.
            {'Volume': ['<', 0.5], 'Porosity': ['>', 0.1]}
    :param templates:
        dict
        templates dictionary as returned from rp_utils.io.project_templates()
    :param legend_items:
        list
        list of Line2D objects that are used in the legends.
        if None, the well names will be used
    :param plot_type:
        str
        plot_type =
            'AI-VpVs': AI versus Vp/Vs plot
            'Phi-Vp': Porosity versus Vp plot
            'I-G': Intercept versus Gradient plot
    :param ref_val:
        dict
        dictionary with wellnames as keys. Each value is a list of reference
        Vp [m/s], Vs [m/s], and rho [g/cm3] that are used
        when calculating the Intercept and gradient
    :param fig:

    :param ax:

    :param block_name:
        str
        Name of the log block from where the logs are picked
    :param color_by:
        str
        If None, data are colored according to each individual wells template, else set it to
        a log type, and it tries to color the points based on that data
    :param savefig
        str
        full pathname of file to save the plot to
    :param backus:
        bool
        If true, Vp, Vs and Rho are Backus averaged before plotting
    :param **kwargs:
        keyword arguments passed on to crossplot.plot()
    :return:
        colorbar, legend
    """
    #
    # some initial setups
    backus_length = 15.
    if ref_val is not None:
        if not isinstance(ref_val, dict):
            raise ValueError('ref_val must be a dictionary with reference Vp, Vs and rho for each well')
    log_table = small_log_table(log_table)
    if block_name is None:
        block_name = cw.def_lb_name

    _savefig = False
    if plot_type is None:
        plot_type = 'AI-VpVs'
    elif plot_type == 'I-G' and ref_val is None:
        raise ValueError('Reference values for each well is needed for creating Intercept vs. Gradient plot')
    logs = [n.lower() for n in list(log_table.values())]
    if savefig is not None:
        _savefig = True

    #
    # set up plotting environment
    if fig is None:
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.subplots()
        else:  # Only ax is set, but not fig. Which means all functionality calling fig must be turned off
            _savefig = False
    elif ax is None:
        ax = fig.subplots()

    # load
    well_names = []
    desc = ''
    n_wells = len(wells)
    counter = 0
    cbar = None
    for wname, _well in wells.items():
        counter += 1
        print('Plotting well {}'.format(wname))
        # create a deep copy of the well so that the original is not altered
        well = deepcopy(_well)
        these_wis = wis[wname]
        if wi_name.upper() not in [_x.upper() for _x in list(these_wis.keys())]:
            warn_txt = 'Working interval {} does not exist in well {}'.format(wi_name, wname)
            print('WARNING: {}'.format(warn_txt))
            logger.warning(warn_txt)
            continue  # this working interval does not exist in current well
        # test if necessary logs are in the well
        skip = False
        for _log in logs:
            if _log not in list(well.block[block_name].logs.keys()):
                warn_txt = '{} log is lacking in {}'.format(_log, wname)
                print('WARNING: {}'.format(warn_txt))
                logger.warning(warn_txt)
                skip = True
        if skip:
            continue
        # create mask based on working interval
        well.calc_mask({}, name='XXX', wis=wis, wi_name=wi_name)
        # apply mask (which also deletes the mask itself)
        well.apply_mask('XXX')
        # create mask based on cutoffs
        if cutoffs is not None:
            well.calc_mask(cutoffs, name='cmask', log_type_input=True, log_table=log_table)
            mask = well.block[block_name].masks['cmask'].data
            desc = well.block[block_name].masks['cmask'].header.desc
        else:
            mask = None

        # Collect data for plot, start with Vp, Vs and Rho
        # TODO check the units, and convert if necessary
        rho = well.block[block_name].logs[log_table['Density'].lower()].data
        rho_unit = well.block[block_name].logs[log_table['Density'].lower()].header.unit
        if 'P velocity' in list(log_table.keys()):
            vp = well.block[block_name].logs[log_table['P velocity'].lower()].data
            vp_unit = well.block[block_name].logs[log_table['P velocity'].lower()].header.unit
            vs = well.block[block_name].logs[log_table['S velocity'].lower()].data
            vs_unit = well.block[block_name].logs[log_table['S velocity'].lower()].header.unit
        else:  # Assume we can use the sonic logs instead.
            vp_unit = 'm/s'
            success, vp = cnvrt(well.block[block_name].logs[log_table['Sonic'].lower()].data,  # convert this data
                       well.block[block_name].logs[log_table['Sonic'].lower()].header.unit,  # from this unit
                       vp_unit)  # to this unit
            vs_unit = 'm/s'
            success, vs = cnvrt(well.block[block_name].logs[log_table['Shear sonic'].lower()].data,
                       well.block[block_name].logs[log_table['Shear sonic'].lower()].header.unit,
                       vs_unit)

        # Apply Backus average if requested
        if backus is True:
            ba = bra.backus(vp, vs, rho, backus_length, well.block[block_name].step.value)
            vp = ba[0]
            vs = ba[1]
            rho = ba[2]

        if plot_type == 'AI-VpVs':
            # Calculating Vp / Vs using vp and vs
            x_data = vp * rho
            x_unit = '{} {}'.format(vp_unit, rho_unit)
            y_data = vp / vs
            y_unit = '-'
            xtempl = {'full_name': 'AI', 'unit': x_unit}
            ytempl = {'full_name': 'Vp/Vs', 'unit': y_unit}

        elif (plot_type == 'Vp-bd') or (plot_type == 'Vp-tvd'):
            x_data = vp
            x_unit = vp_unit
            xtempl = {'full_name': log_table['P velocity'], 'unit': x_unit}
            if plot_type == 'Vp-bd':
                y_data = well.get_burial_depth(templates, block_name)
                ytempl = {'full_name': 'Burial depth', 'unit': 'm'}
            else:
                y_data = well.block[block_name].get_tvd()
                ytempl = {'full_name': 'TVD', 'unit': 'm'}

        elif (plot_type == 'VpVs-bd') or (plot_type == 'VpVs-tvd'):
            x_data = vp / vs
            xtempl = {'full_name': 'Vp/Vs', 'unit': '-'}
            if plot_type == 'VpVs-bd':
                y_data = well.get_burial_depth(templates, block_name)
                ytempl = {'full_name': 'Burial depth', 'unit': 'm'}
            else:
                y_data = well.block[block_name].get_tvd()
                ytempl = {'full_name': 'TVD', 'unit': 'm'}

        elif (plot_type == 'AI-bd') or (plot_type == 'AI-tvd'):
            x_data = vp * rho
            x_unit = '{} {}'.format(vp_unit, rho_unit)
            xtempl = {'full_name': 'AI', 'unit': x_unit}
            if plot_type == 'AI-bd':
                y_data = well.get_burial_depth(templates, block_name)
                ytempl = {'full_name': 'Burial depth', 'unit': 'm'}
            else:
                y_data = well.block[block_name].get_tvd()
                ytempl = {'full_name': 'TVD', 'unit': 'm'}

        elif (plot_type == 'Rho-bd') or (plot_type == 'Rho-tvd'):
            x_data = rho
            x_unit = '{}'.format(rho_unit)
            xtempl = {'full_name': log_table['Density'], 'unit': x_unit}
            if plot_type == 'Rho-bd':
                y_data = well.get_burial_depth(templates, block_name)
                ytempl = {'full_name': 'Burial depth', 'unit': 'm'}
            else:
                y_data = well.block[block_name].get_tvd()
                ytempl = {'full_name': 'TVD', 'unit': 'm'}

        elif plot_type == 'Phi-Vp':
            x_data = well.block[block_name].logs[log_table['Porosity'].lower()].data
            x_unit = '{}'.format(well.block[block_name].logs[log_table['Porosity'].lower()].header.unit)
            y_data = vp
            y_unit = '{}'.format(vp_unit)
            xtempl = {'full_name': 'Porosity', 'unit': x_unit}
            ytempl = {'full_name': 'Vp', 'unit': y_unit}
        elif plot_type in ['Phi-%AI', 'Phi-I', 'Phi-G']:
            if ref_val is None:
                raise ValueError('Reference values for each well is needed')
            x_data = well.block[block_name].logs[log_table['Porosity'].lower()].data
            x_unit = '{}'.format(well.block[block_name].logs[log_table['Porosity'].lower()].header.unit)
            xtempl = {'full_name': 'Porosity', 'unit': x_unit}
            if plot_type == 'Phi-%AI':
                y2_data = vp * rho
                y1_data = ref_val[wname][0] * ref_val[wname][2]
                y_data = 100 * (y2_data - y1_data) / y1_data
                y_unit = '%'
                ytempl = {'full_name': 'rel. AI', 'unit': y_unit}
            elif plot_type == 'Phi-I':
                y_data = rp.intercept(ref_val[wname][0], vp, ref_val[wname][2], rho)
                y_unit = '-'
                ytempl = {'full_name': 'Intercept', 'unit': y_unit}
            else:
                y_data = rp.gradient(ref_val[wname][0], vp, ref_val[wname][1], vs, ref_val[wname][2], rho)
                y_unit = '-'
                ytempl = {'full_name': 'Gradient', 'unit': y_unit}
        elif plot_type == 'I-G':
            x_data = rp.intercept(ref_val[wname][0], vp,
                                  ref_val[wname][2], rho)
            x_unit = '-'
            y_data = rp.gradient(ref_val[wname][0], vp, ref_val[wname][1], vs, ref_val[wname][2], rho)
            y_unit = '-'
            xtempl = {'full_name': 'Intercept', 'unit': x_unit}
            ytempl = {'full_name': 'Gradient', 'unit': y_unit}
        else:
            raise IOError('No known plot type selected')

        well_names.append(wname)
        
        # arrange coloring
        this_color = templates[wname]['color']
        draw_color_bar = False
        if isinstance(color_by, str):
            _color_logs = well.block[block_name].get_logs_of_type(color_by)
            if len(_color_logs) > 0:
                this_color = _color_logs[0].data
                # only draw colorbar for last well
                if counter == n_wells:
                    draw_color_bar = None

        # start plotting

        cbar = xp.plot(
            x_data,
            y_data,
            cdata=this_color,
            cbar=draw_color_bar,
            mdata=templates[wname]['symbol'],
            xtempl=xtempl,
            ytempl=ytempl,
            mask=mask,
            fig=fig,
            ax=ax,
            **kwargs
        )
    ax.autoscale(True, axis='both')
    if ('tvd' in plot_type) or ('bd' in plot_type):
        ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title('{}, {}'.format(wi_name, desc))
    #
    if legend_items is None:
        legend_items = []
        for wname in well_names:
            legend_items.append(
                Line2D([0], [0],
                       color=templates[wname]['color'],
                       lw=0, marker=templates[wname]['symbol'],
                       label=wname))

    legend_names = [x._label for x in legend_items]
    this_legend = ax.legend(
        legend_items,
        legend_names,
        prop=FontProperties(size='smaller'),
        scatterpoints=1,
        markerscale=2,
        loc=1)

    if _savefig:
        fig.savefig(savefig)

    return cbar, this_legend


def plot_rpt(t, rpt, constants, rpt_keywords, sizes, colors, fig=None, ax=None, **kwargs):
    """
    Plot any RPT (rock physics template) that can be described by a function rpt(t), which can be
    evaluated at different values of a constant (eg. the saturation). E.G. rpt(x, const=constants[i])
    E.G. to plot a RPT which is a function of porosity in a Vp/Vs x AI crossplot:
        t = porosity
        x, y = rpt(t)  # x is AI, y is Vp/Vs

    :param t:
        np.array
        array of length N
        t values used to draw the rockphysics template, preferably less than about 10 items long for creating
        nice plots
    :param rpt:
        function
        Rock physics template function of t
        Should take a second argument which is used to parameterize the function
        e.g.
        def rpt(t, c, **rpt_keywords):
            return c*t + rpt_keywords.pop('zero_crossing', 0)
    :param constants:
        list
        list of length M of constants used to parametrize the rpt function
    :param rpt_keywords:
        dict
        Dictionary with keywords passed on to the rpt function
    :param sizes:
        float or np.array (of size N, or M x N)
        determines the size of the markers
        if np.array it must be same size as x
    :param colors
        str or np.array
        determines the colors of the markers
        in np.array it must be same size as x
    :return
        all_x, all_y
        lists of len(constants) ndarray's of x and y data that can be used to
        annotate the plot
    """
    #
    # some initial setups
    all_x = []
    all_y = []
    lw = kwargs.pop('lw', 0.5)
    tc = kwargs.pop('c', 'k')
    edgecolor = kwargs.pop('edgecolor', 'none')
    for test_obj, def_val in zip([sizes, colors], [90., 'red']):
        if test_obj is None:
            test_obj = def_val
        elif isinstance(test_obj, np.ndarray):
            if len(test_obj.shape) == 1:
                # Broadcast 1D array so that it can reused for all elements in constants
                test_obj = np.broadcast_to(test_obj, (len(constants), len(test_obj)))
            elif len(test_obj.shape) == 2:
                if not test_obj.shape == (len(constants), len(t)):
                    raise IOError('Shape of input must match constants, and t: ({}, {})'.format(len(constants), len(t)))
        if def_val == 90.:
            sizes = test_obj
        else:
            colors = test_obj

    #
    # set up plotting environment
    if fig is None:
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.subplots()
    elif ax is None:
        ax = fig.subplots()

    #
    # start drawing the RPT
    if not isinstance(colors, str):
        vmin = np.min(colors)
        vmax = np.max(colors)
        cmap = 'jet'
    else:
        vmin = None; vmax = None; cmap = None
    for i, const in enumerate(constants):
        x, y = rpt(t, const, **rpt_keywords)
        all_x.append(x)
        all_y.append(y)
        # First draw lines of the RPT
        ax.plot(x, y, lw=lw, c=tc, label='_nolegend_', **kwargs)
        # Next draw the points
        ax.scatter(x, y, c=colors if isinstance(colors, str) else colors[i],
                   s=sizes[i] if isinstance(sizes, np.ndarray) else float(sizes),
                   vmin=vmin, vmax=vmax, cmap=cmap, edgecolor=edgecolor,
                   label='_nolegend_', **kwargs)

    return all_x, all_y


def rpt_phi_sw(_phi, _sw, **kwargs):
    plot_type = kwargs.pop('plot_type', 'AI-VpVs')
    ref_val = kwargs.pop('ref_val', None)  # ref values of Vp, Vs and rho for calculating I x G [Vp, Vs, rho]
    model = kwargs.pop('model', 'stiff')
    phi_c = kwargs.pop('phi_c', None)  # critical porosity
    p_conf = kwargs.pop('p_conf', None)  # Confining pressure in MPa (effective pressure?)
    smcf = kwargs.pop('smcf', None)  # shear modulus correction factor  (1=dry pack with perfect adhesion, 0=dry frictionless pack)
    rho_b = kwargs.pop('rho_b', None)  # Brine density  g/cm3
    k_b = kwargs.pop('k_b', None)   # Brine bulk modulus  GPa
    rho_hc = kwargs.pop('rho_hc', None)  # HC density
    k_hc = kwargs.pop('k_hc', None)  # HC bulk modulus
    rho_min = kwargs.pop('rho_min', None)  # Density [g/cm3] of mineral mix
    k_min = kwargs.pop('k_min', None)  # Bulk modulus GPa of mineral mix
    mu_min = kwargs.pop('mu_min', None)  # Shear modulus of mineral mix
    rho_qz = kwargs.pop('rho_qz', None)  # Density [g/cm3] of quartz
    k_qz = kwargs.pop('k_qz', None)  # Bulk modulus GPa of quartz
    mu_qz = kwargs.pop('mu_qz', None)  # Shear modulus of quartz
    vsh = kwargs.pop('vsh', None)  # Volume fraction of shale
    rho_sh = kwargs.pop('rho_sh', None)  # Density [g/cm3] of shale
    k_sh = kwargs.pop('k_sh', None)  # Bulk modulus GPa of shale
    mu_sh = kwargs.pop('mu_sh', None)  # Shear modulus of shale
    ntg = kwargs.pop('ntg', 1)  # Net to Gross, use the shale properties above in the non-Net
    c_n = kwargs.pop('c_n', None)  # Coordination number in the contact cement model
    k_cem = kwargs.pop('k_cem', None)  # Cement bulk modulus in GPa
    mu_cem = kwargs.pop('mu_cem', None)  # Cement shear modulus in GPa
    scheme = kwargs.pop('scheme', None)  # scheme for calculating cement deposition
    apc = kwargs.pop('apc', None)  # absolute percent cement (in %)
    calc_phi_eff = kwargs.pop('calc_phi_eff', False)  # set this to True to calculate effective porosity and use it

    # Check if we need to calculate the average mineral properties, or if they are given
    if (rho_min is None) or (mu_min is None) or (k_min is None):
        k_min = rp.vrh_bounds([vsh, 1 - vsh], [k_sh, k_qz])[2]  # Mineral bulk modulus
        mu_min = rp.vrh_bounds([vsh, 1 - vsh], [mu_sh, mu_qz])[2]  # Mineral shear modulus
        rho_min = rp.vrh_bounds([vsh, 1 - vsh], [rho_sh, rho_qz])[0]  # Density of minerals

    if calc_phi_eff:
        #phi = _phi - vsh * ((rho_min - rho_sh) / (rho_min - rho_b))  # eq. 2 in https://link.springer.com/article/10.1007/s13202-020-01073-2#Fig8
        phi = _phi * (1. - vsh)  # eq. 1.4 in https://www.informit.com/articles/article.aspx?p=1626870&seqNum=2
    else:
        phi = 1. * _phi

    # Apply the rock physics model to modify the mineral properties
    if model == 'stiff':
        k_dry, mu_dry = rp.stiffsand(k_min, mu_min, phi, phi_c, c_n, p_conf, smcf)
    elif model == 'contactcement':
        k_dry, mu_dry = rp.contactcement(k_min, mu_min, phi, phi_c, c_n, k_cem, mu_cem, scheme)
    elif model == 'constantcement':
        k_dry, mu_dry = rp.constantcement(k_min, mu_min, phi, phi_c, apc, c_n, k_cem, mu_cem, scheme, smcf=smcf)
    else:
        model = 'soft'
        k_dry, mu_dry = rp.softsand(k_min, mu_min, phi, phi_c, c_n, p_conf, smcf)
    info_txt = 'Using {} rock physics template'.format(model)
    print(info_txt)
    logger.info(info_txt)

    # Calculate the final fluid properties for the given water saturation
    k_f2 = rp.vrh_bounds([_sw, 1.-_sw], [k_b, k_hc])[1]  # K_f
    rho_f2 = rp.vrh_bounds([_sw, 1.-_sw],  [rho_b, rho_hc])[0]  #RHO_f

    # Use Gassman to calculate the final elastic properties
    vp_2, vs_2, rho_2, k_2 = rp.vels(k_dry, mu_dry, k_min, rho_min, k_f2, rho_f2, phi)

    # if ntg is not 1, the final values are modified with the harmonic mean of the Net to Gross properties
    if ntg != 1:
        vp_2 = rp.vrh_bounds([ntg, 1.-ntg], [vp_2, 1000. * rp.v_p(k_sh, mu_sh, rho_sh)])[1]
        vs_2 = rp.vrh_bounds([ntg, 1.-ntg], [vs_2, 1000. * rp.v_s(mu_sh, rho_sh)])[1]
        rho_2 = rp.vrh_bounds([ntg, 1.-ntg], [rho_2, rho_sh])[1]
        k_2 = rp.vrh_bounds([ntg, 1.-ntg], [k_2, k_sh])[1]

    if plot_type == 'AI-VpVs':
        xx = rho_2*vp_2
        yy = vp_2/vs_2
    elif plot_type == 'AI-Vp':
        xx = rho_2 * vp_2
        yy = vp_2
    elif plot_type == 'Phi-Vp':
        xx = phi
        yy = vp_2
    elif plot_type == 'I-G':
        if ref_val is None:
            raise IOError('List of Vp, Vs, and density must be provided to calculate I and G')
        xx = rp.intercept(ref_val[0], vp_2, ref_val[2], rho_2)
        yy = rp.gradient(ref_val[0], vp_2, ref_val[1], vs_2, ref_val[2], rho_2)
    else:
        raise IOError('No valid plot_type selected')

    return xx, yy

def test():
    from blixt_rp.core.well import Project
    import blixt_utils.io.io as uio
    from blixt_rp.core.minerals import MineralMix
    wi_name = 'SAND E'

    plot_type = 'AI-VpVs'
    ref_val = [2695., 1340., 2.35]  # Kvitnos shale

    fig, ax = plt.subplots()

    wp = Project()
    log_table = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry', 'Porosity': 'phie',
                    'Volume': 'vcl'}
    wis = uio.project_working_intervals(wp.project_table)
    templates = uio.project_templates(wp.project_table)
    cutoffs = {'Volume': ['<', 0.4], 'Porosity': ['>', 0.1]}
    wells = wp.load_all_wells()
    plot_rp(wells, log_table, wis, wi_name, cutoffs, templates,
            plot_type=plot_type, ref_val=ref_val, fig=fig, ax=ax)

    # Calculate average properties of mineral mix in desired working interval
    mm = MineralMix()
    mm.read_excel(wp.project_table)
    _rho_min = np.ones(0)  # zero length empty array
    _k_min = np.ones(0)
    _mu_min = np.ones(0)
    for well in wells.values():
        # calculate the mask for the given cut-offs, and for the given working interval
        well.calc_mask(cutoffs, wis=wis, wi_name=wi_name, name='this_mask', log_table=log_table)
        mask = well.block['Logs'].masks['this_mask'].data

        _rho_min = np.append(_rho_min, well.calc_vrh_bounds(mm, param='rho', wis=wis, method='Voigt')[wi_name][mask])
        _k_min = np.append(_k_min, well.calc_vrh_bounds(mm, param='k', wis=wis, method='Voigt-Reuss-Hill')[wi_name][mask])
        _mu_min = np.append(_mu_min, well.calc_vrh_bounds(mm, param='mu', wis=wis, method='Voigt-Reuss-Hill')[wi_name][mask])

    rho_min = np.nanmean(_rho_min)
    k_min = np.nanmean(_k_min)
    mu_min = np.nanmean(_mu_min)

    phi = np.linspace(0.05, 0.3, 6)
    sw = np.array([0.8, 0.95, 1.])
    sizes = np.empty((sw.size, phi.size))
    colors = np.array([np.ones(6) * 100 * (i + 1) ** 2 for i in range(3)]) / 900.
    # iterate over all phi values
    for i, val in enumerate(phi):
        sizes[:, i] = 10 + (40 * val) ** 2

    xx, yy = plot_rpt(phi, rpt_phi_sw, sw,
             {'plot_type': plot_type,
              'ref_value': ref_val,
              'model': 'stiff',
              'rho_min': rho_min,
              'k_min': k_min,
              'mu_min': mu_min},
             sizes, colors, fig=fig, ax=ax)

    # Annotate rpt
    dx = 0; dy = 0.0
    plt.text(
        xx[-1][-1] + dx, yy[-1][-1] + dy, '$\phi={:.02f}$'.format(phi[-1]),
        **opt1)
    plt.text(
        xx[-1][0] + dx, yy[-1][0] + dy, '$\phi={:.02f}$'.format(phi[0]),
        **opt1)
    for i, _sw in enumerate(sw):
        plt.text(
            xx[i][-1] + dx,
            yy[i][-1] + dy, '$S_w={:.02f}$'.format(_sw),
            **opt2
        )

    plt.show()

if __name__ == '__main__':
    #t = np.linspace(2000, 6000, 6)
    #constants = [0, 1, 2]
    #sizes = np.linspace(100, 200, 6)
    #colors = np.array([np.ones(6)*100*(i+1) for i in range(3)])
    #plot_rpt(t, ex_rpt, constants, {'level': 0}, sizes, colors)
    test()


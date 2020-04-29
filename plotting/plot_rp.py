import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

import rp.rp_core as rp
from plotting import crossplot as xp

logger = logging.getLogger(__name__)
# Styles useful for textboxes
opt1 = {'bbox': {'facecolor': '0.9', 'alpha': 0.5, 'edgecolor': 'none'}}
opt2 = {'ha': 'right', 'bbox': {'facecolor': '0.9', 'alpha': 0.5, 'edgecolor': 'none'}}


def plot_rp(wells, logname_dict, wis, wi_name, cutoffs, templates=None,
            plot_type=None, ref_val=None, fig=None, ax=None, block_name='Logs', savefig=None, **kwargs):
    """
    Plots some standard rock physics crossplots for the given wells

    :param wells:
        dict
        dictionary with well names as keys, and core.well.Well object as values
        As returned from core.well.Project.load_all_wells()
    :param logname_dict:
        dict
        Dictionary of log type: log name key: value pairs to create statistics on
        The Vp, Vs, Rho and Phi logs are necessary for output to RokDoc compatible Sums & Average excel file
        E.G.
            logname_dict = {
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
        wis = utils.io.project_working_intervals(wp.project_table)
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
        templates dictionary as returned from utils.io.project_templates()
    :param plot_type:
        str
        plot_type =
            'AI-VpVs': AI versus Vp/Vs plot
            'Phi-Vp': Porosity versus Vp plot
            'I-G': Intercept versus Gradient plot
    :param ref_val:
        list
        List of reference Vp [m/s], Vs [m/s], and rho [g/cm3] that are used
        when calculating the Intercept and gradient
    :param fig:

    :param ax:

    :param block_name:
        str
        Name of the log block from where the logs are picked
    :param savefig
        str
        full pathname of file to save the plot to
    :return:
    """
    #
    # some initial setups
    if plot_type is None:
        plot_type = 'AI-VpVs'
    elif plot_type == 'I-G' and ref_val is None:
        ref_val = [3500., 1700., 2.6]
    logs = list(logname_dict.values())

    #
    # set up plotting environment
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    if ax is None:
        ax = fig.subplots()

    # load
    well_names = []
    desc = ''
    for wname, _well in wells.items():
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
        well.calc_mask(cutoffs, name='cmask', log_type_input=True)
        mask = well.block[block_name].masks['cmask'].data
        desc = well.block[block_name].masks['cmask'].header.desc

        # collect data for plot
        if plot_type == 'AI-VpVs':
            x_data = well.block[block_name].logs[logname_dict['P velocity']].data * \
                     well.block[block_name].logs[logname_dict['Density']].data
            x_unit = '{} {}'.format(well.block[block_name].logs[logname_dict['P velocity']].header.unit,
                                    well.block[block_name].logs[logname_dict['Density']].header.unit)
            y_data = well.block[block_name].logs[logname_dict['P velocity']].data / \
                well.block[block_name].logs[logname_dict['S velocity']].data
            y_unit = '-'
            xtempl = {'full_name': 'AI', 'unit': x_unit}
            ytempl = {'full_name': 'Vp/Vs', 'unit': y_unit}
        elif plot_type == 'Phi-Vp':
            x_data = well.block[block_name].logs[logname_dict['Porosity']].data
            x_unit = '{}'.format(well.block[block_name].logs[logname_dict['Porosity']].header.unit)
            y_data = well.block[block_name].logs[logname_dict['P velocity']].data
            y_unit = '{}'.format(well.block[block_name].logs[logname_dict['P velocity']].header.unit)
            xtempl = {'full_name': 'Porosity', 'unit': x_unit}
            ytempl = {'full_name': 'Vp', 'unit': y_unit}
        elif plot_type == 'I-G':
            x_data = rp.intercept(ref_val[0], well.block[block_name].logs[logname_dict['P velocity']].data,
                                  ref_val[2], well.block[block_name].logs[logname_dict['Density']].data)
            x_unit = '-'
            y_data = rp.gradient(ref_val[0], well.block[block_name].logs[logname_dict['P velocity']].data,
                                 ref_val[1], well.block[block_name].logs[logname_dict['S velocity']].data,
                                 ref_val[2], well.block[block_name].logs[logname_dict['Density']].data)
            y_unit = '-'
            xtempl = {'full_name': 'Intercept', 'unit': x_unit}
            ytempl = {'full_name': 'Gradient', 'unit': y_unit}
        else:
            raise IOError('No known plot type selected')

        well_names.append(wname)
        # start plotting

        xp.plot(
            x_data,
            y_data,
            cdata=templates[wname]['color'],
            mdata=templates[wname]['symbol'],
            xtempl=xtempl,
            ytempl=ytempl,
            mask=mask,
            fig=fig,
            ax=ax,
            **kwargs
        )
    ax.autoscale(True, axis='both')
    ax.set_title('{}, {}'.format(wi_name, desc))
    legend_elements = []
    for wname in well_names:
        legend_elements.append(
            Line2D([0], [0],
                   color=templates[wname]['color'],
                   lw=0, marker=templates[wname]['symbol'],
                   label=wname))

    # noinspection PyUnusedLocal
    this_legend = ax.legend(
        legend_elements,
        well_names,
        prop=FontProperties(size='smaller'),
        scatterpoints=1,
        markerscale=2,
        loc=1)

    if savefig is not None:
        fig.savefig(savefig)


def ex_rpt(t, c, **kw):
    return t, kw.pop('level', 7.) + np.log(min(t)) - np.log(t) + c


def plot_rpt(t, rpt, constants, rpt_keywords, sizes, colors, fig=None, ax=None, **kwargs):
    """
    Plot any RPT (rock physics template) that can be described by a function rpt(t), which can be
    evaluated at different values of a constant (eg. the saturation). E.G. rpt(x, const=constants[i])
    E.G. to plot a RPT which is a function of porosity in a Vp/Vs x AI crossplot:
        t = porosity
        x, y = rpt(t)  # x is AI, y is Vp/Vs

    :param ax:
    :param fig:
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
    """
    #
    # some initial setups
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
                    raise IOError('Shape of input must match constants, and x: ({}, {})'.format(len(constants), len(t)))
        if def_val == 90.:
            sizes = test_obj
        else:
            colors = test_obj

    #
    # set up plotting environment
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    if ax is None:
        ax = fig.subplots()

    # start drawing the RPT
    if not isinstance(colors, str):
        vmin = np.min(colors)
        vmax = np.max(colors)
        cmap = 'jet'
    else:
        vmin = None
        vmax = None
        cmap = None
    for i, const in enumerate(constants):
        x, y = rpt(t, const, **rpt_keywords)
        # First draw lines of the RPT
        ax.plot(
            x,
            y,
            lw=lw,
            c=tc,
            label='_nolegend_',
            **kwargs
        )
        # Next draw the points
        ax.scatter(
            x,
            y,
            c=colors if isinstance(colors, str) else colors[i],
            s=sizes[i] if isinstance(sizes, np.ndarray) else float(sizes),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            edgecolor=edgecolor,
            label='_nolegend_',
            **kwargs
        )
        if i == len(constants) - 1:
            plt.text(
                x[-1] + 150, y[-1] + .02, '$\phi={:.02f}$'.format(t[-1]),
                **opt1)
            plt.text(
                x[0] + 150, y[0] + .02, '$\phi={:.02f}$'.format(t[0]),
                **opt1)
        plt.text(
            x[-1] - 200,
            y[-1] - .015, '$S_w={:.02f}$'.format(const),
            **opt2
        )


def test():
    vsh = 0.6
    phic = 0.4
    cn = 8
    p = 45
    f = 0.3
    RHO_hc = 0.2
    K_hc = 0.06
    RHO_b = 1.1
    K_b = 2.8
    RHO_qz = 2.6
    K_qz = 37
    MU_qz = 44
    RHO_sh = 2.8
    K_sh = 15
    MU_sh = 5

    phi = np.linspace(0.1, phic, 6)
    sw = np.array([0., 0.5, 1.])
    sizes = np.empty((sw.size, phi.size))
    colors = np.array([np.ones(6) * 100 * (i + 1) ** 2 for i in range(3)]) / 900.
    # iterate over all phi values
    for i, val in enumerate(phi):
        sizes[:, i] = 10 + (40 * val) ** 2

    def rpt(_phi, _sw, _vsh=vsh, _phic=phic, _cn=cn, _p=p, _f=f,
            _rho_b=RHO_b, _k_b=K_b,
            _rho_hc=RHO_hc, _k_hc=K_hc,
            _rho_qz=RHO_qz, _k_qz=K_qz, _mu_qz=MU_qz,
            _rho_sh=RHO_sh, _k_sh=K_sh, _mu_sh=MU_sh):
        print('phic: {}, Cn: {}, P: {}, f: {}'.format(_phic, _cn, _p, _f))

        k_min = rp.vrh_bounds([_vsh, 1 - _vsh], [_k_sh, _k_qz])[2]  # Mineral bulk modulus
        print('k_min: {}'.format(k_min))
        mu_min = rp.vrh_bounds([_vsh, 1 - _vsh], [_mu_sh, _mu_qz])[2]  # Mineral shear modulus
        print('mu_min: {}'.format(mu_min))
        rho_min = rp.vrh_bounds([_vsh, 1 - _vsh], [_rho_sh, _rho_qz])[0]  # Density of minerals
        print('rho_min: {}'.format(rho_min))

        # Apply the RPT on the minerals
        k_dry, mu_dry = rp.stiffsand(k_min, mu_min, _phi, _phic, _cn, _p, _f)
        print('k_dry: {}, mu_dry: {}'.format(k_dry, mu_dry))

        k_f2 = rp.vrh_bounds([_sw, 1. - _sw], [_k_b, _k_hc])[1]  # Bulk modulus of final fluid
        rho_f2 = rp.vrh_bounds([_sw, 1. - _sw], [_rho_b, _rho_hc])[0]  # Density of final fluid
        print('sw: {}, K_f: {}, RHO_f: {}'.format(_sw, k_f2, rho_f2))

        vp_2, vs_2, rho_2, k_2 = rp.vels(k_dry, mu_dry, k_min, rho_min, k_f2, rho_f2, _phi)
        return rho_2 * vp_2, vp_2 / vs_2
        # return _phi, _vp2

    # wp = Project()
    # logname_dict = {'P velocity': 'vp_dry', 'S velocity': 'vs_dry', 'Density': 'rho_dry', 'Porosity': 'phie',
    #                'Volume': 'vcl'}
    # wis = uio.project_working_intervals(wp.project_table)
    # templates = uio.project_templates(wp.project_table)
    # cutoffs = {'Volume': ['<', 0.4], 'Porosity': ['>', 0.1]}
    # wells = wp.load_all_wells()

    fig, ax = plt.subplots()
    # plot_rp(wells, logname_dict, wis, 'SAND E', cutoffs, templates, fig=fig, ax=ax)
    plot_rpt(phi, rpt, sw, {}, sizes, colors, fig=fig, ax=ax)
    plt.show()


if __name__ == '__main__':
    # t = np.linspace(2000, 6000, 6)
    # constants = [0, 1, 2]
    # sizes = np.linspace(100, 200, 6)
    # colors = np.array([np.ones(6)*100*(i+1) for i in range(3)])
    # plot_rpt(t, ex_rpt, constants, {'level': 0}, sizes, colors)
    test()

import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from copy import deepcopy
import logging

from plotting import crossplot as xp

logger = logging.getLogger(__name__)

def plot_rp(wells, logname_dict, wis, wi_name, cutoffs, templates=None,
            plot_type=None, fig=None, ax=None, block_name='Logs', savefig=None, **kwargs):

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
        plot_type = 'AI-VpVs': AI versus Vp/Vs plot
    :param fig:

    :param ax:

    :param block_name:
        str
        Name of the log block from where the logs are picked
    :param savefig
        str
        full pathname of file to save the plot to
    :param **kwargs:
        keyword arguments passed on to crossplot.plot()
    :return:
    """
    #
    # some initial setups
    if plot_type is None:
        plot_type = 'AI-VpVs'
    log_types = list(logname_dict.keys())
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
        x_data = well.block[block_name].logs[logname_dict['P velocity']].data * \
                 well.block[block_name].logs[logname_dict['Density']].data
        x_unit = '{} {}'.format(
            well.block[block_name].logs[logname_dict['P velocity']].header.unit,
                 well.block[block_name].logs[logname_dict['Density']].header.unit)
        y_data = well.block[block_name].logs[logname_dict['P velocity']].data / \
                 well.block[block_name].logs[logname_dict['S velocity']].data
        y_unit = '-'

        well_names.append(wname)
        # start plotting

        xp.plot(
            x_data,
            y_data,
            cdata=templates[wname]['color'],
            mdata=templates[wname]['symbol'],
            xtempl={'full_name': 'AI',
                    'unit': x_unit},
            ytempl={'full_name': 'Vp/Vs',
                    'unit': y_unit},
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

    this_legend = ax.legend(
        legend_elements,
        well_names,
        prop=FontProperties(size='smaller'),
        scatterpoints=1,
        markerscale=2,
        loc=1)

    if savefig is not None:
        fig.savefig(savefig)


def ex_rpt(x, c, **kw):
    return kw.pop('level', 7.)+np.log(min(x)) - np.log(x) + c


def plot_rpt(x, rpt, rpt_keywords, sizes, colors, constants, fig=None, ax=None, **kwargs):
    """
    Plot any RPT (rock physics template) that can be described by a function rpt(x), which can be
    parameterized by a constant. E.G. rpt(x, const=constants[i])

    :param x:
        np.array
        array of length N
        x values used to draw the rockphysics template, preferably less than about 10 items long for creating
        nice plots
    :param rpt:
        function
        Rock physics template function of x
        Should take a second argument which is used to parameterize the function
        e.g.
        def rpt(x, c, **rpt_keywords):
            return c*x + rpt_keywords.pop('zero_crossing', 0)
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
    :param constants:
        list
        list of length M of constants used to parametrize the rpt function
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
                print(test_obj.shape)
            elif len(test_obj.shape) == 2:
                if not test_obj.shape == (len(constants), len(x)):
                    raise IOError('Shape of input must match constants, and x: ({}, {})'.format(len(constants), len(x)))
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
    for i, const in enumerate(constants):
        # First draw lines of the RPT
        ax.plot(
            x,
            rpt(x, const, **rpt_keywords),
            lw=lw,
            c=tc,
            label='_nolegend_',
            **kwargs
        )
        # Next draw the points
        ax.scatter(
            x,
            rpt(x, const, **rpt_keywords),
            c=colors if isinstance(colors, str) else colors[i],
            s=sizes[i] if isinstance(sizes, np.ndarray) else float(sizes),
            edgecolor=edgecolor,
            label='_nolegend_',
            **kwargs
        )


if __name__ == '__main__':
    x = np.linspace(2000, 6000, 6)
    constants = [0, 1, 2]
    #colors = np.linspace(100, 200, 6)
    colors = 'blue'
    #sizes = np.array([np.ones(6)*100*(i+1) for i in range(3)])
    sizes = 800.
    plot_rpt(x, ex_rpt, {'level': 0}, sizes, colors, constants)

    plt.show()

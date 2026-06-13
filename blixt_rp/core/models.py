import matplotlib.pyplot as plt
import matplotlib.lines as m_lines
import numpy as np
import sys
import os
import logging
import types
from copy import deepcopy
import pint
from bokeh.models import ColumnDataSource
from scipy.stats import theilslopes
from typing import Callable

from .log_curve_new import LogCurve, Depth
from ..plotting.log_plotter import add_color_amp
from ..plotting.plot_logs_new import get_wiggles_in_depth
from .core import LithoFluid, LithoFluidsTable, LithoFluids

# sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')
# Instead of sys.path.append load all PyCharm projects in PyCharm, and they gets added to the sys.path automatically

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.plotting.helpers import axis_plot, wiggle_plot, wavelet_plot
from blixt_utils.plotting.crossplot import cnames
import blixt_rp.rp.rp_core as rp
import blixt_utils.misc.wavelets as bumw
from blixt_utils.utils import print_info
from blixt_rp import Q_

logger = logging.getLogger(__name__)

text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}

global_base_case_name = 'Base'

min_amp = 1E6
max_amp = 1E-6

def plot_quasi_2d(model, ax=None, **kwargs):
    show = False
    new_ax = False
    if ax is None:
        # fig, axs = plt.subplots(1, len(model.trace_index_range), figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))
        show = True
        new_ax = True

    # for i, trace_i in enumerate(model.trace_index_range):
        # plot_1d(model, ax=axs[i], index=trace_i, legend=i == 0, yticks=i == 0)
    i_range = model.trace_index_range
    if model.domain == 'TWT':
        last_layer_depth = [model.depth_to_top.to('ms').magnitude]*len(i_range)
    elif model.domain == 'Z':
        last_layer_depth = [model.depth_to_top.to('m').magnitude]*len(i_range)
    else:
        last_layer_depth = None
        print_info('Domain of model is not set properly ({}). Should "TWT" or "Z"'.format(model.domain),
                   'error', logger, 'IOError')
    ax.plot(i_range, last_layer_depth, **kwargs)
    for _layer in model.layers:
        if isinstance(_layer.thickness, pint.Quantity):
            if model.domain == 'TWT':
                _this_thick = _layer.thickness.to('ms').magnitude
            else:
                _this_thick = _layer.thickness.to('m').magnitude
            _h = np.array([_this_thick for _i in i_range])
            _tmp = np.array([last_layer_depth[_i] + _this_thick for _i in i_range])
        else:  # thickness is a function
            if model.domain == 'TWT':
                _this_thick = [_layer.thickness(_i).to('ms').magnitude for _i in i_range]
            else:
                _this_thick = [_layer.thickness(_i).to('m').magnitude for _i in i_range]
            _h = np.array([_this_thick[_i] for _i in i_range])
            _tmp = np.array([last_layer_depth[_i] + _this_thick[_i] for _i in i_range])
        ax.plot(i_range[_h > 0], _tmp[_h > 0], **kwargs)
        last_layer_depth = _tmp
    if new_ax:
        ax.set_ylim(ax.get_ylim()[::-1])
    if show:
        plt.show()


def plot_1d(model, ax=None, index=0, legend=True, yticks=True):
    """
    :param model:
        Model object
    :param ax:
        matplotlib.Axes
        Use this axes to plot the model, if None, a new figure and axes is created
    """
    show = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True

    # extract the elastic properties from the model
    vps, vss, rhos = [], [], []
    target, ntg, gross_vp, gross_vs, gross_rho = [], [], [], [], []
    for layer in model.layers:
        if isinstance(layer.vp, types.FunctionType):
            vps.append(layer.vp(index))
            # print(layer.vp(index))
        else:
            vps.append(layer.vp)

        if isinstance(layer.vs, types.FunctionType):
            vss.append(layer.vs(index))
        else:
            vss.append(layer.vs)

        if isinstance(layer.rho, types.FunctionType):
            rhos.append(layer.rho(index))
        else:
            rhos.append(layer.rho)
        target.append(layer.target)

        if isinstance(layer.ntg, types.FunctionType):
            layer_ntg = layer.ntg(index)
        else:
            layer_ntg = layer.ntg
        ntg.append(layer_ntg)
        if layer_ntg < 1:
            gross_vp.append(layer.gross_vp)
            gross_vs.append(layer.gross_vs)
            gross_rho.append(layer.gross_rho)
        else:
            gross_vp.append(0.)
            gross_vs.append(0.)
            gross_rho.append(0.)

    linestyle_ai = {'lw': 1, 'color': 'k', 'ls': '-'}
    linestyle_vpvs = {'lw': 1, 'color': 'k', 'ls': '--'}
    ai_line = m_lines.Line2D([1], [1], **linestyle_ai)
    vpvs_line = m_lines.Line2D([1], [1], **linestyle_vpvs)

    # the boundaries (bwb) need to add up the respective thicknesses
    bwb = [model.depth_to_top]
    for i in range(len(model)):
        last_top = bwb[i]
        if isinstance(model.layers[i].thickness, types.FunctionType):
            bwb.append(last_top + model.layers[i].thickness(index))
        else:
            bwb.append(last_top + model.layers[i].thickness)

    # realize the model to create data
    twt, layer_index, vp, vs, rho, z = model.realize_model(0.001)
    if model.domain == 'TWT':
        y = twt
    else:
        y = z

    if model.model_type == 'quasi 2D':
        data_ai = vp[index, :] * rho[index, :]
        data_vpvs = vp[index, :] / vs[index, :]
        _vp = vp[index, :]
        _layer_index = layer_index[index, :]
        # print(_vp.min(), _vp.max())
    else:
        data_ai = vp * rho
        data_vpvs = vp / vs
        _vp = vp
        _layer_index = layer_index
    max_ai = max(data_ai)
    data_ai = data_ai / max_ai
    max_vpvs = max(data_vpvs)
    data_vpvs = 0.9 * data_vpvs / max_vpvs

    # create a discrete data set which controls the filling
    # 0: no fill, 1: net reservoir (sand), 2: gross reservoir (shale)
    filler = np.zeros(len(_vp))
    # Step through each layer and search for areas where NTG < 1 and
    # apply the filler there
    for i, layer in enumerate(model.layers):
        if isinstance(layer.vp, types.FunctionType):
            layer_vp = layer.vp(index)
        else:
            layer_vp = layer.vp
        if isinstance(layer.ntg, types.FunctionType):
            layer_ntg = layer.ntg(index)
        else:
            layer_ntg = layer.ntg

        if layer.target and (layer_ntg < 1):
            filler[
                np.array([all(xx) for xx in zip(_layer_index == i, _vp == layer_vp)])
            ] = 1.
            filler[
                np.array([all(xx) for xx in zip(_layer_index == i, _vp == layer.gross_vp)])
            ] = 2.
        elif layer.target:
            filler[
                np.array([all(xx) for xx in zip(_layer_index == i, _vp == layer_vp)])
            ] = 1.

    axis_plot(ax, y, [data_ai], [[0., 1.1]], [linestyle_ai], nxt=0)
    axis_plot(ax, y, [data_vpvs], [[0., 1.1]], [linestyle_vpvs], nxt=0)
    ax.fill_betweenx(y, data_ai, where=(filler == 1.), color='y', alpha=0.3)
    ax.fill_betweenx(y, data_ai, where=(filler == 2.), color='grey', alpha=0.4)

    for i, _y in enumerate(bwb[1:-1]):
        #ax.plot([0., ais[i+1]/max_ai], [_y, _y], **linestyle_ai)
        ax.plot([0., vps[i+1] * rhos[i+1]/max_ai], [_y, _y], **linestyle_ai)

    i = 0
    for _vp, _vs, _rho in zip(vps, vss, rhos):
        info_txt = '{}.\nVp: {:.2f}\nVs: {:.2f}\nRho: {:.2f}'.format(i+1, _vp/1000., _vs/1000., _rho)
        ax.text(0.5*_vp * _rho/max_ai, 0.5*(bwb[i] + bwb[i+1]), info_txt, ha='right', va='center', **text_style)
        i += 1
    i = 0
    for _gvp, _gvs, _grho in zip(gross_vp, gross_vs, gross_rho):
        if target[i] and (ntg[i] < 1):
            info_txt = 'NTG: {:.1f}.\nVp: {:.2f}\nVs: {:.2f}\nRho: {:.2f}'.format(ntg[i], _gvp/1000., _gvs/1000., _grho)
            ax.text(0.6*vps[i] * rhos[i]/max_ai, 0.5*(bwb[i] + bwb[i+1]), info_txt, ha='left', va='center', **text_style)
        i += 1

    ax.set_ylim(ax.get_ylim()[::-1])
    if legend:
        ax.legend([ai_line, vpvs_line], ['AI', 'Vp/Vs'])
    if not yticks:
        ax.get_yaxis().set_ticklabels([])
        ax.tick_params(axis='y', length=0)
    else:
        if model.domain == 'TWT':
            ax.set_ylabel('TWT [s]')
        else:
            # TODO
            # The realization returns the twt
            # ax.set_ylabel('TWT [s]')
            ax.set_ylabel('Z [m]')

    if show:
        plt.show()


def plot_wiggles(model, sample_rate, wavelet, angle=0., eei=False, ax=None, color_by_gradient=False,
                 extract_avo_at=None, avo_angles=None, avo_plot_position=None,
                 extract_amp_at=None, plot_domain=None, overburden_vel=3000.,
                 extract_on='exact', scaling=None, **kwargs):
    """

    Args:
        model:
        sample_rate:
            Always in seconds!
        wavelet:
            dict
            dictionary with three keys:
                'wavelet': contains the wavelet amplitude
                'time': contains the time data [s]
                'header': a dictionary with info about the wavelet
            see blixt_utils.io.io.read_petrel_wavelet() for example
        angle:
            incident angle theta in degrees
        eei:
            Bool
            If true it plots the seismic trace at a given chi angle (Extended Elastic Impedance) instead of
            incidence angle. So the parameter angle is in this case interpreted as the chi angle (deg), which
            should be between -90 and 90 deg.
        ax:
        color_by_gradient:
            Bool
            If True, use the polarity of the gradient to color the wiggle, instead of the polarity of the amplitude
        extract_avo_at:
            two-tuple, or list of two-tuples
            Each two tuple contains the x (index number) and y (TWT [s] or Z [m]) coordinates of where to extract avo curves

        avo_angles:
            list
            List of offset angles to use in plots
            Used quite differently for quasi-2D and 1D plots
            AND in 1D models, when eei is True, these angles are taken to be Chi angles
        avo_plot_position
        extract_amp_at:
            float, or list of floats
            depth (TWT or Z) values at which the seismic amplitude is extracted.
            Same length as number of traces in model.
            If given as a single float, it is repeated to yield a list
        extract_on:
            str
            'exact', 'nearest_min', 'nearest_max'
            Used by the function find_value() to extract amplitude values
        scaling:
            float
            Scaling parameter passed on to wiggle_plot
        kwargs
            keyword arguments passed on to wiggle_plot
    Returns:
        avo_curves, extracted_amplitudes, minimum amplitude, maximum amplitude, distance between min & max
    """
    # TODO
    # We can make the 'distance between min & max' more robust, so that it works more correctly for "non-symmetric"
    # wedge models
    from blixt_utils.utils import find_value

    if avo_plot_position is None:
        avo_plot_position = [0.68, 0.02, 0.3, 0.3]
    if angle is None:
        angle = 0
    extracted_amplitudes = None
    min_amp = None
    max_amp = None
    apparent_thickness = None
    if extract_amp_at is not None:
        extracted_amplitudes = []
    if extract_avo_at is not None and avo_angles is None:
        avo_angles = [0, 10, 20, 30, 40]

    grad = None
    show = False
    tmp_avo_angles = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True

    avo_curves = None
    avo_positions = None
    if (extract_avo_at is not None) or isinstance(extract_avo_at, tuple) or isinstance(extract_avo_at, list):
        avo_curves = {}
        avo_positions = {}
        if isinstance(extract_avo_at, tuple):
            avo_curves[0] = []
            avo_positions[0] = extract_avo_at
        elif isinstance(extract_avo_at, list):
            for _i, _t in enumerate(extract_avo_at):
                avo_curves[_i] = []
                avo_positions[_i] = _t

    if plot_domain is None:
        plot_domain = 'TWT'

    twt, layer_i, vp, vs, rho, z = model.realize_model(sample_rate, overburden_vel=overburden_vel)

    if model.model_type == 'quasi 2D' and model.trace_index_range is not None:
        min_amp = []
        max_amp = []
        apparent_thickness = {'twt': [], 'z': []}
        if avo_angles is None:
            avo_angles = [0, 10, 20, 30, 40]

        for i, trace_i in enumerate(model.trace_index_range):
            calculate_here = False
            this_index = False

            ref = rp.reflectivity(
                vp[trace_i, :], None, vs[trace_i, :], None, rho[trace_i, :], None, eei=eei, along_wiggle=True
            )
            if color_by_gradient:
                grad = rp.gradient(
                    vp[trace_i, :], None, vs[trace_i, :], None, rho[trace_i, :], None, along_wiggle=True
                )
            else:
                grad = None
            # wiggle = calc_wiggle(len(twt), wavelet, ref, angle)
            wiggle = bumw.convolve_with_refl(wavelet['wavelet'], ref(angle))
            # print('plot_wiggles: {}, {}'.format(np.min(wiggle), np.max(wiggle)))
            min_amp.append(np.min(wiggle))
            max_amp.append(np.max(wiggle))
            apparent_thickness['twt'].append(np.abs(twt[wiggle.argmin()] - twt[wiggle.argmax()]))
            apparent_thickness['z'].append(np.abs(z[i, wiggle.argmin()] - z[i, wiggle.argmax()]))

            if extract_amp_at is not None:
                if isinstance(extract_amp_at, float):
                    twt_extract = extract_amp_at
                elif isinstance(extract_avo_at, list):
                    twt_extract = extract_amp_at[i]
                else:
                    raise IOError('extract_amp_at must be either float or list of floats')
                extracted_amplitudes.append(wiggle[np.argmin((twt - twt_extract)**2)])

            if plot_domain == 'TWT':
                wiggle_plot(ax, twt, wiggle, i, scaling=scaling, color_by_gradient=grad, **kwargs)
            else:
                wiggle_plot(ax, z[0, :], wiggle, i, scaling=scaling, color_by_gradient=grad, **kwargs)

            # extract avo curves
            if avo_positions is not None:
                for _i, _t in list(avo_positions.items()):
                    # print('XXX2', _i, _t[0], trace_i, avo_positions[_i])
                    if trace_i == _t[0]:
                        calculate_here = True
                        this_index = _i
                if calculate_here:
                    if eei:  # if EEI is true, then we need to calculate the normal reflectivity
                        ref = rp.reflectivity(
                            vp[trace_i, :], None, vs[trace_i, :], None, rho[trace_i, :], None, eei=False,
                            along_wiggle=True
                        )
                    twt_index = np.argmin((twt - avo_positions[this_index][1])**2)
                    for _ang in avo_angles:
                        # tmp_wiggle = calc_wiggle(len(twt), wavelet, ref, _ang)
                        tmp_wiggle = bumw.convolve_with_refl(wavelet['wavelet'], ref(_ang))
                        wiggle_value, twt_index = find_value(tmp_wiggle, twt_index, snap_to=extract_on)
                        avo_curves[this_index].append(wiggle_value)
                    # avo_positions[this_index] = (this_index, twt[twt_index])
                    avo_positions[this_index] = (trace_i, twt[twt_index])

    elif model.model_type == '1D' and avo_angles is not None:
        if eei:
            my_x_label = 'Chi angle [deg]'
        else:
            my_x_label = 'Incident angle [deg]'
        ref = rp.reflectivity(vp, None, vs, None, rho, None, eei=eei, along_wiggle=True)
        if color_by_gradient:
            grad = rp.gradient(vp, None, vs, None, rho, None, along_wiggle=True)
        else:
            grad = None

        for ang in avo_angles:
            calculate_here = False
            this_index = False

            # wiggle = calc_wiggle(len(twt), wavelet, ref, ang)
            wiggle = bumw.convolve_with_refl(wavelet['wavelet'], ref(ang))

            wiggle_plot(ax, twt, wiggle, ang, scaling=80, color_by_gradient=grad, **kwargs)
            # wiggle_plot(ax, twt, wiggle, ang, scaling=1E-4, color_by_gradient=grad, **kwargs)

            # extract avo curves
            if avo_positions is not None:
                tmp_avo_angles = [0, 10, 20, 30, 40.]
                for _i, _t in list(avo_positions.items()):
                    if ang == _t[0]:
                        calculate_here = True
                        this_index = _i
                if calculate_here:
                    if eei:  # if EEI is true, then we need to calculate the normal reflectivity
                        tmp_ref = rp.reflectivity(
                            vp, None, vs, None, rho, None, eei=False,
                            along_wiggle=True
                        )
                    else:
                        tmp_ref = ref
                    twt_index = np.argmin((twt - avo_positions[this_index][1])**2)
                    for _ang in tmp_avo_angles:
                        # tmp_wiggle = calc_wiggle(len(twt), wavelet, tmp_ref, _ang)
                        tmp_wiggle = bumw.convolve_with_refl(wavelet['wavelet'], tmp_ref(_ang))
                        wiggle_value, twt_index = find_value(tmp_wiggle, twt_index, snap_to=extract_on)
                        # avo_curves[this_index].append(tmp_wiggle[twt_index - 1])
                        avo_curves[this_index].append(wiggle_value)
                    avo_positions[this_index] = (this_index, twt[twt_index])

        ax.set_xlabel(my_x_label)

    elif model.model_type == '1D' and avo_angles is None:
        ref = rp.reflectivity(vp, None, vs, None, rho, None, eei=eei, along_wiggle=True)
        wiggle = bumw.convolve_with_refl(wavelet['wavelet'], ref(0.))
        wiggle_plot(ax, twt, wiggle, 0., scaling=80,  **kwargs)
    else:
        warn_txt = 'Not possible to plot these wiggles'
        print(warn_txt)

    # Add extra information to plots
    if model.model_type == 'quasi 2D' and model.trace_index_range is not None:
        if eei:
            info_txt = r'EEI at $\chi$={}$^\circ$'.format(angle)
            ax.text(0.05, 0.95, info_txt, ha='left', va='top', transform=ax.transAxes, **text_style)
        else:
            if angle > 0:
                info_txt = r'Amp. at $\theta$={}$^\circ$'.format(angle)
                ax.text(0.05, 0.95, info_txt, ha='left', va='top', transform=ax.transAxes, **text_style)

    if avo_curves is not None:
        avo_ax = ax.inset_axes(avo_plot_position)

        for _i, avo in list(avo_curves.items()):
            if plot_domain == 'TWT':
                ax.annotate('{}'.format(_i + 1), avo_positions[_i], bbox={'boxstyle': 'circle', 'color': cnames[_i]})
                print('XXX2', avo_positions[_i])
                # ax.annotate('{}'.format(_i + 1), (10, 2.), bbox={'boxstyle': 'circle', 'color': cnames[_i]})
            else:
                # TODO
                # The Z position is not entirely correct, as we have hard-coded it to use the first
                # vector of the Z array
                twt_index = np.argmin((twt - avo_positions[_i][1]) ** 2)
                ax.annotate('{}'.format(_i + 1),
                            (avo_positions[_i][0], z[0, twt_index]), bbox={'boxstyle': 'circle', 'color': cnames[_i]})
            if tmp_avo_angles:
                avo_ax.plot(tmp_avo_angles, avo, c=cnames[_i], label='{}'.format(_i + 1))
            else:
                # print('XXX1', avo_angles, avo)
                avo_ax.plot(avo_angles, avo, c=cnames[_i], label='{}'.format(_i + 1))
        # avo_ax.legend()
        avo_ax.tick_params(direction='in', labelsize='small')
        avo_ax.tick_params(axis='x', pad=-15)
        avo_ax.set_xlabel('Incident angle', fontsize=8, backgroundcolor='w')
        avo_ax.set_ylabel('Reflectivity', fontsize=8, backgroundcolor='w')
        avo_ax.grid(axis='y')

    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.grid(axis='y')

    show_wavelet = True
    if show_wavelet:
        # length of wavelet relative to model
        wf = len(wavelet['time']) / len(twt)  # assuming the have the same sample rate - which they should!
        if wf > 0.25:
            mini_ax = ax.inset_axes([0.82, 0.8, 0.16, 0.18])
            wavelet_plot(mini_ax, wavelet['time'], wavelet['wavelet'], orientation='down', show_ticks=True)
        else:
            mini_ax = ax.inset_axes([0.8, 0.8, 0.8 * wf, wf])
            wavelet_plot(mini_ax, wavelet['time'], wavelet['wavelet'], orientation='down', show_ticks=False)
        mini_ax.tick_params(labelsize='small')

    plot_outline = False
    if plot_outline:
        plot_quasi_2d(model, ax=ax, c='k', ls='--', lw=0.5)

    if show:
        plt.show()

    return avo_curves, extracted_amplitudes, min_amp, max_amp, apparent_thickness


class Layer:
    """
    Class handling one layer of a model
    """
    from typing import Callable

    def __init__(self,
                 name: str | None = None,
                 case: str | None = None,
                 color: str | None  = '#D9D9D9',
                 thickness: Q_ | Callable[[int], Q_] |  None = None,
                 litho_fluid: LithoFluid | None = None,
                 ntg=None,
                 **kwargs
                 ):
        """
        :param name:
            str
            Optional
            Name of the layer. Should for example indicate stratigraphy.
            Two layers can have the same name, e.g. "Lange SST", but must then represent two
            different cases (see case parameter)
        :param case:
            str
            Optional
            The case name, e.g. "Base" or "Heavy oil", "Brine", ...
            If not provided, it is given the global base case name
        :param color:
            Optional
            String that gives the color of that layer
        :param thickness:
            pint Quantity or function that returns a pint Quantity
            Thickness of layer in m or s, depending on which domain the model.
            If thickness is a function, it is parametrized by an integer i.
            e.g. thickness = f, where f is a function like:
              def f(i):
                  return 10 + i * 0.5
        :param litho_fluid:
            LithoFluid with vp, vs, and rho as either pint.Quantities or functions that return pint Quantities
            if a function, it needs to be parametrized by an integer i
            Say we use the constantcement function to calculate vp where the porosity is assumed to change:
              def vp(i):
                  from blixt_rp.rp.rp_core import constantcement, v_p
                  phi = 0.1 + i * 0.03
                  k_eff, mu_eff = constantcement(37, 45, phi)
                  return v_p(k_eff, mu_eff, 2.6)

        :param ntg:
            float or function
            net-to-gross, 0 <= ntg >= 1
        :param layer_type:
            str
            '1D'
                default
            'quasi 2D'
                Layer properties can change "laterally" according to the functions provided
        :param target:
            bool
            True for target layers
            Not possible to set.
            The combination of name and case determines if it is a target
            A target layer must come in a pair (within a model) with the layer (must have the same name)
            that represents the Base case
        :param gross_litho_fluid:
            elastic parameters in the non-reservoir part (gross) of the layer
        :param thin_bed_factor:
            int
            Number of internal layers in the reservoir (net) part of the layer
            If 1, the net portion is taken up by one homogeneous layer
        """
        # get the required parameters
        self.resolution = None
        self.layer_type = '1D'
        if isinstance(thickness, Callable) or isinstance(ntg, Callable) or litho_fluid.type == 'quasi 2D':
            self.layer_type = 'quasi 2D'

        if thickness is None:
            thickness = Q_(25., 'm')
        domain = check_domain(thickness)
        self.thickness = thickness
        self.domain = domain
        self.name = name
        if case is None:
            case = global_base_case_name
        # Target layers are duplicated layers but with a different case name
        if case != global_base_case_name:
            self.target = True
        else:
            self.target = False
        if self.target and self.name is None:
            print_info('Both name ({}) and case ({}) must be set to have a valid target layer'.format(
                name, case ),
                'error', logger, 'IOError')
        self.case = case
        self.color = color

        if litho_fluid is None:
            self.litho_fluid = LithoFluid(default='brine_sst')
        else:
            self.litho_fluid = litho_fluid

        self.vp = self.litho_fluid.vp
        self.vs = self.litho_fluid.vs
        self.rho = self.litho_fluid.rho

        if ntg is None:
            self.ntg = 1.
        elif (ntg != 1. or isinstance(ntg, Callable)) and not self.target:
            print_info('Only target layers can have a NTG separate from 1.', 'error', logger, 'IOError')
        else:
            self.ntg = ntg

        if not isinstance(ntg, Callable) and not (0 <= self.ntg <= 1):
            raise ValueError('NTG must be between 0 and 1')

        # Get the net-to-gross related keyword arguments
        self.gross_litho_fluid = None
        self.thin_bed_factor = None
        if self.ntg < 1 or isinstance(self.ntg, Callable):
            self.gross_litho_fluid = kwargs.pop('gross_litho_fluid', LithoFluid(default='shale'))
            self.gross_vp = self.gross_litho_fluid.vp
            self.gross_vs = self.gross_litho_fluid.vs
            self.gross_rho = self.gross_litho_fluid.rho
            self.thin_bed_factor = kwargs.pop('thin_bed_factor', 3)

        # for arg in [self.thickness, self.vp, self.vs, self.rho, self.ntg]:
        #     if isinstance(arg, Callable):
        #         self.layer_type = 'quasi 2D'

    # @property
    # def thickness(self):
    #     return self._thickness

    # @thickness.setter
    # def thickness(self, new_thickness):
    #     self.domain = check_domain(new_thickness)
    #     self._thickness = new_thickness

    def realize_layer(self, resolution: Q_, voigt_reuss_hill=False, index=0):
        """
        Realize the current layer by returning arrays of the elastic properties with the given resolution

        :param resolution:
            float
            resolution in TWT [s]
        :param voigt_reuss_hill:
            bool
            If true, the Voigt Reuss Hill average for the given NTG is used when returning the elastic parameters
        :param index:
            int
            Integer that specifies the index of "laterally" varying quasi 2D layer
            e.g. phi = 0.05 + index * 0.05

        """
        self.resolution = resolution

        if isinstance(self.thickness, types.FunctionType):
            layer_thickness = self.thickness(index)
        else:
            layer_thickness = self.thickness

        if isinstance(self.vp, types.FunctionType):
            layer_vp = self.vp(index)
            # print(layer_vp)
        else:
            layer_vp = self.vp

        # # Let the domain decide how the layer is realized
        # if self.domain == 'TWT':
        #     n = int(layer_thickness / resolution)
        # else:
        #     twt_thickness = 2. * layer_thickness / layer_vp
        #     n = int(twt_thickness / resolution)
        n = int(layer_thickness.to(resolution.units) / resolution)

        if isinstance(self.vs, types.FunctionType):
            layer_vs = self.vs(index)
        else:
            layer_vs = self.vs

        if isinstance(self.rho, types.FunctionType):
            layer_rho = self.rho(index)
        else:
            layer_rho = self.rho

        if isinstance(self.ntg, types.FunctionType):
            layer_ntg = self.ntg(index)
        else:
            layer_ntg = self.ntg

        if layer_ntg < 1. and self.target and not voigt_reuss_hill:
            net_group_size = int(np.ceil(layer_ntg * layer_thickness / (self.thin_bed_factor * resolution)))
            gross_group_size = int(np.ceil((1. - layer_ntg) * layer_thickness / (self.thin_bed_factor * resolution)))
            this_vp = []
            this_vs = []
            this_rho = []
            i = 0
            while len(this_vp) <= n:
                #print(n, len(this_vp), net_group_size, gross_group_size)
                if i <= self.thin_bed_factor - 1:
                    this_vp += [layer_vp] * net_group_size
                    this_vp += [self.gross_vp] * gross_group_size
                    this_vs += [layer_vs] * net_group_size
                    this_vs += [self.gross_vs] * gross_group_size
                    this_rho += [layer_rho] * net_group_size
                    this_rho += [self.gross_rho] * gross_group_size
                else:  # only add gross values towards the end. This may make the final NTG wrong, but ensures # layers
                    this_vp += [self.gross_vp] * gross_group_size
                    this_vs += [self.gross_vs] * gross_group_size
                    this_rho += [self.gross_rho] * gross_group_size
                i += 1
            ## Cut, convert and remove unnecessary single values at the edge
            #this_vp = np.array(this_vp)[:n]
            #this_vp[-1] = this_vp[-2]
            #this_vs = np.array(this_vs)[:n]
            #this_vs[-1] = this_vs[-2]
            #this_rho = np.array(this_rho)[:n]
            #this_rho[-1] = this_rho[-2]
        elif layer_ntg < 1. and self.target and voigt_reuss_hill:
            _, _, vp_vrh = rp.vrh_bounds([layer_ntg, 1. - layer_ntg], [layer_vp, self.gross_vp])
            _, _, vs_vrh = rp.vrh_bounds([layer_ntg, 1. - layer_ntg], [layer_vs, self.gross_vs])
            _, _, rho_vrh = rp.vrh_bounds([layer_ntg, 1. - layer_ntg], [layer_rho, self.gross_rho])
            this_vp = np.ones(n) * vp_vrh
            this_vs = np.ones(n) * vs_vrh
            this_rho = np.ones(n) * rho_vrh
        else:
            this_vp = np.ones(n) * layer_vp
            this_vs = np.ones(n) * layer_vs
            this_rho = np.ones(n) * layer_rho

        return np.array(this_vp), np.array(this_vs), np.array(this_rho)


class Model:
    """
    Object that holds a subsurface model with N layers
    """

    def __init__(self,
                 depth_to_top: Q_ | None = None,
                 layers: list | None = None,
                 trace_index_range=None,
                 name: str | None = None):
        """
        :param model_type:
            str
            '1D'
                default
            'quasi 2D'
                Layer properties can change "laterally" according to functions provided to the individual layers
        :param depth_to_top:
            pint Quantity
            depth to top in seconds or meters
        :param layers:
            list of layers, each a Layer object
            The first layer is at the top, and consequent layers beneath
            Target layers must come together (after each other in this list) with the layer that represents the base
            case. And it is the name and case name of the layers that determine if it is a target
            E.G. the following list of four layers represent a three layer model, with a base and oil case
            [
                layer1 (name = 'Top', case='Base'),
                layer2 (name = 'Reservoir', case='Base'),
                layer3 (name = 'Reservoir', case='Oil'),
                layer4 (name = 'Bottom', case='Base')
            ]
            NOTE that the "case='Base'" is optional. If no case name is given, it is by default set to 'Base'

        :param trace_index_range:
            list like object of integers
            used when some layers are quasi 2D layers, to parameterize the "lateral" variation
            eg. np.arange(10)
        :param name:
            str
            Optional string which we can use to add a title to the model
        """

        # set the required parameters
        if layers is None:
            layers = []

        self.model_type = '1D'
        if any([_layer.layer_type == 'quasi 2D' for _layer in layers]):
            self.model_type = 'quasi 2D'

        self._layers = layers

        last_domain = None
        if len(self.layers) > 0:
            last_domain = self.layers[0].domain
            for i, _layer in enumerate(self.layers):
                if _layer.domain != last_domain:
                    raise ValueError('Not all layers have the same domain')

        self.domain = last_domain

        if depth_to_top is None:
            if self.domain == 'TWT':
                self.depth_to_top = Q_(2.0, 's')
            else:
                self.depth_to_top = Q_(3000., 'm')
        else:
            self.depth_to_top = depth_to_top

        if trace_index_range is None:
            self.trace_index_range = None
        else:
            self.trace_index_range = trace_index_range

        self.name = name
        self.base_case_name = global_base_case_name

        self.table_keys = ['name', 'case', 'color', 'thickness', 'litho_fluid']
        self.trace_index = 0

    def __str__(self):
        return '{} model in {} domain with {} layers'.format(self.model_type, self.domain, len(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        if self.model_type == 'quasi 2D' and self.trace_index_range is not None:
            if np.abs(index) >= len(self.trace_index_range):
                raise IndexError('list index out of range')
            if index < 0:
                index = len(self.trace_index_range) + index
            layers = []
            for layer in self.layers:
                if hasattr(layer.thickness, '__call__'):  # True if vp is a function
                    layer_thickness = layer.thickness(index)
                else:
                    layer_thickness = layer.thickness
                if hasattr(layer.vp, '__call__'):  # True if vp is a function
                    layer_vp = layer.vp(index)
                else:
                    layer_vp = layer.vp
                if hasattr(layer.vs, '__call__'):  # True if vp is a function
                    layer_vs = layer.vs(index)
                else:
                    layer_vs = layer.vs
                if hasattr(layer.rho, '__call__'):  # True if vp is a function
                    layer_rho = layer.rho(index)
                else:
                    layer_rho = layer.rho
                if hasattr(layer.ntg, '__call__'):  # True if vp is a function
                    layer_ntg = layer.ntg(index)
                else:
                    layer_ntg = layer.ntg
                if hasattr(layer, 'gross_vp'):
                    layer_gross_vp = layer.gross_vp
                else:
                    layer_gross_vp = 3400.
                if hasattr(layer, 'gross_vs'):
                    layer_gross_vs = layer.gross_vs
                else:
                    layer_gross_vs = 1000.
                if hasattr(layer, 'gross_rho'):
                    layer_gross_rho = layer.gross_rho
                else:
                    layer_gross_rho = 2.
                if hasattr(layer, 'thin_bed_factor'):
                    layer_thin_bed_factor = layer.thin_bed_factor
                else:
                    layer_thin_bed_factor = 3.

                layers.append(
                    Layer(
                        thickness=layer_thickness,
                        target=layer.target,
                        vp=layer_vp,
                        vs=layer_vs,
                        rho=layer_rho,
                        ntg=layer_ntg,
                        layer_type='1D',
                        domain=layer.domain,
                        gross_vp=layer_gross_vp,
                        gross_vs=layer_gross_vs,
                        gross_rho=layer_gross_rho,
                        thin_bed_factor=layer_thin_bed_factor
                    )
                )
            return Model(depth_to_top=self.depth_to_top,
                         layers=layers
                         )
        else:
            return self  # if it is a 1D model, return itself. Does this make sense?

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value: list):
        self._layers = value

    @property
    def layer_names(self) -> list:
        _list = []
        for _i, _l in enumerate(self.layers):
            _list.append(_l.name)
        return list(set(_list))

    @property
    def case_names(self) -> list:
        _list = []
        for _i, _l in enumerate(self.layers):
            _list.append(_l.case)
        return list(set(_list))

    def index_of(self, name=None, case=None) -> int | None:
        if name is not None and case is not None:
            for _i, _l in enumerate(self.layers):
                if _l.name == name and _l.case == case:
                    return _i
        # elif case is None:
        #     for _i, _l in enumerate(self.layers):
        #         if _l.name == name:
        #             return _i
        # elif name is None:
        #     for _i, _l in enumerate(self.layers):
        #         if _l.case == case:
        #             return _i
        return None

    @property
    def target_layers(self) -> list:
        """
        Returns the indexes of all layers that are identified as target layers
        :return:
        """
        _list = []
        for _i, _l in enumerate(self.layers):
            if _l.case != global_base_case_name:
                _list.append(_i)
        return _list

    @property
    def base_case_layers(self) -> list:
        """
        Returns the indexes of all layers that are identified as base case layers
        :return:
        """
        _list = []
        for _i, _l in enumerate(self.layers):
            if _l.case.lower() == global_base_case_name.lower():
                _list.append(_i)
        return _list

    @property
    def base_target_combos(self, verbose=True) -> list:
        """
        Finds all the layers with the same name, and different cases, and returns these as a list of lists
        E.G. for case of
            [
                layer1 (name = 'Top', case='Base'),
                layer2 (name = 'Reservoir', case='Base'),
                layer3 (name = 'Reservoir', case='Oil'),
                layer4 (name = 'Bottom', case='Base')
            ]
        the list "[[1,2]]" is returned
        :return:
        """
        _list = []
        for _name in self.layer_names:
            if _name not in [self.layers[_i].name for _i in self.target_layers]:
                if verbose:
                    print('{} is not a target layer'.format(_name))
                continue
            _k = [self.index_of(_name, global_base_case_name)]
            if verbose:
                print('Layer: {}, and case: {}, has index {}'.format(
                    _name, global_base_case_name, _k[0]
                ))
            if _k[0] is None:
                continue
            for _case in self.case_names:
                _j = None
                if _case.lower() == global_base_case_name.lower():
                    continue
                _j = self.index_of(_name, _case)
                if _j is not None:
                    _k.append(self.index_of(_name, _case))
            _list.append(_k)
        return _list

    def get_layer(self, name: str, case: str) -> Layer | None:
        """
        Returns the layer with the given name and case if that case exists. Else it returns the base case layer
        :param name:
        :param case:
        :return:
        """
        base_case_layer = None
        for _layer in self.layers:
            if _layer.name == name and _layer.case.lower() == global_base_case_name.lower():
                base_case_layer = _layer
            if _layer.name == name and _layer.case.lower() == case.lower():
                return _layer
        return base_case_layer

    def total_thickness(self) -> Q_ | None:
        """
        Returns the total thickness of the model.
        NOTE. Only sums up the thickness of base case layers
        :return:
        """
        from typing import Callable
        total_thickness = None
        for _i, _j in enumerate(self.base_case_layers):
            _layer = self.layers[_j]
            if isinstance(_layer.thickness, Callable):
                this_thickness = _layer.thickness(self.trace_index)
            else:
                this_thickness = _layer.thickness

            if _i == 0:
                total_thickness = this_thickness
            else:
                total_thickness += this_thickness

        return total_thickness

    def depth_array(self, resolution: Q_) -> Q_:
        """
        Returns a pint.Quantity array with length similar to the total thickness of the model (sum layer thicknesses)
        with a sampling matching the desired resolution

        :param resolution:
            pint.Quantity
            Determines the resolution of the layer.
            Must match the dimension of the model
        :return:
        """
        depth = self.depth_to_top + Q_(
            np.arange(
                0.,
                self.total_thickness().magnitude,
                resolution.to(self.total_thickness().units).magnitude),
            self.total_thickness().units)
        return depth

    def interface_indexes(self, resolution: Q_, index: int | None = None) -> list:
        """
        Returns a list of indexes, one for each interface (except top and bottom) for the given resolution
        :param resolution:
            pint.Quantity
            Determines the resolution of the layer.
            Must match the dimension of the model
        :param index
            int
            The 'lateral' index which is used when the thickness is given as a function
        :return:
        """
        if index is None:
            index = self.trace_index
        _list = []
        previous_thickness = 0.
        interface_depth = self.depth_to_top
        depths = self.depth_array(resolution)
        for _i, _j in enumerate(self.base_case_layers):
            interface_depth += previous_thickness
            if _i > 0:
                _x = int( np.argmin(np.sqrt( (depths - interface_depth)**2 )) )
                _list.append(_x)

            if isinstance(self.layers[_j].thickness, Callable):
                _thickness = self.layers[_j].thickness(index)
            else:
                _thickness = self.layers[_j].thickness

            previous_thickness = _thickness

        return _list

    def horizons_cds(self, resolution: Q_ | None = None) -> ColumnDataSource:
        """
        Returns ColumnDataSource (cds) with the top depths of each layer in the model

        :param resolution:
            pint.Quantity or None
            If a resolution is given, it also calculates the depth index for the top of each layer

        :return:
        """
        print('XXX', [self.layers[_i].name for _i in self.base_case_layers])
        _dict = {'Top {}'.format(self.layers[_i].name): [] for _i in self.base_case_layers}
        if resolution is not None:
            depths = self.depth_array(resolution)
            for _layer_i in self.base_case_layers:
                _dict['Top {} index'.format(self.layers[_layer_i].name)] = []
        else:
            depths = None

        if self.trace_index_range is None:
            trace_indexes = [0]
        else:
            trace_indexes = self.trace_index_range

        _dict['x'] = trace_indexes

        for trace_idx in self.trace_index_range:
            previous_thickness = 0.
            interface_depth = self.depth_to_top
            for _i, _j in enumerate(self.base_case_layers):
                top_name = 'Top {}'.format(self.layers[_j].name)
                top_index_name = 'Top {} index'.format(self.layers[_j].name)
                interface_depth += previous_thickness
                _dict[top_name].append(interface_depth.to('m').magnitude)
                if depths is not None:
                    _dict[top_index_name].append(int(np.argmin(np.sqrt( (depths - interface_depth)**2 ))))
                if isinstance(self.layers[_j].thickness, Callable):
                    _thickness = self.layers[_j].thickness(trace_idx)
                else:
                    _thickness = self.layers[_j].thickness
                previous_thickness = _thickness

        return ColumnDataSource(_dict)


    def append(self, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        if layer.domain != self.domain:
            raise ValueError('Appended layer must be in same domain as model')
        self._layers.append(layer)
        if layer.layer_type == 'quasi 2D':
            self.model_type = 'quasi 2D'

    def insert(self, index, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        if layer.domain != self.domain:
            raise ValueError('Inserted layer must be in same domain as model')
        self.layers.insert(index, layer)
        if layer.layer_type == 'quasi 2D':
            self.model_type = 'quasi 2D'

    def realize_model(self, resolution, voigt_reuss_hill=False, overburden_vel=3000., verbose=False):
        """
        Realize the current model so that the parameters are regularly sampled in TWT, regardless if the
        model is given in TWT or Z domain
        :param resolution:
            float
            resolution in TWT [s]
        :param voigt_reuss_hill:
            bool
            If true, the Voigt Reuss Hill average for the given NTG is used when returning the elastic parameters
        :param overburden_vel:
            float
            Approximate vp velocity in the overburden to give a reasonable estimate of top TWT when the model is
            specified in Z

        """

        if self.domain == 'TWT':
            twt_top = self.depth_to_top
            z_top = self.depth_to_top * overburden_vel / 2.
        else:
            twt_top = 2. * self.depth_to_top / overburden_vel
            z_top = self.depth_to_top

        if self.model_type == 'quasi 2D' and self.trace_index_range is not None:

            if self.domain != 'TWT':
                raise NotImplementedError('Realization of 2D models in Z-domain is yet not implemented')

            n = len(self.trace_index_range)
            layer_index, this_vp, this_vs, this_rho = None, None, None, None
            bgrnd_vp, bgrnd_vs, bgrnd_rho, bgrnd_layer_index = None, None, None, None
            thickening = False
            for trace_i in self.trace_index_range:
                tmp_layer_index = np.zeros(0)
                tmp_vp, tmp_vs, tmp_rho = np.zeros(0), np.zeros(0), np.zeros(0)
                for i, layer in enumerate(self.layers):
                    _vp, _vs, _rho = layer.realize_layer(resolution, voigt_reuss_hill=voigt_reuss_hill, index=trace_i)
                    tmp_layer_index = np.append(tmp_layer_index, np.ones(len(_vp)) * i)
                    tmp_vp = np.append(tmp_vp, _vp)
                    tmp_vs = np.append(tmp_vs, _vs)
                    tmp_rho = np.append(tmp_rho, _rho)
                if trace_i == 0:
                    this_vp = np.zeros((n, len(tmp_vp)))
                    this_vs = np.zeros((n, len(tmp_vp)))
                    this_rho = np.zeros((n, len(tmp_vp)))
                    layer_index = np.zeros((n, len(tmp_vp)))
                    # Store background model
                    bgrnd_vp = deepcopy(tmp_vp)
                    bgrnd_vs = deepcopy(tmp_vs)
                    bgrnd_rho = deepcopy(tmp_rho)
                    bgrnd_layer_index = deepcopy(tmp_layer_index)

                if verbose:
                    print('Trace: {}, length: {}'.format(trace_i, len(tmp_vp)))

                # First fill model with background model
                this_vp[trace_i, :] = bgrnd_vp
                this_vs[trace_i, :] = bgrnd_vs
                this_rho[trace_i, :] = bgrnd_rho
                layer_index[trace_i, :] = bgrnd_layer_index

                if len(tmp_vp) > len(bgrnd_vp):
                    thickening = True  # Model gets "thicker" for this trace
                else:
                    thickening = False

                if thickening:
                    this_vp[trace_i, :len(bgrnd_vp)] = tmp_vp[:len(bgrnd_vp)]
                    this_vs[trace_i, :len(bgrnd_vs)] = tmp_vs[:len(bgrnd_vs)]
                    this_rho[trace_i, :len(bgrnd_rho)] = tmp_rho[:len(bgrnd_rho)]
                    layer_index[trace_i, :len(bgrnd_layer_index)] = tmp_layer_index[:len(bgrnd_layer_index)]
                else:
                    this_vp[trace_i, :len(tmp_vp)] = tmp_vp
                    this_vs[trace_i, :len(tmp_vs)] = tmp_vs
                    this_rho[trace_i, :len(tmp_rho)] = tmp_rho
                    layer_index[trace_i, :len(tmp_layer_index)] = tmp_layer_index

            twt = twt_top + np.arange(len(this_vp[0])) * resolution
            this_z = np.zeros(this_vp.shape)
            for trace_i in self.trace_index_range:
                this_z[trace_i, :] = z_top + 0.5 * np.cumsum(this_vp[trace_i, :] * resolution)

        else:
            layer_index, this_vp, this_vs, this_rho = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
            this_z = np.zeros(0)
            last_z = z_top
            for i, layer in enumerate(self.layers):
                _vp, _vs, _rho = layer.realize_layer(resolution, voigt_reuss_hill=voigt_reuss_hill)
                layer_index = np.append(layer_index, np.ones(len(_vp)) * i)
                this_vp = np.append(this_vp, _vp)
                this_vs = np.append(this_vs, _vs)
                this_rho = np.append(this_rho, _rho)
                # print(i)
                # print('this_z', type(this_z))  #, this_z)
                # print('last_z', type(last_z))  #, last_z)
                # print('_vp', type(_vp))  #, _vp)
                # print('resolution', type(resolution))  #, resolution)
                this_z = np.append(this_z, last_z + 0.5 * np.cumsum(_vp * resolution))
                last_z = this_z[-1]

            twt = twt_top + np.arange(len(this_vp)) * resolution

        return twt, layer_index, this_vp, this_vs, this_rho, this_z

    def plot(self, ax=None, index=None, kwargs1d=None):
        if self.model_type == '1D':
            if kwargs1d is None:
                plot_1d(self, ax)
            else:
                plot_1d(self, ax, **kwargs1d)
        elif self.model_type == 'quasi 2D' and index is not None:
            if kwargs1d is None:
                plot_1d(self, ax, index=index)
            else:
                plot_1d(self, ax, index=index, **kwargs1d)
        elif self.model_type == 'quasi 2D':
            plot_quasi_2d(self, ax)


class ModelTable:
    """
    Returns a bokeh DataTable populated with rows of single layers of a model
    """
    from bokeh.models import ColumnDataSource

    def __init__(self,
                 model: Model,
                 width: int | None = None,
                 height: int | None = None,
                 title: str | None = 'Model layers'):
        """

        :param model:
            Model object
        :param width:
        :param height:
        """
        self.model = model
        if width is None:
            width = 700
        self.width = width
        if height is None:
            height = 135
        self.height = height
        self.keys = ['name', 'case', 'color', 'thickness', 'litho_fluid']
        if title is not None:
            title += ', Top at {}'.format(model.depth_to_top)
        self.title = title
        self.trace_index = 0

    @property
    def cds(self):
        _dict = {_x: [] for _x in self.keys}
        _dict['top'] = []
        previous_thickness = 0.
        previous_top = float(self.model.depth_to_top.to('m').magnitude)
        for _i, _layer in enumerate(self.model.layers):
            if isinstance(_layer.thickness, Callable):
                this_thickness = float(_layer.thickness(self.trace_index).to('m').magnitude)
            else:
                this_thickness = float(_layer.thickness.to('m').magnitude)
            if _i == 0:
                _dict['top'].append(previous_top)
                previous_thickness = this_thickness
            else:
                if _layer.case.lower() == global_base_case_name.lower():  # Only add base case layers to the sum of thicknesses
                    _dict['top'].append(previous_top + previous_thickness)
                    previous_top += previous_thickness
                    previous_thickness = this_thickness
                else:
                    _dict['top'].append(previous_top)
                    previous_thickness = this_thickness


            for _key in self.keys:
                if _key == 'thickness':
                    if isinstance(_layer.thickness, Callable):
                        _dict[_key].append(_layer.thickness(self.trace_index).to('m').magnitude)
                    else:
                        _dict[_key].append(_layer.thickness.to('m').magnitude)
                elif _key == 'litho_fluid':
                    _dict[_key].append(_layer.litho_fluid.name)
                elif _key == 'name':
                    _dict[_key].append(_layer.name)
                elif _key == 'case':
                    _dict[_key].append(_layer.case)
                else:
                    _dict[_key].append(_layer.__dict__[_key])
        return ColumnDataSource(_dict)

    def table_columns(self, litho_fluids_cds:ColumnDataSource):
        """
        :param litho_fluids_cds:
            ColumnDataSource of the LithoFluidsTable that contains all the different LithoFluids we can use to
            populate the model
        """
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, HTMLTemplateFormatter)
        from bokeh.models import ColumnDataSource, StringFormatter, IntEditor, NumberEditor, NumberFormatter
        colored_cell_template = """
                <div style="background:<%= 
                    (function color_from_val(){
                        return(color)
                        }()) %>; 
                    color: white"> 
                <%= value %>
                </div>
            """
        formatter = HTMLTemplateFormatter(template=colored_cell_template)

        column_names = [_s.capitalize().replace('_', ' ') for _s in self.keys]
        table_columns = []

        # Create the name selection editor separatel, so that we can updated it later
        name_selector = SelectEditor(options=litho_fluids_cds.data['name'])

        #Now create the table columns
        for i, column_key in enumerate(self.keys):
            if column_key == 'litho_fluid':
                _editor = name_selector
                # _editor = StringEditor(completions=litho_fluids_cds.data['name'])
                _formatter = StringFormatter(font_style='bold')
            elif column_key == 'thickness':
                _editor = NumberEditor()
                _formatter = NumberFormatter(format='0.0[0]')
            elif column_key == 'color':
                _editor=StringEditor()
                _formatter=formatter
            else:
                _editor = StringEditor()
                _formatter = StringFormatter()
            table_columns.append(
                TableColumn(
                    field=column_key,
                    title=column_names[i],
                    editor=_editor,
                    formatter=_formatter

                )
            )

        # Try to modify the options of litho-fluid names if the litho_fluids_cds is changed
        def update_name_options(attr, old, new):
            name_selector.options = litho_fluids_cds.data['name']

        litho_fluids_cds.on_change('data', update_name_options)

        return table_columns

    def realize(self, resolution: Q_, index: int | None = None) -> dict:
        """
        "Realizes" the model according to the given resolution.

        I.E. Before realization a layer has one value for Vp, Vs, ...  After realization we have an array of values
        corresponding to the layer thickness and given resolution

        As a model can contain multiple cases, we need to create one realization per case, so the output
        is a dictionary with the cases as keys, and their respective realizations as values represented as LogCurves
        :param resolution:
            pint.Quantity
            Determines the resolution of the layer.
            Must match the dimension of the model
        :param index:
            int
            The 'lateral' index of where the model is realized
        :return:
            dict
            Dictionary of LogCurves
        """
        if index is None:
            index = self.trace_index

        _dict = {_case: {} for _case in self.model.case_names}

        depth = self.model.depth_array(resolution)
        n = len(depth.magnitude)

        i_inds = self.model.interface_indexes(resolution, index=index)

        for _case in list(_dict.keys()):
            # Populate arrays with the values of each layer. Units are not important as each layer is
            # converted to m/s and gr/cm3 when initiated
            _vp = np.zeros(n)
            _vs = np.zeros(n)
            _rho = np.zeros(n)
            _twt = np.zeros(n)
            for i, j in enumerate(self.model.base_case_layers):
                _name = self.model.layers[j].name
                this_layer = self.model.get_layer(_name, _case)
                this_vp = this_layer.vp
                if isinstance(this_vp, Callable):
                    this_vp = this_vp(index)
                this_vs = this_layer.vs
                if isinstance(this_vs, Callable):
                    this_vs = this_vs(index)
                this_rho = this_layer.rho
                if isinstance(this_rho, Callable):
                    this_rho = this_rho(index)

                if i == 0:  # First layer
                    _vp[:i_inds[i]] = this_vp.magnitude
                    _vs[:i_inds[i]] = this_vs.magnitude
                    _rho[:i_inds[i]] = this_rho.magnitude
                elif i > len(i_inds) - 1:  # last layer
                    _vp[i_inds[i-1]:] = this_vp.magnitude
                    _vs[i_inds[i-1]:] = this_vs.magnitude
                    _rho[i_inds[i-1]:] = this_rho.magnitude
                else:
                    _vp[i_inds[i-1]:i_inds[i]] = this_vp.magnitude
                    _vs[i_inds[i-1]:i_inds[i]] = this_vs.magnitude
                    print('XXX', _case, this_rho)
                    _rho[i_inds[i-1]:i_inds[i]] = this_rho.magnitude

            _twt = np.cumsum(2.0 * resolution.to('m').magnitude / _vp) + 1.0

            _dict[_case]['vp'] = LogCurve('vp_{}'.format(_case), Q_(_vp, 'm/s'), Depth(depth), log_type='P velocity')
            _dict[_case]['vs'] = LogCurve('vs_{}'.format(_case), Q_(_vs, 'm/s'), Depth(depth), log_type='S velocity')
            _dict[_case]['rho'] = LogCurve('rho_{}'.format(_case), Q_(_rho, 'grams/cm^3'), Depth(depth), log_type='Density')
            _dict[_case]['ai'] = LogCurve('ai_{}'.format(_case), (Q_(_vp, 'm/s') * Q_(_rho, 'grams/cm^3')).to('kiloPa * s / m'), Depth(depth), log_type='Impedance')
            _dict[_case]['vpvs'] = LogCurve('vpvs_{}'.format(_case), Q_(_vp, 'm/s') / Q_(_vs, 'm/s'), Depth(depth), log_type='VpVs')
            _dict[_case]['twt'] = LogCurve('twt_{}'.format(_case), Q_(_twt, 's'), Depth(depth), log_type='TWT')

        return _dict

    # def line_cds(self, resolution: Q_, mod_cds: ColumnDataSource) -> ColumnDataSource:
    def line_cds(self, elastics_dict: dict) -> ColumnDataSource:
        """
        Calculates the lines that represent the elastic properties of the model

        :param elastics_dict:
            dict
            A realization of the model table (at a given trace index)
            See
        :return:
        """
        # elastics = self.realize(resolution, mod_cds)
        _dict = modify_dict(elastics_dict)
        return ColumnDataSource(_dict)

    def draw(self, cds: ColumnDataSource, litho_fluids_cds: ColumnDataSource):
        """
        Returns a table that is used for defining a model
        :param cds:
            ColumnDataSource of the initial model, with N layers
        :param litho_fluids_cds:
            ColumnDataSource of the LithoFluidsTable that contains all the different LithoFluids we can use to
            populate the model
        :return:
        """
        from bokeh.models import DataTable, Button, Div
        from bokeh.plotting import column, row

        def add_row_function():
            new_data = dict(cds.data)
            n = len(new_data['name'])
            new_name = None
            for _key in list(new_data.keys()):
                if _key == 'name':
                    new_data[_key].append(new_name)
                elif _key == 'case':
                    new_data[_key].append(global_base_case_name)
                elif _key == 'thickness':
                    new_data[_key].append(30.)
                else:
                    if n == 0:
                        new_data[_key].append(None)
                    else:
                        new_data[_key].append(new_data[_key][-1])
            cds.data = new_data

        def delete_row_function():
            selected_index = cds.selected.indices
            new_data = {_x:[] for _x in self.keys}
            for _i in range(len(cds.data['name'])):
                if _i  in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(cds.data[_x][_i])
            cds.selected.indices = []
            cds.data = new_data

        def update_table_function():
            new_data = dict(cds.data)
            layers = []

            # TODO
            # It doesn't seem to catch if I change name of a layer correctly.
            # The names are updated in the table, but if I by mistake had several layers with the
            # same name and case, I can't fix that interactively

            # Iterate over all layers
            for _i in range(len(new_data['name'])):
                this_lf = new_data['litho_fluid'][_i]
                litho_fluid_i = 0
                try:
                    litho_fluid_i = [_x.lower() for _x in litho_fluids_cds.data['name']].index(this_lf.lower())
                except ValueError as e:
                    warn_txt = 'Litho fluid {} is not found in LithoFluidTable. Using first'.format(this_lf)
                    print_info(warn_txt, 'warning', logger)
                    continue

                # Within the number of original layers. Layers can be quasi 2D
                if _i < len(self.model.layers):
                    orig_layer = self.model.layers[_i]
                    # Check if the input thickness is a function
                    if isinstance(orig_layer.thickness, Callable):
                        print('Model thickness is a function in row {}'.format(_i))
                        # Don't update the thickness when it is a function
                        _thickness = orig_layer.thickness
                        new_data['thickness'][_i] = _thickness(self.trace_index).magnitude
                    else:
                        _thickness =  Q_(new_data['thickness'][_i], 'm')
                    # Also check if input Vp, Vs and Rho are functions
                    if isinstance(orig_layer.vp, Callable):
                        print('Model Vp is a function in row {}'.format(_i))
                        _vp = orig_layer.vp
                    else:
                        _vp = litho_fluids_cds.data['vp'][litho_fluid_i]
                    if isinstance(orig_layer.vs, Callable):
                        print('Model Vs is a function in row {}'.format(_i))
                        _vs = orig_layer.vs
                    else:
                        _vs = litho_fluids_cds.data['vs'][litho_fluid_i]
                    if isinstance(orig_layer.rho, Callable):
                        print('Model Rho is a function in row {}'.format(_i))
                        _rho = orig_layer.rho
                    else:
                       _rho = litho_fluids_cds.data['rho'][litho_fluid_i]
                else:
                    _thickness =  Q_(new_data['thickness'][_i], 'm')
                    _vp = litho_fluids_cds.data['vp'][litho_fluid_i]
                    _vs = litho_fluids_cds.data['vs'][litho_fluid_i]
                    _rho = litho_fluids_cds.data['rho'][litho_fluid_i]

                if _thickness is None:
                        _thickness = Q_(10., 'm')

                this_layer = Layer(
                    name=new_data['name'][_i],
                    case=new_data['case'][_i],
                    color=new_data['color'][_i],
                    thickness=_thickness,
                    litho_fluid=LithoFluid(
                        # TODO Unsure if the name of the lithofluid will update correctly
                        name=litho_fluids_cds.data['name'][litho_fluid_i],
                        vp=_vp,
                        vs=_vs,
                        rho=_rho
                    )
                )
                layers.append(this_layer)
            self.model = Model(
                depth_to_top=self.model.depth_to_top,
                layers=layers,
                trace_index_range=self.model.trace_index_range
            )
            cds.data = new_data

        title_txt = '<div style="font-size:12px; font-weight:600; margin-bottom:0px; text-align:center">\n'
        title_txt += '{}\n'.format(self.title)
        title_txt += '</div>\n'
        title_div = Div(text = title_txt)

        table = DataTable(
            source=cds,
            columns=self.table_columns(litho_fluids_cds),
            editable=True,
            width=self.width,
            height=self.height,
            index_position = -1,
            index_header = 'index'
        )

        if self.title is not None:
            table = column(title_div, table, sizing_mode='stretch_width')

        add_row = Button(label='Add row', button_type='success')
        add_row.on_click(add_row_function)

        delete_row = Button(label='Delete selected rows', button_type='success')
        delete_row.on_click(delete_row_function)

        update_table = Button(label='Update', button_type='success')
        update_table.on_click(update_table_function)

        return table, add_row, delete_row, update_table


class LaminarModel:
    """
    Creates an interactive laminar (1D & Quasi 2D) model together with a display of the synthetic response
    NOTE! Only works in depth domain ('Z')
    """

    from blixt_rp.plotting.cross_plotter import DataSource

    def __init__(self,
                 model: Model | None = None,
                 litho_fluids: LithoFluids | None = None,
                 resolution: Q_ | None = None,
                 wavelet: dict | None = None,
                 avo_or_eei: str | None = None,
                 **kwargs
                 ):
        """

        :param model:
        :param litho_fluids:
        :param resolution:
            pint Quantity
            Depth resolution (in depth, not time) of the realized model
        :param wavelet:
        :param avo_or_eei:
        :param kwargs:
            dt:
                pint Quantity, resolution in time used when calculating the synthetic seismic response (needs to be in
                time)
            freq:
                Wavelet central frequency in Hz, which is used when calculating the synthetic response
            table_height:
                int
                Height of the table showing the layers
        """

        if model is None:
            layers = [
                LithoFluid(default='shale').to_model_layer('Top',thickness=Q_(50., 'm')),
                LithoFluid(default='brine_sst').to_model_layer('Reservoir',thickness=Q_(25., 'm')),
                LithoFluid(default='oil_sst').to_model_layer('Reservoir', case='Oil',thickness=Q_(25., 'm')),
                LithoFluid(default='shale').to_model_layer('Bottom',thickness=Q_(50., 'm'))
            ]
            model = Model(layers=layers, name='Default model')
        self.model = model
        self.model_table = None

        if litho_fluids is None:
            litho_fluids = LithoFluids([
                LithoFluid(default='shale'),
                LithoFluid(default='brine_sst'),
                LithoFluid(default='oil_sst')
            ])
        self._litho_fluids = litho_fluids
        if resolution is None:
            resolution = Q_(0.1, 'm')
        self.resolution = resolution
        self.dt = kwargs.pop('dt', Q_(1., 'millisecond'))
        self.freq = kwargs.pop('freq', 20.)
        self.lf_width = kwargs.pop('lf_width', 300)
        self.model_width = kwargs.pop('model_width', None)
        self.table_height = kwargs.pop('table_height', None)
        self.lf_table = None
        self.previous_cases = None
        if avo_or_eei is None:
            avo_or_eei = 'avo'
        self.avo_or_eei = avo_or_eei
        if self.avo_or_eei == 'avo':
            self.angles = np.arange(0., 40., 2)
        else:
            self.angles = np.arange(-90., 91., 5)
        self.active_2d_case = global_base_case_name

    def initiate_lf_table(self):
        self.lf_table = LithoFluidsTable(self._litho_fluids, width=self.lf_width, advanced=False)

    def draw_lf_table(self, lf_cds: ColumnDataSource):
        # Returns the LithoFluidsTable with control buttons
        # Returns: lf_table, lf_add_row, lf_delete_row, lf_update
        # lf_obj = LithoFluidsTable(self._litho_fluids, width=self.lf_width)
        return self.lf_table.draw(lf_cds)

    # def initiate_model_table(self, lf_cds: ColumnDataSource):
    def initiate_model_table(self):
        mod_table = ModelTable(self.model, width=self.model_width, height=self.table_height)
        self.model_table = mod_table

    def draw_model_table(self, mod_cds: ColumnDataSource, lf_cds:ColumnDataSource):
        # Returns the ModelTable with control buttons
        # Returns: model_table, model_add_row, model_delete_row, model_update
        return self.model_table.draw(mod_cds, lf_cds)

    def draw(self):
        from bokeh.models import Span, Slider, Div, CustomJS
        from bokeh.models import GlyphRenderer
        from bokeh.models.glyphs import Image, ImageRGBA, ImageURL

        from blixt_rp.plotting.log_plotter import LogPlotter, LogColumn, Line, select_column, select_line
        from blixt_rp.core.core import Template
        # Create the litho fluids table
        self.initiate_lf_table()
        cds_lf = self.lf_table.cds
        lf_table, add_row, delete_row, update = self.draw_lf_table(cds_lf)

        # Create the models table
        self.initiate_model_table()
        cds_m = self.model_table.cds
        _model_table, add_row_m, delete_row_m, update_m = self.draw_model_table(cds_m, cds_lf)

        # Create log plot object
        plotter = LogPlotter(width=900, height=600)

        # Create controller widgets
        freq_slider = Slider(title='Wavelet central freq. [Hz]', start=10, end=40, step=5, value=self.freq)

        def freq_slider_change(attr, old, new):
            self.freq = new

        # Update the self.freq attribute depending on the frequency slider.
        freq_slider.on_change('value', freq_slider_change)

        # Set up a fixed resolution in time
        dt = self.dt

        # Initiate the elastics and draw the initial data
        elastics_dict = self.model_table.realize(self.resolution)
        cds_lines = self.model_table.line_cds(elastics_dict)

        # def draw_plot():
        extremes = find_extremes(cds_lines)
        ais_lines = []
        vpvs_lines = []
        for _case in self.model.case_names:
            _line_style = '-'
            _line_color = 'blue'
            _line_width = 1.
            print('XXX', _case)
            if _case != global_base_case_name:
                _line_style = '--'
                _line_color = 'red'
                _line_width = 2.
            ais_lines.append(Line(x='ai_{}'.format(_case), y='depth', cds=cds_lines,
                                  style=Template(**{
                                      'name': 'AI {}'.format(_case),
                                      'line_color': _line_color,
                                      'line_width': _line_width,
                                      'line_style': _line_style,
                                      'min': extremes['ai'][0]*0.95,
                                      'max': extremes['ai'][1]*1.05
                                  })))
            vpvs_lines.append(Line(x='vpvs_{}'.format(_case), y='depth', cds=cds_lines,
                                   style=Template(**{
                                       'name': 'VpVs {}'.format(_case),
                                       'line_color': _line_color,
                                       'line_width': _line_width,
                                       'line_style': _line_style,
                                       'min': extremes['vpvs'][0]*0.95,
                                       'max': extremes['vpvs'][1]*1.05
                                   })))

        # Create two columns to hold the AI and VpVs lines
        ai_column = LogColumn('AI', lines=ais_lines, rel_width=0.5)
        vpvs_column = LogColumn('VpVs', lines=vpvs_lines, rel_width=0.4)  # Make this thinner to compensate for tick marks on ai_column

        # Create two columns that holds the synthetic traces for the Base case and one other case (initially
        # a copy of the first)
        synth_cds_base = calc_synth_cds(
            elastics_dict[global_base_case_name]['vp'],
            elastics_dict[global_base_case_name]['vs'],
            elastics_dict[global_base_case_name]['rho'],
            elastics_dict[global_base_case_name]['twt'],
            dt, self.angles, self.avo_or_eei, freq_slider.value
        )
        synth_cds_variation = calc_synth_cds(
            elastics_dict[global_base_case_name]['vp'],
            elastics_dict[global_base_case_name]['vs'],
            elastics_dict[global_base_case_name]['rho'],
            elastics_dict[global_base_case_name]['twt'],
            dt, self.angles, self.avo_or_eei, freq_slider.value
        )
        # print('XXX', np.min(synth_cds_base.data['value']), np.max(synth_cds_base.data['value']))
        traces_base = create_traces(
            synth_cds_base,
            elastics_dict[global_base_case_name]['vp'].depth.values,
            'Base case', self.angles, self.avo_or_eei)
        traces_variation = create_traces(
            synth_cds_variation,
            elastics_dict[global_base_case_name]['vp'].depth.values,
            'Base case', self.angles, self.avo_or_eei)

        synth_base_column = LogColumn('SYNTH_BASE', seismic_traces=traces_base, rel_width=1.)
        synth_variation_column = LogColumn('SYNTH_VARIATION', seismic_traces=traces_variation, rel_width=1.)

        plotter.columns = [ai_column, vpvs_column, synth_base_column, synth_variation_column]
        # end of draw_plot()

        self.previous_cases = self.model.case_names
        # grid = plotter.draw()
        grid = plotter.draw_ext_toolbar()

        def update_m_function():
            # Update the elastic properties based on the new model
            _elastics_dict = self.model_table.realize(self.resolution)

            # Update the ColumnDataSource of the lines and the model
            cds_lines.data = dict(self.model_table.line_cds(_elastics_dict).data)
            cds_m.data = dict(self.model_table.cds.data)

            # Update the CDS of the synthetics for the base case
            synth_cds_base.data = dict(calc_synth_cds(
                _elastics_dict[global_base_case_name]['vp'],
                _elastics_dict[global_base_case_name]['vs'],
                _elastics_dict[global_base_case_name]['rho'],
                _elastics_dict[global_base_case_name]['twt'],
                dt, self.angles, self.avo_or_eei, freq_slider.value
            ).data)

            # Update the CDS of the synthetics of the other case, if it exists
            # NOTE that we only accept one other case than the base case
            if len(self.model.case_names) > 1:
                cases = self.model.case_names
                cases.remove(global_base_case_name)
                print('Trying to update the synthetics of ', cases[0])
                synth_cds_variation.data = dict(calc_synth_cds(
                    _elastics_dict[cases[0]]['vp'],
                    _elastics_dict[cases[0]]['vs'],
                    _elastics_dict[cases[0]]['rho'],
                    _elastics_dict[cases[0]]['twt'],
                    dt, self.angles, self.avo_or_eei, freq_slider.value
                ).data)
                _p = select_column(grid, 'SYNTH_VARIATION')
                _p.xaxis.axis_label = '{} case: {}'.format(cases[0], self.avo_or_eei)

            # Update the depth range of the synthetics to match the changes in depth
            for _column in [select_column(grid, _col_name) for _col_name in ['SYNTH_BASE', 'SYNTH_VARIATION']]:
                image_renderers = [
                    r for r in _column.renderers
                    if isinstance(r, GlyphRenderer) and isinstance(r.glyph, (Image, ImageRGBA, ImageURL))
                ]
                image_renderers[0].glyph.y = self.model.depth_to_top.to('m').magnitude
                image_renderers[0].glyph.dh = total_cds_thickness(cds_m)

            # Detect changes in number of cases
            case_changes = detect_change_in_cases(self.model.case_names, self.previous_cases)
            if len(case_changes['new']) > 0:
                # get AI, and VpVs columns
                _p_ai = select_column(grid, 'AI')
                _p_vpvs = select_column(grid, 'VpVs')
                for _case in case_changes['new']:
                    _y = 'ai_{}'.format(_case)
                    # print('Trying to plot the new case:', _case, _y)
                    _p_ai.line(x='ai_{}'.format(_case), y='depth', source=cds_lines,
                               **{'line_color': 'red', 'legend_label': 'AI_{}'.format(_case)})
                    _p_vpvs.line(x='vpvs_{}'.format(_case), y='depth', source=cds_lines,
                                 **{'line_color': 'red', 'legend_label': 'VpVs_{}'.format(_case)})

            # Only handle the situation when a new case is added, for now (TODO)
            if len(case_changes['removed']) > 0:
                print_info('Can not handle the situation when a case is removed', 'error', logger, 'IOError')

            self.previous_cases = self.model.case_names

        update_m.on_click(update_m_function)
        # update.on_click(update_m_function)  # When using this the model table returns to its previous state when "Update" is clicked

        if self.model.name is None:
            title = Div(text="<div></div")
        else:
            title_txt = '<div style="font-size:12px; font-weight:600; margin-bottom:0px; text-align:center">\n'
            title_txt += '{}\n </div>'.format(self.model.name)
            title = Div(text=title_txt)

        return (title, lf_table, add_row, delete_row, update,
                _model_table, add_row_m, delete_row_m, update_m,
                grid, freq_slider, synth_cds_base)

    def draw_2d(self):
        from bokeh.models import Span, Slider, Select, Div
        from bokeh.models import GlyphRenderer
        from bokeh.models.glyphs import Image, ImageRGBA, ImageURL
        from bokeh.plotting import row

        from blixt_rp.plotting.log_plotter import LogPlotter, LogColumn, Line, select_column, select_line
        from blixt_rp.core.core import Template

        # Create the litho fluids table
        self.initiate_lf_table()
        cds_lf = self.lf_table.cds
        lf_table, add_row, delete_row, update = self.draw_lf_table(cds_lf)

        # Create the models table
        self.initiate_model_table()
        cds_m = self.model_table.cds
        _model_table, add_row_m, delete_row_m, update_m = self.draw_model_table(cds_m, cds_lf)

        # Create log plot object
        plotter = LogPlotter(width=1100, height=600)

        # Create controller widgets
        trace_selector = Select(
            title='Trace #',
            value=str(self.model.trace_index),
            options=[str(_i) for _i in list(self.model.trace_index_range)])
        case_selector = Select(
            title='Case',
            value=self.active_2d_case,
            options=self.model.case_names)
        freq_slider = Slider(title='Wavelet central freq. [Hz]', start=10, end=40, step=5, value=self.freq)

        def freq_slider_change(attr, old, new):
            self.freq = new

        # Update the self.freq attribute depending on the frequency slider.
        freq_slider.on_change('value', freq_slider_change)

        # Set up a fixed resolution in time
        dt = self.dt

        def draw_lines():
            # Initiate the elastics and draw the initial data
            # In the 2D case, this initial calculation will be for the left-most (index = 0) realization of the model
            _elastics_dict = self.model_table.realize(self.resolution, int(trace_selector.value))
            _cds_lines = self.model_table.line_cds(_elastics_dict)

            # def draw_plot():
            extremes = find_extremes(_cds_lines)
            _ais_lines = []
            _vpvs_lines = []
            for _case in self.model.case_names:
                _line_style = '-'
                _line_color = 'blue'
                _line_width = 1.
                print('XXX', _case)
                if _case != global_base_case_name:
                    _line_style = '--'
                    _line_color = 'red'
                    _line_width = 2.
                _ais_lines.append(Line(x='ai_{}'.format(_case), y='depth', cds=_cds_lines,
                                      style=Template(**{
                                          'name': 'AI {}'.format(_case),
                                          'line_color': _line_color,
                                          'line_width': _line_width,
                                          'line_style': _line_style,
                                          'min': extremes['ai'][0]*0.95,
                                          'max': extremes['ai'][1]*1.05
                                      })))
                _vpvs_lines.append(Line(x='vpvs_{}'.format(_case), y='depth', cds=_cds_lines,
                                       style=Template(**{
                                           'name': 'VpVs {}'.format(_case),
                                           'line_color': _line_color,
                                           'line_width': _line_width,
                                           'line_style': _line_style,
                                           'min': extremes['vpvs'][0]*0.95,
                                           'max': extremes['vpvs'][1]*1.05
                                       })))
            return _cds_lines, _ais_lines, _vpvs_lines

        cds_lines, ais_lines, vpvs_lines = draw_lines()

        # Create two columns to hold the AI and VpVs lines
        ai_column = LogColumn('AI', lines=ais_lines, rel_width=0.3)
        vpvs_column = LogColumn('VpVs', lines=vpvs_lines, rel_width=0.25)  # Make this thinner to compensate for tick marks on ai_column

        def calc_synth_2d_cds(_case_name):
            # Calculate the synthetics for the whole 2d section
            synth_values = []
            _elastics_dict = None
            for _i in self.model.trace_index_range:
                _elastics_dict = self.model_table.realize(self.resolution, _i)

                _synth_cds_base = calc_synth_cds(
                    _elastics_dict[_case_name]['vp'],
                    _elastics_dict[_case_name]['vs'],
                    _elastics_dict[_case_name]['rho'],
                    _elastics_dict[_case_name]['twt'],
                    dt, 0., self.avo_or_eei, freq_slider.value
                )
                synth_values.append(_synth_cds_base.data['value'][0])

            _synth_cds = ColumnDataSource({'value': [np.array(synth_values, dtype=float).T]})
            print('Total size: {}'.format(_synth_cds.data['value'][0].shape))
            return _synth_cds, _elastics_dict

        synth_2d_cds, elastics_dict = calc_synth_2d_cds(global_base_case_name)

        traces_base = create_traces(
            synth_2d_cds,
            elastics_dict[global_base_case_name]['vp'].depth.values,
            'Base case', self.model.trace_index_range, self.avo_or_eei)

        synth_2d_column = LogColumn('SYNTH_2D', seismic_traces=traces_base, rel_width=1.4)

        plotter.columns = [ai_column, vpvs_column, synth_2d_column]

        self.previous_cases = self.model.case_names
        grid = plotter.draw_ext_toolbar()

        # Add vertical line indicating where the 1D model is extracted
        v_line = Span(
            location = int(trace_selector.value) + 0.5,
            dimension='height',
            line_dash='dashed'
        )
        _p = select_column(grid, 'SYNTH_2D')
        _p.add_layout(v_line)

        # Add a modifier to the seismic colorbar
        color_amp_factor = add_color_amp(_p)

        def update_m_function():
            # Update the elastic properties based on the new model
            _elastics_dict = self.model_table.realize(self.resolution, int(trace_selector.value))

            # Update the ColumnDataSource of the lines and the model
            cds_lines.data = dict(self.model_table.line_cds(_elastics_dict).data)
            cds_m.data = dict(self.model_table.cds.data)

            # Update the CDS of the synthetics
            _synth_2d_cds, _elastics_dict = calc_synth_2d_cds(str(case_selector.value))
            synth_2d_cds.data = dict(_synth_2d_cds.data)

            # Update the depth range of the synthetics to match the changes in depth
            for _column in [select_column(grid, _col_name) for _col_name in ['SYNTH_2D']]:
                image_renderers = [
                    r for r in _column.renderers
                    if isinstance(r, GlyphRenderer) and isinstance(r.glyph, (Image, ImageRGBA, ImageURL))
                ]
                image_renderers[0].glyph.y = self.model.depth_to_top.to('m').magnitude
                image_renderers[0].glyph.dh = total_cds_thickness(cds_m)

            # Detect changes in number of cases
            case_changes = detect_change_in_cases(self.model.case_names, self.previous_cases)
            if len(case_changes['new']) > 0:
                # get AI, and VpVs columns
                _p_ai = select_column(grid, 'AI')
                _p_vpvs = select_column(grid, 'VpVs')
                for _case in case_changes['new']:
                    _y = 'ai_{}'.format(_case)
                    # print('Trying to plot the new case:', _case, _y)
                    _p_ai.line(x='ai_{}'.format(_case), y='depth', source=cds_lines,
                               **{'line_color': 'red', 'legend_label': 'AI_{}'.format(_case)})
                    _p_vpvs.line(x='vpvs_{}'.format(_case), y='depth', source=cds_lines,
                                 **{'line_color': 'red', 'legend_label': 'VpVs_{}'.format(_case)})

            # Only handle the situation when a new case is added, for now (TODO)
            if len(case_changes['removed']) > 0:
                print_info('Can not handle the situation when a case is removed', 'error', logger, 'IOError')

            self.previous_cases = self.model.case_names

        def update_trace_function(attr, old, new):
            print('Trace: ', attr, old, new)
            v_line.location = int(new) + 0.5
            # Update the elastic properties based on the new trace index
            _elastics_dict = self.model_table.realize(self.resolution, int(new))

            # Update the ColumnDataSource of the lines
            cds_lines.data = dict(self.model_table.line_cds(_elastics_dict).data)

        def update_case_function(attr, old, new):
            print('Case: ', attr, old, new)
            self.active_2d_case = new

            # Update the CDS of the synthetics
            _synth_2d_cds, _elastics_dict = calc_synth_2d_cds(str(case_selector.value))
            synth_2d_cds.data = dict(_synth_2d_cds.data)

        update_m.on_click(update_m_function)
        # update.on_click(update_m_function)  # When using this the model table returns to its previous state when "Update" is clicked

        trace_selector.on_change("value", update_trace_function)
        case_selector.on_change("value", update_case_function)

        if self.model.name is None:
            title = Div(text="<div></div")
        else:
            title_txt = '<div style="font-size:12px; font-weight:600; margin-bottom:0px; text-align:center">\n'
            title_txt += '{}\n </div>'.format(self.model.name)
            title = Div(text=title_txt)

        return (title, lf_table, add_row, delete_row, update, _model_table,
                add_row_m, delete_row_m, update_m, grid,
                row(trace_selector, case_selector, freq_slider, color_amp_factor), synth_2d_cds)


class WedgeModel(LaminarModel):
    """
    A special instance of the LaminarModel where some restrictions, and extensions, of a typical wedge model are
    incorporated
    """
    def __init__(self,
                 litho_fluids: LithoFluids | None = None,
                 top_layers: list | None = None,
                 resolution: Q_ | None = None,
                 depth_to_top: Q_ | None = None,
                 min_thickness: Q_ | None = None,
                 max_thickness: Q_ | None = None,
                 n_traces: int | None = None,
                 **kwargs
                 ):
        """
        Returns a simple wedge model with constant elastic properties in the three layers of the model

        :param litho_fluids:
            LithoFluids
            Need to contain at least 3 LithoFluids for the set up to work smoothly.
                1 for the top and base
                1 for the wedge
                and 1 for the variant of the wedge
        : param top_layers:
            list
            Optional
            List of model layers (of type Layer) that builds a more complex overburden than the default
            "Top" layer.
        :param resolution:
        :param depth_to_top:
        :param min_thickness:
        :param max_thickness:
        :param n_traces:
        :param kwargs:
        """

        #
        # Set up default values:
        #
        if litho_fluids is None:
            litho_fluids = LithoFluids(
                [LithoFluid(default='shale'), LithoFluid(default='brine_sst'), LithoFluid(default='oil_sst')]
            )
        if resolution is None:
            resolution = Q_(0.1, 'm')
        if depth_to_top is None:
            depth_to_top = Q_(3000., 'm')
        if min_thickness is None:
            min_thickness = Q_(0.1, 'm')
        if max_thickness is None:
            max_thickness = Q_(50., 'm')
        if n_traces is None:
            n_traces = 51

        top_thickness = 1.0 * max_thickness

        self.wedge_thickness = np.linspace(min_thickness.to('m').magnitude, max_thickness.to('m').magnitude, n_traces)

        def wedge(i):
            return min_thickness + (max_thickness - min_thickness) * i / (n_traces - 1)

        def reverse_wedge(i):
            return top_thickness + (max_thickness - min_thickness) - (max_thickness - min_thickness) * i / (n_traces - 1)

        #
        # Create the layers that define the model
        #
        layer1 = Layer(name='Top', thickness=top_thickness, litho_fluid=litho_fluids.litho_fluids[0])
        layer2 = Layer(name='Wedge', thickness=wedge, litho_fluid=litho_fluids.litho_fluids[1])
        layer2_variant = Layer(name='Wedge', case='Perturb', thickness=wedge, litho_fluid=litho_fluids.litho_fluids[2])
        layer3 = Layer(name='Bottom', thickness=reverse_wedge, litho_fluid=litho_fluids.litho_fluids[0])
        if top_layers is not None:
            layers = top_layers + [layer2, layer2_variant, layer3]
        else:
            layers = [layer1, layer2, layer2_variant, layer3]

        model = Model(layers=layers,
                      depth_to_top=depth_to_top,
                      trace_index_range=np.arange(n_traces)
                      )
        super().__init__(
            model=model, litho_fluids=litho_fluids, resolution=resolution
        )

    def draw(self):
        from bokeh.plotting import column, row, figure
        from bokeh.models import Select, HoverTool, Range1d, CrosshairTool, Span
        from bokeh.layouts import gridplot

        from blixt_rp.core.seismic import picks_from_seismic_cds
        from blixt_rp.plotting.log_plotter import default_tools

        global max_amp
        global min_amp

        # Create the default Quasi 2D widgets
        (title, lf_table, add_row, delete_row, update, model_table,
         add_row_m, delete_row_m, update_m, grid,
         controls, synth_2d_cds) = self.draw_2d()

        # Create the widgets for plotting the Top / Base wedge seismic picks, which we calculate the
        # apparent thickness from
        picks = ['global_max', 'global_min', 'extract',
                 'nearest_max', 'nearest_max_above', 'nearest_max_below',
                 'nearest_min', 'nearest_min_above', 'nearest_min_below', 'none']

        horizons_cds = self.model.horizons_cds(resolution=self.resolution)

        # Draw the outline of the wedge
        # for h_name in ['Top Top', 'Top Wedge', 'Top Bottom']:
        for h_name in ['Top Wedge', 'Top Bottom']:
            grid.children[0].children[2][0].scatter(x='x', y=h_name, marker='dash', source=horizons_cds, size=10)

        top_picker = Select(
            title='Pick top wedge:',
            value='none',
            options=picks
        )
        base_picker = Select(
           title='Pick base wedge:',
            value='none',
            options=picks
        )

        depth = self.model.depth_array(self.resolution).to('m').magnitude

        top_amps, top_depths_indxs = picks_from_seismic_cds(synth_2d_cds, top_picker.value, horizons_cds.data['Top Wedge index'])
        base_amps, base_depths_indxs = picks_from_seismic_cds(synth_2d_cds, base_picker.value, horizons_cds.data['Top Bottom index'])

        horizons_cds.data['Thickness'] = self.wedge_thickness
        horizons_cds.data['Top pick'] = [depth[_i] for _i in top_depths_indxs]
        horizons_cds.data['Top amp'] = np.absolute(top_amps)
        horizons_cds.data['Base pick'] = [depth[_i] for _i in base_depths_indxs]
        horizons_cds.data['Base amp'] = np.absolute(base_amps)
        horizons_cds.data['Apparent thickness'] = [_b - _t for _b, _t in zip(
            horizons_cds.data['Base pick'], horizons_cds.data['Top pick'])]

        def find_extreme_amps():
            global max_amp
            global min_amp
            # for amps in [horizons_cds.data['Top amp'], horizons_cds.data['Base amp']]:
            #     if np.max(amps) > max_amp:
            #         max_amp = np.max(amps)
            #     if np.min(amps) < min_amp:
            #         min_amp = np.min(amps)
            max_amp = np.max(synth_2d_cds.data['value'][0])
            min_amp = np.min(synth_2d_cds.data['value'][0])

        find_extreme_amps()


        def do_top_pick():
            _top_amps, _top_depths_indxs = picks_from_seismic_cds(synth_2d_cds, top_picker.value, horizons_cds.data['Top Wedge index'])
            horizons_cds.data['Top pick'] = [depth[_i] for _i in _top_depths_indxs]
            horizons_cds.data['Top amp'] = np.absolute(_top_amps)
            horizons_cds.data['Apparent thickness'] = [_b - _t for _b, _t in zip(
                horizons_cds.data['Base pick'], horizons_cds.data['Top pick'])]

        def do_base_pick():
            _base_amps, _base_depths_indxs = picks_from_seismic_cds(synth_2d_cds, base_picker.value, horizons_cds.data['Top Bottom index'])
            horizons_cds.data['Base pick'] = [depth[_i] for _i in _base_depths_indxs]
            horizons_cds.data['Base amp'] = np.absolute(_base_amps)
            horizons_cds.data['Apparent thickness'] = [_b - _t for _b, _t in zip(
                horizons_cds.data['Base pick'], horizons_cds.data['Top pick'])]

        def top_picker_function(attr, old, new):
            do_top_pick()

        def base_picker_function(attr, old, new):
            do_base_pick()

        #
        # Draw the picks
        #
        grid.children[0].children[2][0].scatter(x='x', y='Top pick', marker='circle', source=horizons_cds, size=7)
        grid.children[0].children[2][0].scatter(x='x', y='Base pick', marker='square', source=horizons_cds, size=7)

        top_picker.on_change('value', top_picker_function)
        base_picker.on_change('value', base_picker_function)
        update_m.on_click(do_top_pick)
        update_m.on_click(do_base_pick)

        # Create new figures to which we can add the new apparent thickness plots
        # First extract the width of the columns in the original gridplot
        _last_width = 0  # Width of the last column which plots the wedge synthetics
        _total_width = 0
        # Now that draw_2d() is using the new draw_ext_toolbar() method of LogPlotter, we need to change this
        # a little
        # for _child in grid.children:
        for _child in grid.children[0].children:
            _last_width = _child[0].width
            _total_width += _last_width
        _height = 300
        p_scatter = figure(width=_total_width - _last_width, height=_height, tools=default_tools)
        p_lines = figure(width=_last_width, height=_height, tools=default_tools)

        p_scatter.scatter(x='Apparent thickness', y='Top amp', source=horizons_cds)
        p_lines.line(
            x='Thickness', y='Apparent thickness', source=horizons_cds, legend_label='Apparent thickness',
            line_color='black', line_width=2.0, line_dash='dashed'
        )
        print('XXX Range1d range: ', min_amp, max_amp)
        p_lines.extra_y_ranges['Amplitude'] = Range1d(min_amp, max_amp)
        p_lines.line(
            x='Thickness', y='Top amp', source=horizons_cds, legend_label='|Top ampl|', y_range_name='Amplitude',
            line_color='red', line_width=2.0
        )
        p_lines.line(
            x='Thickness', y='Base amp', source=horizons_cds, legend_label='|Base ampl|', y_range_name='Amplitude',
            line_color='blue'
        )

        # Style the new plots
        _w = Span(dimension="width", line_dash="dashed", line_width=1)
        _h = Span(dimension="height", line_dash="dashed", line_width=1)
        for _p in [p_scatter, p_lines]:
            _p.toolbar.logo = None
            _p.add_tools(CrosshairTool(overlay=(_w, _h)))
            hover = _p.select(dict(type=HoverTool))
            hover.tooltips = [("(x,y)", "($x, $y)")]
            _p.xaxis.visible = True
            _p.yaxis.visible = False
        p_scatter.yaxis.visible = True
        p_scatter.xaxis.axis_label = 'Apparent thickness [m]'
        p_scatter.yaxis.axis_label = '|Top ampl|'
        p_lines.legend.click_policy = 'hide'
        p_lines.legend.location = 'bottom_right'
        p_lines.legend.label_text_font_size = '8pt'
        p_lines.xaxis.axis_label = 'Wedge thickness [m]'

        new_grid = gridplot([[p_scatter, p_lines]], toolbar_location='right', merge_tools=True)

        return (lf_table, row(add_row, delete_row, update), model_table, row(update_m), grid,
                row(controls, top_picker, base_picker), new_grid, synth_2d_cds)

def build_layered_model(depth_to_target, overburden_thickness, target_thickness,
                        overburden, target, underburden, domain='TWT') -> Model:
    """
    Returns a simple 1D model of three layers
    :param depth_to_target:
        float
        depth in TWT [s] or Z [m] to top of target
    :param overburden_thickness:
        float
        thickness in TWT [s] or Z [m] to top of wedge
    :param target_thickness:
    :param overburden:
        dict
        with keys: 'vp', 'vs', and 'rho', and the Vp, Vs, and density values as values
    :param target:
    :param underburden:
    :param domain:
    :return:
        Model
    """
    top_layer = Layer(thickness=overburden_thickness, **overburden, domain=domain)
    target_layer = Layer(thickness=target_thickness, **target, target=True, domain=domain)
    base_layer = Layer(thickness=overburden_thickness, **underburden, domain=domain)
    return Model(
        depth_to_top=depth_to_target - overburden_thickness,
        layers=[top_layer, target_layer, base_layer]
    )


def build_wedge(
        depth_to_wedge: Q_,
        from_thickness: Q_,
        to_thickness: Q_,
        n_traces: int,
        overburden: LithoFluid,
        target: LithoFluid,
        underburden: LithoFluid):
    """
    Returns a simple wedge model with constant elastic properties in the three layers of the model

    :param depth_to_wedge:
        float
        depth in TWT [s] or Z [m] to top of wedge
    :param from_thickness:
        float
        Thickness, in TWT [s] or Z [m], at the left most side of the wedge
    :param to_thickness:
    :param n_traces:
        int
        Number of traces
    :param overburden:
        LithoFluid
        with keys: 'vp', 'vs', and 'rho'
    :param target:
        LithoFluid
        with keys: 'vp', 'vs', and 'rho'
    :param underburden:
        LithoFluid
        with keys: 'vp', 'vs', and 'rho'

    """
    if from_thickness > to_thickness:
        top_thickness = 1.0 * from_thickness

        def wedge(i):
            return from_thickness - (from_thickness - to_thickness) * i / (n_traces - 1)

        def reverse_wedge(i):
            return top_thickness + (from_thickness - to_thickness) * i / (n_traces - 1)
    else:
        top_thickness = 1.0 * to_thickness

        def wedge(i):
            return from_thickness + (to_thickness - from_thickness) * i / (n_traces - 1)

        def reverse_wedge(i):
            return top_thickness + (to_thickness - from_thickness) - (to_thickness - from_thickness) * i / (n_traces - 1)

    def top(i):
        return top_thickness

    top_layer = Layer(thickness=top, litho_fluid=overburden)
    wedge_layer = Layer(thickness=wedge, litho_fluid=target, target=True)
    base_layer = Layer(thickness=reverse_wedge, litho_fluid=underburden)

    return Model(
        depth_to_top=depth_to_wedge - top_thickness,
        layers=[top_layer, wedge_layer, base_layer],
        trace_index_range=np.arange(n_traces)
    )


def build_saturation_wedge(depth_to_wedge, thickness, to_hc_saturation, n_traces, overburden, brine_target,
                           underburden, poro, rpt_parameters: dict | None = None, domain='TWT'):
    """
    Returns a simple "saturation" wedge model where the HC saturation in the target goes from 0 to 'to_hc_saturation'

    :param depth_to_wedge:
        float
        depth in TWT [s] or Z [m] to top of wedge
    :param thickness:
        float
        Thickness, in TWT [s] or Z [m] of the target
    :param to_hc_saturation:
        float
        Fractional HC saturation in the right side of the 'wedge'
    :param n_traces:
        int
        Number of traces
    :param overburden:
        dict
        with keys: 'vp', 'vs', and 'rho'
    :param brine_target:
        dict
        with keys: 'vp', 'vs', and 'rho'
        This should represent the brine filled target where the brine is gradually replaced with HC towards the right
        of the 'wedge'
    :param underburden:
        dict
        with keys: 'vp', 'vs', and 'rho'
    :param poro:
        float
        Porosity, fractional
    :param rpt_parameters:
        dict
        Dictionary with the rock physics parameters we will use in the Gassmann fluid subsitution.
        See rp_core.rpt_parameters() for more info

    """
    from blixt_rp.rp.rp_core import gassmann_vel
    if 1 >= to_hc_saturation > 0:
        pass
    else:
        raise IOError("'to_hc_saturation' must be larger than 0 and less or equal to 1")

    def replace_fluid(i):
        def s_hc(_i):
            return to_hc_saturation * _i / (n_traces - 1)

        k_f2 = (1. - s_hc(i)) * rpt_parameters['k_b'] + s_hc(i) * rpt_parameters['k_hc']
        rho_f2 = (1. - s_hc(i)) * rpt_parameters['rho_b'] + s_hc(i) * rpt_parameters['rho_hc']
        # print('XXX1', k_f2, rho_f2, s_hc(i))
        return gassmann_vel(
            brine_target['vp'],
            brine_target['vs'],
            brine_target['rho'],
            rpt_parameters['k_b'],
            rpt_parameters['rho_b'],
            k_f2,
            rho_f2,
            rpt_parameters['k_min'], poro
        )

    def vp(i):
        return replace_fluid(i)[0]

    def vs(i):
        return replace_fluid(i)[1]

    def rho(i):
        return replace_fluid(i)[2]

    top_layer = Layer(thickness=0.5 * thickness, **overburden, domain=domain)
    wedge_layer = Layer(thickness=thickness, vp=vp, vs=vs, rho=rho, target=True, domain=domain)
    base_layer = Layer(thickness=0.5 * thickness, **underburden, domain=domain)

    return Model(
        depth_to_top=depth_to_wedge - 0.5 * thickness,
        layers=[top_layer, wedge_layer, base_layer],
        trace_index_range=np.arange(n_traces),
        domain=domain
    )


def tuning_wedge_analysis(depth_to_wedge, from_thickness, to_thickness, n_traces, overburden, target, underburden,
                          sample_rate, wavelet, angle=None, plot_domain='TWT', title=None, savefig=None, overburden_vel=3000.,
                          extract_avo_at=None, scaling=None):
    """
    Tuning wedge is always done on a model in time domain, but the result can be plotted in depth domain
    """
    if scaling is None:
        scaling = 40.
    if angle is None:
        angle = 0.

    if from_thickness > to_thickness:  # Wedge pointing rightwards
        loc = 'lower left'
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [0.2, 1]})
        axs[1, 0].set_axis_off()
        wiggle_ax = axs[0, 1]
        model_ax = axs[0, 0]
        wedge_ax = axs[1, 1]

    else:  # Wedge pointing leftwards
        loc = 'lower right'
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 0.2]})
        axs[1, 1].set_axis_off()
        wiggle_ax = axs[0, 0]
        model_ax = axs[0, 1]
        wedge_ax = axs[1, 0]

    fig.subplots_adjust(wspace=0.)

    if plot_domain == 'TWT':
        wedge_thickness = np.linspace(from_thickness, to_thickness, n_traces) * 1000.  # ms
        wedge_xlabel = 'Wedge thickness [ms]'
    else:
        wedge_thickness = np.linspace(from_thickness, to_thickness, n_traces) * target['vp'] * 0.5  # m
        wedge_xlabel = 'Wedge thickness [m]'

    # TODO
    # Double check that it should be 'TWT' below!
    m = build_wedge(depth_to_wedge, from_thickness, to_thickness, n_traces,
                    overburden, target, underburden, domain='TWT')
    avo_curves, amps, min_amps, max_amps, apparent_thickness = plot_wiggles(
        m, sample_rate, wavelet, angle=angle, ax=wiggle_ax, extract_avo_at=extract_avo_at,
        plot_domain=plot_domain, overburden_vel=overburden_vel, scaling=scaling)
    if title is not None:
        fig.suptitle(title)

    if loc == 'lower left':
        m[0].plot(ax=model_ax, kwargs1d={'yticks': False, 'legend': False})
    else:
        m[-1].plot(ax=model_ax, kwargs1d={'yticks': False, 'legend': False})
    model_ax.autoscale(tight=True)

    wedge_ax.plot(wedge_thickness, np.abs(np.array(max_amps)), label='|Max|')
    wedge_ax.plot(wedge_thickness,  np.abs(np.array(min_amps)), label='|Min|')
    wedge_ax.set_ylabel('Amplitude')
    wedge_ax.set_xlabel(wedge_xlabel)
    wedge_ax.legend(loc=loc)

    ax2 = wedge_ax.twinx()
    color = 'tab:green'
    if plot_domain == 'TWT':
        ax2.plot(wedge_thickness, np.array(apparent_thickness['twt']) * 1000., color=color)
        ax2.set_ylabel('Apparent thickness [ms]', color=color)
    else:
        ax2.plot(wedge_thickness, np.array(apparent_thickness['z']), color=color)
        ax2.set_ylabel('Apparent thickness [m]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if savefig:
        fig.savefig(savefig)
    else:
        plt.show()


def saturation_wedge_analysis(depth_to_wedge, thickness, to_hc_saturation, n_traces, overburden, brine_target,
                              underburden, sample_rate, wavelet, poro, plot_domain='TWT', title=None, savefig=None,
                              overburden_vel=3000., extract_avo_at=None, scaling=None,
                              rpt_parameters: dict | None = None, ):
    if scaling is None:
        scaling = 10.

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 0.2]})
    axs[1, 1].set_axis_off()
    wiggle_ax = axs[0, 0]
    model_ax = axs[0, 1]
    wedge_ax = axs[1, 0]
    fig.subplots_adjust(wspace=0.)

    m = build_saturation_wedge(
        depth_to_wedge, thickness, to_hc_saturation, n_traces,
        overburden, brine_target, underburden, poro, rpt_parameters)

    hc_sat = np.linspace(0, to_hc_saturation, n_traces)

    avo_curves, amps, min_amps, max_amps, apparent_thickness = plot_wiggles(
        m, sample_rate, wavelet, ax=wiggle_ax, extract_avo_at=extract_avo_at,
        plot_domain=plot_domain, overburden_vel=overburden_vel, scaling=scaling)
    # wiggle_ax.set_ylim(wiggle_ax.get_ylim()[::-1])
    m[0].plot(ax=model_ax, kwargs1d={'yticks': False, 'legend': False})
    wedge_ax.plot(hc_sat, np.abs(np.array(min_amps)), label='|Min|')
    wedge_ax.set_ylabel('Amplitude')
    wedge_ax.set_xlabel('HC saturation')
    wedge_ax.legend(loc='upper left')
    wedge_ax.grid(True)
    if title is not None:
        fig.suptitle(title)
    if savefig:
        fig.savefig(savefig)


def laminar_model_analysis(
    model: Model,
    sample_rate: float,
    wavelet: dict,
    plot_domain='TWT',
    title=None, savefig=None,
    overburden_vel=3000.,
    extract_avo_at=None,
    extract_on='exact',
    avo_plot_position=None,
    scaling=None):
    """
    Plot a model together with wiggles and more
    """

    fig, axs = plt.subplots(1, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [1, 0.4]})
    # TODO Try using a shared y axis
    # axs[1, 1].set_axis_off()
    wiggle_ax = axs[0]
    model_ax = axs[1]

    fig.subplots_adjust(wspace=0.)

    avo_curves, amps, min_amps, max_amps, apparent_thickness = plot_wiggles(
        model, sample_rate, wavelet, ax=wiggle_ax, extract_avo_at=extract_avo_at,
        avo_plot_position=avo_plot_position, extract_on=extract_on,
        plot_domain=plot_domain, overburden_vel=overburden_vel, scaling=scaling)
    if title is not None:
        fig.suptitle(title)

    # wiggle_ax.set_ylim(wiggle_ax.get_ylim()[::-1])
    model[0].plot(ax=model_ax, kwargs1d={'yticks': False, 'legend': False})
    model_ax.autoscale(tight=True)

    if savefig:
        fig.savefig(savefig)

def modify_dict(_input):
    _dict = {}
    for _case in list(_input.keys()):
        for _log in list(_input[_case].keys()):
            # Each LogCurve has its own ColumnDataSource (log values and depth)
            _log_cds = _input[_case][_log].cds
            for _var in list(_log_cds.data.keys()):
                # Join all logs to one common ColumnDataSource
                _dict[_var] = _log_cds.data[_var]
    return _dict

def detect_change_in_cases(new_cases, prev_cases):
    _dict = {'new': [], 'removed': []}
    if new_cases == prev_cases:
        return _dict
    for _case in new_cases:
        if _case not in prev_cases:
            _dict['new'].append(_case)
    for _case in prev_cases:
        if _case not in new_cases:
            _dict['removed'].append(_case)
    return _dict

def check_domain(_input):
    """
    Check domain of the input
    :param _input:
        either a function that returns a pint.Quantity, or a pint.Quantity
    :return:
    """
    domain = None
    if isinstance(_input, types.FunctionType):
        # A test if the function returns a length or time value
        if _input(0).check('[length]'):
            domain = 'Z'
        elif _input(0).check('[time]'):
            domain = 'TWT'
    elif _input.check('[length]'):
        domain = 'Z'
    elif _input.check('[time]'):
        domain = 'TWT'
    else:
        error_txt = 'Input must be given in either Time, Depth or a function'
        print_info(error_txt, 'error', logger, 'IOError')

    return domain

def calc_synth_cds(_vp, _vs, _rho, _twt, _dt, _angles, avo_or_eei, freq):
    if avo_or_eei == 'avo':
        _avo_angles = _angles
        _chi_angles = None
    else:
        _avo_angles = None
        _chi_angles = _angles
    _amp = get_wiggles_in_depth(
        _vp, _vs, _rho, _twt.data, _dt, avo_angles=_avo_angles, chi_angles=_chi_angles, center_frequency=float(freq),
        verbose=False
    )
    return ColumnDataSource({'value': [_amp.T]})


def create_traces(_synth_cds, _depth, _title, angles, avo_or_eei):
    from blixt_rp.core.seismic import SeismicTraces
    return SeismicTraces(
        x=angles, y=_depth, traces=None, cds=_synth_cds, trace_type=avo_or_eei, title=_title)


def find_extremes(cds: ColumnDataSource) -> dict:
    """
    Finds the extremes for the AI and VpVs parameters in the given CDS
    :param cds:
    :return:
        dict
        Dictionary with keys: 'AI_XXX' and 'VpVs_XXX', where XXX represent Case names (e.g. 'Oil')
        and list of min and max values for each of this
    """
    _dict = {'ai': [], 'vpvs': []}
    for _type in list(_dict.keys()):
        _last_min = 1E9
        _last_max = -1E9
        for _key in list(cds.data.keys()):
            if _type in _key:
                if min(cds.data[_key]) < _last_min:
                    _last_min = min(cds.data[_key])
                if max(cds.data[_key]) > _last_max:
                    _last_max = max(cds.data[_key])
        _dict[_type].append(_last_min)
        _dict[_type].append(_last_max)
    return _dict

def total_cds_thickness(cds: ColumnDataSource) -> float:
    # Returns the total thickness of the model based on the models CDS, in the units of the CDS
    _i = 0
    _thickness = 0.
    for _c, _t in zip(cds.data['case'], cds.data['thickness']):
        # Only count the thickness of base case layers
        if _c.lower() == global_base_case_name.lower():
            if _i == 0:
                _thickness = _t
            else:
                _thickness += _t
            _i += 1
    return _thickness



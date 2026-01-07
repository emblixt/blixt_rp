import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import os
import logging
import types
from copy import deepcopy
import pint
from bokeh.models import Column, ColumnDataSource, IntEditor

from .log_curve_new import LogCurve, Depth
from .. import ureg, Q_
from ..rp.rp_core_new import LithoFluids, LithoFluidsTable

# sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')
# Instead of sys.path.append load all PyCharm projects in PyCharm, and they gets added to the sys.path automatically

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot, wavelet_plot
from blixt_utils.plotting.crossplot import cnames
import blixt_rp.rp.rp_core as rp
import blixt_utils.misc.wavelets as bumw
import blixt_utils.io.io as uio
from blixt_utils.utils import print_info

logger = logging.getLogger(__name__)

text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}


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
    last_layer_depth = [model.depth_to_top]*len(i_range)
    ax.plot(i_range, last_layer_depth, **kwargs)
    for _layer in model.layers:
        if isinstance(_layer.thickness, float) or isinstance(_layer.thickness, int):
            _h = np.array([_layer.thickness for _i in i_range])
            _tmp = np.array([last_layer_depth[_i] + _layer.thickness for _i in i_range])
        else:
            _h = np.array([_layer.thickness(_i) for _i in i_range])
            _tmp = np.array([last_layer_depth[_i] + _layer.thickness(_i) for _i in i_range])
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
    ai_line = mlines.Line2D([1], [1], **linestyle_ai)
    vpvs_line = mlines.Line2D([1], [1], **linestyle_vpvs)

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


class Model:
    """
    Object that holds a subsurface model with N layers
    """

    def __init__(self,
                 model_type=None,
                 depth_to_top=None,
                 layers=None,
                 trace_index_range=None,
                 domain=None):
        """
        :param model_type:
            str
            '1D'
                default
            'quasi 2D'
                Layer properties can change "laterally" according to functions provided to the individual layers
        :param depth_to_top:
            float
            depth to top in seconds or meters
        :param layers:
            list of layers, each a Layer object
            The first layer is at the top, and consequent layers beneath
        :param trace_index_range:
            list like object of integers
            used when some layers are quasi 2D layers, to parameterize the "lateral" variation
            eg. np.arange(10)
        :param domain:
            string
            'TWT' (time in seconds) or 'Z' (depth in meters)
        """

        # set the required parameters
        if model_type is None:
            self.model_type = '1D'
        else:
            self.model_type = model_type

        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        last_domain = None
        if len(self.layers) > 0:
            last_domain = self.layers[0].domain
            for i, _layer in enumerate(self.layers):
                if _layer.layer_type == 'quasi 2D':
                    self.model_type = 'quasi 2D'
                if _layer.domain != last_domain:
                    raise ValueError('Not all layers have the same domain')

        if domain is None:
            if last_domain is not None:
                self.domain = last_domain
            else:
                self.domain = 'TWT'
        else:
            if (last_domain is not None) and last_domain != domain:
                raise ValueError(
                    'The layers does not have the same domain ({}) as the model ({})'.format(
                        last_domain, domain
                    ))
            else:
                self.domain = domain

        if depth_to_top is None:
            if self.domain == 'TWT':
                self.depth_to_top = 2.0
            else:
                self.depth_to_top = 3000.
        else:
            self.depth_to_top = depth_to_top

        if trace_index_range is None:
            self.trace_index_range = None
        else:
            self.trace_index_range = trace_index_range

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
                         model_type='1D',
                         domain=self.domain,
                         layers=layers
                         )
        else:
            return self  # if it is a 1D model, return itself. Does this make sense?

    def append(self, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        if layer.domain != self.domain:
            raise ValueError('Appended layer must be in same domain as model')
        self.layers.append(layer)
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


class Layer:
    """
    Class handling one layer of a model
    """

    def __init__(self,
                 thickness: pint.Quantity | types.FunctionType |  None = None,
                 target=None,
                 vp: pint.Quantity | types.FunctionType | None = None,
                 vs: pint.Quantity | types.FunctionType | None = None,
                 rho: pint.Quantity | types.FunctionType | None = None,
                 ntg=None,
                 domain=None,
                 **kwargs
                 ):
        """
        :param thickness:
            float or function
            Thickness of layer in m or s, depending on which domain the model.
            If thickness is a function, it is parametrized by an integer i.
            e.g. thickness = f, where f is a function like:
              def f(i):
                  return 10 + i * 0.5
        :param target:
            bool
            True for target layers
        :param vp:
            float or function
            P velocity in m/s
            if vp is a function, it needs to be parametrized by an integer i
            Say we use the constantcement function to calculate vp where the porosity is assumed to change:
              def vp(i):
                  from blixt_rp.rp.rp_core import constantcement, v_p
                  phi = 0.1 + i * 0.03
                  k_eff, mu_eff = constantcement(37, 45, phi)
                  return v_p(k_eff, mu_eff, 2.6)

        :param vs:
            float or function
            Shear velocity in m/s
        :param rho:
            float or function
            Density in g/cm3
        :param ntg:
            float or function
            net-to-gross, 0 <= ntg >= 1
        :param domain:
            string
            'TWT' (time in seconds) or 'Z' (depth in meters)
        :param index:
            int
            index used in the parameterization of an independent variable, such as thickness, or porosity.
            e.g. phi = 0.05 + index * 0.05
        :param layer_type:
            str
            '1D'
                default
            'quasi 2D'
                Layer properties can change "laterally" according to the functions provided

        :param gross_xxx:
            elastic parameters in the non-reservoir part (gross) of the layer
        :param thin_bed_factor:
            int
            Number of internal layers in the reservoir (net) part of the layer
            If 1, the net portion is taken up by one homogeneous layer
        """
        # get the required parameters
        self.resolution = None
        self.layer_type = '1D'

        if target is None:
            self.target = False
        else:
            self.target = target

        if domain is not None:
            raise ValueError('domain is now set automatically, and should not be set manually')

        if thickness is None:
            thickness = Q_(0.1, 's')
        if thickness.check('[length]'):
            domain = 'Z'
        elif thickness.check('[time]'):
            domain = 'TWT'
        else:
            error_txt = 'Thickness must be given in either Time or Depth'
            print_info(error_txt, 'error', logger, 'IOError')
        self._thickness = thickness

        if vp is None:
            self.vp = Q_(3600., 'm/s')
        else:
            self.vp = vp.to('m/s')

        if vs is None:
            self.vs = Q_(1800., 'm/s')
        else:
            self.vs = vs.to('m/s')

        if rho is None:
            self.rho = Q_(2.3, 'gram/cm^3')
        else:
            self.rho = rho.to('gram/cm^3')

        if ntg is None:
            self.ntg = 1.
        else:
            self.ntg = ntg

        self.domain = domain

        if not (0 <= self.ntg <= 1):
            raise ValueError('NTG must be between 0 and 1')

        # Get the net-to-gross related keyword arguments
        if self.ntg < 1 and self.target:
            self.gross_vp = kwargs.pop('gross_vp', 3400.)
            self.gross_vs = kwargs.pop('gross_vs', 1000.)
            self.gross_rho = kwargs.pop('gross_rho', 2.)
            self.thin_bed_factor = kwargs.pop('thin_bed_factor', 3)

        for arg in [self._thickness, self.vp, self.vs, self.rho, self.ntg]:
            if isinstance(arg, types.FunctionType):
                self.layer_type = 'quasi 2D'

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, new_thickness):
        if new_thickness.check('[length]'):
            self.domain = 'Z'
        elif new_thickness.check('[time]'):
            self.domain = 'TWT'
        elif isinstance(new_thickness, types.FunctionType):
            # TODO Insert a test if the function returns a length or time value
            pass
        else:
            error_txt = 'Thickness must be given in either Time, Depth or a function'
            print_info(error_txt, 'error', logger, 'IOError')

        self._thickness = new_thickness

    def realize_layer(self, resolution, voigt_reuss_hill=False, index=0):
        """
        Realize the current layer by returning arrays of the elastic properties with the given resolution
        *Note* that a layer is always realized in TWT domain, so the resolution is interpreted as dt in seconds

        :param resolution:
            float
            resolution in TWT [s]
        :param voigt_reuss_hill:
            bool
            If true, the Voigt Reuss Hill average for the given NTG is used when returning the elastic parameters
        :param index:
            int
            Integer that specifies the index of "laterally" varying quasi 2D layer

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

        # Let the domain decide how the layer is realized
        if self.domain == 'TWT':
            n = int(layer_thickness / resolution)
        else:
            twt_thickness = 2. * layer_thickness / layer_vp
            n = int(twt_thickness / resolution)

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
        layers=[top_layer, target_layer, base_layer],
        domain=domain
    )


def build_wedge(depth_to_wedge, from_thickness, to_thickness, n_traces, overburden, target, underburden, domain='TWT'):
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
        dict
        with keys: 'vp', 'vs', and 'rho'
    :param target:
        dict
        with keys: 'vp', 'vs', and 'rho'
    :param underburden:
        dict
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

    top_layer = Layer(thickness=top, **overburden, domain=domain)
    wedge_layer = Layer(thickness=wedge, **target, target=True, domain=domain)
    base_layer = Layer(thickness=reverse_wedge, **underburden, domain=domain)

    return Model(
        depth_to_top=depth_to_wedge - top_thickness,
        layers=[top_layer, wedge_layer, base_layer],
        trace_index_range=np.arange(n_traces),
        domain=domain
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


class ModelLayer(Layer):
    from blixt_rp.rp.rp_core_new import LithoFluid
    """
    Class to hold one layer of a seismic model
    It has some overlap with the Layer class in blixt_rp.core.models.py, but is tuned towards using bokeh interactive
    plotting
    """
    def __init__(self,
                 number: int,
                 case: str | None,
                 color: str | None  = '#D9D9D9',
                 thickness: pint.Quantity | None  = Q_(25., 'm'),
                 litho_fluid: LithoFluid | None = None,
                 **kwargs
                 ):
        """

        :param number:
            Integer that should reflect the internal order of the different layers.
            number = 1 the layer is at the top, increasing number are deeper down
        :param case:
            String indicating which case the layer represents.
            E.G. Each layer is named 'Base'  by default, and if layer 3 is the target, we can add a new layer 3 with
            case string 'HC' and let that represent the hydrocarbon case
        :param color:
            String that gives the color of that layer0
        :param thickness:
            Thickness in either time or length units.
        :param litho_fluid:
            The LithoFluid class associated with this layer
        """
        domain = None
        self._number = number
        if case is None:
            case = 'Base'
        self._case = case
        if color is None:
            color = '#D9D9D9'
        self.color = color
        if thickness is None:
            thickness = Q_(25., 'm')
        self._litho_fluid = litho_fluid

        super().__init__(
            thickness=thickness,
            target=kwargs.pop('target', False),
            vp=litho_fluid.vp.to('m/s'),
            vs=litho_fluid.vs.to('m/s'),
            rho=litho_fluid.rho.to('grams/cm^3'),
            ntg=kwargs.pop('ntg', 1.),
            domain=domain,
            **kwargs
        )



    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, new_number):
        self._number = new_number

    @property
    def case(self):
        return self._case

    @case.setter
    def case(self, new_case):
        self._case = new_case

    @property
    def litho_fluid(self):
        return self._litho_fluid

    @litho_fluid.setter
    def litho_fluid(self, new_litho_fluid):
        self._litho_fluid = new_litho_fluid
        self.vp = new_litho_fluid.vp.to('m/s').magnitude
        self.vs = new_litho_fluid.vs.to('m/s').magnitude
        self.rho = new_litho_fluid.rho.to('grams/cm^3').magnitude

    def realize_layer(self,
                      resolution: pint.Quantity, voigt_reuss_hill: bool =False, index: int = 0):
        """
        Overrides the 'realize_layer' of the parent Layer class, which is a bit old-fashioned and does not
        take some new methods into account.
        BUT it doesn't accept cases where vp, vs and rho are functions - which the parent class (Layer) does.

        Realize the current layer by returning arrays of the elastic properties with the given resolution

        :param resolution:
            pint.Quantity
            Determines the resolution of the layer.
            Must match the dimension of the thickness of the layer

        :param voigt_reuss_hill:
        :param index:
        :return:
        """
        raise_domain_error = False
        if resolution.check('[length]'):
            if not self.domain == 'Z':
                raise_domain_error = True
        elif resolution.check('[time]'):
            if not self.domain == 'TWT':
                raise_domain_error = True
        if raise_domain_error:
            err_txt = 'The dimensions of resolution ({}) and domain ({}) does not match'.format(
                resolution.units, self.domain
            )
            print_info(err_txt, 'error', logger, 'IOError')

        n = int(self.thickness.magnitude / resolution.to(self.thickness.units).magnitude)
        this_vp = np.ones(n) * self.vp
        this_vs = np.ones(n) * self.vs
        this_rho = np.ones(n) * self.rho
        return this_vp, this_vs, this_rho

class ModelTable:
    """
    Returns a bokeh DataTable populated with rows of single layers
    """
    from bokeh.models import ColumnDataSource

    def __init__(self,
                 layers: list | None = None,
                 litho_fluid_cds:ColumnDataSource | None = None,
                 depth_to_top: pint.Quantity | None = None,
                 base_case_name: str | None = None,
                 width: int | None = None,
                 height: int | None = None):
        """

        :param layers:
            List of TableLayer objects
        :param litho_fluid_cds:
            ColumnDataSource of the LithoFluidsTable that contains all the different LithoFluids we can use to
            populate the model
        :param  base_case_name
            str
            Name of the default base case
        :param width:
        :param height:
        """
        if layers is None:
            layers = []
        self.layers = layers
        self.litho_fluid_cds = litho_fluid_cds
        if base_case_name is None:
            base_case_name = 'Base'
        self.base_case_name = base_case_name
        if depth_to_top is None:
            depth_to_top = Q_(3000., 'm')
        self.depth_to_top = depth_to_top
        if width is None:
            width = 700
        self.width = width
        if height is None:
            height = 135
        self.height = height
        self.keys = ['number', 'case', 'color', 'thickness', 'litho_fluid']

    @property
    def input_litho_fluids(self):
        if self.litho_fluid_cds is not None:
            return self.litho_fluid_cds.data['name']
        else:
            return []

    @property
    def cds(self):
        _dict = {_x: [] for _x in self.keys}
        _dict['top'] = []
        previous_thickness = 0.
        previous_top = float(self.depth_to_top.to('m').magnitude)
        for _i, _layer in enumerate(self.layers):
            # TODO The top attribute is not calculated correctly, it should iterate under 'number' first of all,
            # But there is probably something more to
            if _i == 0:
                _dict['top'].append(previous_top)
                previous_thickness = float(_layer.thickness.to('m').magnitude)
            else:
                _dict['top'].append(previous_top + previous_thickness)
                previous_thickness = float(_layer.thickness.to('m').magnitude)
                previous_top += previous_thickness
            for _key in self.keys:
                if _key == 'thickness':
                    _dict[_key].append(_layer.thickness.magnitude)
                elif _key == 'litho_fluid':
                    _dict[_key].append(_layer.litho_fluid.name)
                elif _key == 'number':
                    _dict[_key].append(_layer.number)
                elif _key == 'case':
                    _dict[_key].append(_layer.case)
                else:
                    _dict[_key].append(_layer.__dict__[_key])
        return ColumnDataSource(_dict)

    @property
    def numbers(self):
        return list(set([int(_n) for _n in self.cds.data['number']]))

    @property
    def duplicate_numbers(self) -> list:
        """
        Identify which layers that have more than one case
        :return:
        """
        seen = set()
        duplicates = []

        for item in [int(_n) for _n in self.cds.data['number']]:
            if item in seen:
                if item not in duplicates:  # Optional: to ensure the duplicates list itself has no repeated entries
                    duplicates.append(item)
            else:
                seen.add(item)
        return duplicates

    @property
    def cases(self):
        return list(set(self.cds.data['case']))

    @property
    def litho_fluids(self):
        return self.cds.data['litho_fluid']

    @litho_fluids.setter
    def litho_fluids(self, new_litho_fluids):
        if isinstance(new_litho_fluids, list):
            self.cds.data['litho_fluid'] = new_litho_fluids
        else:
            print_info('new_litho_fluids is not a list', 'warning', logger)

    def add_litho_fluid(self, litho_fluid: str):
        self.cds.data['litho_fluid'].append(litho_fluid)

    def get_layer(self, number: int, case: str, base_case: str | None = None) -> ModelLayer | None:
        """
        Returns the layer with the given number and case if that case exists. Else it returns the base case layer
        :param number:
        :param case:
        :param base_case:
            str
            Name of the base case scenario
        :return:
        """
        if base_case is None:
            base_case = self.base_case_name
        base_case_layer = None
        for _layer in self.layers:
            if _layer.number == number and _layer.case.lower() == base_case.lower():
                base_case_layer = _layer
            if _layer.number == number and _layer.case.lower() == case.lower():
                return _layer
        return base_case_layer

    def depth_range(self):
        """
        Returns the total thickness of the model.
        NOTE. Only sums up the unique layers (which have different layer numbers)
        :return:
        """
        i = 0
        previous_layers = []
        thickness = None
        for _layer in self.layers:
            if int(_layer.number) in previous_layers:
                # Skip layers which are non-unique
                continue
            previous_layers.append(int(_layer.number))

            if i == 0:
                thickness = _layer.thickness
            else:
                thickness += _layer.thickness
            i += 1
        return thickness

    def md_array(self, resolution: pint.Quantity) -> pint.Quantity:
        """
        Returns a pint.Quantity array with length similar to the total length of the model (sum layer thicknesses)
        with a sampling matching the desired resolution

        :param resolution:
            pint.Quantity
            Determines the resolution of the layer.
            Must match the dimension of the model
        :return:
        """
        md = self.depth_to_top + Q_(
            np.arange(
                0.,
                self.depth_range().magnitude,
                resolution.to(self.depth_range().units).magnitude),
            self.depth_range().units)
        return md

    def interface_depths(self, cds: ColumnDataSource) -> list:
        """
        Returns a list of depths, one for each interface (except top and bottom) that are found in the Column Data Source
        of the ModelTable
        :param cds:
        :return:
        """
        _list = []
        i = 0
        previous_layers = []
        previous_thickness = 0.
        interface_depth = self.depth_to_top
        for _layer in self.layers:
            if int(_layer.number) in previous_layers:
                # Skip layers which are non-unique
                continue
            previous_layers.append(int(_layer.number))

            if i > 0:
                # interface_depth += _layer.thickness
                interface_depth += previous_thickness
                _list.append(interface_depth)
            previous_thickness = _layer.thickness
            i += 1

        return _list

    def interface_indexes(self, md, cds: ColumnDataSource) -> list:
        """
        Returns a list of indexes, one for each interface (except top and bottom) that are found in the Column Data Source
        of the ModelTable
        :param md:
            output from self.md_array()
        :param cds:
            ColumnDataSource of the model table
        :return:
        """

        interface_depths = self.interface_depths(cds)
        i_inds = []
        for _depth in interface_depths:
            i_inds.append(int(
                np.argmin(np.sqrt((md - _depth)**2))
            ))
        return i_inds

    def table_columns(self):
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
        column_names[0] = 'Layer'
        table_columns = []
        for i, column_key in enumerate(self.keys):
            if column_key == 'litho_fluid':
                # _editor = SelectEditor(options=self.input_litho_fluids)
                _editor = StringEditor(completions=self.input_litho_fluids)
                _formatter = StringFormatter(font_style='bold')
            elif column_key == 'thickness':
                _editor = NumberEditor()
                _formatter = NumberFormatter(format='0.0[0]')
            elif column_key == 'number':
                _editor = IntEditor()
                _formatter = NumberFormatter(format='0.')
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
        return table_columns

    def realize(self, resolution: pint.Quantity, mod_cds: ColumnDataSource) -> dict:
        """

        :param resolution:
            pint.Quantity
            Determines the resolution of the layer.
            Must match the dimension of the model
        :param mod_cds:
        :return:
        """
        if len(self.cases) > 2:
            print_info('More than two cases might not be realized as desired', 'error', logger, IOError)
        _dict = {_case: {} for _case in self.cases}
        _nrs = self.numbers  # Layer numbers. Low number are above higher numbers
        _target_layers = self.duplicate_numbers  # Duplicate numbers indicate that a layer has different cases

        md = self.md_array(resolution)

        i_inds = self.interface_indexes(md, mod_cds)

        # iterate over all layers
        for _case in self.cases:
            # Populate arrays with the values of each layer. Units are not important as each layer is
            # converted to m/s and gr/cm3 when initiated
            _vp = np.zeros(len(md))
            _vs = np.zeros(len(md))
            _rho = np.zeros(len(md))
            _twt = np.zeros(len(md))
            for i, n in enumerate(_nrs):  # This is also the correct order, as lower numbers are at the top :-)
                this_layer = self.get_layer(n, _case)
                print('XXX', i, n, _case, this_layer.case)
                if i == 0:  # First layer
                    _vp[:i_inds[i]] = this_layer.vp
                    _vs[:i_inds[i]] = this_layer.vs
                    _rho[:i_inds[i]] = this_layer.rho
                elif i > len(i_inds) - 1:  # last layer
                    _vp[i_inds[i-1]:] = this_layer.vp
                    _vs[i_inds[i-1]:] = this_layer.vs
                    _rho[i_inds[i-1]:] = this_layer.rho
                else:
                    _vp[i_inds[i-1]:i_inds[i]] = this_layer.vp
                    _vs[i_inds[i-1]:i_inds[i]] = this_layer.vs
                    _rho[i_inds[i-1]:i_inds[i]] = this_layer.rho

            _twt = np.cumsum(2.0 * resolution / _vp)

            _dict[_case]['vp'] = LogCurve('vp_{}'.format(_case), Q_(_vp, 'm/s'), Depth(md), log_type='P velocity')
            _dict[_case]['vs'] = LogCurve('vs_{}'.format(_case), Q_(_vs, 'm/s'), Depth(md), log_type='S velocity')
            _dict[_case]['rho'] = LogCurve('rho_{}'.format(_case), Q_(_rho, 'grams/cm^3'), Depth(md), log_type='Density')
            _dict[_case]['ai'] = LogCurve('ai_{}'.format(_case), (Q_(_vp, 'm/s') * Q_(_rho, 'grams/cm^3')).to('kiloPa * s / m'), Depth(md), log_type='Impedance')
            _dict[_case]['vpvs'] = LogCurve('vpvs_{}'.format(_case), Q_(_vp, 'm/s') / Q_(_vs, 'm/s'), Depth(md), log_type='VpVs')
            _dict[_case]['twt'] = LogCurve('twt_{}'.format(_case), Q_(_twt, 's'), Depth(md), log_type='TWT')

        return _dict

    def line_cds(self, resolution: pint.Quantity, mod_cds: ColumnDataSource) -> ColumnDataSource:
        elastics = self.realize(resolution, mod_cds)
        _dict = modify_dict(elastics)
        return ColumnDataSource(_dict)

    def draw(self, cds: ColumnDataSource):
        """
        Returns a table that is used for defining a model
        :param cds:
            ColumnDataSource of the initial model, with N layers
        :return:
        """
        from bokeh.models import DataTable, Button
        from blixt_rp.rp.rp_core_new import LithoFluid

        def add_row_function():
            new_data = dict(cds.data)
            n = len(new_data['number'])
            new_row_number = n + 1
            for _key in list(new_data.keys()):
                if _key == 'number':
                    new_data[_key].append(new_row_number)
                elif _key == 'case':
                    new_data[_key].append(self.base_case_name)
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
            for _i in range(len(cds.data['number'])):
                if _i  in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(cds.data[_x][_i])
            cds.selected.indices = []
            cds.data = new_data

        def update_table_function():
            new_data = dict(cds.data)
            layers = []
            print(self.input_litho_fluids)
            for _i in range(len(new_data['number'])):
                _thickness =  new_data['thickness'][_i]
                if _thickness is None:
                    _thickness = 10.
                this_lf = new_data['litho_fluid'][_i]
                this_i = 0
                try:
                    this_i = [_x.lower() for _x in self.litho_fluid_cds.data['name']].index(this_lf.lower())
                except ValueError as e:
                    warn_txt = 'Litho fluid {} is not found in LithoFluidTable. Using first'.format(this_lf)
                    print_info(warn_txt, 'warning', logger)
                    continue
                this_layer = ModelLayer(
                    number=new_data['number'][_i],
                    case=new_data['case'][_i],
                    color=new_data['color'][_i],
                    thickness=Q_(_thickness, 'm'),
                    litho_fluid=LithoFluid(
                        name=self.litho_fluid_cds.data['name'][this_i],
                        vp=self.litho_fluid_cds.data['vp'][this_i],
                        vs=self.litho_fluid_cds.data['vs'][this_i],
                        rho=self.litho_fluid_cds.data['rho'][this_i]
                    )
                )
                layers.append(this_layer)
            self.layers = layers
            cds.data = new_data
            print(self.depth_range())

        dt = DataTable(
            source=cds,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            height=self.height,
            index_position = -1,
            index_header = 'index'
        )

        add_row = Button(label='Add row', button_type='success')
        add_row.on_click(add_row_function)

        delete_row = Button(label='Delete selected rows', button_type='success')
        delete_row.on_click(delete_row_function)

        update_table = Button(label='Update', button_type='success')
        update_table.on_click(update_table_function)

        return dt, add_row, delete_row, update_table

class LaminarModel:
    """
    Creates an interactive laminar (1D) model together with a display of the synthetic response
    NOTE! Only works in depth domain ('Z')
    """
    from blixt_rp.rp.rp_core_new import LithoFluids
    from blixt_rp.plotting.cross_plotter import DataSource

    def __init__(self,
                 litho_fluids: LithoFluids,
                 resolution: pint.Quantity | None = None,
                 wavelet: dict | None = None,
                 depth_to_top: pint.Quantity | None = None,
                 **kwargs
    ):
        self._litho_fluids = litho_fluids
        self.resolution = resolution
        self.depth_to_top = depth_to_top
        self.lf_width = kwargs.pop('lf_width', 300)
        self.model_width = kwargs.pop('model_width', None)
        self.model = None
        self.lf_table = None
        self.previous_cases = None

    def initiate_lf_table(self):
        self.lf_table = LithoFluidsTable(self._litho_fluids, width=self.lf_width, advanced=False)

    def draw_lf_table(self, lf_cds: ColumnDataSource):
        # Returns the LithoFluidsTable with control buttons
        # Returns: lf_table, lf_add_row, lf_delete_row, lf_update
        # lf_obj = LithoFluidsTable(self._litho_fluids, width=self.lf_width)
        return self.lf_table.draw(lf_cds)

    def initiate_model_table(self, lf_cds: ColumnDataSource):
        initial_layers = [_lf.to_model_layer(_i+1) for _i, _lf in enumerate(self._litho_fluids.litho_fluids)]
        mod_obj = ModelTable(initial_layers, lf_cds, self.depth_to_top)
        self.depth_to_top = mod_obj.depth_to_top
        self.model = mod_obj

    def draw_model_table(self, mod_cds: ColumnDataSource):
        # Returns the ModelTable with control buttons
        # Returns: model_table, model_add_row, model_delete_row, model_update
        return self.model.draw(mod_cds)

    def draw(self):
        from bokeh.models import Span
        from blixt_rp.plotting.log_plotter import LogPlotter, LogColumn, Line, select_column, select_line
        from blixt_rp.core.core import Template
        # Create the litho fluids table
        self.initiate_lf_table()
        cds_lf = self.lf_table.cds
        lf_table, add_row, delete_row, update = self.draw_lf_table(cds_lf)

        # Create the models table
        self.initiate_model_table(cds_lf)
        cds_m = self.model.cds
        model_table, add_row_m, delete_row_m, update_m = self.draw_model_table(cds_m)

        # Initiate the elastics and draw the initial data
        cds_lines = self.model.line_cds(self.resolution, cds_m)

        def spans():
            _tmp = []
            for _i, _name in enumerate(cds_m.data['top']):
                _tmp.append(Span(location=cds_m.data['top'][_i], dimension='width',
                                 line_width=1,
                                 line_dash='solid',
                                 line_color='black'))
            return _tmp

        # Create log plot object
        plotter = LogPlotter(width=800, height=1000)

        def draw_lines():
            # NOTE This does not respond when new lines (cases) are added
            ais_lines = []
            vpvs_lines = []
            for _case in self.model.cases:
                ais_lines.append(Line(x='ai_{}'.format(_case), y='depth', cds=cds_lines,
                                      style=Template(**{'name': 'AI {}'.format(_case)})))
                vpvs_lines.append(Line(x='vpvs_{}'.format(_case), y='depth', cds=cds_lines,
                                       style=Template(**{'name': 'VpVs {}'.format(_case)})))

            ai_column = LogColumn('AI', lines=ais_lines, rel_width=1.)
            vpvs_column = LogColumn('VpVs', lines=vpvs_lines, rel_width=1.)
            plotter.columns = [ai_column, vpvs_column]

        self.previous_cases = self.model.cases
        draw_lines()
        grid = plotter.figure()

        _spans = spans()
        for _span in _spans:
            for i, _child in enumerate(grid.children):
                _child[0].add_layout(_span)


        def update_m_function():
            print(self.model.cases)
            # print(self.model.numbers)
            # print(self.model.duplicate_numbers)

            # Update the ColumnDataSources
            cds_lines.data = dict(self.model.line_cds(self.resolution, cds_m).data)
            cds_m.data = dict(self.model.cds.data)
            print(cds_m.data['top'], cds_m.data['thickness'])

            # Update the Spans
            for _i, _span in enumerate(_spans):
                _span.location = cds_m.data['top'][_i]

            # Detect changes in number of cases
            case_changes = detect_change_in_cases(self.model.cases, self.previous_cases)
            # Only handle the situation when a new case is added, for now (TODO)
            if len(case_changes['new']) > 0:
                # get AI, and VpVs columns
                _p_ai = select_column(grid, 'AI')
                _p_vpvs = select_column(grid, 'VpVs')
                for _case in case_changes['new']:
                    _y = 'ai_{}'.format(_case)
                    print('Trying to plot the new case:', _case, _y)
                    _p_ai.line(x='ai_{}'.format(_case), y='depth', source=cds_lines,
                               **{'line_color': 'red', 'legend_label': 'AI_{}'.format(_case)})
                    _p_vpvs.line(x='vpvs_{}'.format(_case), y='depth', source=cds_lines,
                               **{'line_color': 'red', 'legend_label': 'VpVs_{}'.format(_case)})

            self.previous_cases = self.model.cases

        print('XXX Print TWT data')
        print(cds_lines.data['twt_Base'])
        update_m.on_click(update_m_function)

        return lf_table, add_row, delete_row, update, model_table, add_row_m, delete_row_m, update_m, grid


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
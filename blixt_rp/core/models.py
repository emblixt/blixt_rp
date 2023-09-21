import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import logging
import types
from copy import deepcopy

# sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')
# Instead of sys.path.append load all PyCharm projects in PyCharm, and they gets added to the sys.path automatically

from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot, wavelet_plot
from blixt_utils.plotting.crossplot import cnames
import blixt_rp.rp.rp_core as rp
import blixt_utils.misc.wavelets as bumw
import blixt_utils.io.io as uio

logger = logging.getLogger(__name__)

text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}


def plot_quasi_2d(model, ax=None):
    show = False
    if ax is None:
        fig, axs = plt.subplots(1, len(model.trace_index_range), figsize=(10, 8))
        show = True

    for i, trace_i in enumerate(model.trace_index_range):
        plot_1d(model, ax=axs[i], index=trace_i, legend=i == 0, yticks=i == 0)

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
    y, layer_index, vp, vs, rho = model.realize_model(0.001)
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

    if show:
        plt.show()


def plot_wiggles(model, sample_rate, wavelet, angle=0., eei=False, ax=None, color_by_gradient=False,
                 extract_avo_at=None, avo_angles=None, avo_plot_position=None, **kwargs):
    """

    Args:
        model:
        sample_rate:
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
            Each two tuple contains the x (index number) and y (TWT [s]) coordinates of where to extract avo curves

        avo_angles:
            list
            List of offset angles to use in plots
            Used quite differently for quasi-2D and 1D plots
            AND in 1D models, when eei is True, these angles are taken to be Chi angles
        avo_plot_position
        kwargs
            keyword arguments passed on to wiggle_plot
    Returns:

    """
    if avo_plot_position is None:
        avo_plot_position = [0.68, 0.02, 0.3, 0.3]
    if angle is None:
        angle = 0

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

    # def calc_wiggle(twt_length, _wavelet, _reflectivity, _angle):
    #     _wiggle = np.convolve(_wavelet['wavelet'], np.nan_to_num(_reflectivity(_angle)), mode='same')
    #     while len(_wiggle) < twt_length:
    #         _wiggle = np.append(_wiggle, np.ones(1) * _wiggle[-1])  # extend with one item
    #     if twt_length < len(_wiggle):
    #         _wiggle = _wiggle[:twt_length]
    #     return _wiggle

    twt, layer_i, vp, vs, rho = model.realize_model(sample_rate)

    if model.model_type == 'quasi 2D' and model.trace_index_range is not None:
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

            wiggle_plot(ax, twt, wiggle, i, scaling=40, color_by_gradient=grad, **kwargs)
            # Find out where there is a new layer (layer_i has a unit jump), and annotate it with a horizontal marker
            jumps = twt[np.diff(layer_i[trace_i, :], prepend=[0]) != 0]
            for jump in jumps:
                ax.scatter([i], [jump], c='black', marker='_')

            # extract avo curves
            if avo_positions is not None:
                for _i, _t in list(avo_positions.items()):
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
                        avo_curves[this_index].append(tmp_wiggle[twt_index - 1])

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
            # Find out where there is a new layer (layer_i has a unit jump), and annotate it with a horizontal marker
            jumps = twt[np.diff(layer_i, prepend=[0]) != 0]
            for jump in jumps:
                ax.axhline(jump, linestyle='--')

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
                        avo_curves[this_index].append(tmp_wiggle[twt_index - 1])

        ax.set_xlabel(my_x_label)

    else:
        warn_txt = 'Not possbile to plot this model'
        print(warn_txt)

    # Add extra information to plots
    if model.model_type == 'quasi 2D' and model.trace_index_range is not None:
        if eei:
            info_txt = r'EEI at $\chi$={}$^\circ$'.format(angle)
        else:
            info_txt = r'Amp. at $\theta$={}$^\circ$'.format(angle)
        ax.text(0.05, 0.95, info_txt, ha='left', va='top', transform=ax.transAxes, **text_style)

    if avo_curves is not None:
        avo_ax = ax.inset_axes(avo_plot_position)

        for _i, avo in list(avo_curves.items()):
            ax.annotate('{}'.format(_i + 1), avo_positions[_i], bbox={'boxstyle': 'circle', 'color': cnames[_i]})
            if tmp_avo_angles:
                avo_ax.plot(tmp_avo_angles, avo, c=cnames[_i], label='{}'.format(_i + 1))
            else:
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
    if show:
        plt.show()


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
        if depth_to_top is None:
            self.depth_to_top = 2.0
        else:
            self.depth_to_top = depth_to_top

        if layers is None:
            self.layers = []
        else:
            self.layers = layers



        if domain is None:
            self.domain = 'TWT'
        else:
            self.domain = domain

        for _layer in self.layers:
            if _layer.layer_type == 'quasi 2D':
                self.model_type = 'quasi 2D'

        if trace_index_range is None:
            self.trace_index_range = None
        else:
            self.trace_index_range = trace_index_range

    def __str__(self):
        return '{} model in {} domain with {} layers'.format(self.model_type, self.domain, len(self.layers))

    def __len__(self):
        return len(self.layers)

    def append(self, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        self.layers.append(layer)
        if layer.layer_type == 'quasi 2D':
            self.model_type = 'quasi 2D'

    def insert(self, index, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        self.layers.insert(index, layer)
        if layer.layer_type == 'quasi 2D':
            self.model_type = 'quasi 2D'

    def realize_model(self, resolution, voigt_reuss_hill=False, verbose=False):
        if self.model_type == 'quasi 2D' and self.trace_index_range is not None:

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

            z = self.depth_to_top + np.arange(len(this_vp[0])) * resolution

        else:
            layer_index, this_vp, this_vs, this_rho = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
            for i, layer in enumerate(self.layers):
                _vp, _vs, _rho = layer.realize_layer(resolution, voigt_reuss_hill=voigt_reuss_hill)
                layer_index = np.append(layer_index, np.ones(len(_vp)) * i)
                this_vp = np.append(this_vp, _vp)
                this_vs = np.append(this_vs, _vs)
                this_rho = np.append(this_rho, _rho)
            z = self.depth_to_top + np.arange(len(this_vp)) * resolution

        return z, layer_index, this_vp, this_vs, this_rho

    def plot(self, ax=None, index=None):
        if self.model_type == '1D':
            plot_1d(self, ax)
        elif self.model_type == 'quasi 2D' and index is not None:
            plot_1d(self, ax, index=index)
        elif self.model_type == 'quasi 2D':
            plot_quasi_2d(self, ax)


class Layer:
    """
    Class handling one layer of a model
    """

    def __init__(self,
                 thickness=None,
                 target=None,
                 vp=None,
                 vs=None,
                 rho=None,
                 ntg=None,
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

        if thickness is None:
            self.thickness = 0.1
        else:
            self.thickness = thickness

        if vp is None:
            self.vp = 3600.
        else:
            self.vp = vp

        if vs is None:
            self.vs = 1800.
        else:
            self.vs = vs

        if rho is None:
            self.rho = 2.3
        else:
            self.rho = rho

        if ntg is None:
            self.ntg = 1.
        else:
            self.ntg = ntg

        if not (0 <= self.ntg <= 1):
            raise ValueError('NTG must be between 0 and 1')

        # Get the net-to-gross related keyword arguments
        if self.ntg < 1 and self.target:
            self.gross_vp = kwargs.pop('gross_vp', 3400.)
            self.gross_vs = kwargs.pop('gross_vs', 1000.)
            self.gross_rho = kwargs.pop('gross_rho', 2.)
            self.thin_bed_factor = kwargs.pop('thin_bed_factor', 3)

        for arg in [self.thickness, self.vp, self.vs, self.rho, self.ntg]:
            if isinstance(arg, types.FunctionType):
                self.layer_type = 'quasi 2D'

    def realize_layer(self, resolution, voigt_reuss_hill=False, index=0):
        """
        Realize the current layer by returning arrays of the elastic properties with the given resolution
        :param resolution:
            float
            resolution in TWT [s] or depth [m]
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

        n = int(layer_thickness / resolution)

        if isinstance(self.vp, types.FunctionType):
            layer_vp = self.vp(index)
            # print(layer_vp)
        else:
            layer_vp = self.vp

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

        return this_vp, this_vs, this_rho


def test_plot():
    import blixt_utils.misc.wavelets as bumw
    # first_layer = Layer(thickness=0.1, vp=3300, vs=1500, rho=2.1)
    # second_layer = Layer(thickness=0.2, vp=3500, vs=1600, rho=2.3, target=True, ntg=0.8)
    first_layer = Layer(thickness=0.05, vp=3400, vs=1820, rho=2.6)
    second_layer = Layer(thickness=0.03, vp=3600, vs=2120., rho=2.2, target=True)
    m = Model(depth_to_top=1.94, layers=[first_layer, second_layer])
    m.append(first_layer)
    m.plot()

    wavelet = bumw.ricker(0.096, 0.001, 25)
    plot_wiggles(m, 0.001, wavelet, avo_angles=[0., 5., 10., 15.,  20., 25., 30., 35., 40.])
    plot_wiggles(m, 0.001, wavelet, eei=True, avo_angles=[-90, -70., -50., -30., -15., 0., 15., 30., 50., 70., 90.],
                 extract_avo_at=(-90, 1.99))


def test_ntg():

    thin_bed_factor = 3
    net_vp = 3000.
    resolution = 0.001
    fig, axs = plt.subplots(nrows=11)
    for i in range(11):
        ntg = i/10.

        ntg_layer = Layer(target=True, thickness=0.1, vp=net_vp, ntg=ntg, thin_bed_factor=thin_bed_factor)

        m = Model(layers=[ntg_layer])

        x, _, _ = m.layers[0].realize_layer(resolution)
        net = len(x[x == net_vp])

        print('Requested len: {}, returned len: {}'.format(int(m.layers[0].thickness / resolution), len(x)))
        axs[i].set_title('NTG={}, tbf={}. Observed NTG {:.2f}'.format(ntg, thin_bed_factor, net/len(x)))
        axs[i].plot(x)

    plt.show()


def test_realization():
    first_layer = Layer(thickness=0.1, vp=3300, vs=1500, rho=2.1)
    second_layer = Layer(thickness=0.1, vp=3500, vs=1600, rho=2.3, target=True, ntg=0.8)
    m = Model(layers=[first_layer, second_layer])
    m.append(first_layer)
    twt, li, vp, _, _ = m.realize_model(0.001)
    print(len(vp))
    plt.plot(twt, vp / 1000., twt, li)
    plt.show()


def test_quasi2d():
    import blixt_utils.misc.wavelets as bumw
    from blixt_rp.rp.rp_core import constantcement, v_p, v_s
    n_samplings = 51
    # TODO
    # At the moment, quasi2d models will likely fail when combining a varying thickness with NTG separate from 1
    # when Voigt Reuss Hill average is set to False.

    # for a wedge model to work, we need to counterweight the changing thickness of one layer with an extra layer
    # so that the total height of the model is kept constant
    def wedge(i):
        return 0.1 - 0.1/50 * i

    def reverse_wedge(i):
        return 0.06 + 0.1/50 * i

    def vp(i):
        # phi = np.linspace(0.1, 0.4, n_samplings)
        # k_eff, mu_eff = constantcement(37, 45, phi, apc=2)
        # # print(k_eff, mu_eff)
        # # print(v_p(k_eff[i], mu_eff[i], 2.3))
        # return 1000. * v_p(k_eff[i], mu_eff[i], 2.3)

        # gas lens (gas in the center, brine on the flanks):
        if (i > 17) and (i < 34):
            return 3600.
        else:
            return 3730.

    def vs(i):
        # phi = np.linspace(0.1, 0.4, n_samplings)
        # k_eff, mu_eff = constantcement(37, 45, phi, apc=2)
        # return 1000. * v_s(k_eff[i], 2.3)
        # gas lens:
        if (i > 17) and (i < 34):
            return 2120.
        else:
            return 2070.

    def rho(i):
        # gas lens:
        if (i > 17) and (i < 34):
            return 2.2
        else:
            return 2.3

    # wedge model
    # first_layer = Layer(thickness=0.06, vp=2800., vs=1350, rho=2.46)
    # second_layer = Layer(thickness=wedge, vp=3100, vs=1800, rho=2.3, target=True)
    # third_layer = Layer(thickness=reverse_wedge, vp=2800, vs=1350, rho=2.46, target=False)

    # gas lens model
    first_layer = Layer(thickness=0.05, vp=3400., vs=1820., rho=2.6)
    second_layer = Layer(thickness=0.03, vp=vp, vs=vs, rho=rho, target=True)
    third_layer = Layer(thickness=0.05, vp=3400., vs=1820., rho=2.6)

    m = Model(depth_to_top=1.94, layers=[first_layer, second_layer, third_layer],
              trace_index_range=np.arange(n_samplings))
    # m.plot()

    wavelet = bumw.ricker(0.096, 0.001, 25)

    plot_wiggles(m, 0.001, wavelet, angle=0., eei=True, extract_avo_at=[(8, 1.99), (24, 1.99)])
    plot_wiggles(m, 0.001, wavelet, angle=15., eei=True, extract_avo_at=[(8, 1.99), (24, 1.99)])
    plot_wiggles(m, 0.001, wavelet, angle=-90., eei=True, extract_avo_at=[(8, 1.99), (24, 1.99)])


if __name__ == '__main__':
    # test_realization()
    # test_plot()
    # test_ntg()
    test_quasi2d()

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import logging

sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')
#sys.path.append('C:\\Users\\marten\\PycharmProjects\\blixt_utils')

from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot
import blixt_rp.rp.rp_core as rp

logger = logging.getLogger(__name__)


def plot(model, ax=None):
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
        vps.append(layer.vp)
        vss.append(layer.vs)
        rhos.append(layer.rho)
        target.append(layer.target)
        ntg.append(layer.ntg)
        if layer.ntg < 1:
            gross_vp.append(layer.gross_vp)
            gross_vs.append(layer.gross_vs)
            gross_rho.append(layer.gross_rho)
        else:
            gross_vp.append(0.)
            gross_vs.append(0.)
            gross_rho.append(0.)

    linestyle_ai = {'lw': 1, 'color': 'k', 'ls': '-'}
    linestyle_vpvs = {'lw': 1, 'color': 'k', 'ls': '--'}
    text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}
    ai_line = mlines.Line2D([1], [1], **linestyle_ai)
    vpvs_line = mlines.Line2D([1], [1], **linestyle_vpvs)

    # the boundaries (bwb) need to add up the respective thicknesses
    bwb = [model.depth_to_top]
    for i in range(len(model)):
        last_top = bwb[i]
        bwb.append(last_top + model.layers[i].thickness)

    # realize the model to create data
    y, layer_index, vp, vs, rho = model.realize_model(0.001)
    data_ai = vp * rho
    max_ai = max(data_ai)
    data_ai = data_ai / max_ai
    data_vpvs = vp / vs
    max_vpvs = max(data_vpvs)
    data_vpvs = 0.9 * data_vpvs / max_vpvs

    # create a discrete data set which controls the filling
    # 0: no fill, 1: net reservoir (sand), 2: gross reservoir (shale)
    filler = np.zeros(len(vp))
    # Step through each layer and search for areas where NTG < 1 and
    # apply the filler there
    for i, layer in enumerate(model.layers):
        if layer.target and (layer.ntg < 1):
            filler[
                np.array([all(xx) for xx in zip(layer_index == i, vp == layer.vp)])
            ] = 1.
            filler[
                np.array([all(xx) for xx in zip(layer_index == i, vp == layer.gross_vp)])
            ] = 2.
        elif layer.target:
            filler[
                np.array([all(xx) for xx in zip(layer_index == i, vp == layer.vp)])
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
        info_txt = '{}.\nVp: {:.1f}\nVs: {:.1f}\nRho: {:.1f}'.format(i+1, _vp/1000., _vs/1000., _rho)
        ax.text(0.5*_vp * _rho/max_ai, 0.5*(bwb[i] + bwb[i+1]), info_txt, ha='right', va='center', **text_style)
        i += 1
    i = 0
    for _gvp, _gvs, _grho in zip(gross_vp, gross_vs, gross_rho):
        if target[i] and (ntg[i] < 1):
            info_txt = 'NTG: {:.1f}.\nVp: {:.1f}\nVs: {:.1f}\nRho: {:.1f}'.format(ntg[i], _gvp/1000., _gvs/1000., _grho)
            ax.text(0.6*vps[i] * rhos[i]/max_ai, 0.5*(bwb[i] + bwb[i+1]), info_txt, ha='left', va='center', **text_style)
        i += 1

    ax.set_ylim(ax.get_ylim()[::-1])
    ax.legend([ai_line, vpvs_line], ['AI', 'Vp/Vs'])

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
                 domain=None):
        """
        :param model_type:
            str
            '1D' is the only option
        :param depth_to_top:
            float
            depth to top in seconds or meters
        :param layers:
            list of layers, each a Layer object
            The first layer is at the top, and consequent layers beneath
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

    def __str__(self):
        return '{} model in {} domain with {} layers'.format(self.model_type, self.domain, len(self.layers))

    def __len__(self):
        return len(self.layers)

    def append(self, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        self.layers.append(layer)

    def insert(self, index, layer):
        if not isinstance(layer, Layer):
            raise IOError('Layer must be a Layer object')
        self.layers.insert(index, layer)

    def realize_model(self, resolution, voigt_reuss_hill=False):
        layer_index, this_vp, this_vs, this_rho = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        for i, layer in enumerate(self.layers):
            _vp, _vs, _rho = layer.realize_layer(resolution, voigt_reuss_hill=voigt_reuss_hill)
            layer_index = np.append(layer_index, np.ones(len(_vp)) * i)
            this_vp = np.append(this_vp, _vp)
            this_vs = np.append(this_vs, _vs)
            this_rho = np.append(this_rho, _rho)
        z = self.depth_to_top + np.arange(len(this_vp)) * resolution
        return z, layer_index, this_vp, this_vs, this_rho

    def plot(self, ax=None):
        plot(self, ax)


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
            float
            Thickness of layer in m or s, depending on which domain the model
        :param target:
            bool
            True for target layers
        :param vp:
            float
            P velocity in m/s
        :param vs:
            float
            Shear velocity in m/s
        :param rho:
            float
            Density in g/cm3
        :param ntg:
            float
            net-to-gross, 0 <= ntg >= 1
        :param gross_xxx:
            elastic parameters in the non-reservoir part (gross) of the layer
        :param thin_bed_factor:
            int
            Number of internal layers in the reservoir (net) part of the layer
            If 1, the net portion is taken up by one homogeneous layer
        """
        # get the required parameters
        self.resolution = None
        if thickness is None:
            self.thickness = 0.1
        else:
            self.thickness = thickness
        if target is None:
            self.target = False
        else:
            self.target = target
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

    def realize_layer(self, resolution, voigt_reuss_hill=False):
        """
        Realize the current layer by returning arrays of the elastic properties with the given resolution
        :param resolution:
            float
            resolution in TWT [s] or depth [m]
        :param voigt_reuss_hill:
            bool
            If true, the Voigt Reuss Hill average for the given NTG is used when returning the elastic parameters

        """
        self.resolution = resolution
        n = int(self.thickness / resolution)
        if self.ntg < 1. and self.target and not voigt_reuss_hill:
            net_group_size = int(np.ceil(self.ntg * self.thickness / (self.thin_bed_factor * resolution)))
            gross_group_size = int(np.ceil((1. - self.ntg) * self.thickness / (self.thin_bed_factor * resolution)))
            this_vp = []
            this_vs = []
            this_rho = []
            i = 0
            while len(this_vp) <= n:
                #print(n, len(this_vp), net_group_size, gross_group_size)
                if i <= self.thin_bed_factor - 1:
                    this_vp += [self.vp] * net_group_size
                    this_vp += [self.gross_vp] * gross_group_size
                    this_vs += [self.vs] * net_group_size
                    this_vs += [self.gross_vs] * gross_group_size
                    this_rho += [self.rho] * net_group_size
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
        elif self.ntg < 1. and self.target and voigt_reuss_hill:
            _, _, vp_vrh = rp.vrh_bounds([self.ntg, 1. - self.ntg], [self.vp, self.gross_vp])
            _, _, vs_vrh = rp.vrh_bounds([self.ntg, 1. - self.ntg], [self.vs, self.gross_vs])
            _, _, rho_vrh = rp.vrh_bounds([self.ntg, 1. - self.ntg], [self.rho, self.gross_rho])
            this_vp = np.ones(n) * vp_vrh
            this_vs = np.ones(n) * vs_vrh
            this_rho = np.ones(n) * rho_vrh
        else:
            this_vp = np.ones(n) * self.vp
            this_vs = np.ones(n) * self.vs
            this_rho = np.ones(n) * self.rho

        return this_vp, this_vs, this_rho


def test_plot():
    first_layer = Layer(thickness=0.1, vp=3300, vs=1500, rho=2.1)
    second_layer = Layer(thickness=0.2, vp=3500, vs=1600, rho=2.3, target=True, ntg=0.8)
    m = Model(layers=[first_layer, second_layer])
    m.append(first_layer)
    m.plot()


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


if __name__ == '__main__':
    #test_realization()
    test_plot()
    #test_ntg()

import matplotlib.pyplot as plt
import numpy as np
import sys
import logging

#sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')
sys.path.append('C:\\Users\\MårtenBlixt\\PycharmProjects\\blixt_utils')

from blixt_utils.plotting.helpers import axis_plot, axis_log_plot, annotate_plot, header_plot, wiggle_plot

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
    for layer in model.layers:
        vps.append(layer.vp)
        vss.append(layer.vs)
        rhos.append(layer.rho)

    linestyle_ai = {'lw': 1, 'color': 'k', 'ls': '-'}
    linestyle_vpvs = {'lw': 1, 'color': 'k', 'ls': '--'}
    text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}

    # the boundaries (bwb) need to add up the respective thicknesses
    bwb = [model.depth_to_top]
    for i in range(len(model)):
        last_top = bwb[i]
        bwb.append(last_top + model.layers[i].thickness)

    # Create data containers
    y = np.linspace(bwb[0], bwb[-1],  100)  # fake depth parameter
    data_ai = np.ones(100)
    data_vpvs = np.ones(100)

    ais = [_vp * _rho for _vp, _rho in zip(vps, rhos)]  # Acoustic impedance in m/s g/cm3
    vpvss = [_vp / _vs for _vp, _vs in zip(vps, vss)]  # vp / vs
    max_ai = max(ais)
    max_vpvs = max(vpvss)

    # normalize the data according to max values
    for i in range(len(bwb)):
        if i == 0:
            continue
        data_ai[(y >= bwb[i-1]) & (y <= bwb[i])] = ais[i-1]/max_ai
        data_vpvs[(y >= bwb[i-1]) & (y <= bwb[i])] = vpvss[i-1] * 0.9 / max_vpvs

    axis_plot(ax, y, [data_ai], [[0., 1.1]], [linestyle_ai], nxt=0)
    axis_plot(ax, y, [data_vpvs], [[0., 1.1]], [linestyle_vpvs], nxt=0)
    for i, _y in enumerate(bwb[1:-1]):
        ax.plot([0., ais[i+1]/max_ai], [_y, _y], **linestyle_ai)

    i = 0
    for _vp, _vs, _rho in zip(vps, vss, rhos):
        info_txt = '{}.\nVp: {:.1f}\nVs: {:.1f}\nRho: {:.1f}'.format(i+1, _vp/1000., _vs/1000., _rho)
        ax.text(0.5*ais[i]/max_ai, 0.5*(bwb[i] + bwb[i+1]), info_txt, ha='center', va='center', **text_style)
        i += 1

    ax.set_ylim(ax.get_ylim()[::-1])

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
        self.xx = None
        if self.ntg < 1 and self.target:
            self.gross_vp = kwargs.pop('gross_vp', 3400.)
            self.gross_vs = kwargs.pop('gross_vs', 1000.)
            self.gross_rho = kwargs.pop('gross_rho', 2.)
            self.thin_bed_factor = kwargs.pop('thin_bed_factor', 1)

            # Divide the layer into n sub layers
            if self.ntg >= 0.5:
                n = self.thin_bed_factor / (1. - self.ntg)
            else:
                n = self.thin_bed_factor / self.ntg
            # number of cells in a group to subdivide the net part in to "thin_bed_factor" number of sub layers
            ng = int(np.round(self.ntg / (1. - self.ntg)))
            if self.ntg >= 0.5:
                self.xx = [self.gross_vp]
                while len(self.xx) <= n:
                    self.xx += [self.vp] * ng
                    self.xx += [self.gross_vp]

    def get_xx(self):
        return self.xx


def test():
    first_layer = Layer(thickness=0.1, vp=3300, vs=1500, rho=2.1)
    second_layer = Layer(thickness=0.2, vp=3500, vs=1600, rho=2.3)

    ntg = 0.9
    thin_bed_factor = 1
    net_vp = 3000.
    ntg_layer = Layer(target=True, thickness=0.2, vp=net_vp, ntg=ntg, thin_bed_factor=thin_bed_factor)

    m = Model(layers=[ntg_layer])

    x = m.layers[0].get_xx()
    x = np.array(x)
    net = len(x[x==net_vp])

    print('For NTG={}, and tbf={}, the observed NTG is {}'.format(ntg, thin_bed_factor, net/len(x)))



if __name__ == '__main__':
    test()

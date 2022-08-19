import matplotlib.pyplot as plt
import numpy as np
import sys
import logging

sys.path.append('C:\\Users\\eribli\\PycharmProjects\\blixt_utils')

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
        show=True

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
    y = np.linspace(bwb[0], bwb[-1],  100) # fake depth parameter
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
        for name, param, def_val in zip(
                ['model_type', 'depth_to_top', 'layers', 'domain'],
                [model_type, depth_to_top, layers, domain],
                ['1D', 2.0, [], 'TWT']):
            if param is None:
                super(Model, self).__setattr__(name, def_val)
            else:
                super(Model, self).__setattr__(name, param)

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
        for name, param, def_val in zip(
                ['thickness', 'target', 'vp', 'vs', 'rho', 'ntg'],
                [thickness, target, vp, vs, rho, ntg],
                [0.1, False, 3600, 1800, 2.3, 1.]):
            if param is None:
                super(Layer, self).__setattr__(name, def_val)
            else:
                super(Layer, self).__setattr__(name, param)
        if not (0 <= self.ntg <= 1):
            raise ValueError('NTG must be between 0 and 1')

        # Get the net-to-gross related keyword arguments
        if self.ntg < 1:
            for name, param, def_val in zip(
                    ['gross_vp', 'gross_vs', 'gross_rho', 'thin_bed_factor'],
                    [gross_vp, gross_vs, gross_rho, thin_bed_factor],
                    [3400, 1000, 2.0, 1]):
                param = kwargs.pop(name, def_val)
                super(Layer, self).__setattr__(name, param)
            # Divide the layer into n sub layers
            if self.ntg >= 0.5
                n = self.thin_bed_factor / (1. - self.ntg)
            else:
                n = self.thin_bed_factor /  self.ntg




def test():
    first_layer = Layer(thickness=0.1, vp=3300, vs=1500, rho=2.1)
    second_layer = Layer(thickness=0.2, vp=3500, vs=1600, rho=2.3)

    m = Model(layers=[first_layer, second_layer])
    print(m)
    m.append(first_layer)
    print(m)

    m.plot()


if __name__ == '__main__':
    test()

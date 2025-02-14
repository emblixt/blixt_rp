import unittest
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add to path to avoid having to install libraries, useful in development
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.models import (Model, Layer, build_wedge, plot_wiggles, build_layered_model, laminar_model_analysis,
                                  build_saturation_wedge)
import blixt_utils.misc.wavelets as bumw

l1 = {'vp': 3000, 'vs': 1820, 'rho': 2.6}
l2 = {'vp': 2900, 'vs': 1620, 'rho': 2.2}


class TestCase(unittest.TestCase):

    def test_1d_twt(self):
        first_layer = Layer(thickness=0.1, **l1)
        second_layer = Layer(thickness=0.04, **l2, target=True)
        m = Model(depth_to_top=2.0, layers=[first_layer, second_layer])
        m.append(first_layer)
        m.plot()

    def test_1d_z(self):
        first_layer = Layer(thickness=150, **l1, domain='Z')
        second_layer = Layer(thickness=40, **l2, target=True, domain='Z')
        m = Model(depth_to_top=3000., layers=[first_layer, second_layer])
        m.append(first_layer)
        m.plot()

    def test_wedge(self):
        m = build_wedge(2.0, 0.02, 0.1, 51, l1, l2, l1)
        # m = build_wedge(3000., 10.0, 40., 51, l1, l2, l1, domain='Z')
        m.plot()

    def test_plot(self):
        first_layer = Layer(thickness=0.1, **l1)
        second_layer = Layer(thickness=0.04, **l2, target=True)
        m = Model(depth_to_top=2.0, layers=[first_layer, second_layer])
        m.append(first_layer)

        wavelet = bumw.ricker(0.096, 0.001, 25)
        plot_wiggles(m, 0.001, wavelet, avo_angles=[0., 5., 10., 15., 20., 25., 30., 35., 40.])
        plot_wiggles(m, 0.001, wavelet, eei=True, avo_angles=[-90, -70., -50., -30., -15., 0., 15., 30., 50., 70., 90.],
                     extract_avo_at=(-90, 1.99))

    def test_ntg(self):

        thin_bed_factor = 3
        net_vp = 3000.
        resolution = 0.001
        fig, axs = plt.subplots(nrows=11)
        for i in range(11):
            ntg = i / 10.

            ntg_layer = Layer(target=True, thickness=0.1, vp=net_vp, ntg=ntg, thin_bed_factor=thin_bed_factor)

            m = Model(layers=[ntg_layer])

            x, _, _ = m.layers[0].realize_layer(resolution)
            net = len(x[x == net_vp])

            print('Requested len: {}, returned len: {}'.format(int(m.layers[0].thickness / resolution), len(x)))
            axs[i].set_title('NTG={}, tbf={}. Observed NTG {:.2f}'.format(ntg, thin_bed_factor, net / len(x)))
            axs[i].plot(x)
        plt.show()

    def test_realization(self):
        # in TWT
        first_layer = Layer(thickness=0.1, vp=3300, vs=1500, rho=2.1)
        second_layer = Layer(thickness=0.1, vp=3500, vs=1600, rho=2.3, target=True, ntg=0.8)
        m = Model(layers=[first_layer, second_layer])
        m.append(first_layer)
        m.plot()
        twt, li, vp, _, _, _ = m.realize_model(0.001)
        print(len(vp))
        fig, ax = plt.subplots()
        ax.plot(twt, vp / 1000., twt, li)

        # in Z
        first_layer = Layer(thickness=100, vp=3300, vs=1500, rho=2.1, domain='Z')
        second_layer = Layer(thickness=20., vp=3500, vs=1600, rho=2.3, target=True, ntg=0.8, domain='Z')
        m = Model(layers=[first_layer, second_layer])
        m.append(first_layer)
        m.plot()
        twt, li, vp, _, _, _ = m.realize_model(0.001)
        print(len(vp))
        fig, ax = plt.subplots()
        ax.plot(twt, vp / 1000., twt, li)

    def test_quasi2d(self):
        import blixt_utils.misc.wavelets as bumw
        from blixt_rp.rp.rp_core import constantcement, v_p, v_s
        n_samplings = 51

        # TODO
        # At the moment, quasi2d models will likely fail when combining a varying thickness with NTG separate from 1
        # when Voigt Reuss Hill average is set to False.

        # for a wedge model to work, we need to counterweight the changing thickness of one layer with an extra layer
        # so that the total height of the model is kept constant
        def wedge(i):
            return 0.1 - 0.1 / 50 * i

        def reverse_wedge(i):
            return 0.06 + 0.1 / 50 * i

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

        plot_wiggles(m, 0.001, wavelet, angle=0., eei=True, scaling=80., extract_avo_at=[(8, 1.99), (24, 1.99)])
        plot_wiggles(m, 0.001, wavelet, angle=15., eei=True, scaling=80., extract_avo_at=[(8, 1.99), (24, 1.99)])
        plot_wiggles(m, 0.001, wavelet, angle=-90., eei=True, scaling=80., extract_avo_at=[(8, 1.99), (24, 1.99)])

    def test_layered_model(self):
        wavelet = bumw.ricker(0.096, 0.001, 25)
        length = wavelet['time'][-1] - wavelet['time'][0]
        dt = wavelet['header']['Sample rate']  # should be given in seconds
        top_thickness = length / 2.
        m = build_layered_model(2., top_thickness, 0.05, l1, l2, l1, domain='TWT')
        laminar_model_analysis(m, dt, wavelet, extract_avo_at=(0, 2.01), extract_on='nearest max')
        plt.show()

    def test_saturation_wedge(self):
        from blixt_rp.rp.rp_core import rpt_parameters
        wavelet = bumw.ricker(0.096, 0.001, 25)
        dt = wavelet['header']['Sample rate']  # should be given in seconds
        rpts = rpt_parameters()
        m = build_saturation_wedge(
            2.0, 0.05, 0.9, 10,
            l1, l2, l1, 0.2, rpts)

        hc_sat = np.linspace(0, 0.9, 10)

        # for i in m.trace_index_range:
        #     print(i, m.layers[1].vp(i), m.layers[1].vs(i), m.layers[1].rho(i))
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [1, 0.2]})
        axs[1, 1].set_axis_off()
        wiggle_ax = axs[0, 0]
        model_ax = axs[0, 1]
        wedge_ax = axs[1, 0]
        fig.subplots_adjust(wspace=0.)
        extract_avo_at = (1, 2.)
        plot_domain = 'TWT'
        overburden_vel = 3000.
        scaling = 10.
        avo_curves, amps, min_amps, max_amps, apparent_thickness = plot_wiggles(
            m, dt, wavelet, ax=wiggle_ax, extract_avo_at=extract_avo_at,
            plot_domain=plot_domain, overburden_vel=overburden_vel, scaling=scaling)
        wiggle_ax.set_ylim(wiggle_ax.get_ylim()[::-1])
        m[0].plot(ax=model_ax, kwargs1d={'yticks': False, 'legend': False})
        wedge_ax.plot(hc_sat, np.abs(np.array(min_amps)), label='|Min|')
        wedge_ax.set_ylabel('Amplitude')
        wedge_ax.set_xlabel('HC saturation')
        wedge_ax.legend(loc='upper left')
        plt.show()

import unittest
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_file

from blixt_rp.core.core import LithoFluid, LithoFluids, LithoFluidsTable
from blixt_rp.core.seismic import AvoAnalyzer

# Add to path to avoid having to install libraries, useful in development
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))
output_file(os.path.join(project_dir, 'blixt_rp', 'test_data','plot.html'))

from blixt_rp.core.models import (Model, Layer, ModelTable, LaminarModel, WedgeModel,
                                  plot_wiggles, build_layered_model, laminar_model_analysis,
                                  build_saturation_wedge, detect_change_in_cases, build_wedge)
from blixt_rp.core.core import LithoFluid
import blixt_utils.misc.wavelets as bumw
from blixt_rp import Q_

# l1 = {'vp': 3000, 'vs': 1820, 'rho': 2.6}
# l2 = {'vp': 2900, 'vs': 1620, 'rho': 2.2}
l1 = LithoFluid(default='shale')
l2 = LithoFluid(default='brine_sst')
layer1 = Layer('Top', 'Base', thickness=Q_(50., 'm'), litho_fluid=l1)
layer2 = Layer('Reservoir', 'Base', thickness=Q_(20., 'm'), litho_fluid=l2)
layer3 = Layer('Reservoir', 'Oil', thickness=Q_(20., 'm'), litho_fluid=LithoFluid(default='oil_sst'))
layer4 = Layer('Reservoir', 'Gas', thickness=Q_(20., 'm'), litho_fluid=LithoFluid(default='oil_sst'))
layer5 = Layer('Basement', 'Base', thickness=Q_(50., 'm'), litho_fluid=l1)
layer6 = Layer('Basement', 'Oil', thickness=Q_(50., 'm'), litho_fluid=l1)

class TestCase(unittest.TestCase):

    def test_1d(self):
        m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer1, layer2, layer3, layer4, layer5])
        # print(m.layer_names)
        # print(m.case_names)
        print(m.index_of('Reservoir', 'Base'))
        print(m.index_of('Reservoir', 'Oil'))
        print(m.index_of('Basement', 'Oil'))

        print(m.target_layers)
        print(m.base_case_layers)
        print(m.base_target_combos)

    def test_combos(self):
        m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer1, layer2, layer3, layer4, layer5, layer6])
        print(m.base_target_combos)

    def test_failing_models(self):
        # # Model with one layer being repeated
        # m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer1, layer1, layer1])
        # print(m.base_target_combos)

        # # Model with layers that have no name or case
        # layer0 = Layer(thickness=Q_(50., 'm'), litho_fluid=l1)
        # m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer0, layer0, layer0])
        # print(m.base_target_combos)

        # Model with layers that have no name but different cases
        layer_1 = Layer(case='Base', thickness=Q_(50., 'm'), litho_fluid=l1)
        layer_2 = Layer(case='Oil', thickness=Q_(50., 'm'), litho_fluid=l1)
        m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer_1, layer_2])
        # print(m.layer_names)
        print(m.base_target_combos)

    def test_depth_range(self):
        m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer1, layer2, layer3, layer4, layer5])
        print(m.total_thickness())
        self.assertTrue(m.total_thickness().magnitude == 120.)

    def test_depth_array(self):
        m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer1, layer2, layer3, layer4, layer5])
        print(m.depth_array(Q_(10., 'm')))

    def test_interface_indexes(self):
        m = Model(depth_to_top=Q_(2000, 'm'), layers=[layer1, layer2, layer3, layer5])
        res = Q_(10., 'm')
        print(m.interface_indexes(res))
        print(m.depth_array(res)[m.interface_indexes(res)])
        test = m.depth_array(res)[m.interface_indexes(res)] == Q_([2050., 2070.], 'm')
        print(test)
        self.assertTrue(test.all())

    def test_1d_twt(self):
        first_layer = Layer(name='A', thickness=Q_(0.1, 's'), litho_fluid=l1)
        second_layer = Layer(name='B', thickness=Q_(0.04, 's'), litho_fluid=l2)
        third_layer = Layer(name='C', thickness=Q_(0.04, 's'), litho_fluid=l2)
        m = Model(depth_to_top=Q_(2.0, 's'), layers=[first_layer, second_layer])
        m.append(third_layer)
        print(m.layer_names)
        # m.plot()
        # plt.show()

    def test_1d_z(self):
        first_layer = Layer(thickness=Q_(150, 'm'), litho_fluid=l1)
        second_layer = Layer(thickness=Q_(40, 'm'), litho_fluid=l2)
        m = Model(depth_to_top=Q_(3000., 'm'), layers=[first_layer, second_layer])
        m.append(first_layer)
        m.plot()
        plt.show()

    def test_wedge_old(self):
        m = build_wedge(Q_(2.0, 's'), Q_(0.02, 's'), Q_(0.1, 's'), 51, l1, l2, l1)
        # m = build_wedge(3000., 10.0, 40., 51, l1, l2, l1, domain='Z')
        m.plot()
        plt.show()

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

    def test_quasi2d_layer(self, unit_test=True):
        from blixt_rp.core.core import LithoFluid
        def vp(_i):
            return Q_(3000. +  5. * _i, 'm/s')
        def vs(_i):
            return Q_(1500. +  5. * _i, 'm/s')
        def rho(_i):
            return Q_(2.5 +  0.1 * _i, 'grams/cm^3')
        def thickness_m(_i):
            return Q_(30. + 2 * _i, 'm')
        def thickness_twt(_i):
            return Q_(40. + 2 * _i, 'milliseconds')

        lf = LithoFluid(vp=vp, vs=vs, rho=rho,
                        vp_std_dev=1., vs_std_dev=1., rho_std_dev=0.01,
                        vp_vs_cc=0.9, vp_rho_cc=0.9, vs_rho_cc=0.9)

        l_m = Layer(thickness=thickness_m, litho_fluid=lf)
        l_twt = Layer(thickness=thickness_twt, litho_fluid=lf)

    def test_quasi2d(self, unit_test=True):
        from bokeh.plotting import show, row, column
        import blixt_utils.misc.wavelets as bumw
        from blixt_rp.core.core import LithoFluid, LithoFluids, LithoFluidsTable
        from blixt_rp.rp.rp_core import constantcement, v_p, v_s

        n_samplings = 11

        # TODO
        # At the moment, quasi2d models will likely fail when combining a varying thickness with NTG separate from 1
        # when Voigt Reuss Hill average is set to False.

        # for a wedge model to work, we need to counterweight the changing thickness of one layer with an extra layer
        # so that the total height of the model is kept constant
        def wedge(i):
            return Q_(50. - i, 'm')

        def reverse_wedge(i):
            return Q_(25. + i, 'm')

        def vp(i):
            # phi = np.linspace(0.1, 0.4, n_samplings)
            # k_eff, mu_eff = constantcement(37, 45, phi, apc=2)
            # # print(k_eff, mu_eff)
            # # print(v_p(k_eff[i], mu_eff[i], 2.3))
            # return 1000. * v_p(k_eff[i], mu_eff[i], 2.3)

            # gas lens (gas in the center, brine on the flanks):
            if (i > 17) and (i < 34):
                return Q_(3600., 'm/s')
            else:
                return Q_(3730., 'm/s')

        def vs(i):
            # phi = np.linspace(0.1, 0.4, n_samplings)
            # k_eff, mu_eff = constantcement(37, 45, phi, apc=2)
            # return 1000. * v_s(k_eff[i], 2.3)
            # gas lens:
            if (i > 17) and (i < 34):
                return Q_(2120., 'm/s')
            else:
                return Q_(2070., 'm/s')

        def rho(i):
            # gas lens:
            if (i > 17) and (i < 34):
                return Q_(2.2, 'grams/cm^3')
            else:
                return Q_(2.3, 'grams/cm^3')

        # Litho-fluids
        lfs = LithoFluids([
            LithoFluid(default='shale'),
            LithoFluid(name='Quasi 2D', vp=vp, vs=1800, rho=2.3),
            LithoFluid(name='Basement', vp=2800, vs=1350, rho=2.46)
        ])
        table_lf = LithoFluidsTable(lfs, width=300, advanced=False)
        cds_lf = table_lf.cds
        lf_table, add_row, delete_row, update = table_lf.draw(cds_lf)

        # wedge model
        first_layer = Layer(name='First', thickness=Q_(50., 'm'), litho_fluid=lfs.litho_fluids[0])
        second_layer = Layer(name='Second', thickness=wedge, litho_fluid=lfs.litho_fluids[1])
        third_layer = Layer(name='Third', thickness=reverse_wedge, litho_fluid=lfs.litho_fluids[2])

        # # gas lens model
        # first_layer = Layer(thickness=Q_(0.05, 's'), vp=Q_(3400., 'm/s'), vs=Q_(1820., 'm/s'), rho=Q_(2.6, 'gram/cm^3'))
        # second_layer = Layer(thickness=Q_(0.03, 's'), vp=Q_(vp, 'm/s'), vs=Q_(vs, 'm/s'), rho=Q_(rho, 'gram/cm^3'), target=True)
        # third_layer = Layer(thickness=Q_(0.05, 's'), vp=Q_(3400., 'm/s'), vs=Q_(1820., 'm/s'), rho=Q_(2.6, 'gram/cm^3'))

        m = Model(depth_to_top=Q_(2000., 'm'), layers=[first_layer, second_layer, third_layer],
                  trace_index_range=np.arange(n_samplings))
        # m.plot()
        table_model = ModelTable(m, cds_lf)
        cds_model = table_model.cds
        model_table, add_row_m, delete_row_m, update_m = table_model.draw(cds_model)

        # wavelet = bumw.ricker(0.096, 0.001, 25)

        # plot_wiggles(m, 0.001, wavelet, angle=0., eei=True, scaling=80., extract_avo_at=[(8, 1.99), (24, 1.99)])
        # plot_wiggles(m, 0.001, wavelet, angle=15., eei=True, scaling=80., extract_avo_at=[(8, 1.99), (24, 1.99)])
        # plot_wiggles(m, 0.001, wavelet, angle=-90., eei=True, scaling=80., extract_avo_at=[(8, 1.99), (24, 1.99)])

        if unit_test:
            show(
                row(
                    column(model_table, row(add_row_m, delete_row_m, update_m)),
                    column(lf_table, row(add_row, delete_row, update))
                )
            )
            return None
        else:
            return model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update

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

    def test_model_table(self, unit_test=True, only_verbatim=True):
        from bokeh.plotting import show, row, column

        # When creating output to Bokeh server, we need to set only_verbatim False
        if not unit_test:
            only_verbatim = False

        lfs = LithoFluids(
            [LithoFluid(default='shale'), LithoFluid(default='brine_sst'), LithoFluid(default='oil_sst')]
        )
        table_lf = LithoFluidsTable(lfs, width=300, advanced=False)
        cds_lf = table_lf.cds
        lf_table, add_row, delete_row, update = table_lf.draw(cds_lf)

        # layers = [_lf.to_model_layer(_name) for _name, _lf in zip(['Top', 'Middle', 'Bottom'], lfs.litho_fluids)]
        layers = [
            LithoFluid(default='shale').to_model_layer('Top',thickness=Q_(50., 'm')),
            LithoFluid(default='brine_sst').to_model_layer('Reservoir',thickness=Q_(25., 'm')),
            LithoFluid(default='oil_sst').to_model_layer('Reservoir', case='Oil',thickness=Q_(25., 'm')),
            LithoFluid(default='shale').to_model_layer('Bottom',thickness=Q_(50., 'm')),
            LithoFluid(default='shale').to_model_layer('Basement',thickness=Q_(50., 'm'))
        ]
        # layers = []
        model = Model(layers=layers)
        table_model = ModelTable(model, cds_lf)
        cds_model = table_model.cds
        model_table, add_row_m, delete_row_m, update_m = table_model.draw(cds_model)

        if only_verbatim:
            for _key in list(cds_model.data.keys()):
                if _key == 'top' or _key == 'thickness':
                    print(_key)
                    print(cds_model.data[_key])
            realized_cases = table_model.realize(Q_(5.0, 'm'))
            # for _key in list(realized_cases.keys()):
            #     print(_key)
            #     print(realized_cases[_key]['vp'].values)
        else:
            if unit_test:
                show(
                    row(
                        column(model_table, row(add_row_m, delete_row_m, update_m)),
                        column(lf_table, row(add_row, delete_row, update))
                    )
                )
                return None
            else:
                return model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update
        return None

    def test_laminar_model(self, unit_test=True):
        from bokeh.plotting import show, row, column

        # Create the laminar model
        lm = LaminarModel(resolution=Q_(0.1, 'm'), avo_or_eei='avo')

        (title, lf_table, add_row, delete_row, update, model_table, add_row_m, delete_row_m, update_m, grid,
         controls, seismic_cds) = lm.draw()

        if unit_test:
            show(column(column(title, grid, sizing_mode='stretch_width'), controls,
                        row(
                            column(model_table, row(add_row_m, delete_row_m, update_m)),
                            column(lf_table, row(add_row, delete_row, update))
                        ))
                 )
            return None
        else:
            return model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update, grid, controls

    def test_laminar_q2d_model(self, unit_test=True):
        from bokeh.plotting import show, row, column

        def vp(_i):
            return Q_(3000. +  5. * _i, 'm/s')
        def vs(_i):
            return Q_(1500. +  5. * _i, 'm/s')
        def rho(_i):
            return Q_(2.5 +  0.1 * _i, 'grams/cm^3')
        def wedge(i):
            return Q_(50. - i, 'm')
        def reverse_wedge(i):
            return Q_(25. + i, 'm')

        # Litho-fluids
        lfs = LithoFluids([
            LithoFluid(default='shale'),
            LithoFluid(default='oil_sst'),
            LithoFluid(name='Quasi 2D', vp=vp, vs=vs, rho=rho),
            LithoFluid(name='Basement', vp=2800, vs=1350, rho=2.46)
        ])

        # wedge model
        first_layer = Layer(name='First', thickness=Q_(50., 'm'), litho_fluid=lfs.litho_fluids[0])
        second_layer = Layer(name='Second', thickness=wedge, litho_fluid=lfs.litho_fluids[2])
        second_layer_oil = Layer(name='Second', case='Oil', thickness=wedge, litho_fluid=lfs.litho_fluids[1])
        third_layer = Layer(name='Third', thickness=reverse_wedge, litho_fluid=lfs.litho_fluids[3])
        layers = [first_layer, second_layer, second_layer_oil, third_layer]
        model = Model(layers=layers, trace_index_range=np.arange(11))

        # Create the laminar model
        lm = LaminarModel(model=model, litho_fluids=lfs, resolution=Q_(0.1, 'm'), avo_or_eei='avo')

        # lf_table, add_row, delete_row, update, model_table, add_row_m, delete_row_m, update_m, grid, controls = lm.draw()
        (title, lf_table, add_row, delete_row, update, model_table, add_row_m, delete_row_m, update_m, grid, controls,
         synth_2d_cds) = lm.draw_2d()

        if unit_test:
            show(column(grid, controls,
                row(
                    column(model_table, row(add_row_m, delete_row_m, update_m)),
                    column(lf_table, row(add_row, delete_row, update))
                ))
            )
            return None
        else:
            return model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update, grid, controls

    def test_wedge(self, unit_test=True):
        from bokeh.plotting import show, row, column
        wm = WedgeModel(n_traces=51)
        lf_table, lf_controls, model_table, model_controls, grid, controls, new_grid, synth_2d_cds = wm.draw()

        # Test adding some extraction points
        analyze_avo = AvoAnalyzer(grid, 'line section', None, model=wm)
        points_cds = analyze_avo.cds
        points_table, update_avo, avo_figure, ixg_figure, eei_figure, chi_input, avo_cds = analyze_avo.draw(points_cds, synth_2d_cds)
        if unit_test:
            show(row(column(grid, controls, new_grid,
                        row(
                            column(model_table, model_controls),
                            column(lf_table, lf_controls)
                        )),
                 column(points_table, row(update_avo, chi_input), avo_figure, ixg_figure, eei_figure))
                 )
            return None
        else:
            return (model_table, model_controls, lf_table, lf_controls, grid, controls, new_grid, points_table,
                    update_avo, avo_figure, ixg_figure, eei_figure, chi_input)

    def test_catch_change_in_cases(self):
        prev_cases = [[1, 2], [1,2,3], [1,2], [1,4]]
        new_cases =  [[1,2], [1,2], [1,2,3], [1,3]]
        for new_case, prev_case in zip(new_cases, prev_cases):
            _dict = detect_change_in_cases(new_case, prev_case)
            print('New:', _dict['new'], ', Removed: ', _dict['removed'])

    def test_horizon_cds(self):
        def wedge(i):
            return Q_(50. - i, 'm')
        def reverse_wedge(i):
            return Q_(25. + i, 'm')
        first_layer = Layer(name='First', thickness=Q_(50., 'm'), litho_fluid=LithoFluid(default='shale'))
        second_layer = Layer(name='Second', thickness=wedge, litho_fluid=LithoFluid(default='brine_sst'))
        second_layer_oil = Layer(name='Second', case='Oil', thickness=wedge, litho_fluid=LithoFluid(default='oil_sst'))
        third_layer = Layer(name='Third', thickness=reverse_wedge, litho_fluid=LithoFluid(default='shale'))
        layers = [first_layer, second_layer, second_layer_oil, third_layer]
        model = Model(layers=layers, trace_index_range=np.arange(11))
        cds = model.horizons_cds()
        for _key in list(cds.data.keys()):
            print(_key, cds.data[_key])

import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import pint.errors
from math import isclose
from .. import ureg, Q_

from blixt_rp.core.log_curve_new import (LogCurve, Depth, is_equivalent, read_las,
                                         read_general_ascii, _to_twt, _to_depth)
from blixt_rp.core.core import Template, CutoffRule, Cutoffs, LogTable, Header

test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table.xlsx')

las_file1 = os.path.join(test_file_dir, "L-30.las")
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"
las_file3 = os.path.join(test_file_dir, "Well F.las")
data_file1 = os.path.join(test_file_dir, "Well A checkshot.txt")
data_file2 = "S:\\Well\\UTM32_Mid_Norway_All\\Q-6406\\6406_11_1_S\\6406_11_1_S___checkshot.txt"

n = 1500
# create a regularly sampled data set
data1 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth1 = Q_(np.linspace(24, 3430, n), 'm')
depth1_short = Q_(np.linspace(100, 3300, n), 'm')

# create an irregularly sampled data set
data2 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth2 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'FT')

n = 1400
# create a shorter regularly sampled data set
data3 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth3 = Q_(np.linspace(24, 3430, n), 'm')
depth3_short = Q_(np.linspace(100, 3300, n), 'm')

# create a shorter irregularly sampled data set
data4 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth4 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'feet')

data5 = Q_(np.linspace(6, 8, n) + np.random.random(n), 'Ohmm')
depth5 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'feet')

# cut off rules:
cr1 = CutoffRule('DataPair1', '><', [Q_(2.9, 'us/feet'), Q_(3.1, 'us/feet')])
cr2 = CutoffRule('Depth', '><', [Q_(1000., 'm'), Q_(2000., 'm')])
cr3 = CutoffRule('TEST', '><', [Q_(2.9, 'us/feet'), Q_(3.1, 'us/feet')])
cr4 = CutoffRule('DataPair1', '<', Q_(5, 'us/feet'))  # all should be included
cr5 = CutoffRule('DataPair1', '>', Q_(5, 'us/feet'))  # all should be excluded

# Useful log table
log_table1 = LogTable({'Density': 'rhob', 'Sonic': 'dt'})
log_table2 = LogTable({'Resistivity': 'rdep', 'Sonic': 'dt'})
log_table2_multi = LogTable({'Resistivity': ['rdep', 'rmed'], 'Sonic': ['dt']})
log_table3 = LogTable({'Density': 'rho_brine', 'P velocity': 'vp_brine', 'S velocity': 'vs_brine'})
log_table4 = LogTable({'Density': 'rhob', 'Sonic': 'dt', 'Gamma ray': 'grd'})

class LogCurveTestCase(unittest.TestCase):

    def test_data_types(self):
        print(type(data1))
        self.assertIsInstance(data1, pint.Quantity, '')
        d = Depth(depth1)
        self.assertIsInstance(d, Depth, '')
        print(depth2.units)
        d = Depth(depth2)  # Automatically converts from feet to meter
        print(d.units)
        self.assertTrue(is_equivalent(d.units, pint.Unit('meter')))
        self.assertFalse(is_equivalent(d.units, pint.Unit('feet')))

    def test_LogCurve_new(self):
        lc = LogCurve(
            'my_well_log',
            data1,
            Depth(depth1)
        )
        print(lc.depth_type)
        print(lc.depth_units)
        print(lc.is_evenly_spaced)
        print(lc.step())
        # Try convert units of data
        print(lc.data.max())
        lc.data = lc.data.to('s/feet')
        print(lc.data.max())
        lc = LogCurve(
            'my_well_log',
            data1,
            Depth(depth1),
            style={'units': 's/feet'}
        )
        print(lc.is_evenly_spaced)
        print(lc.step())
        print(lc.data.max())
        lc.data = lc.data.to('s/m')
        print(lc.data.max())
        print(lc.units)
        print(lc.get_line().min, lc.get_line().max, lc.get_line().x_range)

    def test_unit_convert_using_style(self):
        # This should fail, and raise warning, because the data pair has units us/ft,
        # which can't be converted to 'kg' which the # style asks for
        print('This should fail:')
        style = {'units': 'kg'}
        lc = LogCurve('', data1, Depth(depth1), None, None, Template(**style), None)
        print(lc.units, type(lc.units))
        print(lc.style.units, type(lc.style.units))
        self.assertTrue(is_equivalent(lc.units, ureg.Unit(lc.style.units)))

        print('This should work:')
        style = {'units': 'us/m'}
        lc = LogCurve('', data1, Depth(depth1), None, None, Template(**style), None)
        print(lc.units, type(lc.units))
        print(lc.style.units, type(lc.style.units))
        self.assertTrue(is_equivalent(lc.units, ureg.Unit(lc.style.units)))

        print('And now this works too :-)')
        style = {'units': 's/m'}
        lc.style = Template(**style)
        print(lc.units, type(lc.units))
        print(lc.style.units, type(lc.style.units))
        self.assertTrue(is_equivalent(lc.units, ureg.Unit(lc.style.units)))

        print('And this also')
        style = {'units': 's/feet'}
        lc.style = Template(**style)
        print(lc.units, type(lc.units))
        print(lc.style.units, type(lc.style.units))
        self.assertTrue(is_equivalent(lc.units, ureg.Unit(lc.style.units)))

    def test_built_in_unit_convert(self):
        lc = LogCurve(
            'test',
            data2,
            Depth(depth2)
        )
        print(lc.style.units)
        print('Original max of data: {}'.format(lc.data.max()))
        lc.units = 'us/m'
        #  lc.convert_to('us/m')
        print(lc.style.units)
        print('Max of data after conversion: {}'.format(lc.data.max()))
        print(lc.header.modification_history)
        print('Original depth {}'.format(lc.depth.values[0]))
        lc.depth_units = 'feet'
        # lc.convert_depth_to('feet')
        print(lc.header.modification_history)
        print('Depth after conversion {}'.format(lc.depth.values[0]))

    def test_calculate_velocity(self):
        lc_p = LogCurve(
            'test',
            data1,
            Depth(depth1),
            log_type='Sonic')
        lc_s = LogCurve(
            'test',
            data4,
            Depth(depth4),
            log_type='Shear sonic')

        lc_vp = lc_p.velocity_from_sonic('Vp_from_sonic')
        print(lc_vp.name)
        print(lc_vp.log_type)
        print(lc_vp.units)
        lc_vs = lc_s.velocity_from_sonic('Vs_from_sonic')
        print(lc_vs.name)
        print(lc_vs.log_type)
        print(lc_vs.units)

        print(lc_vs.header)
        print(lc_vs.style)

        # Following should fail because data are not sonic even if we say so
        lc_s_fail = LogCurve(
            'Failes',
            data5,
            Depth(depth5),
            log_type='Shear sonic')
        self.assertRaises(pint.DimensionalityError, lc_s_fail.velocity_from_sonic, 'Test')

    def test_take_sampling(self):
        lc1 = LogCurve( 'test', data1, Depth(depth1_short))
        lc3 = LogCurve('test', data3, Depth(depth3))
        print('Well 1. Top {}, Step {}, Base {}. Length {}'.format(lc1.top, lc1.step(), lc1.base, len(lc1)))
        print('Well 2. Top {}, Step {}, Base {}. Length {}'.format(lc3.top, lc3.step(), lc3.base, len(lc3)))
        lc_new = lc1.take_sampling_from(lc3)
        print('Well 1 takes sampling from Well 2:')
        print('        Top {}, Step {}, Base {}. Length {}'.format(lc_new.top, lc_new.step(), lc_new.base, len(lc_new)))
        fig, ax = plt.subplots()
        lc1.plot(ax=ax); lc_new.plot(ax=ax)
        plt.show()

    def test_LogCurve_copy(self):
        lc = LogCurve(
            'test',
            data1,
            Depth(depth1)
        )
        lc2 = lc.copy()
        print(lc2.header)

    def test_print(self):
        lc = LogCurve(
            'test',
            data1,
            Depth(depth1),
            header={'name': 'Yalla'}
        )
        print(lc)

    def test_log_type(self):
        lc = LogCurve(
            'My log',
            data1,
            Depth(depth1),
            header={'log_type': 'TEST'}
        )
        print(lc.log_type)
        lc = LogCurve(
            'Another log',
            data1,
            Depth(depth1)
        )
        lc.log_type = 'TEST 2'
        print(lc.log_type, lc.header.log_type)

    def test_failing_log_type(self):
        # This should raise an IOError because log type is not the same in header and initialization
        header = {'name': 'My log', 'log_type': 'TEST'}
        self.assertRaises(IOError, LogCurve, 'Name', data1, Depth(depth1), 'FAIL', None, None, header)
        # Below works too. Don't understand why
        # self.assertRaises(OSError, LogCurve, dp2, 'FAIL', None, True, None, header)

    def test_masks(self):
        lc1 = LogCurve(
            'DataPair1',
            data1,
            Depth(depth1),
            log_type='TEST'
        )
        # cutoffs1 = {'DataPair1': ['><', [2.9, 3.1]]}
        # cutoffs2 = {'Depth': ['><', [1000., 2000.]]}
        # cutoffs3 = {'DataPair1': ['><', [2.9, 3.1]], 'Depth': ['><', [1000., 2000.]]}
        # log_table1 = {'TEST': 'DataPair1'}
        # cutoffs4 = {'TEST': ['><', [2.9, 3.1]]}
        # cutoffs5 = {'TEST': ['><', [2.9, 3.1]], 'Depth': ['><', [1000., 2000.]]}
        # cutoffs6 = {'DataPair1': ['<',  5]}  # all should be included
        # cutoffs7 = {'DataPair1': ['>',  5]}  # all should be excluded

        # log tables
        log_table1 = LogTable('my log table', {'TEST': 'DataPair1'})

        # cut offs
        cutoffs1 = Cutoffs(cutoffs=[cr1])
        cutoffs2 = Cutoffs(cutoffs=[cr2])
        cutoffs3 = Cutoffs(cutoffs=[cr1, cr2])
        cutoffs4 = Cutoffs(log_table=log_table1, cutoffs=[cr3])
        cutoffs5 = Cutoffs(log_table=log_table1, cutoffs=[cr2, cr3])
        cutoffs6 = Cutoffs(cutoffs=[cr4])
        cutoffs7 = Cutoffs(cutoffs=[cr5])


        self.assertIsInstance(lc1.calc_mask(cutoffs1, verbose=True), LogCurve, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs2, verbose=True), LogCurve, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs3, verbose=True), LogCurve, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs4, verbose=True), LogCurve, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs5, verbose=True), LogCurve, '')
        self.assertLessEqual(np.max(lc1.values[lc1.calc_mask(cutoffs1, verbose=True).values]), 3.1, '')
        self.assertTrue(lc1.calc_mask(cutoffs6, verbose=True).values.all())
        self.assertTrue(np.sum(lc1.calc_mask(cutoffs7, verbose=True).values) < 1)

        # self.assertIsInstance(lc1.calc_mask(cutoffs1, verbose=True), LogCurve, '')
        # self.assertIsInstance(lc1.calc_mask(cutoffs2, verbose=True), LogCurve, '')
        # self.assertIsInstance(lc1.calc_mask(cutoffs3, verbose=True), LogCurve, '')
        # self.assertIsInstance(lc1.calc_mask(cutoffs4, verbose=True, log_table=log_table1), LogCurve, '')
        # self.assertIsInstance(lc1.calc_mask(cutoffs5, verbose=True, log_table=log_table1), LogCurve, '')
        # self.assertLessEqual(np.max(lc1.values[lc1.calc_mask(cutoffs1, verbose=True).values]), 3.1, '')
        # self.assertTrue(lc1.calc_mask(cutoffs6, verbose=True).values.all())
        # self.assertTrue(np.sum(lc1.calc_mask(cutoffs7, verbose=True).values) < 1)

    def test_read(self):
        # # las_file = 'C:\\Users\\marte\\PycharmProjects\\blixt_rp\\test_data\\Well E_CPI.las'
        # log_curves, well_info = read_las(las_file1)
        # print(len(log_curves))
        # lc = log_curves['grd']
        # print(lc.well, lc.name, lc.log_type, lc.min, lc.max, lc.units, lc.depth_type, lc.depth.top,
        #       lc.depth.base, lc.depth_units)

        # print('\nNow using log_table')
        # log_curves, well_info = read_las(las_file1, log_table=log_table1)
        # print(len(log_curves))
        # lc = log_curves['dt']
        # print(lc.well, lc.name, lc.log_type, lc.min, lc.max, lc.units, lc.depth_type, lc.depth.top,
        #       lc.depth.base, lc.depth_units)

        # print(lc.header)

        # print('\nNow using erroneous log_table')
        # log_table = LogTable({'Density': 'xxx', 'Sonic': 'yyy'})
        # log_curves, well_info = read_las(las_file1, log_table=log_table)
        # print(len(log_curves))

        print('Reading a more complex las file')
        # log_curves, well_info = read_las(las_file2, log_table=log_table2)
        # lc = log_curves['rdep']
        log_curves, well_info = read_las(las_file2, log_table=log_table2_multi)
        lc = log_curves['rmed']
        print(lc.well, lc.name, lc.log_type, lc.min, lc.max, lc.units, lc.depth_type, lc.depth.top,
              lc.depth.base, lc.depth_units)

    def test_read_and_rename(self):

        rename_logs = {'phie': ['cpi_phie']}
        # The LogTable contains the translated log name, which is necessary as the same LogTable should be
        # able to be used across multiple las files, which all use different names for Effective Porosity:w
        log_table = LogTable(
            {'Resistivity': ['rdep', 'rmed'], 'Sonic': ['dt'], 'Porosity': ['PHIE', 'CPI_PHIT']})
        log_curves, well_info = read_las(las_file2, log_table=log_table, rename_logs=rename_logs)
        print(list(log_curves.keys()))
        lc = log_curves['rmed']
        print(lc.well, lc.name, lc.log_type, lc.min, lc.max, lc.units, lc.depth_type, lc.depth.top,
              lc.depth.base, lc.depth_units)

    def test_read_seismic_from_las(self):
        las_file = "G:\\My Drive\\Work - Current and Recent\\GeoMind\\Clients\\AkerBP\\PL932 Kaldafjell AVO feasibility\\Wells\\34_3_3S_seismic.las"
        t = Template(**{'name': 'Seismic', 'units': 'dimensionless', 'line_color': 'b'})
        log_table = LogTable({'Seismic':
                                  ['CGG18M01-PSDM-FIN-FULL-T-Denoise2',
                                    'CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2']})
        rename_logs = {'full': ['CGG18M01-PSDM-FIN-FULL-T-Denoise2'],
                       'mid': ['CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2']}
        log_curves, well_info = read_las(las_file, log_table=log_table, template=t, rename_logs=rename_logs)
        for key, lc in log_curves.items():
            print('-x-')
            print(key, '\n', lc.name, '\n', lc.well, '\n', lc.style, '\n', lc.log_type)

    def test_Template(self):
        t = Template(**{'name': 'MY NAME', 'well': 'MY WELL', 'units': 'METER'})
        print(list(t.keys()))
        print(t.name, t.units)
        t = Template()
        print(t)
        t.get_from_project(project_table, 'Resistivity')
        t.max = 999
        t.colormap = 'A GIANT COLOR'
        print(t)

    def test_a_lot(self):
        # fig, ax = plt.subplots()
        log_curves, well_info = read_las(las_file2, log_table=log_table2)
        lc = log_curves['dt']
        # lc.plot(ax=ax)
        cutoffs = Cutoffs([CutoffRule('dt', '>', Q_(140., 'us/feet'))])
        # lc_smooth10 = lc.smooth(Q_(10., 'm'))
        # lc_smooth10.plot(ax=ax)
        # lc_smooth500 = lc.smooth(Q_(500., 'm'))
        # lc_smooth500.plot(ax=ax)
        # lc_clean = lc_smooth500.clean_data()
        # print(len(lc), len(lc_smooth500), len(lc_clean))
        # fig, ax = plt.subplots()
        # lc.plot(ax=ax); lc_smooth500.plot(ax=ax); lc_clean.plot(ax=ax)
        # lc_smooth500.clean_data(overwrite=True)
        # fig, ax = plt.subplots()
        # lc.plot(ax=ax); lc_smooth500.plot(ax=ax)
        # print(lc_smooth500.header)
        lc_mask = lc.calc_mask(cutoffs, verbose=True)

        fig, ax = plt.subplots()
        lc = log_curves['rdep']
        lc_smooth_with_mask = lc.smooth(Q_(10., 'm'), mask=lc_mask.values)
        lc_smooth_with_mask.plot(ax=ax)
        lc_smooth10 = lc.smooth(Q_(500., 'ft'), discrete_intervals=[Q_(_z, 'km').to('m') for _z in [1., 2., 3.]])
        print(lc_smooth10.header)
        lc_smooth10.plot(ax=ax)

        plt.show()
        self.assertTrue(True)

    def test_fill(self):
        fig, ax = plt.subplots()
        log_curves, well_info = read_las(las_file2, log_table=log_table2)
        lc = log_curves['dt']
        # lc.plot(ax=ax)
        cutoffs = Cutoffs([CutoffRule('dt', '>', Q_(140., 'us/feet'))])
        lc_mask = lc.calc_mask(cutoffs, verbose=True)

        # lc = log_curves['rdep']
        # lc_fill1 = lc.fill_gaps()
        lc_fill1 = lc.fill_gaps(mask=lc_mask.values)
        # lc_fill1.plot(ax=ax)
        # lc_fill2 = lc.fill_gaps(extrapolate=True)
        # lc_fill2 = lc.fill_gaps('fill_with_values', fill_values=np.ones(len(lc)))
        # lc_fill2 = lc.fill_gaps('fill_with_values', fill_values=np.ones(len(lc)), mask=lc_mask.values)
        lc_fill1.plot(ax=ax)
        # ; lc.data.plot(); lc_fill2.data.plot.line('k--')
        lc.plot(ax=ax)

        plt.show()
        self.assertTrue(True)

    def test_despike(self):
        fig, ax = plt.subplots()
        log_curves, well_info = read_las(las_file1, log_table=log_table4)
        lc = log_curves['grd']  # despike fails with gamma ray data because of the pint units
        # lc = log_curves['dt']
        lc.plot(ax=ax)
        # lc_despiked = lc.despike(Q_(20, 'us/ft'), window_len=Q_(20., 'm'), suffix='despike')
        lc_despiked = lc.despike(20, window_len=Q_(20., 'm'), suffix='despike')
        lc_despiked.plot(ax=ax)
        print(lc_despiked.header)
        plt.show()
        self.assertTrue(True)

    def test_calc_trend(self):
        log_curves, well_info = read_las(las_file2, log_table=log_table2)
        lc = log_curves['rdep']
        res = lc.calc_depth_trend(verbose=True, discrete_intervals=[Q_(_z, 'km').to('m') for _z in [1., 2., 3.]])
        res = lc.calc_depth_trend(verbose=True)
        plt.show()
        # print(res)
        self.assertIsInstance(res, list)

    def test_de_trend(self):
        log_curves, well_info = read_las(las_file2, log_table=log_table2)
        lc = log_curves['dt']
        res = lc.calc_depth_trend(verbose=True)
        lc_de_trend = lc.de_trend(Q_(2., 'km'), None, *res[0].x, suffix='test', verbose=True)
        plt.show()
        print(lc_de_trend.header)
        self.assertIsInstance(res, list)

    def test_depth(self):
        from blixt_rp.core.log_curve_new import Depth
        coords_list = [10.0, (2., 3.), np.linspace(5., 10., 3), 1.,
                       10.0, (2., 3.), np.linspace(5., 10., 3), 1.,
                       10.0, (2., 3.), np.linspace(5., 10., 3), 1.]
        coord_units_list = ['s', 's', 's', 's',
                            'FT', 'FT', 'FT', 'FT',
                            'M', 'M', 'M', 'M']
        coord_types_list = ['twt', 'md', 'owt', 'tvd',
                            'twt', 'md', 'owt', 'tvd',
                            'twt', 'md', 'owt', 'tvd']
        successes = [True, False, True, False,
                     False, True, False, True,
                     False, True, False, True]

        for i, coords in enumerate(coords_list):
            coord_units = coord_units_list[i]
            coord_type = coord_types_list[i]
            success = successes[i]
            if success:
                print('{}: This should work'.format(i))
                print(coord_type, coord_units)
                a = Depth(coords, units=coord_units, depth_type=coord_type)
                if coord_type == 'twt':
                    if isinstance(a.values, float):
                        self.assertTrue(isclose(a.values, coords * 1000.))
                    else:
                        self.assertTrue(isclose(a.values[0], coords[0] * 1000.))
                    a.coord_type = 'TEST'
                    a.coords = Q_(0.1, 'hours')
                    print(a.coord_type, a.coords)
                elif coord_units == 'FT':
                    if isinstance(a.values, float):
                        self.assertTrue(isclose(a.values, coords / 3.281, rel_tol=1.e-4))
                    else:
                        self.assertTrue(isclose(a.values[0], coords[0] / 3.281, rel_tol=1.e-4))
                elif coord_units == 'M':
                    if isinstance(a.values, float):
                        self.assertTrue(isclose(a.values, coords))
                    else:
                        self.assertTrue(isclose(a.values[0], coords[0]))
            else:
                print('{}: This should fail'.format(i))
                self.assertRaises(pint.errors.DimensionalityError, Depth, coords, coord_units, coord_type, True)

        dpth1 = Depth(Q_(np.linspace(10, 20, 11), 'm'))
        dpth2 = Depth(Q_(np.linspace(10, 20, 9), 'm'))
        dpth3 = Depth(Q_(np.linspace(10, 20, 11), 'feet'))  # Automatically converts to meters
        print(dpth1.unique_depths(dpth2).values)
        print(dpth1.unique_depths(dpth3).values)
        self.assertTrue(is_equivalent(dpth1.units, dpth3.units))

    def test_rename(self):
        lc = LogCurve(
            'test',
            data2,
            Depth(depth2)
        )
        print('Original header:\n', lc.header)

        lc.name = 'renamed log curve'
        lc.well = 'renamed well name'
        print('Modified header:\n', lc.header)
        print('Well and log curve names: ', lc.well, lc.name)

    def test_dimensionless_units(self):
        # log_curves, well_info = read_las(las_file1, log_table=log_table4)
        # lc = log_curves['grd']
        # print(lc.units, np.nanmin(lc.data), np.nanmax(lc.data))
        # print('Simple multiplication: ', np.nanmin(lc.data * 10.))
        # # print('Simple subtraction: ', np.nanmin(lc.data * 1.1 - lc.data * 0.9))  # this fails
        t1 = Q_(np.random.normal(size=10), 'API')
        t2 = Q_(np.random.normal(size=10), 'API')
        print(t1 - t2)


class DomainConversionTest(unittest.TestCase):
    def test_to_twt(self):
        twt_in = Q_(np.array([10, 15, 21, 27, 33, 40, 46, 52, 58, 64]), 'millisecond')
        dt = Q_(5., 'millisecond')
        data = np.linspace(2, 4, len(twt_in))
        twt_out, data_out = _to_twt(twt_in, data, dt)
        fig, ax = plt.subplots()
        ax.scatter(twt_in, data)
        ax.scatter(twt_out, data_out)

        data_twt = _to_depth(twt_in, Q_(twt_out, 'millisecond'), data_out)
        ax.scatter(twt_in, data_twt, marker='x', color='yellow')
        plt.show()

    def test_convert_real_data(self):
        log_curves, well_info = read_las(las_file2, log_table=log_table3)
        data_curves = read_general_ascii(
            data_file2,
            'space',
            4,
            ['md', 'owt'],
            [0, 1],
            ['m', 'millisecond'],
            ['MD', 'One-way time'])

        owt_lc = data_curves['owt'].take_sampling_from(log_curves['vp_brine'])
        twt = 2.*owt_lc.data
        # Convert the data to the time domain, with a regular sampling
        dt = Q_(1, 'millisecond')
        vp_twt = log_curves['vp_brine'].to_twt(twt, dt)
        vs_twt = log_curves['vs_brine'].to_twt(twt, dt)
        rho_twt = log_curves['rho_brine'].to_twt(twt, dt)
        print(vp_twt.header)
        vp_twt.plot()

    def test_twt_from_owt(self):
        data_curves = read_general_ascii(
            data_file2,
            'space',
            4,
            ['md', 'owt'],
            [0, 1],
            ['m', 'millisecond'],
            ['MD', 'One-way time'])

        owt = data_curves['owt']
        twt = data_curves['md'].get_twt_from_owt()
        twt = data_curves['owt'].get_twt_from_owt()
        print('OWT: ', owt.base, owt.top, owt.min, owt.max, owt.units, len(owt), owt.log_type)
        print('TWT: ', twt.base, twt.top, twt.min, twt.max, twt.units, len(twt), twt.log_type)


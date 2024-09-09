import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import pint.errors
import xarray as xr
from math import isclose
from .. import ureg, Q_

from blixt_rp.core.param import Param

from blixt_rp.core.log_curve_new import LogCurve, LogCurve2dNew, Depth, is_equivalent, read_las
from blixt_rp.core.template_new import Template
from blixt_rp.core.header_new import Header

test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table.xlsx')

las_file1 = os.path.join(test_file_dir, "L-30.las")
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"

n = 1500
# create a regularly sampled data set
data1 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth1 = Q_(np.linspace(24, 3430, n), 'm')

# create an irregularly sampled data set
data2 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth2 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'FT')

n = 1400
# create a shorter regularly sampled data set
data3 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth3 = Q_(np.linspace(24, 3430, n), 'm')

# create a shorter irregularly sampled data set
data4 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth4 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'feet')

data5 = Q_(np.linspace(6, 8, n) + np.random.random(n), 'Ohmm')
depth5 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'feet')


class WellTestCase(unittest.TestCase):

    def test_data_types(self):
        print(type(data1))
        self.assertIsInstance(data1, pint.Quantity, '')
        d = Depth(depth1)
        self.assertIsInstance(d, Depth, '')
        print(depth2.units)
        d = Depth(depth2)  # Automatically converts from feet to meter
        print(d.units)
        self.assertTrue(is_equivalent(d.units, pint.Unit('meter')))

    def test_LogCurve_new(self):
        lc = LogCurve2dNew(
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
        lc = LogCurve2dNew(
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

    def test_failing_unit_convert(self):
        # This should fail, and raise warning, because the data pair has units us/ft,
        # which can't be converted to 'kg' which the # style asks for
        style = {'units': 'kg'}
        lc = LogCurve2dNew('', data1, Depth(depth1), None, None, style, None)
        print(lc.units)
        print(lc.style)

    def test_built_in_unit_convert(self):
        lc = LogCurve2dNew(
            'test',
            data2,
            Depth(depth2)
        )
        print(lc.style.units)
        print('Original max of data: {}'.format(lc.data.max()))
        lc.convert_to('us/m')
        print(lc.style.units)
        print('Max of data after conversion: {}'.format(lc.data.max()))
        print(lc.header.modification_history)
        print('Original depth {}'.format(lc.depth.values[0]))
        lc.convert_depth_to('feet')
        print(lc.header.modification_history)
        print('Depth after conversion {}'.format(lc.depth.values[0]))

    def test_calculate_velocity(self):
        lc_p = LogCurve2dNew(
            'test',
            data1,
            Depth(depth1),
            log_type='Sonic')
        lc_s = LogCurve2dNew(
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
        lc_s_fail = LogCurve2dNew(
            'Failes',
            data5,
            Depth(depth5),
            log_type='Shear sonic')
        self.assertRaises(pint.DimensionalityError, lc_s_fail.velocity_from_sonic, 'Test')

    def test_take_sampling(self):
        lc1 = LogCurve2dNew(
            'test',
            data1,
            Depth(depth1)
        )
        lc3 = LogCurve2dNew('test', data3, Depth(depth3))
        lc_new = lc1.take_sampling_from(lc3)
        print(lc1.step())
        print(lc3.step())
        print(lc_new.step())
        fig, ax = plt.subplots()
        lc1.plot(ax=ax); lc_new.plot(ax=ax)
        print(lc1.name, len(lc1), lc3.name, len(lc3), lc_new.name, len(lc_new))
        plt.show()

    def test_LogCurve_copy(self):
        lc = LogCurve2dNew(
            'test',
            data1,
            Depth(depth1)
        )
        lc2 = lc.copy()
        print(lc2.header)

    def test_print(self):
        lc = LogCurve2dNew(
            'test',
            data1,
            Depth(depth1),
            header={'name': 'Yalla'}
        )
        print(lc)

    def test_log_type(self):
        lc = LogCurve2dNew(
            'My log',
            data1,
            Depth(depth1),
            header={'log_type': 'TEST'}
        )
        print(lc.log_type)
        lc = LogCurve2dNew(
            'Another log',
            data1,
            Depth(depth1)
        )
        lc.log_type = 'TEST 2'
        print(lc.log_type, lc.header.log_type)

    def test_failing_log_type(self):
        # This should raise an IOError because log type is not the same in header and initialization
        header = {'name': 'My log', 'log_type': 'TEST'}
        self.assertRaises(IOError, LogCurve2dNew, 'Name', data1, Depth(depth1), 'FAIL', None, None, header)
        # Below works too. Don't understand why
        # self.assertRaises(OSError, LogCurve2dNew, dp2, 'FAIL', None, True, None, header)

    def test_masks(self):
        lc1 = LogCurve2dNew(
            'DataPair1',
            data1,
            Depth(depth1),
            log_type='TEST'
        )
        cutoffs1 = {'DataPair1': ['><', [2.9, 3.1]]}
        cutoffs2 = {'Depth': ['><', [1000., 2000.]]}
        cutoffs3 = {'DataPair1': ['><', [2.9, 3.1]], 'Depth': ['><', [1000., 2000.]]}
        log_table1 = {'TEST': 'DataPair1'}
        cutoffs4 = {'TEST': ['><', [2.9, 3.1]]}
        cutoffs5 = {'TEST': ['><', [2.9, 3.1]], 'Depth': ['><', [1000., 2000.]]}
        cutoffs6 = {'DataPair1': ['<',  5]}  # all should be included
        cutoffs7 = {'DataPair1': ['>',  5]}  # all should be excluded
        self.assertIsInstance(lc1.calc_mask(cutoffs1, verbose=True), LogCurve2dNew, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs2, verbose=True), LogCurve2dNew, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs3, verbose=True), LogCurve2dNew, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs4, verbose=True, log_table=log_table1), LogCurve2dNew, '')
        self.assertIsInstance(lc1.calc_mask(cutoffs5, verbose=True, log_table=log_table1), LogCurve2dNew, '')
        self.assertLessEqual(np.max(lc1.values[lc1.calc_mask(cutoffs1, verbose=True).values]), 3.1, '')
        self.assertTrue(lc1.calc_mask(cutoffs6, verbose=True).values.all())
        self.assertTrue(np.sum(lc1.calc_mask(cutoffs7, verbose=True).values) < 1)

    def test_read(self):
        # las_file = 'C:\\Users\\marte\\PycharmProjects\\blixt_rp\\test_data\\Well E_CPI.las'
        log_curves, well_info = read_las(las_file1)
        print(len(log_curves))
        lc = log_curves['grd']
        print(lc.well, lc.name, lc.log_type, lc.min, lc.max, lc.units, lc.depth_type, lc.depth.top,
              lc.depth.base, lc.depth_units)

        print('\nNow using log_table')
        log_table = {'Density': 'rhob', 'Sonic': 'dt'}
        log_curves, well_info = read_las(las_file1, log_table=log_table)
        print(len(log_curves))
        lc = log_curves['dt']
        print(lc.well, lc.name, lc.log_type, lc.min, lc.max, lc.units, lc.depth_type, lc.depth.top,
              lc.depth.base, lc.depth_units)

        print(lc.header)

        print('\nNow using erroneous log_table')
        log_table = {'Density': 'xxx', 'Sonic': 'yyy'}
        log_curves, well_info = read_las(las_file1, log_table=log_table)
        print(len(log_curves))

        print('Reading a more complex las file')
        log_table = {'Resistivity': 'rdep', 'Sonic': 'dt'}
        log_curves, well_info = read_las(las_file2, log_table=log_table)
        lc = log_curves['rdep']
        print(lc.well, lc.name, lc.log_type, lc.min, lc.max, lc.units, lc.depth_type, lc.depth.top,
              lc.depth.base, lc.depth_units)


    def test_LogCurve(self):
        lc = LogCurve(
            name='test_data',
            data=data1,
            start=Param(name='start', value=24, unit='m'),
            stop=Param(name='stop', value=3430, unit='m'),
            style={'full_name': 'A test',
                   'min': 1,
                   'max': 10},
            header={'unit': 'X'}
        )
        fig, ax = plt.subplots()
        mask = np.ma.masked_inside(lc.get_depth(), 1000., 1750.).mask
        # lc.depth_plot(mask=mask, mask_desc='Inside 1000 - 1750 m MD', ax=ax)
        lc.smooth(window_len=50, overwrite=True)
        # lc.depth_plot(ax=ax)
        fit_parameters = lc.calc_depth_trend(
            lc.get_depth(),
            down_weight_outliers=True,
            verbose=False
        )
        fitted_log = lc.apply_trend_function(lc.get_depth(), fit_parameters, verbose=False)
        ax.plot(fitted_log, lc.get_depth(), '--')
        return lc

    def test_Template(self):
        t = Template({'name': 'MY NAME', 'well': 'MY WELL', 'units': 'METER'})
        print(list(t.keys()))
        print(t.name, t.units)
        t = Template()
        print(t)
        t.get_from_project(project_table, 'Resistivity')
        t['max'] = 999
        t.colormap = 'A GIANT COLOR'
        print(t)

    def test_Header(self):
        h = Header({'name': 'MY NAME', 'well': 'MY WELL'})
        print(h.keys())
        print(h.name, h.creation_date)
        time.sleep(1)
        h.orig_filename = 'A TEST'
        print(h.orig_filename, h.creation_date, h.modification_date)

    def test_a_lot(self):
        # fig, ax = plt.subplots()
        log_table = {'Resistivity': 'rdep', 'Sonic': 'dt'}
        log_curves, well_info = read_las(las_file2, log_table=log_table)
        lc = log_curves['dt']
        # lc.plot(ax=ax)
        cutoffs = {'dt': ['>', 140.]}
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
        log_table = {'Resistivity': 'rdep', 'Sonic': 'dt'}
        log_curves, well_info = read_las(las_file2, log_table=log_table)
        lc = log_curves['dt']
        # lc.plot(ax=ax)
        cutoffs = {'dt': ['>', 140.]}
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
        log_table = {'Resistivity': 'rdep', 'Sonic': 'dt'}
        log_curves, well_info = read_las(las_file2, log_table=log_table)
        lc = log_curves['dt']
        lc.plot(ax=ax)
        lc_despiked = lc.despike(Q_(20, 'us/ft'), window_len=Q_(20., 'm'), suffix='despike')
        lc_despiked.plot(ax=ax)
        print(lc_despiked.header)
        plt.show()
        self.assertTrue(True)

    def test_calc_trend(self):
        log_table = {'Resistivity': 'rdep', 'Sonic': 'dt'}
        log_curves, well_info = read_las(las_file2, log_table=log_table)
        lc = log_curves['rdep']
        res = lc.calc_depth_trend(verbose=True, discrete_intervals=[Q_(_z, 'km').to('m') for _z in [1., 2., 3.]])
        res = lc.calc_depth_trend(verbose=True)
        plt.show()
        self.assertIsInstance(res, list)

    def test_de_trend(self):
        log_table = {'Resistivity': 'rdep', 'Sonic': 'dt'}
        log_curves, well_info = read_las(las_file2, log_table=log_table)
        lc = log_curves['dt']
        res = lc.calc_depth_trend(verbose=True)
        lc_de_trend = lc.de_trend(Q_(2., 'km'), None, *res[0], suffix='test')
        lc_de_trend.plot()
        plt.show()
        print(lc_de_trend.header)
        self.assertIsInstance(res, list)

    def test_coords(self):
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

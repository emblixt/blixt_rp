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

from blixt_rp.core.log_curve_new import LogCurve, LogCurve2dNew, create_data_array
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
data1 = np.linspace(2, 4, n) + np.random.random(n)
depth1 = np.linspace(24, 3430, n)

# create an irregularly sampled data set
data2 = np.linspace(6, 8, n) + np.random.random(n)
depth2 = np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n)

n = 1400
# create a shorter regularly sampled data set
data3 = np.linspace(2, 4, n) + np.random.random(n)
depth3 = np.linspace(24, 3430, n)

# create a shorter irregularly sampled data set
data4 = np.linspace(6, 8, n) + np.random.random(n)
depth4 = np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n)

dp1 = create_data_array(
    'DataPair1',
    data1,
    'us/feet',
    depth1,
    'm',
    'md')

dp2 = create_data_array(
    'DataPair2',
    data2,
    's/m',
    depth2,
    'FT',
    'md')

dp3 = create_data_array(
    'DataPair3',
    data3,
    'us/feet',
    depth3,
    'm',
    'md')

dp4 = create_data_array(
    'DataPair4',
    data4,
    's/m',
    depth4,
    'feet',
    'md')


class WellTestCase(unittest.TestCase):

    def test_data_pair(self):
        print(type(dp2))
        print(dp2.name)
        if isinstance(dp2, xr.DataArray):
            print('It is recognised as a DataArray')
        else:
            print('It is NOT recognized')
        self.assertIsInstance(dp2, xr.DataArray, '')

    def test_LogCurve_new(self):
        lc = LogCurve2dNew(
            dp2
        )
        print(lc.coord_type)
        print(lc.coord_units)
        print(lc.coords)
        print(lc.is_evenly_spaced)
        print(lc.step)
        # Try convert units of data
        print(lc.d_arr.data.max())
        lc.d_arr = lc.d_arr.pint.to('s/feet')
        print(lc.d_arr.data.max())
        lc = LogCurve2dNew(
            dp1,
            style={'units': 's/feet'}
        )
        print(lc.is_evenly_spaced)
        print(lc.step)
        print(lc.d_arr.data.max())
        lc.d_arr = lc.d_arr.pint.to('s/m')
        print(lc.d_arr.data.max())
        print(lc.units)
        print(lc.values)

    def test_failing_unit_convert(self):
        # This should fail because the data pair has units us/ft, which can't be converted to 'kg' which the
        # style asks for
        style = {'units': 'kg'}
        self.assertRaises(ValueError, LogCurve2dNew, dp2, None, None, style, None)

    def test_built_in_unit_convert(self):
        lc = LogCurve2dNew(
            dp2
        )
        print(lc.style)
        print(lc.d_arr.data.max())
        lc.convert_to('us/m')
        print(lc.style)
        print(lc.d_arr.data.max())
        print(lc.header)
        print(lc.d_arr.coords.values())
        lc.convert_coords_to('feet')
        print(lc.style)
        print(lc.d_arr.data.max())
        print(lc.header)
        print(lc.d_arr.coords.values())

    def test_take_sampling(self):
        lc1 = LogCurve2dNew(dp1)
        lc3 = LogCurve2dNew(dp3)
        lc_new = lc1.take_sampling_from(lc3)
        print(lc1.step)
        print(lc3.step)
        print(lc_new.step)
        lc1.d_arr.plot()
        lc3.d_arr.plot()
        lc_new.d_arr.plot()
        print(lc1.name, len(lc1), lc3.name, len(lc3), lc_new.name, len(lc_new))
        plt.show()

    def test_LogCurve_copy(self):
        lc = LogCurve2dNew(
            dp2
        )
        lc2 = lc.copy()

    def test_print(self):
        lc = LogCurve2dNew(
            dp2,
            header={'name': dp2.name}
        )
        print(lc)

    def test_log_type(self):
        lc = LogCurve2dNew(
            dp2,
            header={'name': dp2.name, 'log_type': 'TEST'}
        )
        print(lc.log_type)
        lc = LogCurve2dNew(
            dp2,
            header={'name': dp2.name}
        )
        lc.log_type = 'TEST 2'
        print(lc.log_type)

    def test_failing_log_type(self):
        # This should raise an IOError because log type is not the same in header and initialization
        header = {'name': dp2.name, 'log_type': 'TEST'}
        self.assertRaises(IOError, LogCurve2dNew, dp2, 'FAIL', None,  None, header)
        # Below works too. Don't understand why
        # self.assertRaises(OSError, LogCurve2dNew, dp2, 'FAIL', None, True, None, header)

    def test_masks(self):
        lc1 = LogCurve2dNew(
            dp1,
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
        lc = LogCurve2dNew(None)
        las_file = 'C:\\Users\\marte\\PycharmProjects\\blixt_rp\\test_data\\Well E_CPI.las'
        lc.read('phie', las_file, 'las', True)
        print(lc.name, np.nanmin(lc.values), np.nanmax(lc.values), lc.units, lc.coord_type, np.min(lc.coords),
              np.max(lc.coords), lc.coord_units)
        self.assertIsInstance(lc, LogCurve2dNew, '')

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
        lc = LogCurve2dNew(None)
        cutoffs = {'dt': ['>', 140.]}
        # lc.read('dt', las_file1, 'las')
        # lc_smooth10 = lc.smooth(Q_(10., 'm'))
        # lc_smooth10.d_arr.plot()
        # lc_smooth500 = lc.smooth(Q_(500., 'm'))
        # lc_smooth500.d_arr.plot()
        # lc_clean = lc_smooth500.clean_data()
        # print(len(lc), len(lc_smooth500), len(lc_clean))
        # lc.d_arr.plot(); lc_smooth500.d_arr.plot(); lc_clean.d_arr.plot()
        # lc_mask = lc.calc_mask(cutoffs, verbose=True)

        lc.read('rdep', las_file2, 'las')
        # lc_smooth_with_mask = lc.smooth(Q_(10., 'm'), mask=lc_mask.values)
        # lc_smooth_with_mask.d_arr.plot()
        lc_smooth10 = lc.smooth(Q_(500., 'ft'), discrete_intervals=[Q_(_z, 'km').to('m').magnitude for _z in [1., 2., 3.]])
        print(lc_smooth10.header)
        # lc_smooth10.d_arr.plot()
        lc_fill1 = lc.fill_gaps()
        # lc_fill2 = lc.fill_gaps(extrapolate=True)
        lc_fill2 = lc.fill_gaps('fill_with_values', fill_values=np.ones(len(lc)))
        lc_fill1.d_arr.plot(); lc.d_arr.plot(); lc_fill2.d_arr.plot.line('k--')

        plt.show()
        self.assertTrue(True)

    def test_coords(self):
        from blixt_rp.core.log_curve_new import Coordinate
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
                a = Coordinate(coords, units=coord_units, coord_type=coord_type)
                if coord_type == 'twt':
                    if isinstance(a.data, float):
                        self.assertTrue(isclose(a.data, coords * 1000.))
                    else:
                        self.assertTrue(isclose(a.data[0], coords[0] * 1000.))
                    a.coord_type = 'TEST'
                    a.coords = Q_(0.1, 'hours')
                    print(a.coord_type, a.coords)
                elif coord_units == 'FT':
                    if isinstance(a.data, float):
                        self.assertTrue(isclose(a.data, coords / 3.281, rel_tol=1.e-4))
                    else:
                        self.assertTrue(isclose(a.data[0], coords[0] / 3.281, rel_tol=1.e-4))
                elif coord_units == 'M':
                    if isinstance(a.data, float):
                        self.assertTrue(isclose(a.data, coords))
                    else:
                        self.assertTrue(isclose(a.data[0], coords[0]))
            else:
                print('{}: This should fail'.format(i))
                self.assertRaises(pint.errors.DimensionalityError, Coordinate, coords, coord_units, coord_type, True)

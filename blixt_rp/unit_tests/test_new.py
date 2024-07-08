import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import xarray as xr
from pint import UnitRegistry

from blixt_rp.core.param import Param

from blixt_rp.core.log_curve_new import LogCurve, LogCurve2dNew, create_data_array, is_equivalent
from blixt_rp.core.template_new import Template
from blixt_rp.core.header_new import Header

ureg = UnitRegistry()

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
    'feet',
    'md')

dp3 = create_data_array(
    'DataPair1',
    data3,
    'us/feet',
    depth3,
    'm',
    'md')

dp4 = create_data_array(
    'DataPair2',
    data4,
    's/m',
    depth4,
    'feet',
    'md')


class WellTestCase(unittest.TestCase):

    def test_data_pair(self):
        print(type(dp1))
        print(dp1.name)
        if isinstance(dp1, xr.DataArray):
            print('It is recognised as a DataArray')
        else:
            print('It is NOT recognized')

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
        print(lc.data.data.max())
        lc.data = lc.data.pint.to('s/feet')
        print(lc.data.data.max())
        lc = LogCurve2dNew(
            dp1,
            style={'units': 's/feet'}
        )
        print(lc.is_evenly_spaced)
        print(lc.step)
        print(lc.data.data.max())
        lc.data = lc.data.pint.to('s/m')
        print(lc.data.data.max())
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
        print(lc.data.data.max())
        lc.convert_to('us/m')
        print(lc.style)
        print(lc.data.data.max())
        print(lc.header)


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
        t.get_from_project("C:\\Users\\marte\\PycharmProjects\\blixt_rp\\excels\\project_table.xlsx", 'Resistivity')
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

    def test_units(self):
        nM = ureg.Unit('nM')
        nmol_L = ureg.Unit('nmol/L')
        m = ureg.Unit('m')
        ft = ureg.Unit('ft')
        print(is_equivalent(nM, nmol_L))  # True
        print(is_equivalent(m, ft))  # False



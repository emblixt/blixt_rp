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
data1 = np.linspace(2, 4, n) + np.random.random(n)
depth1 = np.linspace(24, 3430, n)
data2 = np.linspace(6, 8, n) + np.random.random(n)
depth2 = np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n)

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


class WellTestCase(unittest.TestCase):

    def test_data_pair(self):
        print(type(dp1))
        print(dp1.name)
        if isinstance(dp1, xr.DataArray):
            print('It is recognised as a DataArray')
        else:
            print('It is NOT recognized')

    def test_LogCurve_units(self):
        lc = LogCurve2dNew(
            dp2,
            style={'units': 'kg'}
        )

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

    def test_log_type(self ):
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
        t = Template({'name': 'MY NAME', 'well': 'MY WELL', 'unit': 'METER'})
        print(list(t.keys()))
        print(t.name, t.unit)
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



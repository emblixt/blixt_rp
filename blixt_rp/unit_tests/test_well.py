import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import pint.errors
from math import isclose
from .. import Q_


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

    def test_create_well(self):
        from blixt_rp.core.well_new import Well
        w = Well()
        self.assertIsInstance(w, Well, 'Failed')
        w.read_las(las_file1, True)
        self.assertIsInstance(w.logs, list, 'Well.logs should be a list, not {}'.format(type(w.logs)))
        for i, key in enumerate(list(w.header.well_info.keys())):
            print(key, w.header.well_info[key])
            if i > 5:
                break
        for log in w.logs:
            print(log.name, log.log_type)

        this_log = w.get_log_curve('cals')
        print('\nFetched this log: ', this_log.name)
        print(' -x-')

        w = Well()
        w.read_las(las_file1, True, log_table={'Sonic': 'dt'})
        for log in w.logs:
            print(log.name, log.log_type)

        w = Well()
        w.read_las(las_file1, True, inv_log_table={'cald': 'Caliper', 'cals': 'Caliper'})
        for log in w.logs:
            print(log.name, log.log_type)

        print(w.name)

        w.name = 'TEST'
        print(w.name)

    def test_add_log(self):
        from blixt_rp.core.well_new import Well
        w = Well()
        w.read_las(las_file1, True, log_table={'Sonic': 'dt'})
        for log in w.logs:
            print(log.name, log.log_type)
        mod_date1 = w.get_log_curve('dt').header.modification_date
        w.read_las(las_file1, True, log_table={'Sonic': 'dt', 'Caliper': 'cald'}, if_log_exists='overwrite')
        for log in w.logs:
            print(log.name, log.log_type)
        mod_date2 = w.get_log_curve('dt').header.modification_date
        w.read_las(las_file1, True, log_table={'Sonic': 'dt', 'Caliper': 'cald'}, if_log_exists='ignore')
        for log in w.logs:
            print(log.name, log.log_type)
        mod_date3 = w.get_log_curve('dt').header.modification_date
        w.read_las(las_file1, True, log_table={'Sonic': 'dt', 'Caliper': 'cald'}, if_log_exists='ask')
        for log in w.logs:
            print(log.name, log.log_type)
        print(mod_date1, mod_date2, mod_date3)

    def test_read_las(self):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable, Template
        log_table = LogTable({
            'P velocity': 'Vp_brine',
            'S velocity': 'Vs_brine',
            'Density': 'Rho_brine'
        })
        well1 = Well()
        well1.read_las(las_file2, log_table=log_table, template_file=project_table)
        print(well1.name, well1.get_log_names)
        lc = well1.get_log_curve('vs_brine')
        print(lc.style)
        self.assertIsInstance(lc.style, Template)


import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import pint.errors
from math import isclose
from .. import Q_

from blixt_rp.core.log_curve_new import LogCurve, Depth, is_equivalent, read_las
from blixt_rp.core.template_new import Template
from blixt_rp.core.header_new import Header
from blixt_rp.core.well_new import Well

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

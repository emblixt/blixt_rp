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

test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table.xlsx')

las_file1 = os.path.join(test_file_dir, "L-30.las")
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"

class HeaderTestCase(unittest.TestCase):

    def test_Header(self):
        h = Header({'name': 'MY NAME', 'well': 'MY WELL'})
        print(list(h.keys()))
        print(h.name, h.creation_date)
        time.sleep(1)
        h.orig_filename = 'A TEST'
        print(h.orig_filename, h.creation_date, h.modification_date)
        print(h)


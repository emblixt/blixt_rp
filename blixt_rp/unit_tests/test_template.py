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
import blixt_utils.io.io as uio

test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table.xlsx')

las_file1 = os.path.join(test_file_dir, "L-30.las")
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"


class TemplateTestCase(unittest.TestCase):

    def test_from_scratch(self):
        t = Template({'name': 'MY NAME', 'well': 'MY WELL', 'units': 'METER'})
        print(list(t.keys()))
        print(t.name, t.units)
        t = Template()
        print(t)

    def test_from_project_table(self):
        t = Template()
        t.get_from_project(project_table, 'Resistivity')
        t['max'] = 999
        t.colormap = 'A GIANT COLOR'
        print(t)

    def test_from_project_table_error(self):
        t = Template()
        t.get_from_project(project_table, 'XXX')
        print(t)

    def test_from_table(self):
        import pandas as pd
        templates = pd.read_excel(project_table, header=1, sheet_name='Templates', engine='openpyxl')
        t = Template()
        # self.assertRaises(KeyError, t.get_from_table, templates, 'Resistivity')
        t.get_from_table(templates, 'Resistivity')
        print(t)
        self.assertIsInstance(t, Template)

    def test_from_table_fail(self):
        templates = uio.project_templates(project_table)
        t = Template()
        self.assertRaises(KeyError, t.get_from_table, templates, 'Resistivity')
        # t.get_from_table(templates, 'Resistivity')

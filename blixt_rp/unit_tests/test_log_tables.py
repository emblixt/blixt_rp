import unittest
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from pint import Quantity as Q_

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.core.log_tables as brcl

log_table = {
    'P velocity': 'vp',
    'S velocity': 'vs',
    'Density': 'rhob',
    'Porosity': 'phie',
    'Volume': 'vcl'}


class SomeTests(unittest.TestCase):
    def test_lt_init(self):
        lt = brcl.LogTable('Test', log_table)
        print(lt.name)
        print(list(lt.keys()))
        inv = lt.invert
        print(inv)


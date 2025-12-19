import unittest
import numpy as np
import os, sys
import matplotlib.pyplot as plt

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp.rp_core_new as brrp

excel_file = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\PL1221\\SumsAndAverages.xlsx"

class SomeTests(unittest.TestCase):
    def test_init(self):
        lf = brrp.LithoFluid(name='Test', vp=4000)
        print(lf.name, lf.vp)
        self.assertTrue(True)

    def test_read_litho_fluid(self):
        lf = brrp.LithoFluid()
        lf.from_excel(excel_file, 'ROGN FM_oil_gor40')
        print(lf.name, lf.vp)
        print(list(lf.__dict__.keys()))
        lf.name = 'TEST'
        print(lf.name, lf.vs)
        self.assertTrue(True)

    def test_read_litho_fluids(self):
        lfs = brrp.LithoFluids()
        lfs.from_excel(excel_file)
        for lf in lfs.litho_fluids:
            print(lf.name)
        self.assertTrue(True)

    def test_default_litho_fluid(self):
        lf = brrp.LithoFluid(default='shale')
        print(lf.name, lf.vp)
        self.assertTrue(True)

    def test_plot_litho_fluid(self):
        lf = brrp.LithoFluid(default='shale')
        lf.plot()
        plt.show()
        self.assertTrue(True)

    def test_litho_fluid_table(self, unit_test=True):
        from bokeh.io import output_file
        from bokeh.plotting import show, row, column
        output_file('C:\\Users\\emb\\Downloads\\plot.html')
        lfs = brrp.LithoFluids()
        lfs.from_excel(excel_file)
        # table = brrp.LithoFluidsTable(lfs, advanced=True)
        table = brrp.LithoFluidsTable(lfs, advanced=False)
        source = table.source
        table, add_row, delete_row, update = table.draw(source)
        if unit_test:
            show(column(table, row(add_row, delete_row, update)))
            return None
        else:
            return table, add_row, delete_row, update


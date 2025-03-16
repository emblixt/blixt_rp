import unittest
import numpy as np
import os, sys
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, Spacer
from bokeh.models import Button, CustomJS
from bokeh.io import output_file

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp.rp_core_new as brrp
import blixt_rp.plotting.avo_chi_plotter as brap

output_file('C:\\Users\emb\Documents\plot.html')
excel_file = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\PL1221\\SumsAndAverages.xlsx"

class SomeTests(unittest.TestCase):
    def test_litho_fluid_table(self):
        litho_fluids_list = []
        for _name in ['shale', 'brine_sst', 'oil_sst']:
            rt = brrp.LithoFluid(default=_name)
            litho_fluids_list.append(rt)
        # print(rock_types_list)
        litho_table = brap.create_litho_fluids_table(litho_fluids_list, 800)
        print(litho_table.source.data['name'])
        print(litho_table.source.data['vp'][0], type(litho_table.source.data['vp'][0]))
        show(litho_table)
        self.assertTrue(True)

    def test_get_litho_fluid(self):
        litho_fluids_list = []
        for _name in ['shale', 'brine_sst', 'oil_sst']:
            rt = brrp.LithoFluid(default=_name)
            litho_fluids_list.append(rt)
        litho_table = brap.create_litho_fluids_table(litho_fluids_list, 800)
        result = brap.get_litho_fluid_from_table('shale', litho_table)
        print(result)
        self.assertIsInstance(result, dict)

    def test_interface_table(self):
        litho_fluids_list = []
        for _name in ['shale', 'brine_sst', 'oil_sst']:
            rt = brrp.LithoFluid(default=_name)
            litho_fluids_list.append(rt)
        lf_table = brap.create_litho_fluids_table(litho_fluids_list, 800)
        data_table = brap.create_interface_table(lf_table, 600)
        show(data_table)
        self.assertTrue(True)

    def test_active_interfaces_with_button(self):
        litho_fluids_list = []
        code = """
        var data = source.data
        console.log('Vp: ' + data['vp'][0])
        """
        for _name in ['shale', 'brine_sst', 'oil_sst']:
            rt = brrp.LithoFluid(default=_name)
            litho_fluids_list.append(rt)
        lf_table = brap.create_litho_fluids_table(litho_fluids_list, 800)
        interface_table = brap.create_interface_table(lf_table, 600)
        button = Button(label="Run", button_type="success")
        call_back = CustomJS(
            args=dict(source=lf_table.source),
            code=code)
        button.js_on_click(call_back)
        print(brap.get_active_interfaces(interface_table))
        show(column(lf_table, interface_table, button))
        self.assertTrue(True)

    def test_active_litho_fluids(self):
        litho_fluids_list = []
        for _name in ['shale', 'brine_sst', 'oil_sst']:
            rt = brrp.LithoFluid(default=_name)
            litho_fluids_list.append(rt)
        lf_table = brap.create_litho_fluids_table(litho_fluids_list, 800)
        interface_table = brap.create_interface_table(lf_table, 600)
        active_interface_name =brap.get_active_interfaces(interface_table)[0]
        print(brap.get_interface_litho_fluids(active_interface_name, interface_table, lf_table))
        self.assertTrue(True)

    def test_build_button_python_callback(self):
        # This script is called  by the main.py script under blixt_projects/bokeh_testing
        # And can be invoked by calling:
        # C:\Users\emb\Documents\PycharmProjects\blixt_projects>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show bokeh_testing
        import blixt_rp.rp.rp_core_new as brrp
        import blixt_rp.plotting.avo_chi_plotter as brap
        litho_fluids_list = []
        for _name in ['shale', 'brine_sst', 'oil_sst']:
            rt = brrp.LithoFluid(default=_name)
            litho_fluids_list.append(rt)
        lf_table = brap.create_litho_fluids_table(litho_fluids_list, 800)
        interface_table = brap.create_interface_table(lf_table, 600)

        def callback():
            _active_interfaces = brap.get_active_interfaces(interface_table)
            _litho_fluids = {}
            for _interface in _active_interfaces:
                _litho_fluids[_interface] = brap.get_interface_litho_fluids(_interface, interface_table, lf_table)

        button = Button(label="Run", button_type="success")
        button.on_click(callback)
        return interface_table, lf_table, button

    def test_create_mc_avo_plot(self):
        h, p, it, lt, b, c, n = brap.create_mc_avo_plot()
        show(column([h, p, row(it, c), lt]))  #, sizing_mode='stretch_height'))
        self.assertTrue(True)

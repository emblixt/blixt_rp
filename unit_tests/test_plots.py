import unittest
import matplotlib.pyplot as plt
import numpy as np
import os

import blixt_utils.io.io as uio
import plotting.plot_logs as ppl
from blixt_utils.plotting.helpers import header_plot, axis_plot
from core.well import Project
from core.well import Well


class PlotTestCase(unittest.TestCase):
    wp = Project()
    well_table = {os.path.join(wp.working_dir, 'test_data/Well D.las'):
                      {'Given well name': 'WELL_D',
                       'logs': {
                           'ac': 'Sonic',
                           'acs': 'Shear sonic',
                           'cali': 'Caliper',
                           'den': 'Density',
                           'neu': 'Neutron density',
                           'gr': 'Gamma ray',
                           'rdep': 'Resistivity',
                           'rmed': 'Resistivity',
                           'rsha': 'Resistivity',
                           'neu': 'Neutron density'},
                       'Note': 'Some notes for well A'}}
    wis = {'WELL_D': {
        'SAND C': [1585.0, 1826.0],
        'SHALE C': [1585.0, 1826.0],
        'SAND D': [1826.0, 1878.0],
        'SAND E': [1878.0, 1984.0],
        'SAND F': [1984.0, 2158.0],
        'SHALE G': [2158.0, 2211.0],
        'SAND H': [2211.0, 2365.0]
    }}

    w = Well()
    w.read_well_table(
        well_table,
        0,
        block_name='Logs')

    def test_axis_header(self):
        templ = uio.project_templates(PlotTestCase.wp.project_table)
        fig, ax = plt.subplots()
        log_types = ['P velocity', 'Density', 'Caliper', 'Resistivity']
        limits = [[templ[x]['min'], templ[x]['max']] for x in log_types]
        legends = ['test [{}]'.format(templ[x]['unit']) for x in log_types]
        styles = [{'lw': templ[x]['line width'],
                   'color': templ[x]['line color'],
                   'ls': templ[x]['line style']} for x in log_types]
        header_plot(ax, limits, legends, styles)
        plt.show()
        with self.subTest():
            self.assertTrue(True)

    def test_axis_plot(self):
        templ = uio.project_templates(PlotTestCase.wp.project_table)
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        log_types = ['Gamma ray', 'Caliper']
        limits = [[templ[x]['min'], templ[x]['max']] for x in log_types]
        data = [PlotTestCase.w.get_logs_of_type(x)[0].data for x in log_types]
        y = PlotTestCase.w.block['Logs'].logs['depth'].data
        styles = [{'lw': templ[x]['line width'],
                   'color': templ[x]['line color'],
                   'ls': templ[x]['line style']} for x in log_types]
        axis_plot(ax, y, data, limits, styles, nxt=2)

        plt.show()


    def test_plot_logs(self):
        from blixt_utils.io.io import invert_well_table
        w = PlotTestCase.w
        templates = uio.project_templates(PlotTestCase.wp.project_table)
        ppl.plot_logs(w, invert_well_table(PlotTestCase.well_table, 'WELL_D', rename=False),
                      PlotTestCase.wis, "SAND E", templates, buffer=50. )
                      #savefig='C:/users/mblixt/PycharmProjects/blixt_rp/results_folder/test.png')
                      #savefig='C:/Users/marten/PycharmProjects/blixt_rp/results_folder/test.png')
        with self.subTest():
            self.assertTrue(True)


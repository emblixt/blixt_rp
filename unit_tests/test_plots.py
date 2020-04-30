import unittest
import matplotlib.pyplot as plt

import utils.utils as uu
import utils.io as uio
from core.well import Project


import numpy as np


class PlotTestCase(unittest.TestCase):
    wp = Project()
    templates = uio.project_templates(wp.project_table)

    print(templates.keys())
    def test_axis_header(self):
        templ = PlotTestCase.templates
        fig, ax = plt.subplots()
        log_types = ['P velocity', 'Density', 'Caliper', 'Resistivity']
        lines = [[templ[x]['min'], templ[x]['max']] for x in log_types]
        legends = ['test [{}]'.format(templ[x]['unit']) for x in log_types]
        styles = [{'lw': templ[x]['line width'],
                   'color': templ[x]['line color'],
                   'ls': templ[x]['line style']} for x in log_types]
        uu.axis_header(ax, lines, legends, styles)
        plt.show()
        with self.subTest():
            self.assertTrue(True)



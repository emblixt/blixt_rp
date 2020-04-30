import unittest
import matplotlib.pyplot as plt
import utils.utils as uu


import numpy as np


class PlotTestCase(unittest.TestCase):

    def test_axis_header(self):
        fig, ax = plt.subplots()
        lines = [[10, 20], [100, 200], [50, 75], [1000., 1.E6]]
        legends = ['test1', 'test2', 'test3', 'test4']
        styles = [{'lw': 1, 'color': 'k', 'ls': '-'},
                  {'lw': 2, 'color': 'r', 'ls': '-'},
                  {'lw': 0.5, 'color': 'k', 'ls': '--'},
                  {'lw': 3, 'color': 'b', 'ls': 'dotted'}]
        uu.axis_header(ax, lines, legends, styles)
        plt.show()
        with self.subTest():
            self.assertTrue(True)



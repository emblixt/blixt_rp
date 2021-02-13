import unittest
import os
import blixt_utils.misc.io as uio
from blixt_utils.misc.attribdict import AttribDict
from core.well import Well
from core.well import Project
from blixt_utils.misc.io import convert


import numpy as np

def_lb_name = 'Logs'  # default Block name
def_msk_name = 'Mask'  # default mask name


class ProjectTestCase(unittest.TestCase):

    def test_create_project(self):
        wp = Project(name='MyProject', log_to_stdout=True)
        with self.subTest():
            print(type(wp))
            self.assertTrue(isinstance(wp, Project))

    def test_load_wells(self):
        wp = Project(name='MyProject', log_to_stdout=True)
        wells = wp.load_all_wells()
        for _, well in wells.items():
            with self.subTest():
                self.assertTrue(well, Well)

    def test_load_selected_wells(self):
        """
        For this test to work, the listed wells below need to have "use = Yes" in the project table
        """
        these_wells = ['WELL_A', 'WELL_B']
        wp = Project(name='MyProject', log_to_stdout=True)
        wells = wp.load_all_wells(include_these_wells=these_wells)
        for wname, _ in wells.items():
            with self.subTest():
                self.assertTrue(wname in these_wells)
            with self.subTest():
                self.assertFalse(wname == 'WELL_C')

    def test_load_selected_intervals(self):
        these_intervals = ['Sand E', 'Sand F']
        #these_intervals = ['Sand F']
        wp = Project(name='MyProject', log_to_stdout=True)
        wells = wp.load_all_wells(include_these_intervals=these_intervals)
        for _, well in wells.items():
            md = well.block[def_lb_name].get_md()
            print(well.well, md.min(), md.max())
        with self.subTest():
            self.assertTrue(True)

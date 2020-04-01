import unittest
import utils.io as uio
import os
from core.well import Project
from core.well import Well
import numpy as np

def_lb_name = 'Logs'  # default Block name
def_msk_name = 'Mask'  # default mask name

# TODO
# It might be smarter to first set up a set of test cases in a TestCase class
# and then run the tests in a Test class, as demonstrated in
# https://stackoverflow.com/questions/9829331/how-do-i-handle-multiple-asserts-within-a-single-python-unittest

def create_test_data(var_name):
    w = Well()
    w.read_las(
        MasksTestCase.las_file,
        block_name=def_lb_name,
        only_these_logs=MasksTestCase.my_logs)

    # extract the phie log, and apply a mask on it
    return w, w.block[def_lb_name].logs[var_name].data


class MasksTestCase(unittest.TestCase):
    wp = Project(name='MyProject', log_to_stdout=True)
    well_table = uio.project_wells(wp.project_table, wp.working_dir)
    #las_file = list(well_table.keys())[0]
    las_file = os.path.join(wp.working_dir, 'test_data', 'Well A.las')
    #my_logs = well_table[las_file]['logs']
    my_logs = None

    def test_calc_mask(self):
        lmt = 0.1
        w, phie = create_test_data('phie')
        masked_length = len(phie[phie < lmt])

        # Create the same mask using the create mask function
        w.calc_mask({'phie': ['<', lmt]}, name=def_msk_name)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test length
        with self.subTest():
            print(masked_length, len(phie[msk]))
            self.assertEqual(masked_length, len(phie[msk]))
        # Test value
        with self.subTest():
            print(np.nanmax(phie[msk]), lmt)
            self.assertLess(np.nanmax(phie[msk]), lmt)


    def test_append_mask(self):
        w, phie = create_test_data('phie')

        lmt1 = 0.05; lmt2 = 0.1
        masked_length = len(phie[(phie > lmt1) & (phie < lmt2)])

        # create first mask
        w.calc_mask({'phie': ['>', lmt1]}, name=def_msk_name)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test value
        with self.subTest():
            print(np.nanmin(phie[msk]), lmt1)
            self.assertGreater(np.nanmin(phie[msk]), lmt1)

        # append second mask
        w.calc_mask({'phie': ['<', lmt2]}, name=def_msk_name, append=True)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test value
        with self.subTest():
            print(np.nanmax(phie[msk]), lmt2)
            self.assertLess(np.nanmin(phie[msk]), lmt2)
        # Test length
        with self.subTest():
            print(masked_length, len(phie[msk]))
            print(w.block[def_lb_name].masks[def_msk_name].header.desc)
            self.assertEqual(masked_length, len(phie[msk]))






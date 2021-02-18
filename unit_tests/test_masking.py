import unittest
import blixt_utils.io.io as uio
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
    w.read_well_table(
        MasksTestCase.well_table,
        0,
        block_name=def_lb_name)

    # extract the phie log, and apply a mask on it
    return w, w.block[def_lb_name].logs[var_name].data


class MasksTestCase(unittest.TestCase):
    wp = Project(name='MyProject', log_to_stdout=True)
    # Instead of creating the well table directly from the project table,
    # we can assure the well table to contain the  desired well by
    # writing it explicitly
    well_table = {os.path.join(wp.working_dir, 'test_data/Well A.las'):
                      {'Given well name': 'WELL_A',
                       'logs': {
                           'vp_dry': 'P velocity',
                           'vp_sg08': 'P velocity',
                           'vp_so08': 'P velocity',
                           'vs_dry': 'S velocity',
                           'vs_sg08': 'S velocity',
                           'vs_so08': 'S velocity',
                           'rho_dry': 'Density',
                           'rho_sg08': 'Density',
                           'rho_so08': 'Density',
                           'phie': 'Porosity',
                           'vcl': 'Volume'},
                       'Note': 'Some notes for well A'}}

    # las_file = os.path.join(wp.working_dir, 'test_data', 'Well A.las')
    # my_logs = None
    tops = {
        'WELL_A': {'TOP A': 408.0,
                   'TOP B': 1560.0,
                   'TOP C': 1585.0,
                   'TOP D': 1826.0,
                   'TOP E': 1878.0,
                   'TOP F': 1984.0,
                   'BASE F': 2158.0,
                   'TOP G': 2158.0,
                   'TOP H': 2211.0,
                   'BASE H': 2365.0,
                   'TOP I': 2365.0,
                   'TOP J': 2452.0}
    }
    wis = {'WELL_A': {
        'SAND C': [1585.0, 1826.0],
        'SHALE C': [1585.0, 1826.0],
        'SAND D': [1826.0, 1878.0],
        'SAND E': [1878.0, 1984.0],
        'SAND F': [1984.0, 2158.0],
        'SHALE G': [2158.0, 2211.0],
        'SAND H': [2211.0, 2365.0]
    }}

    def test_calc_mask(self):
        lmt = 0.1
        w, phie = create_test_data('phie')
        masked_length = len(phie[phie < lmt])

        #
        # Create the same mask using the create mask function
        #
        cutoff = {'phie': ['<', lmt]}
        w.calc_mask(cutoff, name=def_msk_name, log_type_input=False)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test length
        with self.subTest():
            print(masked_length, len(phie[msk]))
            self.assertEqual(masked_length, len(phie[msk]))
        # Test value
        with self.subTest():
            print(np.nanmax(phie[msk]), lmt)
            self.assertLess(np.nanmax(phie[msk]), lmt)
        del (w.block[def_lb_name].masks[def_msk_name])

        #
        # Create the same mask using the create mask function with log type
        #
        print('Testing log type input: Porosity')
        w.calc_mask({'Porosity': ['<', lmt]}, name=def_msk_name, log_type_input=True)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test length
        with self.subTest():
            print(masked_length, len(phie[msk]))
            self.assertEqual(masked_length, len(phie[msk]))
        # Test value
        with self.subTest():
            print(np.nanmax(phie[msk]), lmt)
            self.assertLess(np.nanmax(phie[msk]), lmt)
        del (w.block[def_lb_name].masks[def_msk_name])

        #
        # Create mask applying tops
        #
        print('Test masking with tops input')
        w.calc_mask({'Porosity': ['<', lmt]}, name=def_msk_name,
                    tops=MasksTestCase.tops, use_tops=['TOP C', 'BASE F'], log_type_input=True)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test value
        with self.subTest():
            print(np.nanmax(phie[msk]), lmt)
            self.assertLess(np.nanmax(phie[msk]), lmt)
        del (w.block[def_lb_name].masks[def_msk_name])

        #
        # Create mask from empty cutoffs, allow all data
        #
        print('Test mask with empty cutoffs')
        w.calc_mask({})
        msk = w.block[def_lb_name].masks[def_msk_name].data
        with self.subTest():
            print(len(phie), len(phie[msk]))
            self.assertEqual(len(phie), len(phie[msk]))

        del (w.block[def_lb_name].masks[def_msk_name])

        #
        # Ask for a mask for a log that does not exists
        #
        print('Test mask without any valid logs')
        any_error = False
        try:
            w.calc_mask({'Saturation': ['<', lmt]}, name=def_msk_name, log_type_input=True)
        except:
            any_error = True
        # Test if error was raised
        with self.subTest():
            self.assertFalse(any_error)

        #
        # Create mask applying working intervals
        #
        print('Test masking with working intervals input')
        w.calc_mask({'Porosity': ['<', lmt]}, name=def_msk_name,
                    wis=MasksTestCase.wis, wi_name='SAND E', log_type_input=True)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test value
        with self.subTest():
            print(np.nanmax(phie[msk]), lmt)
            self.assertLess(np.nanmax(phie[msk]), lmt)
        del (w.block[def_lb_name].masks[def_msk_name])

        #
        # Create mask using working intervals, but no cutoffs
        #
        print('Test masking with working intervals, but no cutoff')
        w.calc_mask({}, name=def_msk_name,
                    wis=MasksTestCase.wis, wi_name='SAND E')
        msk = w.block[def_lb_name].masks[def_msk_name].data
        print('SAND E MD limits in WELL_A:', MasksTestCase.wis['WELL_A']['SAND E'])
        depths = w.block[def_lb_name].get_md()[msk]
        print('Masked MD min and max:', depths.min(), depths.max())

        # w.calc_mask({'sw': ['<', lmt]}, name=def_msk_name, log_type_input=False)
        # msk = w.block[def_lb_name].masks[def_msk_name].data
        ## Test length
        # with self.subTest():
        #    print(masked_length, len(phie[msk]))
        #    self.assertEqual(masked_length, len(phie[msk]))

    def test_apply_mask(self):
        lmt = 0.1
        w, phie = create_test_data('phie')
        masked_length = len(phie[phie < lmt])

        cutoff = {'phie': ['<', lmt]}
        w.calc_mask(cutoff, name=def_msk_name, log_type_input=False)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        w.apply_mask(def_msk_name)
        m_phie = w.block[def_lb_name].logs['phie'].data

        # Test length
        with self.subTest():
            print(masked_length, len(m_phie))
            self.assertEqual(masked_length, len(m_phie))
        # Test value
        with self.subTest():
            print(np.nanmax(m_phie), lmt)
            self.assertLess(np.nanmax(m_phie), lmt)

        w.calc_mask({}, name=def_msk_name,
                    wis=MasksTestCase.wis, wi_name='Sand F')
        msk = w.block[def_lb_name].masks[def_msk_name].data
        w.apply_mask(def_msk_name)
        print('Sand F MD limits in WELL_A:', MasksTestCase.wis['WELL_A']['SAND F'])
        depths = w.block[def_lb_name].get_md()
        print('Masked MD min and max:', depths.min(), depths.max())

    def test_append_mask(self):
        w, phie = create_test_data('phie')

        lmt1 = 0.05;
        lmt2 = 0.1
        masked_length = len(phie[(phie > lmt1) & (phie < lmt2)])

        # create first mask
        cutoff_1 = {'phie': ['>', lmt1]}
        w.calc_mask(cutoff_1, name=def_msk_name, log_type_input=False)
        msk = w.block[def_lb_name].masks[def_msk_name].data
        # Test value
        with self.subTest():
            print(np.nanmin(phie[msk]), lmt1)
            self.assertGreater(np.nanmin(phie[msk]), lmt1)

        # append second mask
        cutoff_2 = {'phie': ['<', lmt2]}
        w.calc_mask(cutoff_2, name=def_msk_name, append='AND', log_type_input=False)
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

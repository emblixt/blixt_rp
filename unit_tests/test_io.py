import unittest
import os
import utils.io as uio
from utils.attribdict import AttribDict
from core.well import Well
from core.well import Project
from utils.io import convert


import numpy as np

def_lb_name = 'LogBlock'  # default LogBlock name
def_msk_name = 'Mask'  # default mask name


def create_test_data(var_name):
    w = Well()
    w.read_las(
        LasTestCase.las_file,
        block_name=def_lb_name,
        only_these_logs=LasTestCase.my_logs,
        note='Test note')

    # extract the phie log, and apply a mask on it
    return w, w.log_blocks[def_lb_name].logs[var_name].data


def read_las(lfile):
    with open(lfile, 'r') as f:
        lines = f.readlines()
    null_value, gen_keys, well_dict = convert(lines)
    return null_value, gen_keys, well_dict


class LasTestCase(unittest.TestCase):
    wp = Project(name='MyProject', log_to_stdout=True)
    well_table = uio.project_wells(wp.project_table, wp.working_dir)
    #las_file = list(well_table.keys())[0]
    las_file = os.path.join(wp.working_dir, 'test_data/Well A.las')
    #my_logs = well_table[las_file]['logs']
    my_logs = None

    def test_read_las(self):
        null_value, gen_keys, well_dict = read_las(LasTestCase.las_file)
        with self.subTest():
            print(len(gen_keys), len(list(well_dict['data'].keys())))
            self.assertEqual(len(gen_keys), len(list(well_dict['data'].keys())))

    def test_headers(self):
        """
        All header keys in the well header, and LogBlock header should be of the AttribDict type
        :return:
        """
        w, data = create_test_data('phie')
        success = True
        for key in list(w.header.keys()):
            if not isinstance(w.header[key], AttribDict):
                success = False
        for lblock in list(w.log_blocks.keys()):
            for key in list(w.log_blocks[lblock].header.keys()):
                if not isinstance(w.log_blocks[lblock].header[key], AttribDict):
                    success = False
        self.assertTrue(success)

    def test_null_value(self):
        """
        When reading in a las file, any 'null values' should be replaced with np.nan
        :return:
        """
        null_value, gen_keys, well_dict = read_las(LasTestCase.las_file)
        var_name = gen_keys[1]
        data = np.array(well_dict['data'][var_name])
        # data should not contain explicit 'null_value's, so below null_data should have length zero
        null_data = data[data == float(null_value)]
        with self.subTest():
            print(var_name, null_value, len(null_data))
            self.assertEqual(0, len(null_data))
        with self.subTest():
            # these test data should contain NaN's, so t below should contain some True elements
            t = np.isnan(data)
            print('Data contains NaN:', any(t))
            self.assertTrue(any(t))




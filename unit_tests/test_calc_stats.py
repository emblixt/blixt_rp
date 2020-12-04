import unittest
from core.well import Project
import utils.calc_stats as uc

def_lb_name = 'Logs'  # default Block name
def_msk_name = 'Mask'  # default mask name


class StatsTestCase(unittest.TestCase):

    def test_create_containers(self):
        log_table = {'Saturation': 'sw', 'S velocity': 'vs'}
        results, results_per_well, depth_from_top = uc.create_containers(list(log_table.values()))
        with self.subTest():
            print("Is output of right type?")
            self.assertTrue(isinstance(results, dict))
            self.assertTrue(isinstance(results_per_well, dict))
            self.assertTrue(isinstance(depth_from_top, dict))
        with self.subTest():
            print("Is size of output correct?")
            self.assertEqual(len(log_table), len(results))

    def test_collect_data(self):
        wp = Project(name='MyProject', log_to_stdout=True)
        wells = wp.load_all_wells()
        wis = wp.load_all_wis()
        log_table = {'Porosity': 'phie', 'S velocity': 'vs'}
        results, results_per_well, depth_from_top = uc.create_containers(list(log_table.values()))
        interval_names = ['Sand C', 'Shale C', 'Sand D', 'Sand E', 'Sand F', 'Shale G', 'Sand H']
        for wi_name in interval_names:
            uc.collect_data_for_this_interval(wells, list(log_table.values()), wis, wi_name, results, results_per_well,
                                              depth_from_top, {}, log_table)
            with self.subTest():
                self.assertTrue(True)

import unittest
import os
import sys
import matplotlib.pyplot as plt


import pint
from .. import ureg, Q_

# Add to path to avoid having to install libraries, useful in development
project_dir = os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests','')
sys.path.append(os.path.join(str(project_dir), 'blixt_utils'))

from blixt_utils.utils import fix_well_name

class TestCase(unittest.TestCase):

    def test_create_well(self):
        from blixt_rp.core.intervals import SingleInterval, IntervalInfo, Well, Intervals
        well = Well('TestWell', {})
        interval1 = SingleInterval('Test FM', 'WrongWellName', 'Test FM', Q_(100., 'm'), Q_(200., 'm'))
        interval2 = SingleInterval('Test FM', 'TestWell', 'Test FM', Q_(100., 'm'), Q_(200., 'm'))
        well.add_interval(interval1)
        well.add_interval(interval2)
        well.add_interval(interval2)
        interval_info = IntervalInfo('Test FM', level=-1, color='green')
        wis = Intervals()
        wis.add_interval(interval_info)
        wis.add_well(well)

        print(wis)
        print('-x-')
        print(well)
        print('-x-')
        print(interval2)

        self.assertIsInstance(wis, Intervals)

    def test_read(self):
        from blixt_rp.core.intervals import SingleInterval, IntervalInfo, Well, Intervals
        f = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Sodir_well_tops_NorwegianSea.xlsx"
        wis = Intervals(name='Working intervals')
        wis.read_sodir_tops(f)
        print(wis)

        for well in wis.well_names():
            print(wis.get_well(well))

        fig, ax = plt.subplots()
        wis.get_well(well).plot_intervals(2, ax)
        plt.show()
        self.assertIsInstance(wis, Intervals)



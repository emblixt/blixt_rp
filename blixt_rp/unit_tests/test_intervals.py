import unittest
import os
import sys
import matplotlib.pyplot as plt


import pint
from .. import ureg, Q_

# Add to path to avoid having to install libraries, useful in development
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.utils import fix_well_name


class TestCase(unittest.TestCase):

    def test_create_well(self):
        from blixt_rp.core.intervals import SingleInterval, IntervalInfo, Well, Intervals
        well = Well('TestWell', {})
        wis = Intervals(name='TEST')
        wis.add_well(well)
        interval1 = SingleInterval('Test FM', 1, Q_(100., 'm'), Q_(200., 'm'), well='TestWell')
        interval2 = SingleInterval('Test FM', 1, Q_(100., 'm'), Q_(200., 'm'), well='WrongWell')
        interval3 = SingleInterval('Test FM', 1, Q_(100., 'm'), Q_(200., 'm'), well='TestWell')
        wis.add_single_interval(interval1)
        wis.add_single_interval(interval2)
        wis.add_single_interval(interval3)

        print(wis)
        print('-x-')
        print(wis.well_names())
        print('-x-')
        print(wis.interval_names())
        print('-x-')
        for i in well.intervals.values():
            print(i)

        self.assertIsInstance(wis, Intervals)

    def test_read(self):
        from blixt_rp.core.intervals import SingleInterval, IntervalInfo, Well, Intervals
        f = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Sodir_well_tops_BarentsSea.xlsx"
        wis = Intervals(name='Working intervals')
        wis.read_sodir_tops(f)

        # for well in wis.well_names():
        #     print('Well: {}'.format(well))
        #     print(' levels: {}'.format(wis.get_well(well).get_levels))
        #     print(' depths: {}'.format(wis.get_well(well).get_depth_range))
        #     for interval_uid in wis.get_well(well).interval_uids():
        #         this_interval = wis.get_well(well).get_interval(interval_uid)
        #         print('  Interval: {}: {} - {}'.format(this_interval.name, this_interval.top.magnitude,
        #                                                this_interval.base.magnitude))
        # print(wis.interval_names())

        # fig, axs = plt.subplots(ncols=2)
        # wis.get_well('6201_11_2').plot_intervals(3, axs[0])
        # wis.get_well('6201_11_2').plot_intervals(2, axs[1], depth_range=(3000., 3800.))
        # plt.show()

        out_file = os.path.join(project_dir, 'test.xlsx')
        test = wis.write_to_excel(out_file, 'Working intervals', 'Interval info',
                                  append=False)

        print(test)
        self.assertIsInstance(wis, Intervals)

    def test_append(self):
        from blixt_rp.core.intervals import SingleInterval, IntervalInfo, Well, Intervals
        append_file = os.path.join(project_dir, 'test.xlsx')
        well = Well('TestWell', {})
        wis = Intervals(name='TEST')
        wis.add_well(well)
        interval1 = SingleInterval('Test FM', 1, Q_(100., 'm'), Q_(200., 'm'), well='TestWell')
        interval2 = SingleInterval('Test FM', 1, Q_(100., 'm'), Q_(200., 'm'), well='WrongWell')
        interval3 = SingleInterval('Test FM', 1, Q_(100., 'm'), Q_(200., 'm'), well='TestWell')
        wis.add_single_interval(interval1)
        wis.add_single_interval(interval2)
        wis.add_single_interval(interval3)

        wis.read_blixt_tops(append_file)

        print(wis)
        self.assertIsInstance(wis, Intervals)



import unittest
import os
import sys
import matplotlib.pyplot as plt


import pint
from .. import ureg, Q_

# Add to path to avoid having to install libraries, useful in development
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.intervals import Interval, StratUnit, Intervals

fm1 = StratUnit('Test1 FM', 1)
fm2 = StratUnit('Test2 FM', 1)
gp1 = StratUnit('Test1 GP', 2)
interval1 = Interval('TestWell', Q_(100., 'm'), Q_(200., 'm'), fm1)
interval2 = Interval('TestWell', Q_(200., 'm'), Q_(300., 'm'), fm2)
interval3 = Interval('TestWell', Q_(100., 'm'), Q_(300., 'm'), gp1)


class TestCase(unittest.TestCase):

    def test_create_intervals(self):
        wis = Intervals('TEST', [interval1, interval2, interval3])

        print(wis)
        print('-x-')
        print(wis.well_names())
        print('-x-')
        print(wis.interval_names())
        print('-x-')
        print(wis.get_well_depth_range('TestWell'))
        print('-x-')
        _i = wis.get_interval('Test1 FM', 'TestWell')
        print(_i.thickness)
        print('-x-')
        print(_i.mid)
        print('-x-')
        print(_i.distance_to_top(Q_(0, 'm')))
        print('-x-')

        self.assertIsInstance(wis, Intervals)

    def test_read(self):
        f = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Sodir_well_tops_BarentsSea.xlsx"
        wis = Intervals(name='Working intervals')
        wis.read_sodir_tops(f)

        for well in wis.well_names():
            print('Well: {}'.format(well))
            print(' depths: {}'.format(wis.get_well_depth_range(well)))
            print(' Number of intervals: {}'.format(len(wis.get_well_intervals(well))))
            for _interval in wis.get_well_intervals(well):
                print('  Interval: {}: {} - {}'.format(_interval.name, _interval.top.magnitude,
                                                       _interval.base.magnitude))
        print(wis.interval_names())

        # fig, axs = plt.subplots(ncols=2)
        # wis.get_well('7119_7_1').plot_intervals(3, axs[0])
        # wis.get_well('7119_7_1').plot_intervals(2, axs[1], depth_range=(3000., 3800.))
        # plt.show()

        # out_file = os.path.join(project_dir, 'test.xlsx')
        # test = wis.write_to_excel(out_file, 'Working intervals', 'Interval info',
        #                           append=False)
        # print(test)

        self.assertIsInstance(wis, Intervals)

    def test_read_project_intervals(self):
        project_file = os.path.join(project_dir,"blixt_rp\\excels\\project_table_new.xlsx")
        wis = Intervals()
        wis.read_blixt_tops(project_file)
        print(wis)
        for _i in wis.intervals:
            print(_i)

    def test_append(self):
        append_file = os.path.join(project_dir, 'test.xlsx')
        wis = Intervals(name='TEST', intervals=[interval1, interval2])
        wis.add_interval(interval3)

        #wis.read_blixt_tops(append_file)

        print(wis)
        self.assertIsInstance(wis, Intervals)

    def test_plot(self):
        wis = Intervals(name='TEST', intervals=[interval1, interval2, interval3])

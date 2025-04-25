import unittest
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from pint import Quantity as Q_

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
# sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.core import (Interval, StratUnit, Intervals, Template, Header, CutoffRule, Cutoffs, LogTable,
                                templates_from_table)



test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table.xlsx')

las_file1 = os.path.join(test_file_dir, "L-30.las")
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"

rule1 = CutoffRule('name1', '>', Q_(100, 'm'))
rule2 = CutoffRule('name2', '<', Q_(10, 'm'))
rule3 = CutoffRule('name3', '==', Q_(1000, 'm'))
rule4 = CutoffRule('name4', None, 'my_interval')
rule5 = CutoffRule('name5', '><', [Q_(10, 'm'), Q_(1000, 'm')])

log_table = {
    'P velocity': 'vp',
    'S velocity': 'vs',
    'Density': 'rhob',
    'Porosity': 'phie',
    'Volume': 'vcl'}

log_table_multi = {
    'P velocity': ['vp_virg', 'vp_brine', 'vp_oil', 'vp_gas'],
    'S velocity': ['vs_virg', 'vs_brine', 'vs_oil', 'vs_gas'],
    'Density': ['rhob_virg', 'rhob_brine', 'rhob_oil', 'rhob_gas']}

fm1 = StratUnit('Test1 FM', 1)
fm2 = StratUnit('Test2 FM', 1)
gp1 = StratUnit('Test1 GP', 2)
interval1 = Interval('TestWell', Q_(100., 'm'), Q_(200., 'm'), fm1)
interval2 = Interval('TestWell', Q_(200., 'm'), Q_(300., 'm'), fm2)
interval3 = Interval('TestWell', Q_(100., 'm'), Q_(300., 'm'), gp1)

class CutoffTests(unittest.TestCase):
    def test_rule_init(self):
        limits = [Q_(4,'m'), [Q_(4,'m'), Q_(10,'m')], 'interval_name']
        for limit in limits:
            print(limit)
            rule = CutoffRule('test', '>', limit)
            self.assertIsInstance(rule, CutoffRule)
        rule = CutoffRule('', None, 'test_interval')
        self.assertIsInstance(rule, CutoffRule)

    def test_failed_rule_init(self):
        limits = [4.0, [4.0, 4.0], 11]
        for limit in limits:
            print(limit)
            self.assertRaises(IOError, CutoffRule, param='test', operator='>', limit=limit)
        self.assertRaises(IOError, CutoffRule, param='', operator=None, limit=4)
        self.assertRaises(IOError, CutoffRule, param='', operator='XXX', limit=4)

    def test_print_rules(self):
        for rule in [rule1, rule2, rule3, rule4, rule5]:
            print(rule)

    def test_failed_init(self):
        cutoffs = Cutoffs
        self.assertRaises(IOError, cutoffs, name='test', alpha='JA', beta='Nej' )

    def test_init(self):
        cutoffs = Cutoffs(cutoffs=[rule1, rule2])
        print(len(cutoffs))
        cutoffs.append([rule3, rule4])
        cutoffs.append(rule1)
        cutoffs.append(rule5)
        print(len(cutoffs))
        print(cutoffs.cutoff_names)
        print(cutoffs)
        print(cutoffs.get_dict())
        self.assertTrue(True)


class LogTableTests(unittest.TestCase):
    def test_lt_init(self):
        lt = LogTable( log_table)
        print(list(lt.keys()))
        lt.test = 'A log name'  # this has no function now
        lt['TEST'] = 'A log name' # But this work when __setitem__ is kept untouched
        print(lt.keys())
        inv = lt.invert
        print(lt.log_names)
        print(lt.log_types)
        print(inv)
        print(lt.dict)


    def test_multi(self):
        lt = LogTable('Multiple logs', log_table_multi)
        print(lt.name)
        print(list(lt.keys()))
        lt['Density'].append('my other rhob')
        print(lt.log_names)
        print(lt.log_types)
        print(lt.invert)
        print(lt.dict)

    def test_mixing(self):
        lt1 = LogTable('Test', log_table)
        lt2 = LogTable('Multiple logs', log_table_multi)
        # lt1['Should fail'] = [1,2,3]
        lt2['Should fail'] = 'Vp'


class TemplateTestCase(unittest.TestCase):

    def test_from_scratch(self):
        t = Template(**{'name': 'MY NAME', 'well': 'MY WELL', 'units': 'METER'})
        print(list(t.keys()))
        print(t.name, t.units, t.well)
        t = Template()
        print(t)

    def test_from_project_table(self):
        t = Template()
        t.get_from_project(project_table, 'Resistivity')
        t.max = 999
        t.colormap = 'A GIANT COLOR'
        print(t)
        d = t.__dict__
        print(type(t), type(d))

    def test_dictionary_from_project_table(self):
        table = pd.read_excel(project_table, header=1, sheet_name='Templates', engine='openpyxl')
        template_dict = templates_from_table(table)
        for i, _key in enumerate(list(template_dict.keys())):
            print(_key)
            print(template_dict[_key])
            if i > 0:
                break
    def test_from_project_table_error(self):
        t = Template()
        t.get_from_project(project_table, 'XXX')
        print(t)

class IntervalTests(unittest.TestCase):

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
        # print(wis)
        # for _i in wis.intervals:
        #     print(_i)
        print(wis.get_strat_units())
        print(wis.get_intervals_dict('Well_F'))

    def test_append(self):
        append_file = os.path.join(project_dir, 'test.xlsx')
        wis = Intervals(name='TEST', intervals=[interval1, interval2])
        wis.add_interval(interval3)

        #wis.read_blixt_tops(append_file)

        print(wis)
        self.assertIsInstance(wis, Intervals)

    def test_plot(self):
        wis = Intervals(name='TEST', intervals=[interval1, interval2, interval3])


class HeaderTestCase(unittest.TestCase):

    def test_Header(self):
        h = Header({'name': 'MY NAME', 'well': 'MY WELL'})
        print(list(h.keys()))
        print(h.name, h.creation_date)
        time.sleep(1)
        h.orig_filename = 'A TEST'
        print(h.orig_filename, h.creation_date, h.modification_date)
        print(h)

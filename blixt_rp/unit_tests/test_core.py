import unittest
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from bokeh.plotting import show

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
# sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp import Q_
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
rule6 = CutoffRule('P velocity', '><', [Q_(10, 'm/s'), Q_(1000, 'm/s')])
rule7 = CutoffRule('Density', '><', [Q_(1.5, 'G/cc'), Q_(2.5, 'G/cc')])
rule8 = CutoffRule('Porosity', '==', Q_(0.3, ''))
rule9 = CutoffRule('Porosity', '<', Q_(0.35, 'dimensionless'))

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
interval4 = Interval('AnotherWell', Q_(100. - 10., 'm'), Q_(200. + 10., 'm'), fm1)
interval5 = Interval('AnotherWell', Q_(200. - 10., 'm'), Q_(300. + 10., 'm'), fm2)
interval6 = Interval('AnotherWell', Q_(100. - 10., 'm'), Q_(300. + 10., 'm'), gp1)

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
        for rule in [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]:
            print(rule)

    def test_from_string(self):
        for rule in [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]:
            new_rule = CutoffRule(from_string=str(rule))
            print(rule, new_rule)

        string = "VCL>0.8[], PHIE<0.1[], vp><[2000, 4000][m/s], interval:Heather FM[]"
        cutoffs = Cutoffs(from_string=string)
        print(cutoffs)

    def test_failed_init(self):
        cutoffs = Cutoffs
        self.assertRaises(TypeError, cutoffs, name='test', alpha='JA', beta='Nej' )

    def test_init(self):
        cutoffs = Cutoffs(cutoffs=[rule1, rule2])
        print(len(cutoffs))
        cutoffs.append([rule3, rule4])
        cutoffs.append(rule1)
        cutoffs.append([rule1])
        cutoffs.append(rule5)
        print(len(cutoffs))
        print(cutoffs.cutoff_params)
        print(cutoffs)
        print(cutoffs.get_dict())
        self.assertTrue(True)

    def test_classification_table(self):
        from bokeh.io import output_file
        from bokeh.plotting import column
        from blixt_rp.core.core import CutoffRule, ClassificationTable
        output_file('C:\\Users\\emb\\Downloads\\plot.html')
        # output_file('C:\\Users\\marte\\Downloads\\plot.html')

        rule1 = CutoffRule('param1', '>', Q_(100, 'm'))
        rule2 = CutoffRule('param2', '<', Q_(10, 'm'))
        rule3 = CutoffRule('param3', '==', Q_(1000, 'm'))
        rule5 = CutoffRule('param5', '><', [Q_(10, 'm'), Q_(1000, 'm')])
        ct = ClassificationTable([rule1, rule2, rule3, rule5])
        # ct = ClassificationTable()
        rules_cds = ct.cds
        # print(ct.cds.data)
        # table, add_row, delete_row, update, use = ct.draw(rules_cds, ['log A', 'log B'], units=['m', 'km'])
        table, add_row, delete_row, update, use = ct.draw(rules_cds)
        print(ct.active_cutoffs(rules_cds))
        rules_cds.data['use'] = [True, True, True, True]
        print(ct.active_cutoffs(rules_cds))

        show(column(table, add_row, delete_row, update, use))
        # return table, add_row, delete_row, update, use

    def test_classification_mask(self):
        from bokeh.io import output_file
        from bokeh.plotting import column
        from blixt_rp.core.core import CutoffRule, ClassificationTable
        output_file('C:\\Users\\emb\\Downloads\\plot.html')
        # output_file('C:\\Users\\marte\\Downloads\\plot.html')

        rule1 = CutoffRule('param1', '>', Q_(100, 'm'))
        rule2 = CutoffRule('param2', '<', Q_(10, 'm'))
        ct = ClassificationTable([rule1, rule2])
        rules_cds = ct.cds
        table, add_row, delete_row, update, use = ct.draw(rules_cds, ['param1', 'param2'], units=['m', 'km'])

        data_cds = ColumnDataSource(dict(
            mask=np.ones(10, dtype=bool),
            param1=np.linspace(10, 200, 10),
            param2 = np.linspace(1, 12, 10)
        ))

        # Force the rules to be active as cutoffs
        rules_cds.data['use'] = [True, True]
        mask = ct.create_mask(rules_cds, data_cds)
        print('Both cutoff rules are active: ', mask)

        rules_cds.data['use'] = [True, False]
        mask = ct.create_mask(rules_cds, data_cds)
        print('First cutoff rules is active: ', mask)

        rules_cds.data['use'] = [False, True]
        mask = ct.create_mask(rules_cds, data_cds)
        print('Second cutoff rules is active: ', mask)

        # show(column(table, add_row, delete_row, update, use))

    def test_log_types(self):
        cutoffs = Cutoffs(cutoffs=[rule5, rule6, rule7])
        # rule5 can not be converted using the LogTable because it is not contained in it
        print(cutoffs.cutoff_params)
        new_cutoffs = cutoffs.use_log_table(LogTable(log_table))
        print(new_cutoffs.cutoff_params)

        # Raise an NotImplementedError because we don't support multi log LogTables yet
        self.assertRaises(NotImplementedError, cutoffs.use_log_table, LogTable(log_table_multi))

    def test_create_mask_from_cutoff_rule(self):
        data = {'name2': np.linspace(5., 20, 10)}
        mask = rule2.create_mask(data)
        print(mask)
        self.assertTrue(
            all(_x == _y for _x, _y in zip(mask, [True, True, True, False, False, False, False, False, False, False])))

    def test_create_mask_from_cutoffs(self):
        data_no_units = {'name1': np.linspace(90., 190, 10), 'name2': np.linspace(5., 20, 10)}
        data_with_units = {'name1': Q_(np.linspace(90., 190, 10), 'm'), 'name2': Q_(np.linspace(5., 20, 10), 'm')}
        cfs = Cutoffs(cutoffs=[rule1, rule2])
        mask = cfs.create_mask(data_with_units)
        print(mask)
        self.assertTrue(
            all(_x == _y for _x, _y in zip(mask, [False, True, True, False, False, False, False, False, False, False])))

class LogTableTests(unittest.TestCase):
    def test_lt_init(self):
        lt = LogTable( log_table)
        for _key in list(lt.keys()):
            print('{}: {}'.format(_key, lt[_key]))
        lt.test = 'A log name'  # this has no function now
        lt['TEST'] = 'A log name' # But this work when __setitem__ is kept untouched
        print(lt.keys())
        inv = lt.invert
        print(lt.log_names)
        print(lt.log_types)
        print(inv)
        print(lt.dict)
        print(lt['P velocity'])

    def test_from_invert(self):
        inv_dict = {'bs': 'Bit size', 'cali': 'Caliper', 'cpi_phie': 'Porosity', 'cpi_sw': 'Saturation', 'gr': 'Gamma ray', 'rho_fsub_gas80': 'Density', 'rho_fsub_oil80': 'Density', 'rho_insitu_conditioned': 'Density', 'vcl_used': 'Volume', 'vp_fsub_gas80': 'P velocity', 'vp_fsub_oil80': 'P velocity', 'vp_insitu_conditioned': 'P velocity', 'vs_fsub_gas80': 'S velocity', 'vs_fsub_oil80': 'S velocity', 'vs_insitu_conditioned': 'S velocity', 'vsh_mineral': 'Volume'}
        lt = LogTable()
        lt.from_invert(inv_dict)
        print(lt.log_types)
        print(lt.log_names)
        print(lt['Bit size'])

    def test_multi(self):
        lt = LogTable(log_table_multi)
        print(list(lt.keys()))
        lt['Density'].append('my other rhob')
        print(lt.log_names)
        print(lt.log_types)
        print(lt.invert)
        print(lt.dict)

    def test_mixing(self):
        lt1 = LogTable( log_table)
        lt2 = LogTable( log_table_multi)
        # lt1['Should fail'] = [1,2,3]
        lt2['Should fail'] = 'Vp'

    # def test_un_translate(self):
    #     lt1 = LogTable(log_table)
    #     rename_logs = {'phie': ['cpi_phie', 'xxx_phie']}
    #     lt2 = lt1.un_translate(rename_logs=rename_logs)
    #     print(lt2.log_names)

    #     print('Using a multi_log LogTable:')
    #     lt1 = LogTable(log_table_multi)
    #     rename_logs = {'vp_virg': ['vp_dry', 'vp_insitu']}
    #     lt2 = lt1.un_translate(rename_logs=rename_logs)
    #     print(lt2.log_names)

    def test_keep(self):
        lt1 = LogTable(log_table)
        keep = ['vp', 'rhob']
        lt2 = lt1.keep(keep)
        print(lt2.log_types)
        print(lt2.log_names)

        lt1 = LogTable(log_table_multi)
        keep = ['vp_brine', 'vp_oil', 'rhob_virg']
        lt2 = lt1.keep(keep)
        print(lt2)


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
            if i == 0:
                print(template_dict[_key])

        table = pd.read_excel(project_table, header=1, sheet_name='Well settings', engine='openpyxl')
        template_dict = templates_from_table(table, well_style=True)
        for i, _key in enumerate(list(template_dict.keys())):
            print(_key)
            if i == 0:
                print(template_dict[_key])

    def test_from_project_table_error(self):
        t = Template()
        t.get_from_project(project_table, 'XXX')
        print(t)

    def test_templates_table(self):
        from blixt_rp.core.core import TemplatesTable
        from bokeh.io import output_file
        output_file('C:\\Users\\emb\\Downloads\\plot.html')
        project_table = "C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx"

        templates = []
        for log_type in ['Resistivity', 'P velocity', 'S velocity']:
            t = Template()
            t.get_from_project(project_table, log_type)
            t.name = log_type  # The original template has no name
            templates.append(t)
        print(templates)
        tmpl_table = TemplatesTable(templates)
        print(tmpl_table.templates)
        for _t in tmpl_table.templates:
            print(_t.name)
        tmpl_cds = tmpl_table.cds
        print(tmpl_cds.data)
        table = tmpl_table.draw(tmpl_cds)
        show(table)


class IntervalTests(unittest.TestCase):

    def test_create_intervals(self):
        wis = Intervals('TEST', [interval1, interval2, interval3, interval4, interval5, interval6])

        print(wis)
        print('-x1-')
        print(wis.well_names())
        print('-x2-')
        print(wis.interval_names())
        print('-x3-')
        print(wis.get_well_depth_range('TestWell'))
        print('-x4-')
        print(wis.get_well_depth_range('AnotherWell'))
        print('-x5-')
        _i = wis.get_interval('Test1 FM', 'TestWell')
        print(_i.thickness)
        print('-x6-')
        print(_i.mid)
        print('-x7-')
        print(_i.distance_to_top(Q_(0, 'm')))
        print('-x8-')

        self.assertIsInstance(wis, Intervals)

    def test_read(self):
        f = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Sodir_well_tops_BarentsSea.xlsx"
        wis = Intervals(name='Working intervals')
        wis.read_sodir_tops(f)

        # Test removing most well
        wis.keep_wells(['7016_2_1', '7119_7_1'])

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

    def test_cutoff_rule(self):
        from blixt_utils.misc.masks import create_mask
        project_file = os.path.join(project_dir,"blixt_rp\\excels\\project_table_new.xlsx")
        wis = Intervals()
        wis.read_blixt_tops(project_file)
        _rule1 = wis.get_cutoff_rule('Shale C', 'WELL_F')
        _rule2 = wis.get_cutoff_rule('NO INTERVAL', 'WELL_F')
        print(_rule1, _rule2)
        md = Q_(np.linspace(1000., 3000., 20), 'm')
        for _rule in [_rule1, _rule2]:
            mask = create_mask(md, _rule.operator, _rule.limit)
            print(md[mask])



    def test_append(self):
        append_file = os.path.join(project_dir, 'test.xlsx')
        wis = Intervals(name='TEST', intervals=[interval1, interval2])
        wis.add_interval(interval3)

        #wis.read_blixt_tops(append_file)

        print(wis)
        self.assertIsInstance(wis, Intervals)

    def test_plot(self):
        wis = Intervals(name='TEST', intervals=[interval1, interval2, interval3])

    def test_intervals_table(self):
        from blixt_rp.core.core import Intervals, WorkingIntervalsTable
        from bokeh.io import output_file
        output_file(os.path.join(test_file_dir,'plot.html'))
        wis = Intervals()
        wis.read_blixt_tops(project_table)
        wis.keep_wells(['WELL_B', 'WELL_C', 'WELL_F'])
        wis_table = WorkingIntervalsTable(wis)
        wis_cds = wis_table.cds
        print(wis_cds.data['use'])
        print('-x1-')
        print(wis_table.active_intervals_dict(wis_cds))
        print(wis_table.active_intervals(wis_cds))
        # Modify Column Data Source
        for i in range(len(wis_cds.data['use'])):
            wis_cds.data['use'][i] = False
        # Add it back to wis_tabl
        wis_table.cds = wis_cds
        # Read it out again
        wis_cds = wis_table.cds
        print(wis_table.active_intervals(wis_cds))

        table = wis_table.draw(wis_table.cds)
        show(table)

    def test_intervals_table_2(self):
        from blixt_rp.core.core import Intervals, WorkingIntervalsTable
        wis = Intervals(
            name='test', intervals=[
                Interval( well='one', top=Q_(1050, 'm'), base=Q_(1150, 'm'), interval_info=StratUnit('wi_1', 1)),
                Interval( well='one', top=Q_(1200, 'm'), base=Q_(1400, 'm'), interval_info=StratUnit('wi_2', 1)),
                Interval( well='two', top=Q_(1550, 'm'), base=Q_(1650, 'm'), interval_info=StratUnit('wi_1', 1)),
                Interval( well='two', top=Q_(1800, 'm'), base=Q_(1950, 'm'), interval_info=StratUnit('wi_2', 1))
            ] )
        wis_table = WorkingIntervalsTable(wis)
        wis_cds = wis_table.cds
        mask = wis_table.create_mask(wis_cds, md=Q_(np.linspace(1000., 2000., 10), 'm'),
                                     wells=None)
        self.assertTrue(np.array_equal(mask, np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)))
        print('-x1-')
        mask = wis_table.create_mask(wis_cds, md=Q_(np.linspace(1000., 2000., 10), 'm'),
                                     wells=['one'] * 10)
        self.assertTrue(np.array_equal(mask, np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)))
        print('-x2-')
        mask = wis_table.create_mask(wis_cds, md=Q_(np.linspace(1000., 2000., 10), 'm'),
                                     wells=['two'] * 10)
        self.assertTrue(np.array_equal(mask, np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=bool)))


class HeaderTestCase(unittest.TestCase):

    def test_Header(self):
        import time
        h = Header({'name': 'MY NAME', 'well': 'MY WELL'})
        print(list(h.keys()))
        print(h.name, h.creation_date)
        time.sleep(1)
        h.orig_filename = 'A TEST'
        print(h.orig_filename, h.creation_date, h.modification_date)
        print(h)
        print(list(h.keys()))


class LithoFluidTests(unittest.TestCase):

    def litho_fluids(self):
        """
        Create a set of useful litho-fluids
        :return:
        """
        from blixt_rp.core.core import LithoFluid
        lf1 = LithoFluid(None, 3440., 2170., 2.17)
        lf2 = LithoFluid('Oseberg SST', 3400., 2050., 2.21)
        lf3 = LithoFluid('Fluid sub. problem', 3200., 2050., 2.21)

        return lf1, lf2, lf3

    def test_fluid_sub(self):
        for lf in self.litho_fluids():
            lf_hc = lf.fluid_sub(verbose=True)
            # print(lf_hc.name, lf_hc.vp.magnitude, lf_hc.vs.magnitude, lf_hc.rho.magnitude)
        # In RokDoc we have the following results for these values:
        #  lf1 -> 80% oil: 3311.	2194.	2.12
        #  lf1 -> 100% gas: 3274.	2305.	1.92
        #  lf2 -> 80% oil: 3269.	2070.	2.17
        #  lf3 -> 80% oil: Fails because of K_dry
        lf = self.litho_fluids()[0]
        lf_gas = lf.fluid_sub(s_w2=Q_(0., ''), hc=dict(rho=Q_(0.15, 'G/cc'), k=Q_(0.037, 'GPa')), verbose=True)
        # print('Gas: ', lf_gas.name, lf_gas.vp.magnitude, lf_gas.vs.magnitude, lf_gas.rho.magnitude)
        lf_heavy_oil = lf.fluid_sub(hc=dict(rho=Q_(1.1, 'G/cc'), k=Q_(2.757, 'GPa')), verbose=True)
        # print('Heavy oil: ', lf_heavy_oil.name, lf_heavy_oil.vp.magnitude, lf_heavy_oil.vs.magnitude, lf_heavy_oil.rho.magnitude)


    def test_2d_lf(self):
        from blixt_rp.core.core import LithoFluid
        def vp(_i):
            return Q_(3000. +  5. * _i, 'm/s')
        def vs(_i):
            return Q_(1500. +  5. * _i, 'm/s')
        def rho(_i):
            return Q_(2.5 +  0.1 * _i, 'grams/cm^3')

        lf = LithoFluid(vp=vp, vs=vs, rho=rho,
                        vp_std_dev=1., vs_std_dev=1., rho_std_dev=0.01,
                        vp_vs_cc=0.9, vp_rho_cc=0.9, vs_rho_cc=0.9)
        print(lf.vp(10))
        print(lf.to_sums_dict(10))
        self.assertEqual(3050., lf.vp(10).magnitude)

    def test_plot(self):
        from blixt_rp.core.core import LithoFluid
        lf = LithoFluid()
        lf.plot()
        plt.show()
        self.assertTrue(True)

    def test_table(self, unit_test=True):
        from bokeh.io import output_file
        from bokeh.plotting import show, row, column
        from blixt_rp.core.core import LithoFluid, LithoFluids, LithoFluidsTable
        output_file('C:\\Users\\marten.blixt\\Downloads\\plot.html')

        lfs = LithoFluids(litho_fluids=[LithoFluid(default='shale'), LithoFluid(default='brine_sst')])

        lft = LithoFluidsTable(lfs)

        cds = lft.cds
        dt, add_row, delete_row, update_table = lft.draw(cds)
        self.assertTrue(True)
        if unit_test:
            show(column(
                dt,
                row(add_row, delete_row, update_table)
            ))
            return None
        else:
            return dt, add_row, delete_row, update_table

    def test_quasi_2D_table(self, unit_test=True):
        from bokeh.io import output_file
        from bokeh.plotting import show, row, column
        from blixt_rp.core.core import LithoFluid, LithoFluids, LithoFluidsTable
        output_file('C:\\Users\\marten.blixt\\Downloads\\plot.html')

        def vp(_i):
            return Q_(3000. +  5. * _i, 'm/s')
        lf1 = LithoFluid(name='Quasi 2D', vp=vp)
        lfs = LithoFluids(litho_fluids=[LithoFluid(default='shale'), lf1, LithoFluid(default='brine_sst')])
        lft = LithoFluidsTable(lfs)

        cds = lft.cds
        dt, add_row, delete_row, update_table = lft.draw(cds)
        self.assertTrue(True)
        if unit_test:
            show(column(
                dt,
                row(add_row, delete_row, update_table)
            ))
            return None
        else:
            return dt, add_row, delete_row, update_table

import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import pint.errors
from math import isclose
from .. import Q_
from blixt_rp.core.core import LogTable, Cutoffs, CutoffRule

test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table_new.xlsx')

las_file1 = os.path.join(test_file_dir, "L-30.las")
log_table1 = LogTable({
    'Caliper': ['CALD', 'CALS'],
    'Density': ['DHRO'],
    'Sonic': ['DT'],
    'Gamma ray': ['GRD', 'GRS']
})
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"
log_table2 = LogTable({
    'P velocity': 'Vp_brine',
    'S velocity': 'Vs_brine',
    'Density': 'Rho_brine'
})

las_file3 = os.path.join(test_file_dir, "Well F.las")
las_fileA = os.path.join(test_file_dir, "Well A.las")
data_file1 = os.path.join(test_file_dir, "Well A checkshot.txt")
data_file2 = "S:\\Well\\UTM32_Mid_Norway_All\\Q-6406\\6406_11_1_S\\6406_11_1_S___checkshot.txt"
data_file2_wellpath = "S:\\Well\\UTM32_Mid_Norway_All\\Q-6406\\6406_11_1_S\\6406_11_1_S___wellpath.txt"

n = 1500
# create a regularly sampled data set
data1 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth1 = Q_(np.linspace(24, 3430, n), 'm')

# create an irregularly sampled data set
data2 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth2 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'FT')

n = 1400
# create a shorter regularly sampled data set
data3 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth3 = Q_(np.linspace(24, 3430, n), 'm')

# create a shorter irregularly sampled data set
data4 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth4 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'feet')

data5 = Q_(np.linspace(6, 8, n) + np.random.random(n), 'Ohmm')
depth5 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'feet')

def simple_test_well():
    from blixt_rp.core.well_new import Well
    from blixt_rp.core.log_curve_new import Depth, LogCurve
    lc1 = LogCurve('Long and regular', data1, Depth(depth1))
    lc2 = LogCurve('Long and irregular', data2, Depth(depth2))
    lc3 = LogCurve('Short and regular', data3, Depth(depth3))
    lc4 = LogCurve('Short and irregular', data4, Depth(depth4))
    w = Well()
    for lc in [lc1, lc2, lc3, lc4]:
        w.add_log(lc)

    return w

class WellTestCase(unittest.TestCase):

    def test_create_well(self):
        w = simple_test_well()
        print(w.get_log_names)
        for lc in w.logs:
            print(lc.name, len(lc), lc.is_evenly_spaced, lc.units, lc.depth_units,  lc.log_type)

    def test_harmonize_logs(self):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.log_curve_new import Depth, LogCurve

        _n = 1400
        # Create a well where the log with the greatest depth range is regular
        _data1 = Q_(np.linspace(2, 4, _n) + np.random.random(n), 'us/feet')
        _depth1 = Depth(Q_(np.linspace(24, 3430, _n), 'm'))
        _lc1 = LogCurve('Long and regular', _data1, _depth1)
        _data2 = Q_(np.linspace(6, 8, _n) + np.random.random(n), 's/m')
        _depth2 = Depth(Q_(np.linspace(50, 3000, _n) + np.random.random(n), 'm'))
        _lc2 = LogCurve('Short and irregular', _data2, _depth2)
        w1 = Well()
        for lc in [_lc1, _lc2]:
            w1.add_log(lc)

        # Create a well where the log with the greatest depth range is irregular
        _data1 = Q_(np.linspace(2, 4, _n) + np.random.random(n), 'us/feet')
        _depth1 = Depth(Q_(np.linspace(24, 3430, _n) + np.random.random(n), 'm'))
        _lc1 = LogCurve('Long and irregular', _data1, _depth1)
        _data2 = Q_(np.linspace(6, 8, _n) + np.random.random(n), 's/m')
        _depth2 = Depth(Q_(np.linspace(50, 3000, _n), 'm'))
        _lc2 = LogCurve('Short and regular', _data2, _depth2)
        w2 = Well()
        for lc in [_lc1, _lc2]:
            w2.add_log(lc)

        # Create a well from las file
        las_file =  os.path.join(test_file_dir, 'Well A.las')
        lt = LogTable({'Porosity': 'phie', 'Volume': 'vcl', 'MD': 'dept'})
        w3 = Well()
        w3.read_las(las_file, log_table=lt, template_file=project_table)

        # Test if the dict method ruins the harmonization
        # NOTE
        #  OF COURSE THE DATA ISN'T EVENLY SPACED AFTER WE HAVE APPLIED A CUTOFF!
        w4 = Well()
        cr1 = CutoffRule('vcl', '<', Q_(0.4, ''))
        cr2 = CutoffRule('phie', '>', Q_(0.1, ''))
        ct = Cutoffs([cr1, cr2])
        w_dict = w3.dict(cutoffs=ct, use_cutoffs=True)
        depth = Depth(w_dict['dept'])
        lc_vcl = LogCurve('vcl', w_dict['vcl'], depth, log_type='Volume')
        if not lc_vcl.is_evenly_spaced:
            print(' vcl log is not evenly spaced')
        w4.add_log(lc_vcl)

        lc_phie = LogCurve('phie', w_dict['phie'], depth, log_type='Porosity')
        if not lc_phie.is_evenly_spaced:
            print(' phie log is not evenly spaced')
        w4.add_log(lc_phie)

        lc_dept = LogCurve('dept', w_dict['dept'], depth, log_type='MD')
        if not lc_dept.is_evenly_spaced:
            print(' dept log is not evenly spaced')
        w4.add_log(lc_dept)


        # harmonize logs
        for w in [w1, w2, w3]:
            print('Before: Is well {} evenly spaced?: {}'.format(w.name, w.is_evenly_spaced))
            print('Before: Are logs in well {} of the same length?: {}'.format(w.name, w.is_of_equal_length))
            w.harmonize_logs()
            print('After: Is well {} evenly spaced?: {}'.format(w.name, w.is_evenly_spaced))
            print('After: Are logs in well {} of the same length?: {}'.format(w.name, w.is_of_equal_length))
            w.harmonize_logs()
            for lc in w.logs:
                print(lc.name, len(lc), lc.is_evenly_spaced, lc.units, lc.depth_units,  lc.base - lc.top)



    def test_create_well_from_las(self):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable
        w = Well()
        self.assertIsInstance(w, Well, 'Failed')
        w.read_las(las_file1, True)
        self.assertIsInstance(w.logs, list, 'Well.logs should be a list, not {}'.format(type(w.logs)))
        for i, key in enumerate(list(w.header.well_info.keys())):
            print(key, w.header.well_info[key])
            if i > 5:
                break
        for log in w.logs:
            print(log.name, log.log_type)

        this_log = w.get_log_curve('cals')
        print('\nFetched this log: ', this_log.name)
        print(' -x-')

        w = Well()
        w.read_las(las_file1, True, log_table=LogTable({'Sonic': 'dt'}))
        for log in w.logs:
            print(log.name, log.log_type)

        w = Well()
        w.read_las(las_file1, True, log_table=LogTable({'Caliper': ['cald', 'cals']}))
        for log in w.logs:
            print(log.name, log.log_type)
        d = w.dict()
        for _key, _value in d.items():
            print(_key, len(_value))

        print(w.name)

        w.name = 'TEST'
        print(w.name)

    def test_add_log(self):
        from blixt_rp.core.well_new import Well
        w = Well()
        w.read_las(las_file1, True, log_table={'Sonic': 'dt'})
        for log in w.logs:
            print(log.name, log.log_type)
        mod_date1 = w.get_log_curve('dt').header.modification_date
        w.read_las(las_file1, True, log_table={'Sonic': 'dt', 'Caliper': 'cald'}, if_log_exists='overwrite')
        for log in w.logs:
            print(log.name, log.log_type)
        mod_date2 = w.get_log_curve('dt').header.modification_date
        w.read_las(las_file1, True, log_table={'Sonic': 'dt', 'Caliper': 'cald'}, if_log_exists='ignore')
        for log in w.logs:
            print(log.name, log.log_type)
        mod_date3 = w.get_log_curve('dt').header.modification_date
        w.read_las(las_file1, True, log_table={'Sonic': 'dt', 'Caliper': 'cald'}, if_log_exists='ask')
        for log in w.logs:
            print(log.name, log.log_type)
        print(mod_date1, mod_date2, mod_date3)

    def test_read_las(self):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable, Template
        well1 = Well()
        well1.read_las(las_file1, log_table=log_table1, template_file=project_table)
        print(well1.get_log_names)
        well1.create_md_log()
        print(well1.get_log_names)
        print(str(well1.style))
        lc = well1.get_log_curve('dt')
        # well1.read_las(las_file2, log_table=log_table2, template_file=project_table)
        # lc = well1.get_log_curve('vs_brine')
        #print(well1.name, well1.get_log_names)
        for _key in list(well1.header.keys()):
            print(_key, well1.header[_key])
        #print(lc.style)
        self.assertIsInstance(lc.style, Template)

    def test_old_well_type(self):
        from blixt_rp.core.well import Well as OldWell
        w0 = OldWell()
        w0.read_las(las_file2)
        print(w0.header, w0.block['Logs'].header, w0.block['Logs'].logs)

    def test_write_las(self):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable
        log_table = LogTable({
            'P velocity': 'Vp_brine',
            'S velocity': 'Vs_brine',
            'Density': 'Rho_brine'
        })
        well1 = Well()
        well1.read_las(las_file2, log_table=log_table, template_file=project_table)
        well1.write_las('test.las')

    def test_las_and_data(self):
        from blixt_rp.core.well_new import Well
        well = Well()
        well.read_las(las_file2, log_table=log_table2, template_file=project_table)
        print(well.get_log_names)
        rho = well.get_log_curve('rho_brine')
        print('Rho:', rho.base, rho.top, rho.step(), len(rho), rho.log_type)

        well.read_general_ascii(data_file2,
                                'space',
                                4,
                                ['md', 'owt'],
                                [0, 1],
                                ['m', 'millisecond'],
                                ['MD', 'One-way time'])
        print(well.get_log_names)
        owt = well.get_log_curve('owt')
        print('OWT:', owt.base, owt.top, owt.step(), len(owt), owt.log_type)
        print(owt.style)

    def test_create_md_log(self):
        from blixt_rp.core.well_new import Well
        well = Well()
        well.read_las(las_file2, log_table=log_table2, template_file=project_table)
        well.read_general_ascii(data_file2,
                                'space',
                                4,
                                ['md', 'owt'],
                                [0, 1],
                                ['m', 'millisecond'],
                                ['MD', 'One-way time'])

        # Now we create a LogCurve of MD data from the LogCurve that has the longest depth range
        md = well.get_md_log()

        # Now we take the MD LogCurve that was generated from data_file2
        md_log = well.get_log_curve('md')

        print(md.header, '\n', md.style, '\n', md.units)
        print(well.get_log_names)
        print(md_log.values[:5], md_log.values[-5:])
        print(md.values[:5], md.values[-5:])


    def test_well_trajectory(self):
        from blixt_rp.core.well_new import Well, WellTrajectory
        well = Well()
        well.read_las(las_file2, log_table=log_table2, template_file=project_table)
        well.read_general_ascii(data_file2_wellpath,
                                'space',
                                1,
                                ['md', 'tvd', 'inc'],
                                [4, 5, 6],
                                ['m', 'm',  'degree'],
                                ['MD', 'TVD', 'INC'])

        # Now we create a LogCurve of MD data from the LogCurve that has the longest depth range
        md = well.get_md_log()
        wt = WellTrajectory(
            md=md.data,
            tvd_kb=well.get_log_curve('tvd'),
            inc=well.get_log_curve('inc'),
            verbose=True
        )
        print(len(md), len(wt.tvd_kb), len(wt.inc))

    def test_data_source(self):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable
        w = Well()
        w.read_las(las_file1, True)
        wds = w.data_source()
        print('Data and units:')
        print('  #', len(wds.variables), wds.variables)
        print('  #', len(wds.units), wds.units)
        for key in list(wds.data.keys()):
            print(key, len(wds.data[key]))
        print('Downsample to every 10th:')
        wds = w.data_source(down_sample=10)
        for key in list(wds.data.keys()):
            print(key, len(wds.data[key]))

        # Add a test of calculating a mask, e.g wds.calc_mask(cutoffs, logtable)
        w = Well()
        w.read_las(las_file1, True)
        cutoffs = Cutoffs(cutoffs=[
            CutoffRule('rhob', '<', Q_(2.1, 'g/cm**3'))
        ])
        print('Mask out densities above 2.1 g/cm3:')
        wds = w.dict(cutoffs=cutoffs, use_cutoffs=True)
        for key in list(wds.keys()):
            print(key, len(wds[key]))
        print(max(wds['rhob']))

        # Note! The cutoffs should potentially hold a working interval!!

    def test_data_source_with_logtable(self):
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.core import LogTable
        w = Well()
        w.read_las(las_fileA, True)
        lt1 = LogTable(name='ptb5', log_table={'P velocity': 'vp_ptb5', 'S velocity': 'vs_ptb5', 'Density': 'rho_ptb5'})
        wds1 = w.data_source(log_table=lt1, verbose=True)
        lt2 = LogTable(name='so08', log_table={'P velocity': 'vp_so08', 'S velocity': 'vs_so08', 'Density': 'rho_so08'})
        wds2 = w.data_source(log_table=lt2)
        lt3 = LogTable(name='sg08', log_table={'P velocity': 'vp_sg08', 'S velocity': 'vs_sg08', 'Density': 'rho_sg08'})
        wds3 = w.data_source(log_table=lt3)
        print('Data and units:')
        for data_source in [wds1, wds2, wds3]:
            print(data_source.name)
            print('  #', len(data_source.variables), data_source.variables)
            print('  #', len(data_source.units), data_source.units)
            print('  -x-')

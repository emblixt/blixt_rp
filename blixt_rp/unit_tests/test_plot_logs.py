import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from bokeh.models import ColumnDataSource, Spinner, Select, Button, Slider, CheckboxGroup, Div
from bokeh.plotting import show, row, column
from bokeh.io import output_file
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename

import bruges.rockphysics.anisotropy as bra
import pint.errors
from math import isclose

from .. import ureg, Q_

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

# from AkerBP_PL932.PL923_AVO import cutoffs
from blixt_rp.core.log_curve_new import LogCurve, Depth, is_equivalent, read_las, read_general_ascii
from blixt_rp.core.core import Template, LogTable, Intervals, Cutoffs, CutoffRule
from blixt_rp.core.well_new import Well
from blixt_rp.core.project_new import Project
import blixt_rp.plotting.plot_logs_new as bupp
from blixt_rp.plotting.log_plotter import LogColumn, Line, add_lines, add_strat_table, LogPlotter, select_column
from blixt_rp.core.seismic import SeismicTraces, interpolate_along_offset

output_file('C:\\Users\emb\\Documents\\plot.html')
test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_rp\\unit_tests',
    'test_data'))

project_table = os.path.join(test_file_dir.replace('test_data', 'excels'), 'project_table_new.xlsx')
project_table2 = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\PL1221\\PL1221 project_new.xlsx"

las_file1 = os.path.join(test_file_dir, "L-30.las")
las_file2 = "T:\\ROKDOC\\PL936\\To Partners from Ikon\\To Partners\\las files\\6406_11_1s.las"
las_file3 = os.path.join(test_file_dir, "Well F.las")
data_file2 = "S:\\Well\\UTM32_Mid_Norway_All\\Q-6406\\6406_11_1_S\\6406_11_1_S___checkshot.txt"

n = 1500
# create a regularly sampled data set
data1 = Q_(np.linspace(2, 4, n) + np.random.random(n), 'us/feet')
depth1 = Q_(np.linspace(24, 3430, n), 'm')

# create an irregularly sampled data set
data2 = Q_(np.linspace(6, 8, n) + np.random.random(n), 's/m')
depth2 = Q_(np.linspace(1400. * 3, 2500. * 3., n) + np.random.random(n), 'FT')

# Useful log table
log_table1 = LogTable({'Density': 'rhob', 'Sonic': 'dt'})
log_table2 = LogTable({'Resistivity': 'rdep', 'Sonic': 'dt'})
log_table3 = LogTable({'Density': 'rho_brine', 'P velocity': 'vp_brine', 'S velocity': 'vs_brine'})

class TestPlot(unittest.TestCase):

    def test_log_plotter(self):
        from blixt_rp.plotting.log_plotter import select_column, select_line
        log_table = LogTable({
            'P velocity': ['Vp_dry', 'Vp_Sg08'],
            'Porosity': ['PHIE'],
            'Volume': ['VSH']
        })
        well1 = Well()
        well1.read_las(las_file3, log_table=log_table, template_file=project_table)
        print(well1.name)
        column_content = [['vsh'], ['phie'], ['vp_dry', 'vp_sg08']]
        plotter = bupp.plot_logs(well1, column_content, rel_widths=[1., 1., 1.])
        grid = plotter.figure()
        _p_one = select_column(grid, 'column_1')
        print(_p_one)
        _p_none = select_column(grid, 'XXX')
        print(_p_none)
        _line_vp_dry = select_line(grid, 'vp_dry')
        print(_line_vp_dry)
        # show(grid)

    def test_log_plotter_with_settings(self):
        from blixt_rp.plotting.log_plotter import select_column, select_line
        log_table = LogTable({
            'P velocity': ['Vp_dry', 'Vp_Sg08'],
            'Porosity': ['PHIE'],
            'Volume': ['VSH']
        })
        well1 = Well()
        well1.read_las(las_file3, log_table=log_table, template_file=project_table)
        print(well1.name)
        column_content = [['vsh'], ['phie'], ['vp_dry', 'vp_sg08']]
        plotter = bupp.plot_logs(well1, column_content, rel_widths=[1., 1., 1.])
        grid = plotter.figure()
        print(plotter.line_cds.data.keys())
        settings = plotter.add_settings(grid)
        show(row(grid, settings))

    def test_log_plotter_with_cutoffs(self):
        rule1 = CutoffRule('vp_dry', '>', Q_(3000, 'm/s'))
        rule2 = CutoffRule('phie', '<', Q_(0.1, ''))
        rule3 = CutoffRule('vsh', '>', Q_(0.4, ''))

        log_table = LogTable({
            'P velocity': ['Vp_dry', 'Vp_Sg08'],
            'Porosity': ['PHIE'],
            'Volume': ['VSH']
        })
        well1 = Well()
        well1.read_las(las_file3, log_table=log_table, template_file=project_table)
        print(well1.name)
        column_content = [['vsh'], ['phie'], ['vp_dry', 'vp_sg08']]
        plotter = bupp.plot_logs(well1, column_content, rel_widths=[1., 1., 1.])
        grid = plotter.figure()
        cutoffs, add_row, delete_row, update, use  = plotter.add_cutoffs(grid, rules=[rule1, rule2, rule3])
        show(row(grid, cutoffs))

    def test_plot_with_backus(self):
        log_table = LogTable({
            'Resistivity': ['rdep', 'rmed', 'rmic'],
            'Sonic': ['dt', 'dt_edit'],
            'P velocity': ['Vp_brine', 'Vp_Oil70', 'Vp_Gas70'],
            'S velocity': ['Vs_brine', 'Vs_Oil70', 'Vs_Gas70'],
            'Density': ['Rho_brine', 'Rho_Oil70', 'Rho_Gas70']
        })
        well1 = Well()
        well1.read_las(las_file2, log_table=log_table)
        for curve in well1.logs:
            if curve.name == 'rdep':
                _t = Template(**{'min': 0.1, 'max': 100., 'line_color': 'blue', 'scale': 'log'})
                curve.style = _t
            elif curve.name == 'rmed':
                curve.style = Template(**{'min': 0.1, 'max': 100., 'line_color': 'red', 'scale': 'log'})
            elif curve.name == 'rmic':
                curve.style = Template(**{'min': 0.1, 'max': 100., 'line_color': 'green', 'scale': 'log'})
            elif curve.name == 'dt':
                curve.style = Template(**{'min': 40., 'max': 180., 'line_color': 'blue'})
            elif curve.name == 'dt_edit':
                curve.style = Template(**{'min': 40., 'max': 180., 'line_color': 'red'})

        # Create a two-column plot
        plotter = bupp.plot_logs(well1, [['rdep', 'rmed', 'rmic'], ['vp_brine', 'vp_oil70', 'vp_gas70']],
                                 scales=['log', 'linear'])
        # Add a new column
        ai_column = LogColumn('AI plots')
        plotter.add_column(ai_column, keep_column_width=False)

        # "Realize" the figure
        grid = plotter.figure()

        # access the last figure in grid plot, and let it display the x-axis
        _p = grid.children[-1][0]
        _p.xaxis.visible = True

        #  Play with the Backus average
        ba_length = Spinner(title="Backus avg. window length [m]", low=5, high=100, step=5., value=20, width=80)
        vp0 = well1.get_log_curve('vp_brine').values
        vs0 = well1.get_log_curve('vs_brine').values
        rho0 = well1.get_log_curve('rho_brine').values
        md = well1.get_log_curve('vp_brine').depth
        ba = bra.backus(vp0, vs0, rho0, ba_length.value, md.step().magnitude)
        line_cds_dict = dict( ai=vp0 * rho0, ai_backus=ba[0] * ba[2], md=md.values )
        line_cds = ColumnDataSource(line_cds_dict)
        orig_template = Template(**dict(line_color='black', line_style = 'solid', line_width = 1, name = 'Orig AI'))
        orig_line = Line(x='ai', y='md', cds=line_cds, style=orig_template)
        backus_template = Template(**dict(line_color='red', line_style = 'solid', line_width = 2,
                                          name = 'AI {}m Backus'.format(ba_length.value),
                                          min=orig_line.x_range()[0], max=orig_line.x_range()[1]))
        backus_line = Line(x='ai_backus', y='md', cds=line_cds, style=backus_template)
        ai_column.lines = [backus_line, orig_line]
        add_lines(_p, ai_column)
        _p.legend.click_policy = 'hide'
        show(row(grid, ba_length))

    def test_interactive_edit(self):
        # This function has now been implemented in plot_logs_new.py, as interactive_edits()


        # This script is called  by the main.py script under blixt_projects/bokeh_testing
        # And can be invoked by calling:
        # C:\Users\emb\Documents\PycharmProjects\blixt_projects>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show bokeh_testing

        select_editors = CheckboxGroup(
            labels=['Fill gaps', 'Remove spikes', 'Smooth', 'Backus average'],
            active=[2])
        despike_clip_sel = Select(title='Clip level', value='moderate', options= ['low', 'moderate', 'high'],
                                  description=' Lower clip level and longer window length clips away more data')
        despike_window_len = Slider(title='Despike window length [m]', start=1, end=20, step=1, value=2)
        backus_window_len = Slider(title='Backus window length [m]', start=2, end=18, step=1, value=5)
        smooth_method_sel = Select(title='Smoothing method', value='median', options= ['convolution', 'median'])
        smooth_window_sel = Select(title='Smoothing window type', value='hanning',
                                   options=['flat', 'hanning', 'hamming', 'bartlett', 'blackman'])
        smooth_window_len = Slider(title='Smoothing window length [m]', start=5, end=100, step=5, value=11)
        run_smoothing = Button(label='Apply smoothing', button_type='success')
        save_result = Button(label='Save to .las file', button_type='success')
        log_table = LogTable({
            'P velocity': 'Vp_brine',
            'S velocity': 'Vs_brine',
            'Density': 'Rho_brine'
        })
        well1 = Well()
        well1.read_las(las_file2, log_table=log_table, template_file=project_table)

        # Create a three-column plot
        column_content = [['vp_brine'], ['vs_brine'], ['rho_brine']]
        plotter = bupp.plot_logs(well1, column_content)

        # add the smoothened data to each column
        line_cdss = []
        for _i, _column in enumerate(plotter.columns):
            # print(_column)
            _log_name = column_content[_i][0]
            _this_smooth_res = well1.get_log_curve(_log_name).smooth(
                window_len=smooth_window_len.value,
                method=smooth_method_sel.value,
                window=smooth_window_sel.value
            )
            _this_cds = ColumnDataSource(dict(smooth=_this_smooth_res.values, md=_this_smooth_res.depth.values))
            _this_style = _this_smooth_res.style
            _this_style.line_width = 3.
            _this_style.line_color = 'red'
            _this_style.name = _this_style.name + '_smooth'
            line_cdss.append(_this_cds)
            _column.add_line(Line(x='smooth', y='md', cds=_this_cds, style=_this_style))

        def callback():
            _step = None
            for _i, _column in enumerate(plotter.columns):
                _log_name = column_content[_i][0]
                _this_smooth_res = well1.get_log_curve(_log_name).copy()
                if _step is None:
                    _step = _this_smooth_res.step().magnitude
                _this_step = _this_smooth_res.step().magnitude
                if _this_step != _step:
                    raise IOError('Current depth step ({}) is different from last ({})'.format(_this_step, _step))
                if select_editors.active.count(0) > 0:  # Fill gaps
                    print('Fill gaps')
                    _this_smooth_res = _this_smooth_res.fill_gaps()
                if select_editors.active.count(1) > 0:  # Remove spikes
                    print('Remove spikes')
                    level = _this_smooth_res.std
                    if despike_clip_sel.value == 'low':
                        level = 0.1 * level
                    elif despike_clip_sel.value == ' moderate':
                        level = 0.5 * level
                    elif despike_clip_sel.value ==' high':
                        level = 1.0 * level
                    _this_smooth_res = _this_smooth_res.despike(max_clip=level, window_len=despike_window_len.value)
                if select_editors.active.count(2) > 0:  # smooth data
                    _this_smooth_res = well1.get_log_curve(_log_name).smooth(
                        window_len=smooth_window_len.value,
                        method=smooth_method_sel.value,
                        window=smooth_window_sel.value
                    )
                line_cdss[_i].data['smooth'] = _this_smooth_res.values
            if select_editors.active.count(3) > 0:  # Run Backus average
                # HARD CODED TO ONLY WORK WHEN Vp, Vs and Rho COMES IN THIS SPECIFIC ORDER
                _vp = line_cdss[0].data['smooth']
                _vs = line_cdss[1].data['smooth']
                _rho = line_cdss[2].data['smooth']
                ba = bra.backus(_vp, _vs, _rho, backus_window_len.value, _step)
                for _i in range(3):
                    line_cdss[_i].data['smooth'] = ba[_i]

        def save():
            file_name = save_as_las()
            well1.write_las(file_name)

        run_smoothing.on_click(callback)
        save_result.on_click(save)
        # "Realize" the figure
        grid = plotter.figure()

        # show(grid)
        return (grid, select_editors, despike_clip_sel, despike_window_len,
                smooth_method_sel, smooth_window_sel, smooth_window_len, backus_window_len, run_smoothing, save_result)

    def test_interactive_edits(self):
        # This script is called  by the main.py script under blixt_projects/bokeh_testing
        # And can be invoked by calling:
        # C:\Users\emb\Documents\PycharmProjects\blixt_projects>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show bokeh_testing

        well1 = Well()
        log_table = LogTable({'Gamma ray': 'grd', 'Density': 'rhob', 'Sonic': 'dt'})
        well1.read_las(las_file1, log_table=log_table,  template_file=project_table)
        well1.name = 'Well_L'
        log_columns = [['grd'], ['rhob'], ['dt']]
        # log_columns = [['rhob'], ['dt']]

        wis = Intervals()
        wis.read_blixt_tops(project_table)

        return bupp.interactive_edits(well1, log_columns, wis=wis)

    def test_add_strat_table(self):
        log_table = LogTable({
            'P velocity': ['Vp_dry', 'Vp_Sg08'],
            'Porosity': ['PHIE'],
            'Volume': ['VSH']
        })
        well1 = Well()
        well1.read_las(las_file3, log_table=log_table, template_file=project_table)
        print(well1.name)
        wis = Intervals()
        wis.read_blixt_tops(project_table)

        # Create a four-column plot
        column_content = [['vsh'], [], ['phie'], ['vp_dry', 'vp_sg08']]
        plotter = bupp.plot_logs(well1, column_content, rel_widths=[1., 0.3, 1., 1.])
        grid = plotter.figure()
        data_table = add_strat_table(grid,
                                     stratigraphy=wis.get_intervals_dict(well1.name), width=800, column_index=1)
        title = Div(text='<h2>{}</h2>'.format(well1.name))
        show(column(title, grid, data_table))

    def test_get_wiggles(self):
        log_curves, well_info = read_las(las_file2, log_table=log_table3)
        data_curves = read_general_ascii(
            data_file2,
            'space',
            4,
            ['md', 'owt'],
            [0, 1],
            ['m', 'millisecond'],
            ['md', 'owt'])
        owt_lc = data_curves['owt'].take_sampling_from(log_curves['vp_brine'])
        time_depth_twt = 2.*owt_lc.data
        dt = Q_(1, 'millisecond')

        result = bupp.get_wiggles_in_depth(
            log_curves['vp_brine'],
            log_curves['vs_brine'],
            log_curves['rho_brine'],
            time_depth_twt,
            dt,
            avo_angles=np.linspace(0, 35, 36),
            chi_angles=np.arange(-90, 91, 1)
        )
        plt.imshow(result[:, ::100])
        plt.show()

    def test_plot_wiggles(self):
        well = Well()
        well.read_las(las_file2, log_table=log_table3, template_file=project_table)

        vp = well.get_log_curve('vp_brine')
        vs = well.get_log_curve('vs_brine')
        rho = well.get_log_curve('rho_brine')
        data_curves = read_general_ascii(
            data_file2,
            'space',
            4,
            ['md', 'owt'],
            [0, 1],
            ['m', 'millisecond'],
            ['MD', 'One-way time'])
        owt_lc = data_curves['owt'].take_sampling_from(vp)
        time_depth_twt = 2.*owt_lc.data
        dt = Q_(1, 'millisecond')

        avo_angles = np.linspace(0, 35, 36)
        chi_angles = np.arange(-90, 91, 1)
        result = bupp.get_wiggles_in_depth(
            vp,
            vs,
            rho,
            time_depth_twt,
            dt,
            avo_angles=None,
            chi_angles=chi_angles,
            center_frequency=20.)
        seismic_traces = SeismicTraces(
            x=chi_angles, y=vp.depth.values,
            # traces=result, trace_type='chi')
            traces=None,
            cds=ColumnDataSource({'value': [result.T]}),
            trace_type='chi')

        plotter = LogPlotter(width=800, height=1000)
        line = Line(x=vp.values * rho.values, y=vp.depth.values, style=Template(
            **{'name': 'AI'} ))
        c1 = LogColumn('AI', lines=[line], rel_width=1)
        c2 = LogColumn('AVO', seismic_traces=seismic_traces, rel_width=2)
        plotter.columns = [c1, c2]
        grid = plotter.figure()
        show(grid)

    def test_chi_rotation(self):
        well = Well()
        log_table = LogTable({'Density': ['rho_brine', 'rho_gas70'], 'P velocity': ['vp_brine', 'vp_gas70'],
                             'S velocity': ['vs_brine', 'vs_gas70']})
        brine_log_table = LogTable({'Density': 'rho_brine', 'P velocity': 'vp_brine', 'S velocity': 'vs_brine'})
        hc_log_table = LogTable({'Density': 'rho_gas70', 'P velocity': 'vp_gas70', 'S velocity': 'vs_gas70'})

        well.read_las(las_file2, log_table=log_table, template_file=project_table)
        well.read_general_ascii(data_file2,
                                'space',
                                4,
                                ['md', 'owt'],
                                [0, 1],
                                ['m', 'millisecond'],
                                ['MD', 'One-way time'])

        wis = Intervals()
        wis.read_blixt_tops(project_table2)

        grid, chi_slider, backus_slider, frq_slider, data_table, xplot, hplot = bupp.plot_chi_rotation(well, brine_log_table, hc_log_table, wis=wis)
        show(row(column(grid, chi_slider, backus_slider, frq_slider), column(hplot, xplot)))

    def test_compare_seismic(self):
        # Create real seismic traces from las file with seismic
        las_file = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Tampen\\Wells\\34_3_3S_seismic.las"
        las_file_logs = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\\Tampen\\Wells\\34_3_3S.las"
        t = Template(**{'name': 'Seismic', 'units': 'dimensionless', 'line_color': 'b'})
        log_table = LogTable({'Seismic':
                                  ['CGG18M01-NVG-PSDM-ANGLE-NEAR-05-15-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-FAR-21-31-TPL932MSMTMADF2denoise2',
                                   'CGG18M01-NVG-PSDM-ANGLE-ULTRAFAR-29-39-TPL932MSMTMADF2denoise2'
                                   ]})
        rename_logs = {
            'near': ['CGG18M01-NVG-PSDM-ANGLE-NEAR-05-15-TPL932MSMTMADF2denoise2'],
            'mid': ['CGG18M01-NVG-PSDM-ANGLE-MID-13-23-TPL932MSMTMADF2denoise2'],
            'far': ['CGG18M01-NVG-PSDM-ANGLE-FAR-21-31-TPL932MSMTMADF2denoise2'],
            'ufar': ['CGG18M01-NVG-PSDM-ANGLE-ULTRAFAR-29-39-TPL932MSMTMADF2denoise2']
        }
        log_curves, well_info = read_las(las_file, log_table=log_table, template=t, rename_logs=rename_logs)

        offset_angles = pint.Quantity(np.array([10., 18., 26., 34.]), 'deg')
        traces = np.zeros((len(offset_angles), len(log_curves['near'])))
        traces[0,:] = log_curves['near'].values
        traces[1,:] = log_curves['mid'].values
        traces[2,:] = log_curves['far'].values
        traces[3,:] = log_curves['ufar'].values
        # interpolate the seismic traces
        _traces = interpolate_along_offset(traces,
                                           offset_angles,
                                           Q_(np.arange(10, 34, 1), 'deg'))
        st_orig = SeismicTraces(x=offset_angles.magnitude,
                                y=log_curves['near'].depth.values,
                                # traces=traces,
                                traces=None,
                                cds=ColumnDataSource({'value': [_traces.T]}),
                                trace_type='avo',
                                title='CGG18M01-NVG-PSDM'
                                )

        well = Well()
        vp_vs_rho_table = LogTable({'Density': 'lfp_rhob_virgin', 'P velocity': 'lfp_vp_virgin',
                              'S velocity': 'lfp_vs_virgin'})
        ref_log_table = LogTable({'Volume': 'vsh', 'Gamma ray': 'gr'})

        well.read_las(las_file_logs, log_table=vp_vs_rho_table, template_file=project_table)
        well.read_las(las_file_logs, log_table=ref_log_table, template_file=project_table)
        # Fake OWT data from another well
        well.read_general_ascii(data_file2,
                                'space',
                                4,
                                ['md', 'owt'],
                                [0, 1],
                                ['m', 'millisecond'],
                                ['MD', 'One-way time'])
        print(well.get_log_names)

        return bupp.compare_synth_with_seismic(well, ref_log_table, vp_vs_rho_table, st_orig)

    def test_plot_trends(self):
        # Only have Well A active in the project table
        log_table = LogTable({'Density': 'rho_dry', 'P velocity': 'vp_dry', 'S velocity': 'vs_dry',
                              'Porosity': 'PHIE', 'Volume': 'VCL'})
        wp = Project(name='MyProject', log_to_stdout=True)
        wp.load_all_wells(log_table=log_table)
        wis = wp.load_all_wis()
        wi_name = 'Sand D'
        de_trend_loc = 1850.
        de_trend_scale = 2.
        rule1 = CutoffRule('vcl', '<', Q_(0.5, ''))
        rule2 = CutoffRule('phie', '>', Q_(0.05, ''))
        cutoffs = Cutoffs(cutoffs=[rule1, rule2])
        bupp.plot_trends('tvd', wp.wells, log_table, wis, wi_name, cutoffs,
                         de_trend_loc=de_trend_loc, de_trend_scale=de_trend_scale,
                         results_folder="C:\\Users\\emb\\Downloads", verbose=True)



def save_as_las():
    root = Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    file_name = asksaveasfilename(filetypes=(("las file", "*.las"),("All Files", "*.*")))
    return file_name

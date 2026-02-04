import matplotlib as mpl
import os, sys
from copy import deepcopy

import bokeh.plotting
import numpy as np
from openpyxl.styles.builtins import title
from pandas import DataFrame
from typing import Literal
from IPython.core.magics.code import extract_code_ranges
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, Spacer
from bokeh.models import (Slider, ColorPicker, Line, LinearAxis, Span, Legend, ColumnDataSource, Text,
                          CustomJS, LinearColorMapper, NumberFormatter, Button, CheckboxGroup)
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, ColorBar, LogColorMapper
from bokeh.models import (DataTable, NumberEditor, SelectEditor, StringEditor, StringFormatter,
                          IntEditor, TableColumn, CheckboxEditor, NumericInput, VArea)
from bokeh.io import output_file, curdoc
from bokeh.layouts import gridplot

import bruges

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp_utils.avo_monte_carlo as havo
from blixt_rp.core.core import LithoFluid, LithoFluids
import blixt_rp.rp.rp_core as rp

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    CrosshairTool(),
    ResetTool(),
    SaveTool()
]

# Global avo angle parameter, assuming n_angles samples from 0 to 40 deg incidence angle
n_angles = 20
theta = np.linspace(0, 40, n_angles)

def clean_name(in_string: str) -> str:
        return in_string.lower().replace(' ', '_').replace('-', '_')


def create_litho_fluids_table(
                         _litho_fluids: list,
                         _width: int | None = None) -> bokeh.models.DataTable:
    """
    Adds the different litho fluids listed in _litho_fluids to a table which can be further used in the AVO modelling
    :param _litho_fluids:
        List of litho fluids LithoFluid
    :param _width:
    :return:
    """
    if _width is None:
        _width = 400

    # Create table columns
    column_keys = list(_litho_fluids[0].__dict__.keys())  # Get name of content from first item in _litho_fluids
    column_names = [_s.capitalize().replace('_', ' ') for _s in column_keys]
    table_columns = []
    for i, column_key in enumerate(column_keys):
        if column_key == 'name':
            _editor = StringEditor()
            _formatter = StringFormatter(font_style='bold')
        else:
            _editor = NumberEditor()
            _formatter = NumberFormatter(format='0.0[0]')
        table_columns.append(
            TableColumn(
                field=column_key,
                title=column_names[i],
                editor=_editor,
                formatter=_formatter

            )
        )

    # Create table content from _litho_fluids:
    _content_dict = {_x: [] for _x in column_keys}
    for litho_fluid in _litho_fluids:
        for _key in column_keys:
            # append data from each litho_fluid to the dict where the units are taken care of.
            if _key == 'name':
                _content_dict[_key].append(litho_fluid.__dict__[_key])
            elif _key in ['vp', 'vs', 'vp_std_dev', 'vs_std_dev']:
                _content_dict[_key].append(litho_fluid.__dict__[_key].to('m/s').magnitude)
            elif _key in ['rho', 'rho_std_dev']:
                _content_dict[_key].append(litho_fluid.__dict__[_key].to('g/cm**3').magnitude)
            elif _key in ['vp_vs_cc', 'vp_rho_cc', 'vs_rho_cc']:
                _content_dict[_key].append(litho_fluid.__dict__[_key].magnitude)
            else:
                raise IOError('key: {}, is unknown and is likely handled correctly')


    source = ColumnDataSource(_content_dict)

    return DataTable(
        source=source,
        columns=table_columns,
        editable=True,
        width=_width,
        index_position=-1,
        index_header='row index',
        index_width=60
    )

def get_litho_fluid_from_table(litho_fluid_name: str,
                               litho_fluid_table: bokeh.models.DataTable) -> dict:
    """
    Gets the litho fluid properties of the named litho fluid from the input table in a format
    compatible with elastics_from_stats() in avo_monte_carlo.py
    :param litho_fluid_name:
    :param litho_fluid_table:
    :return:
        Dictionary with the litho fluid properties in a format specific for elastics_from_stats() in avo_monte_carlo.py
    """
    if litho_fluid_name not in litho_fluid_table.source.data['name']:
        raise IOError('Litho fluid {} not present in litho fluid table'.format(litho_fluid_name))
    for i, _litho_fluid_name in enumerate(litho_fluid_table.source.data['name']):
        if _litho_fluid_name == litho_fluid_name:
            # print(litho_fluid_table.source.data['name'][i], litho_fluid_table.source.data['vp'][i])
            t_data = litho_fluid_table.source.data
            return {'VpMean': t_data['vp'][i],
                   'VsMean': t_data['vs'][i],
                   'RhoMean': t_data['rho'][i],
                   'VpStdDev': t_data['vp_std_dev'][i],
                   'VsStdDev': t_data['vs_std_dev'][i],
                   'RhoStdDev': t_data['rho_std_dev'][i],
                   'VpVsCorrCoef': t_data['vp_vs_cc'][i],
                   'VpRhoCorrCoef': t_data['vp_rho_cc'][i],
                   'VsRhoCorrCoef': t_data['vs_rho_cc'][i]}
    return {}

def create_interface_table(litho_fluid_table: bokeh.models.DataTable,
                           _width: int | None = None) -> bokeh.models.DataTable:
    """
    Creates a table that defines which litho fluid classes that should build up the half-space model interface
    :param litho_fluid_table:
        DataTable from create_litho_fluids_table
    :param _width:
    :return:
    """
    if _width is None:
        _width = 400
    n_rows = 4  # more than 4 interfaces makes the plot messy. Max 3 is advisable
    litho_fluids = list(litho_fluid_table.source.data['name'])
    marker_types = ['circle', 'diamond', 'hex', 'inverted_triangle', 'plus', 'square', 'triangle']
    if len(litho_fluids) < 2:
        raise IOError('There must at least be two "litho fluid" classes available to create an interface')
    table_columns = [
        TableColumn(field='use', title='Use',
                    editor=CheckboxEditor(),
                    width=30),
        TableColumn(field='name', title='Name',
                    editor=StringEditor(),
                    formatter=StringFormatter(font_style='bold')),
        TableColumn(field='top', title='Top',
                    editor = SelectEditor(options=litho_fluids)),
        TableColumn(field='base', title='Base',
                    editor = SelectEditor(options=litho_fluids)),
        TableColumn(field='color', title='Color',
                    editor=StringEditor()),
        TableColumn(field='marker', title='Marker',
                    editor=SelectEditor(options=marker_types)),
        TableColumn(field='size', title='Size',
                    editor=NumberEditor())
    ]
    first_name = '{} : {}'.format(litho_fluid_table.source.data['name'][0], litho_fluid_table.source.data['name'][1])
    source_dict = {
        'use': [False] * n_rows,
        'name': [first_name] * n_rows,
        'top': [litho_fluid_table.source.data['name'][0]] * n_rows,
        'base': [litho_fluid_table.source.data['name'][1]] * n_rows,
        'color': ['red'] * n_rows,
        'marker': marker_types[:n_rows],
        'size': [10.] * n_rows
    }
    source_dict['use'][0] = True
    source_dict['color'][0] = 'blue'

    source = ColumnDataSource(source_dict)

    return DataTable(
        source=source,
        columns = table_columns,
        editable = True,
        width = _width,
        height = 140,
        # height_policy = 'max',
        index_position = -1,
        index_header = 'row index',
        index_width = 60)

def get_active_interfaces(interface_table: bokeh.models.DataTable) -> list:
    """
    Returns a list of the [names of] active half-space model interfaces from the input interface_table
    :param interface_table:
    :return:
    """
    active_interfaces = []
    for i, _name in enumerate(interface_table.source.data['name']):
        if interface_table.source.data['use'][i]:
            active_interfaces.append(_name)

    return active_interfaces

def get_interface_litho_fluids(
        interface_name: str,
        interface_table: bokeh.models.DataTable,
        litho_fluids_table: bokeh.models.DataTable) -> dict:
    """
    Returns a dictionary with keys 'top' and 'base', which each consists of a dictionary consistent with the input to
    'elastics_from_stats()' function in avo_monte_carlo.py

    :param interface_name:
    :param interface_table:
    :param litho_fluids_table:
    :return:
    """
    if interface_name not in list(interface_table.source.data['name']):
        raise IOError('Interface {} not present in interface_table'.format(interface_name))
    i, _name = None, None
    for i, _name in enumerate(list(interface_table.source.data['name'])):
        if _name == interface_name:
            break

    _top_name, _base_name, _color, _marker, _size = None, None, None, None, None
    if _name is not None:
        _top_name = interface_table.source.data['top'][i]
        _base_name = interface_table.source.data['base'][i]
        _color = interface_table.source.data['color'][i]
        _marker = interface_table.source.data['marker'][i]
        _size = interface_table.source.data['size'][i]

    return dict(top=get_litho_fluid_from_table(_top_name, litho_fluids_table),
                base=get_litho_fluid_from_table(_base_name, litho_fluids_table),
                color=_color,
                marker=_marker,
                size=_size)


def create_mc_avo_plot(sums_average_file: str | None = None,
                    avg_type: str | None = None):
    """
    Should create a figure, or its sub-elements, that draws the Monte Carlo simulated IxG results for one or
    multiple half-space models (interfaces)
    :return:
    """
    if sums_average_file is None:
        litho_fluids_list = []
        for _name in ['shale', 'brine_sst', 'oil_sst', 'gas_sst']:
            rt = LithoFluid(default=_name)
            litho_fluids_list.append(rt)
    else:
        lfs = LithoFluids()
        lfs.from_excel(sums_average_file, avg_type)
        litho_fluids_list = lfs.litho_fluids


    p = figure(width=600, height=600, tools=tools)
    p.toolbar.logo = None
    p2 = figure(width=700, height=400, tools=tools)
    p2.toolbar.logo = None
    h = figure(width=600, height=200, tools=[PanTool(), WheelZoomTool()])
    h.toolbar.logo = None
    lf_table = create_litho_fluids_table(litho_fluids_list, 800)
    # TODO Try using the new LithoFluidsTable object in core.py instead
    # my_lfs =  LithoFluidsTable(lfs)
    # my_source = my_lfs.source
    # lf_table, add_row, delete_row, update_table = my_lfs.draw(my_source)
    # TODO It doesn't interact well with the other objects. Needs to be checked

    interface_table = create_interface_table(lf_table, 600)

    chi_input = NumericInput(value=26, low=-90, high=90, title="Chi angle:",
                             description='Chi angle between -90 and 90 deg')

    n_iter_input = NumericInput(value=400, low=10, high=1000, title="Iterations:",
                             description='Number of iterations in the Monte Carlo simulation')

    show_std = CheckboxGroup(labels=['Show uncertainty area (1 std)', 'Show uncertainty lines (1 std)'], active=[1])

    def return_chi_line(y_half_range, chi):
        """
        Returns the end points of a line that goes through the
        origo from -y_half_range to +y_half_range

        :param y_half_range:
        :param chi:
            float
            Chi angle in deg
        :return:
        """
        _xl = [-1.1 * y_half_range * np.tan(np.pi * chi / 180.), 1.1 * y_half_range * np.tan(np.pi * chi / 180.)]
        _yl = [1.1 * y_half_range, -1.1 * y_half_range]
        return _xl, _yl

    def calc_eei(_intercept, _gradient, chi_angle):
        return _intercept * np.cos(chi_angle * np.pi / 180.) + _gradient * np.sin(chi_angle * np.pi / 180.)

    def return_data_dicts():
        _active_interfaces = get_active_interfaces(interface_table)
        eei_dict = None
        amp_dict = None
        for i, _interface in enumerate(_active_interfaces):
            print('Interface: {}'.format(_interface))
            interface_dict = get_interface_litho_fluids(_interface, interface_table, lf_table)
            print(interface_dict['color'])
            vp_t, vs_t, rho_t = havo.elastics_from_stats(interface_dict['base'], n_iter_input.value)
            vp_bg, vs_bg, rho_bg = havo.elastics_from_stats(interface_dict['top'], n_iter_input.value)
            intercepts = rp.intercept(vp_bg, vp_t, rho_bg, rho_t)
            gradients = rp.gradient(vp_bg, vp_t, vs_bg, vs_t, rho_bg, rho_t)
            eei = calc_eei(intercepts, gradients, chi_input.value)
            # print('XXX', chi_input.value, eei[:10])
            # Calculate reflection coefficient as a function of theta for all simulated half-space results
            avos = np.full((n_iter_input.value, n_angles), np.nan)
            for j, params in enumerate(zip(vp_bg, vp_t, vs_bg, vs_t, rho_bg, rho_t)):
                avos[j, :] = rp.reflectivity(*params)(theta)

            if i == 0:
                eei_dict = dict(
                    x=intercepts,
                    y=gradients,
                    eei=eei,
                    label=[_interface] * len(intercepts),
                    size=[interface_dict['size']] * len(intercepts),
                    color=[interface_dict['color']] * len(intercepts),
                    marker=[interface_dict['marker']] * len(intercepts)
                )
                amp_dict = dict(
                    amp=np.mean(avos, axis=0),
                    top=np.mean(avos, axis=0) + np.std(avos, axis=0),
                    base=np.mean(avos, axis=0) - np.std(avos, axis=0),
                    label=[_interface] * n_angles,
                    size=[interface_dict['size']] * n_angles,
                    color=[interface_dict['color']] * n_angles,
                    marker=[interface_dict['marker']] * n_angles,
                    theta=theta
                )
            else:
                eei_dict['x'] = np.append(eei_dict['x'], intercepts)
                eei_dict['y'] = np.append(eei_dict['y'], gradients)
                eei_dict['eei'] = np.append(eei_dict['eei'], eei)
                eei_dict['label'] = eei_dict['label'] + [_interface] * len(intercepts)
                eei_dict['size'] = eei_dict['size'] + [interface_dict['size']] * len(intercepts)
                eei_dict['color'] = eei_dict['color'] + [interface_dict['color']] * len(intercepts)
                eei_dict['marker'] = eei_dict['marker'] + [interface_dict['marker']] * len(intercepts)
                amp_dict['amp'] = np.append(amp_dict['amp'], np.mean(avos, axis=0))
                amp_dict['top'] = np.append(amp_dict['top'], np.mean(avos, axis=0) + np.std(avos, axis=0))
                amp_dict['base'] = np.append(amp_dict['base'], np.mean(avos, axis=0) - np.std(avos, axis=0))
                amp_dict['theta'] = np.append(amp_dict['theta'], theta)
                amp_dict['label'] = amp_dict['label'] + [_interface] * n_angles
                amp_dict['size'] = amp_dict['size'] + [interface_dict['size']] * n_angles
                amp_dict['color'] = amp_dict['color'] + [interface_dict['color']] * n_angles
                amp_dict['marker'] = amp_dict['marker'] + [interface_dict['marker']] * n_angles
        return eei_dict, amp_dict

    # Instantiate the plot with the first active interface
    data, amps = return_data_dicts()
    # data = return_data_dicts()

    source = ColumnDataSource(data)
    amp_source = ColumnDataSource(amps)
    # Calculate histogram
    bins = np.linspace(-0.3, 0.3, 40)
    def draw_histogram():
        h.legend.items = []
        h.renderers = []
        # groups = list(set(source.data['label']))
        # colors = list(set(source.data['color']))
        groups = list(dict.fromkeys(source.data['label']))
        colors = list(dict.fromkeys(source.data['color']))
        for i, group in enumerate(groups):
            _x = source.data['eei'][[_l == group for _l in source.data['label']]]
            hist, edges = np.histogram(_x, density=True, bins=bins)
            h.quad(top=hist, bottom=0, fill_color=colors[i], alpha=0.7,
                   line_color='white', left=edges[:-1], right=edges[1:], legend_label=group)
        h.legend.title = 'Chi: {}'.format(chi_input.value)
        h.legend.click_policy = 'hide'
        h.xaxis.axis_label = 'EEI at Chi {}'.format(chi_input.value)
    draw_histogram()

    p.scatter(
        x='x',
        y='y',
        source=source,
        legend_field='label',
        size='size',
        color='color',
        marker='marker',
        alpha=0.7)

    p.xaxis.axis_label = 'Intercept'
    p.yaxis.axis_label = 'Gradient'
    p.legend.title = 'Chi: {}'.format(chi_input.value)
    for span in ['width', 'height']:
        p.add_layout(Span(location=0., dimension=span, line_width=1, line_color='black', line_dash='dashed'))

    glyph = VArea(x="theta", y1="top", y2="base", fill_color='gray', fill_alpha= 0.3)
    p2.add_glyph(amp_source, glyph)
    top_lines = p2.scatter(x='theta', y='top', source=amp_source, color='color', marker='dash')
    base_lines = p2.scatter(x='theta', y='base', source=amp_source, color='color', marker='dash')
    p2.scatter(
        x='theta',
        y='amp',
        source=amp_source,
        legend_field='label',
        size='size',
        color='color',
        marker='marker',
        alpha=0.7)
    p2.xaxis.axis_label = 'Incident angle [deg.]'
    p2.yaxis.axis_label = '(Mean) reflectivity'
    p2.add_layout(Span(location=0., dimension='width', line_width=1, line_color='black', line_dash='dashed'))

    show_std_callback = CustomJS(args=dict(vareas=glyph, top_line=top_lines, base_line=base_lines, checkbox=show_std), code="""
        const show_area = checkbox.active.includes(0) === true ? 1.0 : 0.0
        vareas.fill_alpha = show_area * 0.3
        top_line.visible = checkbox.active.includes(1)
        base_line.visible = checkbox.active.includes(1)
        console.log('Data changed:', show_area, checkbox.active.includes(1))
    """)

    show_std.js_on_change('active', show_std_callback)

    def callback():
        # _data = return_data_dicts()
        _data, _amps = return_data_dicts()
        source.data = _data
        # source.data.label = _data['label']
        amp_source.data = _amps
        # amp_source.data.label = _amps['label']
        draw_histogram()
        print(source.data['color'][0], source.data['color'][-1])


    run_button = Button(label="Run", button_type="success")
    run_button.on_click(callback)

    y_range = np.max([np.abs(np.min(source.data['y'])), np.abs(np.max(source.data['y']))])
    x_range = np.max([np.abs(np.min(source.data['x'])), np.abs(np.max(source.data['x']))])

    # Plot initial Chi projection line
    _x, _y = return_chi_line( y_range, chi_input.value)
    line_data = dict( x=_x, y=_y )
    line_source = ColumnDataSource(line_data)
    p.line( x='x', y='y', source=line_source)

    def chi_callback(attr, old, new):
        # print(chi_input.value, attr, old, new)
        _x, _y = return_chi_line(
            y_range, chi_input.value)
        line_source.data = dict(x=_x, y=_y)
        source.data['eei'] = calc_eei(source.data['x'], source.data['y'], chi_input.value)
        draw_histogram()
        p.legend.title = 'Chi: {}'.format(chi_input.value)

    p.y_range.start = -1.05 * y_range
    p.y_range.end = 1.05 * y_range
    p.x_range.start = -1.05 * x_range
    p.x_range.end = 1.05 * x_range

    chi_input.on_change('value', chi_callback)

    return h, p, p2, interface_table, lf_table, run_button, chi_input, n_iter_input, show_std








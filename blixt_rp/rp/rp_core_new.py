from cmath import acosh

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
# from dataclasses import dataclass
from copy import deepcopy
import os, sys
from typing import Literal
import bruges.rockphysics.rockphysicsmodels as brr

from bokeh.models import ColumnDataSource, StringFormatter, NumberEditor, NumberFormatter
# from pandas.io.formats.string import StringFormatter

from .. import ureg, Q_
import pint

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.basename(__file__).replace('blixt_rp\\blixt_rp\\rp', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp_utils.definitions as udo
import blixt_utils.io.io as bui
from blixt_utils.plotting import crossplot as xp
from blixt_utils.plotting import helpers as uhelp

logger = logging.getLogger(__name__)

class LithoFluid:
    """
    Class containing the elastic properties, and the statistics describing it, for one single "litho fluid"
    (an element that describes a specific lithology with a specific fluid content)
    """
    def __init__(self,
                 name: str | None = None,
                 vp: float | pint.Quantity | None = None,
                 vs: float | pint.Quantity | None = None,
                 rho: float | pint.Quantity | None = None,
                 vp_std_dev: float | pint.Quantity | None = None,
                 vs_std_dev: float | pint.Quantity | None = None,
                 rho_std_dev: float | pint.Quantity | None = None,
                 vp_vs_cc: float | pint.Quantity | None = None,
                 vp_rho_cc: float | pint.Quantity | None = None,
                 vs_rho_cc: float | pint.Quantity | None = None,
                 default: str | None = None
                 ):
        self.name = name
        # Avoid integer input
        vp = float(vp) if isinstance(vp, int) else vp
        vs = float(vs) if isinstance(vs, int) else vs
        rho = float(rho) if isinstance(rho, int) else rho
        vp_std_dev = float(vp_std_dev) if isinstance(vp_std_dev, int) else vp_std_dev
        vs_std_dev = float(vs_std_dev) if isinstance(vs_std_dev, int) else vs_std_dev
        rho_std_dev = float(rho_std_dev) if isinstance(rho_std_dev, int) else rho_std_dev



        # Give default units if not specified
        self.vp = Q_(vp, 'm/s') if isinstance(vp, float) else (None if vp is None else vp)
        self.vs = Q_(vs, 'm/s') if isinstance(vs, float) else (None if vs is None else vs)
        self.rho = Q_(rho, 'g/cm**3') if isinstance(rho, float) else (None if rho is None else rho)
        self.vp_std_dev = Q_(vp_std_dev, 'm/s') if isinstance(vp_std_dev, float) else (None if vp_std_dev is None else vp_std_dev)
        self.vs_std_dev = Q_(vs_std_dev, 'm/s') if isinstance(vs_std_dev, float) else (None if vs_std_dev is None else vs_std_dev)
        self.rho_std_dev= Q_(rho_std_dev, 'g/cm**3') if isinstance(rho_std_dev, float) else (None if rho_std_dev is None else rho_std_dev)
        self.vp_vs_cc = Q_(vp_vs_cc) if isinstance(vp_vs_cc, float) else (None if vp_vs_cc is None else vp_vs_cc)
        self.vp_rho_cc = Q_(vp_rho_cc) if isinstance(vp_rho_cc, float) else (None if vp_rho_cc is None else vp_rho_cc)
        self.vs_rho_cc = Q_(vs_rho_cc) if isinstance(vs_rho_cc, float) else (None if vs_rho_cc is None else vs_rho_cc)

        if default is not None:
            if default == 'shale':
                self.name = default
                self.vp = Q_(3445, 'm/s')
                self.vs = Q_(1767, 'm/s')
                self.rho = Q_(2.6, 'g/cm**3')
                self.vp_std_dev = Q_(208, 'm/s')
                self.vs_std_dev = Q_(171, 'm/s')
                self.rho_std_dev = Q_(0.036123123, 'g/cm**3')
                self.vp_vs_cc = Q_(0.89)
                self.vp_rho_cc = Q_(0.08)
                self.vs_rho_cc = Q_(-0.07)
            elif default == 'brine_sst':
                self.name = default
                self.vp = Q_(3675, 'm/s')
                self.vs = Q_(2086, 'm/s')
                self.rho = Q_(2.4, 'g/cm**3')
                self.vp_std_dev = Q_(260, 'm/s')
                self.vs_std_dev = Q_(188, 'm/s')
                self.rho_std_dev = Q_(0.092, 'g/cm**3')
                self.vp_vs_cc = Q_(0.92)
                self.vp_rho_cc = Q_(0.62)
                self.vs_rho_cc = Q_(0.65)
            elif default == 'oil_sst':
                self.name = default
                self.vp = Q_(3587, 'm/s')
                self.vs = Q_(2097, 'm/s')
                self.rho = Q_(2.3, 'g/cm**3')
                self.vp_std_dev = Q_(287, 'm/s')
                self.vs_std_dev = Q_(187, 'm/s')
                self.rho_std_dev = Q_(0.098, 'g/cm**3')
                self.vp_vs_cc = Q_(0.91)
                self.vp_rho_cc = Q_(0.58)
                self.vs_rho_cc = Q_(0.63)
            elif default == 'gas_sst':
                self.name = default
                self.vp = Q_(3535, 'm/s')
                self.vs = Q_(2127, 'm/s')
                self.rho = Q_(2.26, 'g/cm**3')
                self.vp_std_dev = Q_(307, 'm/s')
                self.vs_std_dev = Q_(181, 'm/s')
                self.rho_std_dev = Q_(0.12, 'g/cm**3')
                self.vp_vs_cc = Q_(0.89)
                self.vp_rho_cc = Q_(0.49)
                self.vs_rho_cc = Q_(0.58)
            else:
                raise IOError('default must be either "shale", "brine_sst", "oil_sst" or "gas_sst", not {}'.format(default))


    def to_sums_dict(self) -> dict:
        """
        Returns a dictionary which is compatible with the RokDoc "sums and average" file format
        :return:
        """
        return_dict = {'VpMean': self.vp.to('m/s').magnitude,
                       'VsMean': self.vs.to('m/s').magnitude,
                       'RhoMean': self.rho.to('g/cm**3').magnitude,
                       'VpStdDev': self.vp_std_dev.to('m/s').magnitude,
                       'VsStdDev': self.vs_std_dev.to('m/s').magnitude,
                       'RhoStdDev': self.rho_std_dev.to('g/cm**3').magnitude,
                       'VpVsCorrCoef': self.vp_vs_cc.magnitude,
                       'VpRhoCorrCoef': self.vp_rho_cc.magnitude,
                       'VsRhoCorrCoef': self.vs_rho_cc.magnitude}
        return return_dict



    def from_excel(self,
                   excel_file: str,
                   name: str,
                   avg_type: str | None = None):
        """
        Reads litho fluid properties from a RokDoc 'sums and averages' excel file
        :param excel_file:
        :param name:
        :param avg_type:
        :return:
        """
        postfix = ''
        if avg_type is None:
            avg_type = 'mean'
        if avg_type == 'mean':
            postfix = 'Mean'
        elif avg_type == 'median':
            postfix = 'Median'
        elif avg_type == 'mode':
            postfix = 'Mode'
        else:
            raise IOError('avg_type can be either "mean", "median" or "mode", not {}'.format(avg_type))

        all_data = bui.read_sums_and_averages(excel_file)

        self.name = name
        self.vp = Q_(all_data[name]['Vp'+postfix], 'm/s')
        self.vs = Q_(all_data[name]['Vs'+postfix], 'm/s')
        self.rho= Q_(all_data[name]['Rho'+postfix], 'g/cm**3')
        self.vp_std_dev = Q_(all_data[name]['VpStdDev'], 'm/s')
        self.vs_std_dev = Q_(all_data[name]['VsStdDev'], 'm/s')
        self.rho_std_dev= Q_(all_data[name]['RhoStdDev'], 'g/cm**3')
        self.vp_vs_cc = Q_(all_data[name]['VpVsCorrCoef'])
        self.vp_rho_cc = Q_(all_data[name]['VpRhoCorrCoef'])
        self.vs_rho_cc= Q_(all_data[name]['VsRhoCorrCoef'])

    def plot(self,
             ax: mpl.axes._axes.Axes | None = None,
             label: str | None = None,
             color: str | None = None):
        """
        Plot this litho fluid in a AI vs Vp/Vs plot with its statistics
        :param ax:
            matplot
        :param label:
        :param color:
        :return:

        """
        import blixt_rp.rp_utils.avo_monte_carlo as havo
        if label is None:
            label = self.name
        if color is None:
            color = 'b'
        if ax is None:
            fig, ax = plt.subplots()

        n_samps = 1000
        vp, vs, rho = havo.elastics_from_stats(self.to_sums_dict(), n_samps)

        x_data = vp * rho
        x_unit = 'g/cc m/s'
        y_data = vp / vs
        y_unit = '-'
        xtempl = {'full_name': 'AI', 'unit': x_unit}
        ytempl = {'full_name': 'Vp/Vs', 'unit': y_unit}


        _ = xp.plot(
            x_data,
            y_data,
            cdata=color,
            cbar=False,
            xtempl=xtempl,
            ytempl=ytempl,
            edge_color=False,
            ax=ax
        )
        uhelp.confidence_ellipse(x_data, y_data, ax, n_std=1.0, edgecolor='k')
        uhelp.confidence_ellipse(x_data, y_data, ax, n_std=2.0, edgecolor='k', linestyle='--')

    def to_model_layer(self,
                       number,
                       case: str | None = None,
                       color: str | None = None,
                       thickness: pint.Quantity = Q_(25., 'm')):
        from blixt_rp.core.models import ModelLayer
        return ModelLayer(number, case, color, thickness, self)


class LithoFluids:
    """
    Class containing the properties of the different litho fluids that can be modelled
    """
    def __init__(self,
                 litho_fluids: list | None = None
                 ):
        if litho_fluids is None:
            litho_fluids = []
        self._litho_fluids = litho_fluids

    @property
    def litho_fluids(self):
        return self._litho_fluids

    @litho_fluids.setter
    def litho_fluids(self, value: list):
       self._litho_fluids = value

    def __add__(self, other: LithoFluid):
        self._litho_fluids.append(other)

    def from_excel(self,
                   excel_file: str,
                   avg_type: str | None = None):
        """
        Reads litho fluid properties from a RokDoc 'sums and averages' excel file, and adds them to the list of
        LithoFluids
        :param excel_file:
        :param avg_type:
        :return:
        """
        postfix = ''
        if avg_type is None:
            avg_type = 'mean'
        if avg_type == 'mean':
            postfix = 'Mean'
        elif avg_type == 'median':
            postfix = 'Median'
        elif avg_type == 'mode':
            postfix = 'Mode'
        else:
            raise IOError('avg_type can be either "mean", "median" or "mode", not {}'.format(avg_type))

        all_data = bui.read_sums_and_averages(excel_file)

        all_litho_fluids = []
        for _name in list(all_data.keys()):
            all_litho_fluids.append(
                LithoFluid(
                    name=_name,
                    vp=Q_(all_data[_name]['Vp'+postfix], 'm/s'),
                    vs=Q_(all_data[_name]['Vs'+postfix], 'm/s'),
                    rho=Q_(all_data[_name]['Rho'+postfix], 'g/cm**3'),
                    vp_std_dev=Q_(all_data[_name]['VpStdDev'], 'm/s'),
                    vs_std_dev=Q_(all_data[_name]['VsStdDev'], 'm/s'),
                    rho_std_dev=Q_(all_data[_name]['RhoStdDev'], 'g/cm**3'),
                    vp_vs_cc=Q_(all_data[_name]['VpVsCorrCoef']),
                    vp_rho_cc=Q_(all_data[_name]['VpRhoCorrCoef']),
                    vs_rho_cc=Q_(all_data[_name]['VsRhoCorrCoef'])
                )
            )
            self.litho_fluids = all_litho_fluids

class LithoFluidsTable:
    """
    Returns a table of "litho-fluids" that are useful for populating a model
    """
    def __init__(self,
                 litho_fluids: LithoFluids | None = None,
                 advanced: bool = True,
                 width: int | None = None,
                 height: int | None = None):
        """

        :param litho_fluids:
            LithoFluids object
        :param width:
            int
        """
        if width is None:
            width = 700
        self.width = width
        if height is None:
            height = 135
        self.height = height
        if litho_fluids is None:
            litho_fluids = LithoFluids()
        self.litho_fluids = litho_fluids
        if advanced:
            self.keys = ['name', 'vp', 'vs', 'rho',
                         'vp_std_dev', 'vs_std_dev', 'rho_std_dev',
                         'vp_vs_cc', 'vp_rho_cc', 'vs_rho_cc']
        else:
            self.keys = ['name', 'vp', 'vs', 'rho']
        self.advanced = advanced

    @property
    def cds(self) -> ColumnDataSource:
        _dict = {_x:[] for _x in self.keys}
        for _litho_fluid in self.litho_fluids.litho_fluids:
            for _key in self.keys:
                if _key == 'name':
                    _dict['name'].append(_litho_fluid.name)
                else:
                    _dict[_key].append(_litho_fluid.__dict__[_key].magnitude)
        return ColumnDataSource(_dict)

    @property
    def litho_fluid_names(self):
        _litho_fluid_names = []
        for _litho_fluid in self.litho_fluids.litho_fluids:
            _this_name = _litho_fluid.name
            if _this_name not in _litho_fluid_names:
                _litho_fluid_names.append(_this_name)
        return _litho_fluid_names

    def table_columns(self):
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, CheckboxEditor)
        column_names = [_s.capitalize().replace('_', ' ') for _s in self.keys]
        table_columns = []
        for i, column_key in enumerate(self.keys):
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

        return table_columns

    def draw(self,
             cds: ColumnDataSource):
        from bokeh.models import DataTable, Button, CheckboxGroup

        def add_row_function():
            new_data = dict(cds.data)
            lf = LithoFluid(default='brine_sst')
            for _key in list(new_data.keys()):
                if _key == 'name':
                    new_data[_key].append('brine sst')
                else:
                    new_data[_key].append(lf.__dict__[_key])
            cds.data = new_data

        def delete_row_function():
            selected_index = cds.selected.indices
            new_data = {_x:[] for _x in self.keys}
            for _i in range(len(cds.data['name'])):
                if _i  in selected_index:
                    continue
                for _x in self.keys:
                    new_data[_x].append(cds.data[_x][_i])
            cds.selected.indices = []
            cds.data = new_data

        def update_table_function():
            new_data = dict(cds.data)
            _litho_fluids = []
            for _i in range(len(new_data['name'])):
                # print(' - Iteration: {} of {}'.format( _i, len(new_data['name'])))
                if self.advanced:
                    this_litho_fluid = LithoFluid(
                        name=new_data['name'][_i],
                        vp=Q_(new_data['vp'][_i], 'm/s'),
                        vs=Q_(new_data['vs'][_i], 'm/s'),
                        rho=Q_(new_data['rho'][_i], 'gram / cm^3'),
                        vp_std_dev=Q_(new_data['vp_std_dev'][_i], 'm/s'),
                        vs_std_dev=Q_(new_data['vs_std_dev'][_i], 'm/s'),
                        rho_std_dev=Q_(new_data['rho_std_dev'][_i], 'gram / cm^3'),
                        vp_vs_cc=Q_(new_data['vp_vs_cc'][_i], ''),
                        vp_rho_cc=Q_(new_data['vp_rho_cc'][_i], ''),
                        vs_rho_cc=Q_(new_data['vs_rho_cc'][_i], '')
                    )
                else:
                    this_litho_fluid = LithoFluid(
                        name=new_data['name'][_i],
                        vp=Q_(new_data['vp'][_i], 'm/s'),
                        vs=Q_(new_data['vs'][_i], 'm/s'),
                        rho=Q_(new_data['rho'][_i], 'gram / cm^3')
                    )

                _litho_fluids.append(this_litho_fluid)

            self.litho_fluids.litho_fluids = _litho_fluids

            cds.data = new_data

        dt = DataTable(
            source=cds,
            columns=self.table_columns(),
            editable=True,
            width=self.width,
            height=self.height,
            index_position = -1,
            index_header = 'index'
        )

        add_row = Button(label='Add row', button_type='success')
        add_row.on_click(add_row_function)

        delete_row = Button(label='Delete selected rows', button_type='success')
        delete_row.on_click(delete_row_function)

        update_table = Button(label='Update', button_type='success')
        update_table.on_click(update_table_function)

        return dt, add_row, delete_row, update_table


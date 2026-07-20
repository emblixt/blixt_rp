# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:31:28 2020
Module for handling rockphysics fluid models
This version uses the pint library instead of selfmade Param

NOTE, THIS FUNCTIONALITY WILL BE REPLACED WITH THE NEWER fluid_and_minerals.py

:copyright: Erik Mårten Blixt (marten.blixt@gmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)       
@author: mblixt
"""
import logging
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import unittest
import os, sys
from typing import Literal, List, Any
from pathlib import Path
from dataclasses import dataclass

from bokeh.io import output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import column, row
import pint

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp import Q_
import blixt_rp.rp.rp_core as rp
from blixt_rp.core.well_new import Well
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.utils import isnan, print_info
from blixt_rp.core.core import Interval, Intervals


logger = logging.getLogger(__name__)

def_fluids = dict(
    # From RokDoc
    brine=dict(
        rho=Q_(1.0, 'G/cc'), vp=Q_(1600.0, 'm/s'), k=Q_(2.56, 'GPa'), vs=Q_(0.0, 'm/s'), mu=Q_(0.0, 'GPa')),
    oil=dict(
        rho=Q_(0.8, 'G/cc'), vp=Q_(1200.0, 'm/s'), k=Q_(1.152, 'GPa'), vs=Q_(0.0, 'm/s'), mu=Q_(0.0, 'GPa')),
    gas=dict(
        rho=Q_(0.15, 'G/cc'), vp=Q_(500.0, 'm/s'), k=Q_(0.0375, 'GPa'), vs=Q_(0., 'm/s'), mu=Q_(0.0, 'GPa')),
    condensate=dict(
        rho=Q_(0.45, 'G/cc'), vp=Q_(850.0, 'm/s'), k=Q_(0.325125, 'GPa'), vs=Q_(0.0, 'm/s'), mu=Q_(0.0, 'GPa')),
    co2=dict(
        rho=Q_(0.15, 'G/cc'), vp=Q_(500.0, 'm/s'), k=Q_(0.0375, 'GPa'), vs=Q_(0.0, 'm/s'), mu=Q_(0.0, 'GPa')),
    heavy_oil=dict(
        rho=Q_(1.1, 'G/cc'), vp=Q_(1600.0, 'm/s'), k=Q_(2.7573, 'GPa'), vs=Q_(200.0, 'm/s'), mu=Q_(0.044, 'GPa'))
)

def_fluid_vals = dict(
    k=Q_(2, 'GPa'),
    mu=Q_(2, 'GPa'),
    rho=Q_(2, 'gram / cm^3'),
    calculation_method='Batzle and Wang',
    temp_gradient=Q_(0.03, 'degC/m'),
    # temp_ref=Q_(4.0, 'degC'),
    pressure_gradient=Q_(0.0107, 'MPa / m'),
    # pressure_ref=Q_(0, 'MPa'),
    salinity=Q_(70000.0, 'ppm'),
    gor=Q_(1., ''),
    oil_api=Q_(30., ''),
    gas_gravity=Q_(0.6, ''),
    gas_mixing='Brie',
    brie_exponent=Q_(2., ''),
    status=None
)

def get_duplicates(_list: List[str]):
    """
    Finds duplicates in input list and returns the duplicates
    :param _list:
    :return:
    """
    unique_list = []
    dupl_list = []
    for _x in _list:
        if _x not in unique_list:
            unique_list.append(_x)
        else:
            dupl_list.append(_x)
    return dupl_list

def read_all_mother_fluids_from_excel(filename, fluid_sheet: str = 'Fluids', fluid_header: int = 1) -> dict:
    # First read in all mother_fluids defined in the project table
    all_mother_fluids = {}
    fluids_table = pd.read_excel(filename,
                                 sheet_name=fluid_sheet, header=fluid_header, engine='openpyxl')

    for i, name in enumerate(fluids_table['Name']):
        if isnan(name):
            continue  # Avoid empty lines
        this_fluid = MotherFluid(
            calculation_method='User specified' if isnan(fluids_table['Calculation method'][i]) else fluids_table['Calculation method'][i],
            k=Q_(float(fluids_table['Bulk moduli [GPa]'][i]), 'GPa'),
            mu=Q_(float(fluids_table['Shear moduli [GPa]'][i]), 'GPa'),
            rho=Q_(float(fluids_table['Density [g/cm3]'][i]), 'gram / cm^3'),
            temp_gradient=Q_(float(fluids_table['T gradient [deg C/m]'][i]), 'degC / m'),
            # temp_ref=Q_(float(fluids_table['T ref [C]'][i]), 'degC'),
            pressure_gradient=Q_(float(fluids_table['P gradient [MPa/m]'][i]), 'MPa / m'),
            # pressure_ref=Q_(float(fluids_table['P ref [MPa]'][i]), 'MPa'),
            salinity=Q_(float(fluids_table['Salinity [ppm]'][i]), 'ppm'),
            gor=Q_(float(fluids_table['GOR'][i]), ''),
            oil_api=Q_(float(fluids_table['Oil API'][i]), ''),
            gas_gravity=Q_(float(fluids_table['Gas gravity'][i]), ''),
            gas_mixing=fluids_table['Gas mixing'][i],
            brie_exponent=Q_(float(fluids_table['Brie exponent'][i]), ''),
            name=name.lower(),
        )
        all_mother_fluids[name.lower()] = this_fluid

    return all_mother_fluids

def read_fluidmixes_from_excel_test(
        filename: str,
        intervals: Intervals,
        fluid_sheet: str = 'Fluids', fluid_header: int = 1,
        mix_sheet: str = 'Fluid mixtures', mix_header: int = 1):
    """
    The new version tries to collect all single fluids within a well, interval and substitution order
    into one VRH averaged fluid

    NOTE, it requires that all fluids that share the same substitution order (e.g. 'Initial') within a well and an interval
    comes in consecutive rows in the excel sheet

    :param filename:
    :param intervals:
        Intervals object
    :param fluid_sheet:
    :param fluid_header:
    :param mix_sheet:
    :param mix_header:
    :return:
    """

    # First read in all mother_fluids defined in the project table
    all_mother_fluids = read_all_mother_fluids_from_excel(filename, fluid_sheet, fluid_header)

    # Emtpy list of fluids
    fluids = []

    # # Then read in the fluid mixes
    # fluid_mixes = {
    #     'initial': {},
    #     'final': {}
    # }

    # Read the fluid mixes sheet
    mix_table = pd.read_excel(filename, sheet_name=mix_sheet, header=mix_header, engine='openpyxl')
    last_subst_order = 'None'
    fluid_group = []
    volume_fraction_group = []
    fluid_type_group = []

    # Loop over all rows in the excel file
    j = 0
    this_subst = None
    for i, name in enumerate(mix_table['Fluid name']):
        if mix_table['Use'][i] != 'Yes':
            continue

        vf = mix_table['Volume fraction'][i]
        # When 'volume_fraction' is a string, make it low case
        try:
            vf = vf.lower()
        except AttributeError:
            pass
        if isnan(vf):
            continue  # Avoid mother_fluids where the volume fraction is not set

        ftype = mix_table['Fluid type'][i]
        if not isinstance(ftype, str) and np.isnan(ftype):
            ftype = ''
        ftype = ftype.lower()
        if ' ' in ftype:
            ftype = ftype.replace(' ', '_')

        # # Need to pair fluid name with fluid type to get unique fluid names in fluid mixture
        # this_name = '{}_{}'.format(name.lower(), '' if isnan(ftype) else ftype.lower())
        this_well = mix_table['Well name'][i].lower()
        this_wi = mix_table['Interval name'][i].lower()
        this_mother_fluid = deepcopy(all_mother_fluids[name.lower()])
        try:
            this_tag = mix_table['Tag'][i]
        except KeyError:
            this_tag = None

        # Try to extract the Interval from the given Intervals
        this_interval = intervals.get_interval(this_wi, this_well)
        if this_interval is None:
            print_info('Interval: {} was not found in well: {}'.format(this_wi, this_well), 'warning', logger)
            continue

        this_subst = mix_table['Substitution order'][i].lower()
        print(i, j, this_subst, this_well, this_wi, name)

        # Whenever substitution order changes it means that we encounter a new fluid
        if this_subst != last_subst_order.lower():
            # New fluid
            if j > 0:
                # We are not on the first line, and need to combine all the components of the last fluid (group)
                # into one fluid
                print(' These {} fluids are to be combined and saved: '.format(last_subst_order))
                for _f, _t, _v in zip(fluid_group, fluid_type_group, volume_fraction_group):
                    print('   {}, {}, {}'.format(_f.name, _t, _v))
            fluid_group = [this_mother_fluid]
            fluid_type_group = [ftype]
            volume_fraction_group = [vf]
        else:
            fluid_group.append(this_mother_fluid)
            fluid_type_group.append(ftype)
            volume_fraction_group.append(vf)

        last_subst_order = this_subst
        j += 1
    # Also save the last groups
    print(' These {} fluids are to be combined and saved: '.format(this_subst))
    for _f, _t, _v in zip(fluid_group, fluid_type_group, volume_fraction_group):
        print('   {}, {}, {}'.format(_f.name, _t, _v))

    return fluids

def read_fluidmixes_from_excel(
        filename: str,
        intervals: Intervals,
        fluid_sheet: str = 'Fluids', fluid_header: int = 1,
        mix_sheet: str = 'Fluid mixtures', mix_header: int = 1):
    """

    :param filename:
    :param intervals:
        Intervals object
    :param fluid_sheet:
    :param fluid_header:
    :param mix_sheet:
    :param mix_header:
    :return:
    """

    # First read in all mother_fluids defined in the project table
    all_mother_fluids = read_all_mother_fluids_from_excel(filename, fluid_sheet, fluid_header)

    # Emtpy list of fluids
    fluids = []

    # # Then read in the fluid mixes
    # fluid_mixes = {
    #     'initial': {},
    #     'final': {}
    # }

    # Read the fluid mixes sheet
    mix_table = pd.read_excel(filename, sheet_name=mix_sheet, header=mix_header, engine='openpyxl')
    for i, name in enumerate(mix_table['Fluid name']):
        if mix_table['Use'][i] != 'Yes':
            continue

        vf = mix_table['Volume fraction'][i]
        # When 'volume_fraction' is a string, make it low case
        try:
            vf = vf.lower()
        except AttributeError:
            pass


        ftype = mix_table['Fluid type'][i]
        if not isinstance(ftype, str) and np.isnan(ftype):
            ftype = ''
        ftype = ftype.lower()
        if ' ' in ftype:
            ftype = ftype.replace(' ', '_')

        # Need to pair fluid name with fluid type to get unique fluid names in fluid mixture
        this_name = '{}_{}'.format(name.lower(), '' if isnan(ftype) else ftype.lower())
        if isnan(vf):
            continue  # Avoid mother_fluids where the volume fraction is not set
        this_subst = mix_table['Substitution order'][i].lower()
        this_well = mix_table['Well name'][i].lower()
        this_wi = mix_table['Interval name'][i].lower()
        this_mother_fluid = deepcopy(all_mother_fluids[name.lower()])
        # Why should we rename the mother fluid??
        # this_mother_fluid.name = this_name
        try:
            this_tag = mix_table['Tag'][i]
        except KeyError:
            this_tag = None

        # Try to extract the Interval from the given Intervals
        this_interval = intervals.get_interval(this_wi, this_well)
        if this_interval is None:
            print_info('Interval: {} was not found in well: {}'.format(this_wi, this_well), 'warning', logger)
            continue

        fluids.append(
            Fluid(
                mother_fluid=this_mother_fluid,
                interval=this_interval,
                substitution_order=this_subst,
                fluid_type=ftype,
                volume_fraction=vf,
                status='from excel',
                tag=this_tag
            )
        )

    return fluids


class Header(AttribDict):
    """
    Class for fluid header information
    A ``Header`` object may contain all header information (also known as meta
    data) of a Fluid object. 
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every variable import and export modules
    
    :param
        header: Dictionary containing meta information of a single
        Fluid object. 
        Possible keywords are
        summarized in the following `Default Attributes`_ section.
    
    .. rubric:: _`Default Attributes`
    
    """
    defaults = {
        'name': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat(),
        'modification_date': None,
        'modification_history': '',
        }

    def __init__(self, header=None):
        """
        """
        if header is None:
            header = {}
        super(Header, self).__init__(header)

    def __setitem__(self, key, value):
        """
        """
        # keys which shouldn't be modified
        if key in ['creation_date', 'modification_date']:
            pass
        super(Header, self).__setitem__('modification_date', datetime.now().isoformat())

        # all other keys
        if isinstance(value, dict):
            super(Header, self).__setitem__(key, AttribDict(value))
        else:
            super(Header, self).__setitem__(key, value)

    __setattr__ = __setitem__

    def __str__(self):
        """
        Return better readable string representation of Header object.
        """
        keys = list(self.keys())
        return self._pretty_str(keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

class MotherFluid(object):
    """
    A "mother fluid" is a set of fluid properties that can be used to create different kind of fluids (e.g. oil, brine, ...)
    Or it can be fixed with a given set of k, mu and rho
    """
    def __init__(self,
                 calculation_method: str | None = None,  # or Batzle and Wang',  # or 'User specified'
                 k: pint.Quantity | None = None,  # Bulk modulus in GPa
                 mu: pint.Quantity | None = None,  # Shear modulus in GPa
                 rho: pint.Quantity | None = None,  # Density in g/cm3
                 temp_gradient: pint.Quantity | None = None,
                 pressure_gradient: pint.Quantity | None = None,
                 salinity: pint.Quantity | None = None,
                 gor: pint.Quantity | None = None,
                 oil_api: pint.Quantity | None = None,
                 gas_gravity: pint.Quantity | None = None,
                 gas_mixing: str | None = None,
                 brie_exponent: pint.Quantity | None = None,
                 name: str | None = 'Default',
                 header: dict | None = None):

        if header is None:
            header = {}
        if name is not None:
            header['name'] = name
        elif 'name' not in list(header.keys()):
            header['name'] = None

        self.header = Header(header)
        self.calculation_method = calculation_method
        self._k = k
        self._mu = mu
        self._rho = rho
        self._bd = None  # Burial depth
        self.temp_gradient = temp_gradient
        self.pressure_gradient = pressure_gradient
        self.salinity = salinity
        self.gor = gor
        self.oil_api = oil_api
        self.gas_gravity = gas_gravity
        self.gas_mixing = gas_mixing
        self.brie_exponent = brie_exponent
        self._name = header['name']
        # self.status = status

    def __str__(self):
        keys = list(self.__dict__.keys())

        pattern = "%%%ds: %%s" % len(keys)

        head = [ pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.header['name'] = value

    def print_fluid(self, verbose=False):
        out = '  {}\n'.format(self.name)
        if verbose:
            out = str(self)
        else:
            out += '      K: {}, Mu: {}, Rho {}\n'.format(
                self.k, self.mu, self.rho)
            out += '      Calculation method: {}\n'.format(self.calculation_method)
        return out

    def keys(self):
        return self.__dict__.keys()

    @property
    def k(self):
        return self._k

    @property
    def mu(self):
        return self._mu

    @property
    def rho(self):
        return self._rho

    @property
    def bd(self):
        return self._bd

    def calc_elastics(self, fluid_type: str, burial_depth: Q_):
        if self.calculation_method == 'Batzle and Wang':
            _this_rho = None
            _this_k = None
            _s = self.salinity.to('ppm').magnitude
            # _p = self.pressure_ref + self.pressure_gradient * burial_depth
            # We ignore the pressure reference now. Will be calculated more exactly in FluidMix
            _p = Q_(0.0, 'MPa') + self.pressure_gradient * burial_depth
            _p = _p.to('MPa').magnitude
            # _t = self.temp_ref + self.temp_gradient * burial_depth
            _t = Q_(4.0, 'degC') + self.temp_gradient * burial_depth
            _t = _t.to('degC').magnitude
            if fluid_type.lower() == 'brine':
                _this_rho = rp.rho_b(_s, _p,  _t).value
                v_p_b = rp.v_p_b(_s, _p, _t).value
                _this_k = v_p_b**2 * _this_rho * 1.E-6
                calc_success = True
            elif fluid_type.lower() == 'oil':
                _this_k, _this_rho = rp.k_and_rho_o(
                    self.oil_api.magnitude,
                    self.gas_gravity.magnitude,
                    self.gor.magnitude,
                    _p,
                    _t
                )
                calc_success = True
            elif fluid_type.lower() == 'gas':
                _this_k, _this_rho = rp.k_and_rho_g(self.gas_gravity.magnitude, _p, _t)
                calc_success = True
            else:
                calc_success = False
                print_info('No calculation done for fluid type {}'.format(fluid_type),
                           'warning', logger)
            if calc_success:
                self._bd = burial_depth
                self._k = Q_(_this_k, 'GPa')
                self._mu = Q_(np.nan, 'GPa')
                self._rho = Q_(_this_rho, 'GPa')

class FluidsTable:
    """
    Returns a table and bokeh CDS for the provided mother_fluids
    """
    def __init__(self,
                 mother_fluids: list | None = None,
                 title: str | None = 'Fluids',
                 width: int | None = None,
                 height: int | None = None,
                 default_fluids_only: bool = False):
        """

        :param mother_fluids:
            list of MotherFluid objects
        :param width:
            int
        :param default_fluids_only:
            bool
            If True we draw a simplified table containing the mother_fluids defined in def_fluids
        """
        self.title = title
        if width is None:
            width = 70
        if height is None:
                height = 135
        self.height = height
        if mother_fluids is None:
            mother_fluids = []
        self.fluids = mother_fluids
        self.keys = ['name', 'k', 'mu', 'rho', 'calculation_method',
                     'temp_gradient', 'pressure_gradient',
                     'salinity', 'gor', 'oil_api', 'gas_gravity', 'gas_mixing', 'brie_exponent']
        self.default_fluids_only = default_fluids_only
        if default_fluids_only:
            self.fluids = [
                Fluid(
                    k=_val['k'],
                    mu=_val['mu'],
                    rho=_val['rho'],
                    name=_name) for _name, _val in def_fluids.items()
            ]
            self.keys = ['name', 'k', 'mu', 'rho']

    @property
    def cds(self) -> ColumnDataSource:
        _dict = {_x:[] for _x in self.keys}
        for _fluid in self.fluids:
            for _key in self.keys:
                if _key == 'name':
                    _dict['name'].append(_fluid.header.name)
                elif _key in ['bd']:
                    _dict[_key].append(None)
                elif _key in ['k', 'mu', 'rho']:
                    _dict[_key].append(_fluid.__dict__['_{}'.format(_key)].magnitude)
                elif _key in ['gas_mixing', 'calculation_method']:
                    _dict[_key].append(_fluid.__dict__[_key])
                else:
                    _dict[_key].append(_fluid.__dict__[_key].magnitude)
        return ColumnDataSource(_dict)

    @property
    def fluid_names(self):
        _fluid_names = []
        for _fluid in self.fluids:
            _this_name = _fluid.name
            if _this_name not in _fluid_names:
                _fluid_names.append(_this_name)
        return _fluid_names

    def table_columns(self):
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, CheckboxEditor)
        _calc_methods = ['Batzle and Wang', 'User specified']
        _fluid_types = ['-', 'Brine', 'Oil', 'Gas']
        table_columns = [
            TableColumn(field='name', title='Name'),
            # TableColumn(field='fluid_type', title='Fluid type',
            #             editor=SelectEditor(options=_fluid_types)),
            TableColumn(field='k', title='K [GPa]'),
            TableColumn(field='mu', title='G [GPa]'),
            TableColumn(field='rho', title='Den. [g/cm3]'),
            # TableColumn(field='bd', title='Burial depth [m]'),
            TableColumn(field='calculation_method', title='Calc. meth.',
                        editor=SelectEditor(options=_calc_methods)),
            # TableColumn(field='temp_ref', title='T. ref. [deg C]'),
            TableColumn(field='temp_gradient', title='T. grad. [deg C/m]'),
            # TableColumn(field='pressure_ref', title='P. ref. [MPa]'),
            TableColumn(field='pressure_gradient', title='P. grad. [MPa/m]'),
            TableColumn(field='salinity', title='Salinity [ppm]'),
            TableColumn(field='gor', title='GOR'),
            TableColumn(field='oil_api', title='Oil API'),
            TableColumn(field='gas_gravity', title='Gas gravity')
        ]

        if self.default_fluids_only:
            table_columns = table_columns[:4]

        return table_columns

    def draw(self,
             cds: ColumnDataSource):
        from bokeh.models import DataTable, Button, CheckboxGroup, Div

        def add_row_function():
            new_data = dict(cds.data)
            for _key in list(new_data.keys()):
                if _key == 'name':
                    new_data[_key].append('default')
                else:
                    new_data[_key].append(def_fluid_vals[_key])
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
            _fluids = []
            for _i in range(len(new_data['name'])):
                # Avoid empty rows
                if new_data['name'][_i] is None or new_data['name'][_i] == '':
                    continue
                if self.default_fluids_only:
                    this_fluid = Fluid(
                        k=Q_(new_data['k'][_i], 'GPa'),
                        mu=Q_(new_data['mu'][_i], 'GPa'),
                        rho=Q_(new_data['rho'][_i], 'gram / cm^3'),
                        name=new_data['name'][_i])
                else:
                    this_fluid = Fluid(
                        calculation_method=new_data['calculation_method'][_i],
                        k=Q_(new_data['k'][_i], 'GPa'),
                        mu=Q_(new_data['mu'][_i], 'GPa'),
                        rho=Q_(new_data['rho'][_i], 'gram / cm^3'),
                        temp_gradient=Q_(new_data['temp_gradient'][_i], 'degC / meter'),
                        # temp_ref=Q_(new_data['temp_ref'][_i], 'degC'),
                        pressure_gradient=Q_(new_data['pressure_gradient'][_i], 'MPa / meter'),
                        # pressure_ref=Q_(new_data['pressure_ref'][_i], 'MPa'),
                        salinity=Q_(new_data['salinity'][_i], 'ppm'),
                        gor=Q_(new_data['gor'][_i], ''),
                        oil_api=Q_(new_data['oil_api'][_i], ''),
                        gas_gravity=Q_(new_data['gas_gravity'][_i], ''),
                        name=new_data['name'][_i])
                _fluids.append(this_fluid)

            self.fluids = _fluids

            print(self.fluid_names)
            cds.data = new_data

        title = Div(text =
                    """
                    <div style="font-size:12px; font-weight:600; margin-bottom:0px; text-align:center">
                    """ + self.title +  """
                    </div>
               """)

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

        return column(title, dt, sizing_mode='stretch_width'), add_row, delete_row, update_table

# TODO Rewrite the following class so that there is one Fluid per well, interval and subst_order
# This fluid has the VRH bounded average properties of each of the constituent mother fluids
# To make this possible, I think the mother_fluid, fluid_type and volume_fraction must be
# input as lists that are later VRH averaged over
class Fluid:
    """
    A Fluid takes the general properties of a MotherFluid and sets into work for specific setting
    e.g. for a specific well, interval and fluid type
    """
    def __init__(self,
                 mother_fluid: MotherFluid,
                 interval: Interval,
                 substitution_order: Literal['Initial', 'Final'],
                 fluid_type: str,
                 volume_fraction: str,
                 status: str | None,
                 tag: str):
        """

        :param mother_fluid:
        :param interval:
        :param substitution_order:
        :param fluid_type:
            str
            'Oil', 'Gas' or 'Brine' (or '' or 'User specified' for  User specified fluids)
        :param volume_fraction:
        :param tag:
        """
        self.mother_fluid = mother_fluid
        self.interval = interval
        self.substitution_order = substitution_order
        self.fluid_type = fluid_type
        self.volume_fraction = volume_fraction
        if status is None:
            status = ''
        self.status = status
        self.tag = tag
        self.k = None
        self.mu = None
        self.rho = None
        self.bd = None

    @property
    def name(self):
        if len(self.tag) > 0:
            return '{}_{}_{}_{}'.format(self.substitution_order, self.mother_fluid.name,self.fluid_type, self.tag)
        else:
            return '{}_{}_{}'.format(self.substitution_order, self.mother_fluid.name,self.fluid_type)

    def calc_elastics(self, burial_depth: Q_):
        self.mother_fluid.calc_elastics(self.fluid_type, burial_depth)
        self.k = self.mother_fluid.k
        self.mu = self.mother_fluid.mu
        self.rho = self.mother_fluid.rho
        self.bd = burial_depth

class Fluid_test:
    """
    A Fluid_test takes the general properties of a *list* of MotherFluids and calculates the VRH bounds
    for this combination of fluids
    e.g. for a specific well, interval and fluid type
    """
    def __init__(self,
                 mother_fluids: List[MotherFluid],
                 fluid_types: List[str],
                 volume_fractions: list,
                 well_name: str,
                 interval_name: str,
                 substitution_order: str,
                 status: str | None = None,
                 header: dict | None = None,
                 tag: str | None = None):
        """

        :param mother_fluid:
        :param interval:
        :param substitution_order:
        :param fluid_type:
            str
            'Oil', 'Gas' or 'Brine' (or '' or 'User specified' for  User specified fluids)
        :param volume_fraction:
        :param tag:
        """
        self.mother_fluids = mother_fluids
        self.fluid_types = fluid_types
        self.volume_fractions = volume_fractions
        self.well_name = well_name
        self.interval_name = interval_name
        self.substitution_order = substitution_order
        if status is None:
            status = ''
        self.status = status
        if header is None:
            header = {}
        if 'name' not in list(header.keys()):
            header['name'] = None
        self.header = Header(header)
        self.tag = tag
        self.k = None
        self.mu = None
        self.rho = None
        self.bd = None



    @property
    def name(self):
        if len(self.tag) > 0:
            return '{}_{}_{}_{}'.format(self.substitution_order, self.well_name,self.interval_name, self.tag)
        else:
            return '{}_{}_{}'.format(self.substitution_order, self.well_name,self.interval_name)

    def calc_elastics(self, wells: dict, intervals: Intervals, burial_depth: Q_):
        """
        Calculates the Voigt-Reuss-Hill averages of the constituent mother fluids
        :param wells:
            Dictionary with well name: Well object as key: value pairs
        :param intervals:
            Interval object
        :param burial_depth:
        :return:
        """
        self.bd = burial_depth
        info_txt = 'Calculating {} bounds of {} for well {} in interval {}\n'.format(
            'VRH', 'parameters', self.well_name, self.interval_name
        )

        ks = []  # container for all bulk modulus constituents
        mus = []  # container for all shear modulus constituents
        rhos = []  # container for all density constituents
        # Iterate over all constituent mother fluids
        for _i, _f in enumerate(self.mother_fluids):
            info_txt += '  - {}: {} with fraction {}'.format(_f.name, self.fluid_types[_i], self.volume_fractions[_i])
            _f.calc_elastics(self.fluid_types[_i], burial_depth)
            ks.append(_f.k)
            mus.append(_f.mu)
            rhos.append(_f.rho)

class FluidMix(object):
    """
    Object containing the initial, and final, mother_fluids for given wells and intervals
    """
    def __init__(self,
                 name: str | None = None,
                 fluids: List[Fluid] | None = None,
                 header: dict | None = None):

        if header is None:
            header = {}
        if name is not None:
            header['name'] = name
        elif 'name' not in list(header.keys()):
            header['name'] = None

        self.header = Header(header)

        if fluids is None:
            fluids = []
        self.fluids = fluids

        self._name = header['name']

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.header['name'] = value

    def with_key(self, _key ):
        """
        Returns a list of fluids attributes for the given _key
        :param _key:
        :return:
        """
        return list(
            set(
                [_f.__dict__[_key] for _f in self.fluids]
            )
        )

    def fluid_types(self):
        """
        Returns a list of all fluid types present among the fluids
        :return:
        """
        return self.with_key('fluid_type')

    def tags(self):
        return self.with_key('tag')

    def volume_fractions(self):
        return self.with_key('volume_fraction')

    def necessary_logs(self):
        """
        Return the logs listed among the volume fractions
        :return:
        """
        _list = list(self.volume_fractions())  # creates a copy of volume fractions
        for _item in list(_list):
            if isinstance(_item, int) or isinstance(_item, float):
                _list.remove(_item)
            if isinstance(_item, str) and _item.lower() in ['complement']:
                _list.remove(_item)
        return _list


    def substitution_orders(self):
        return self.with_key('substitution_order')

    def statuses(self):
        return self.with_key('status')

    def well_names(self):
        return list(
            set(
                [_f.interval.well.lower() for _f in self.fluids]
            )
        )

    def interval_names(self):
        return list(
            set(
                [_f.interval.name.lower() for _f in self.fluids]
            )
        )

    def intervals(self):
        _list = []
        for _f in self.fluids:
            # print('XXX', _f.interval.name, [_i.name for _i in _list])
            if _f.interval.name not in [_i.name for _i in _list]:
                _list.append(_f.interval)

        return Intervals(name=self.name + ' Intervals', intervals=_list)

    def mother_fluid_names(self):
        return list(
            set(
                [_f.mother_fluid.name for _f in self.fluids]
            )
        )

    def mother_fluids(self):
        _list = []
        for _f in self.fluids:
            if _f.mother_fluid.name.lower() not in [_l.name for _l in _list]:
                _list.append(_f.mother_fluid)
        return _list


    # TODO
    # TODO Rewrite the following code so that it returns one fluid per: subst_order, well_name & wi_name
    # This Fluid object should have the VRH bounded average properties of each mother fluid constituent
    def get_fluids(self,
                   subst_order: str | None = None,
                   well_name: str | None = None,
                   wi_name: str | None = None,
                   mother_fluid_name: str | None = None,
                   fluid_type: str | None = None,
                   volume_fraction: str | float | None = None,
                   status: str | None = None,
                   tag: str | None = None,
                   verbose: bool = False) -> list:
        """
        Returns a list of fluids that matches the set requirements

        :param subst_order:
            str
            Not case-sensitive
        :param well_name:
        :param wi_name:
        :param mother_fluid_name:
        :param fluid_type:
        :param volume_fraction:
            str | float
            If a string, the matching is case-sensitive
        :param status:
            str
        :param tag:
            str
            The matching is case-sensitive
        :param verbose:
        :return:
        """

        # Check consistency of input data
        if subst_order is not None and subst_order.lower() not in self.substitution_orders():
            print_info('Substitution order: {} not present among subst. orders ([{}]) in FluidMix: {}'.format(
                subst_order, ', '.join(self.substitution_orders()), self.name),
                       'warning', logger)
            return []
        if well_name is not None and well_name.lower() not in self.well_names():
            print_info('Well: {} not present among well names ([{}]) in FluidMix: {}'.format(
            well_name, ', '.join(self.well_names()), self.name),
                       'warning', logger)
            return []
        if wi_name is not None and wi_name.lower() not in self.interval_names():
            print_info('Interval: {} not present among interval names ([{}]) in FluidMix: {}'.format(
            wi_name, ', '.join(self.interval_names()), self.name),
                       'warning', logger)
            return []
        if mother_fluid_name is not None and mother_fluid_name.lower() not in self.mother_fluid_names():
            print_info('Mother fluid: {} not present among mother fluids ([{}]) in FluidMix: {}'.format(
            mother_fluid_name, ', '.join(self.mother_fluid_names()), self.name),
                       'warning', logger)
            return []
        if fluid_type is not None and fluid_type.lower() not in self.fluid_types():
            print_info('Fluid type: {} not present among fluid types ([{}]) in FluidMix: {}'.format(
            fluid_type, ', '.join(self.fluid_types()), self.name),
                       'warning', logger)
            return []
        if volume_fraction is not None and volume_fraction not in self.volume_fractions():
            print_info('Volume fraction: {} not present among volume fractions ([{}]) FluidMix: {}'.format(
                volume_fraction, ', '.join([str(_x) for _x in self.volume_fractions()]), self.name),
                       'warning', logger)
            return []
        if status is not None and status not in self.statuses():
            print_info('Status: {} not present among statuses ([{}]) in FluidMix: {}'.format(
                tag, ', '.join(self.statuses()), self.name),
                       'warning', logger)
            return []
        if tag is not None and tag not in self.tags():
            print_info('Tag: {} not present among tags ([{}]) in FluidMix: {}'.format(
                tag, ', '.join(self.tags()), self.name),
                'warning', logger)
            return []


        # Initiate a list of all fluid names
        list_of_names = [_f.name for _f in self.fluids]

        # The below method requires that each fluid in the fluid mix has a unique name
        dupl_names = get_duplicates(list_of_names)
        if len(dupl_names) > 0:
            print_info('There are multiple fluids with the same name ([{}])'.format(', '.join(dupl_names)),
                       'error', logger, 'IOError')

        # Then start removing the ones that doesn't match the requirements
        # Do this by iterating over a copy of the original list
        for fluid_name in list(list_of_names):
            _fluid = [_f for _f in self.fluids if _f.name == fluid_name][0]
            if subst_order is not None and subst_order.lower() != _fluid.substitution_order.lower():
                try:
                    list_of_names.remove(fluid_name)
                    if verbose:
                        print('Subst. order: Removing fluid: {}'.format(_fluid.name))
                except ValueError:
                    if verbose:
                        print('get_fluids(): Subst. order: Tried to remove fluid: {}'.format(_fluid.name))
                    continue
            if well_name is not None and well_name.lower() != _fluid.interval.well.lower():
                try:
                    list_of_names.remove(fluid_name)
                    if verbose:
                        print('get_fluids(): Well name: Removing fluid: {}'.format(_fluid.name))
                except ValueError:
                    if verbose:
                        print('get_fluids(): Well name: Tried to remove fluid: {}'.format(_fluid.name))
                    continue
            if wi_name is not None and wi_name.lower() != _fluid.interval.name.lower():
                try:
                    list_of_names.remove(fluid_name)
                    if verbose:
                        print('get_fluids(): Interval name: Removing fluid: {}'.format(_fluid.name))
                except ValueError:
                    if verbose:
                        print('get_fluids(): Interval name: Tried to remove fluid: {}'.format(_fluid.name))
                    continue
            if mother_fluid_name is not None and mother_fluid_name.lower() != _fluid.mother_fluid.name.lower():
                try:
                    list_of_names.remove(fluid_name)
                    if verbose:
                        print('get_fluids(): Mother fluid name: Removing fluid: {}'.format(_fluid.name))
                except ValueError:
                    if verbose:
                        print('get_fluids(): Mother fluid name: Tried to remove fluid: {}'.format(_fluid.name))
                    continue
            if fluid_type is not None and fluid_type.lower() != _fluid.fluid_type.lower():
                try:
                    list_of_names.remove(fluid_name)
                    if verbose:
                        print('get_fluids(): Fluid type: Removing fluid: {}'.format(_fluid.name))
                except ValueError:
                    if verbose:
                        print('get_fluids(): Fluid type: Tried to remove fluid: {}'.format(_fluid.name))
                    continue
            if volume_fraction is not None and volume_fraction != _fluid.volume_fraction:
                try:
                    list_of_names.remove(fluid_name)
                    if verbose:
                        print('get_fluids(): Volume fraction: Removing fluid: {}'.format(_fluid.name))
                except ValueError:
                    if verbose:
                        print('get_fluids(): Volume fraction: Tried to remove fluid: {}'.format(_fluid.name))
                    continue
            if tag is not None and tag != _fluid.tag:
                try:
                    list_of_names.remove(fluid_name)
                    if verbose:
                        print('get_fluids(): Tag: Removing fluid: {}'.format(_fluid.name))
                except ValueError:
                    if verbose:
                        print('get_fluids(): Tag: Tried to remove fluid: {}'.format(_fluid.name))
                    continue

        return [_f for _f in self.fluids if _f.name in list_of_names]

    def print_all_fluids(self, verbose=False):
        out = ''
        for key in self.substitution_orders():
            for w in self.well_names():
                for wi in self.interval_names():
                    out += self.print_fluids(key, w, wi, verbose)
        return out

    def print_fluids(self, subst_order, well_name, wi_name, verbose=False):
        for _fluid in self.get_fluids(subst_order=subst_order, well_name=well_name, wi_name=wi_name, verbose=verbose):
            out = '-- Fluid mixture {}: {}, {}, {} --'.format(self.name, well_name, wi_name, subst_order)
            for m in ['fluid_type', 'volume_fraction', 'status', 'tag', 'bd', 'k', 'mu', 'rho']:
                out += '\n{}: {}'.format(m, _fluid.__dict__[m])
            # out += '\n' + _fluid.mother_fluid.print_fluid(verbose)
        return out + '\n'

    def read_excel(self, filename,
                   intervals: Intervals,
                   fluid_sheet: str = 'Fluids', fluid_header: int = 1,
                   mix_sheet: str = 'Fluid mixtures', mix_header: int = 1):

        self.fluids = read_fluidmixes_from_excel(filename, intervals, fluid_sheet, fluid_header, mix_sheet, mix_header)
        self.header['orig_file'] = filename

    def calc_elastics(self, wells: dict, debug=False):
        """
        Calculates k, mu, and rho for all mother_fluids for each well and working interval they are defined in, and where the
        calculation method is not 'User specified'
        :param wells:
            dict
            {well name: core.wells.Well} key: value pairs
        :param debug:
            bool
            if True, generate verbose information and create some plots
        :return:
        """
        # 1'st level: Initial and Final mother_fluids
        for key in list(self.substitution_orders()):

            # 2'nd level: loop over all wells
            for w in list(self.well_names()):
                w = w.lower()

                if w not in [_w.lower() for _w in list(wells.keys())]:
                    warn_txt = 'Well {} not present among the input wells ([{}])'.format(
                        w, ', '.join(list(wells.keys())))
                    print_info(warn_txt, 'warning', logger)
                    continue

                # test if this will is listed in the working intervals
                if w.lower() not in [_w.lower() for _w in self.intervals().well_names()]:
                    warn_txt = 'Well {} not present among the working intervals'.format(w)
                    print_info(warn_txt, 'warning', logger)
                    continue

                # Extract the measured and burial depth for this well
                wells[w].create_tvd_ml_log()
                bd = wells[w].get_log_curve('tvd_ml')
                md = wells[w].get_md_log()

                # 3'rd level: loop over all working intervals
                for wi in list(self.interval_names()):
                    # print(self.print_fluids(key, w, wi, verbose=False))
                    # Extract the mean burial depth for this working interval
                    _wi = self.intervals().get_interval(wi, w)
                    wi_md_i = np.nanargmin((md.data - _wi.mid)**2)
                    wi_bd = bd.data[wi_md_i]

                    # 4th level: Loop over all the different fluid types we have at this level
                    # TODO This can perhaps fail if the fluid type is set to None or ''? TEST
                    for _i, _fluid_type in enumerate(list(self.fluid_types())):
                        info_txt = ''
                        _fluids = self.get_fluids(subst_order=key, well_name=w, wi_name=wi, fluid_type=_fluid_type, verbose=False)
                        if len(_fluids) == 0:
                            continue
                        if len(_fluids) > 1:
                            err_txt = 'There should only be one fluid for the given subst. order' + \
                            ' ({}), well ({}), interval ({}) and fluid type ({})'.format(
                                key, w, wi, _fluid_type
                            )
                            print_info(err_txt, 'error', logger, raiser='IOError')

                        # Get the selected (unique) fluid
                        _fluid = _fluids[0]

                        info_txt += '- Subst. Order: {}, Well: {}, interval: {}\n'.format(key, w, wi)
                        info_txt += '  Fluid #{}: {}, type: {}, mother fluid: {}, burial depth: {:.2f}\n'.format(
                            _i, _fluid.name, _fluid.fluid_type, _fluid.mother_fluid.name, wi_bd
                        )
                        if _fluid.mother_fluid.calculation_method == 'User specified':
                            _fluid.calc_elastics(wi_bd)
                            info_txt += '    User specified elastic values.'
                            if debug:
                                print(info_txt)
                            continue
                        if _fluid.status.lower() == 'from excel':
                            # start calculating fluid properties
                            info_txt += "   Has status 'from excel', "
                            if _fluid.mother_fluid.calculation_method.lower() == 'Batzle and Wang'.lower():
                                info_txt += "and will be calculated using 'Batzle and Wang'. "
                                _fluid.calc_elastics(wi_bd)
                                _fluid.status = \
                                    'calculated using Batzle and Wang at burial depth {:.2f}'.format(wi_bd)
                            if debug:
                                print(info_txt)
                        else:
                            if debug:
                                info_txt += '   Has already been calculated'
                                print(info_txt)
                            continue

class TestCases(unittest.TestCase):
    def test_read_fluid_mixes(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        intervals = Intervals(name='MyIntervals')
        intervals.read_blixt_tops(project_table)
        # my_fluid_mixes = read_fluidmixes_from_excel(project_table, intervals)
        my_fluid_mixes = read_fluidmixes_from_excel_test(project_table, intervals)

        # fmix = FluidMix(name='MyFluidMix', fluids=my_fluid_mixes)
        # print(fmix.necessary_logs())
        # print(fmix.print_all_fluids(verbose=False))
        # for _fluid in fmix.get_fluids(
        #     subst_order='fInal',
        #         # subst_order=None,
        #     well_name='wElL_F',
        #     # well_name=None,
        #     wi_name='sand E',
        #     # wi_name=None,
        #     # mother_fluid_name='DEFAULT',
        #     mother_fluid_name=None,
        #     # fluid_type='bRiNe',
        #     fluid_type=None,
        #     # volume_fraction='complement',
        #     volume_fraction=None,
        #     # tag='XX',
        #     tag=None,
        #     verbose=True
        # ):
        #     info_txt = ' Fluid: {}, type: {}, mother fluid: {}, '.format(
        #         _fluid.name, _fluid.fluid_type, _fluid.mother_fluid.name )
        #     print(info_txt)

    def test_calc_elastics(self):
        from blixt_rp.core.project_new import Project

        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        wp = Project(name='MyProject', project_table=project_table)
        wp.load_all_wells()
        wells = {_w.name.lower(): _w for _w in wp.wells}

        intervals = Intervals(name='MyIntervals')
        intervals.read_blixt_tops(project_table)

        fmix = FluidMix(name='Fluids')
        fmix.read_excel(project_table, intervals)
        # print(fmix.print_all_fluids())

        fmix.calc_elastics(wells, True)

        print(fmix.print_all_fluids(verbose=True))

    def test_data(self):
        import os

        from blixt_rp.core.well import Project
        wp = Project(name='MyProject')

        # myfluids = FluidMix()
        # myfluids.read_excel(wp.project_table)
        myfluids = read_fluidmixes_from_excel(wp.project_table)
        all_mother_fluids = read_all_mother_fluids_from_excel(wp.project_table)
        return all_mother_fluids, myfluids

    def test_fluid(self):
        default = def_fluid_vals
        f1 = Fluid(**default)
        print(f1)
        for ft in ['Brine', 'Oil', 'Gas']:
            f1.calc_elastics(ft, Q_(1000., 'm'))
            print('{}: '.format(ft), f1.k, f1.mu, f1.rho)


    def test_fluid_table(self, unit_test=True):

        all_fluids, fluid_mix = self.test_data()
        ft = FluidsTable(mother_fluids=list(all_fluids.values()))
        # ft = FluidsTable(default_fluids_only=True)
        cds = ft.cds
        # for _key in list(cds.data.keys()):
        #     print(_key, cds.data[_key])
        table, add_row, delete_row, update = ft.draw(cds)
        if unit_test:
            from bokeh.io import output_file
            from bokeh.plotting import show, row, column
            output_file(os.path.join(project_dir, 'blixt_rp\\results_folder\\plot.html'))
            show(column(table, row(add_row, delete_row, update)))
            return None, None, None, None
        else:
            return table, add_row, delete_row, update

    def test_fluidsub(self):
        import matplotlib.pyplot as plt
        from blixt_rp.core.core import LogTable, Cutoffs, CutoffRule
        from blixt_rp.core.well_new import Well
        from blixt_rp.core.log_curve_new import Depth, LogCurve
        from blixt_rp.plotting.cross_plotter import CrossPlotter

        file_dir = str(os.path.dirname(__file__).replace(
            'blixt_rp\\core',
            ''))
        project_table = os.path.join(file_dir, 'excels\\project_table_new.xlsx')

        las_file =  os.path.join(file_dir, 'test_data/Well A.las')
        _output_file = os.path.join(file_dir, 'results_folder/plot.html')

        logs = {'vp_dry': 'P velocity', 'vp_so08': 'P velocity', 'vp_sg08': 'P velocity',
                'vs_dry': 'S velocity', 'vs_so08': 'S velocity', 'vs_sg08': 'S velocity',
                'rho_dry': 'Density', 'rho_so08': 'Density', 'rho_sg08': 'Density',
                'phie': 'Porosity', 'vcl': 'Volume', 'dept': 'MD'}
        lt = LogTable()
        lt.from_invert(logs)

        w1 = Well()
        w1.read_las(las_file, log_table=lt, template_file=project_table)

        # Calculate and apply this mask:
        # w.calc_mask({'vcl': ['<', 0.4], 'phie': ['>', 0.1]}, name='sand')  # calc_mask only works for the old Well
        cr1 = CutoffRule('vcl', '<', Q_(0.4, ''))
        cr2 = CutoffRule('phie', '>', Q_(0.1, ''))
        ct = Cutoffs([cr1, cr2])

        w_dict = w1.dict(cutoffs=ct, use_cutoffs=True)
        # w_dict = w1.dict(cutoffs=ct, use_cutoffs=False)  # Now it creates and an extra parameter 'mask'

        test = 'array'  # 'constants'
        # Test with constant Vsh and constant Sw
        v_sh = 0.2
        s_w = 0.2
        if test == 'array':  # test with arrays of v_sh and s_w
            v_sh = w_dict['vcl'].magnitude
            # create a mock-up water saturation
            s_w = 0.2 + v_sh
            s_w[s_w > 1.0] = 1.0

        # Mineral models, K & MU in GPa, rho in g/cm3
        rho_qz = 2.6;   k_qz = 37;  mu_qz = 44
        rho_sh = 2.8;   k_sh = 15;  mu_sh = 5

        # Mineral shear and bulk modulus using Voigt-Reuss-Hill average for given Vsh
        k0 = rp.vrh_bounds([v_sh, (1.-v_sh)], [k_sh, k_qz])[-1]  # GPa
        mu0 = rp.vrh_bounds([v_sh, (1.-v_sh)], [mu_sh, mu_qz])[-1]  # GPa
        rho0 = v_sh*rho_sh + (1. - v_sh)*rho_qz  # g/cm3

        # Brine model, rho in g/cm3, K in GPa
        rho_b = 1.1;   k_b = 2.8

        # Hydrocarbon model, K in GPa
        fluid = 'oil'
        rho_o = 0.8;    k_o = 0.9
        rho_g = 0.2;    k_g = 0.06
        (k_hc, rho_hc) = (k_g, rho_g) if fluid == 'gas' else (k_o, rho_o)

        # intitial fluid
        rho_f1 = rho_b; k_f1 = k_b

        # After fluid substitution
        rho_f2 = s_w*rho_b + (1.-s_w)*rho_hc
        k_f2 = rp.vrh_bounds([s_w, (1.-s_w)], [k_b, k_hc])[1]  # Reuss uniform fluid mix

        w2 = Well()
        w2.name = 'Fluid Sub'
        v_p_2, v_s_2, rho_2, k_2 = rp.gassmann_vel(
            w_dict['vp_dry'].magnitude,
            w_dict['vs_dry'].magnitude,
            w_dict['rho_dry'].magnitude,
            k_f1, rho_f1, k_f2, rho_f2, k0,
            w_dict['phie'].magnitude)
        # Add fluid subst. logs to well
        depth = Depth(w_dict['dept'])
        lc_vp = LogCurve('vp_so08', Q_(v_p_2, 'm/s'), depth, log_type='P velocity', style=deepcopy(w1.get_log_curve('vp_dry').style))
        w2.add_log(lc_vp)
        lc_vs = LogCurve('vs_so08', Q_(v_s_2, 'm/s'), depth, log_type='S velocity', style=deepcopy(w1.get_log_curve('vs_dry').style))
        w2.add_log(lc_vs)
        lc_rho = LogCurve('rho_so08', Q_(rho_2, 'g/cm**3'), depth, log_type='Density', style=deepcopy(w1.get_log_curve('rho_dry').style))
        w2.add_log(lc_rho)
        lc_depth = LogCurve('dept', Q_(depth.magnitude, depth.units), depth, log_type='MD')
        w2.add_log(lc_depth)


        xp = CrossPlotter({w.name: w.data_source() for w in [w1, w2]})
        cds = xp.cds
        xp.show_plot(cds, out_file=_output_file)


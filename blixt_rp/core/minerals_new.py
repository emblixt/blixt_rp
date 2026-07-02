# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:27:45 2020

Module for handling rockphysics mineral models

A mineral is defined by its:
    Name: str
    k, Bulk moduli [GPa]: float
    mu, Shear moduli [GPa]: float
    rho, Density, [g/cm3]: float

:copyright: Erik Mårten Blixt (marten.blixt@gmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)       

@author: mblixt
"""
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
import sys
import os
from bokeh.models import ColumnDataSource
from bokeh.plotting import column, row
import pint
from pandas.core.indexes.base import maybe_extract_name
import logging

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp import Q_
from blixt_utils.misc.attribdict import AttribDict
from blixt_rp.rp_utils.version import info
from blixt_utils.utils import isnan, print_info
import blixt_utils.io.io as uio
from blixt_rp import rp as rp
import blixt_rp.rp_utils.definitions as ud
from blixt_rp.core.core import Cutoffs, LogTable, Intervals

logger = logging.getLogger(__name__)

def_minerals = dict(
    # From RokDoc, and are also stored in the project_table.xlsx
    shale=dict(
        rho=Q_(2.35, 'G/cc'), vp=Q_(2560.0, 'm/s'), k=Q_(11.4, 'GPa'), vs=Q_(1130.0, 'm/s'), mu=Q_(3.0, 'GPa')),
    quartz=dict(
        rho=Q_(2.65, 'G/cc'), vp=Q_(6038.0, 'm/s'), k=Q_(36.6, 'GPa'), vs=Q_(4121.0, 'm/s'), mu=Q_(45.0, 'GPa')),
    calcite=dict(
        rho=Q_(2.71, 'G/cc'), vp=Q_(6253.0, 'm/s'), k=Q_(63.7, 'GPa'), vs=Q_(3420., 'm/s'), mu=Q_(31.7, 'GPa'))
)

class MineralsTable:
    """
    Class for handling and drawing a simple table that contain minerals
    """
    def __init__(self,
                 minerals: list | None = None,
                 title: str | None = 'Minerals',
                 width: int | None = None,
                 height: int | None = None,
                 default_minerals_only: bool = False ):
        """

        :param minerals:
            list
            List of Mineral objects
        :param title:
            str
            Title given to table
        :param width:
            int
        :param height:
            int
        :param default_minerals_only:
            bool
            When True, we create a simpler table only consisting of the above given default minerals (def_minerals)
        """

        self.title = title
        if width is None:
            width = 700
        self.width = width
        if height is None:
            height = 135
        self.height = height
        if minerals is None:
            minerals =[]
        self.minerals = minerals
        self.keys = ['name', 'k', 'mu', 'rho', 'calculation_method', 'cutoffs', 'volume_fraction', 'status']
        self.default_minerals_only = default_minerals_only
        if default_minerals_only:
            self.minerals = [
                Mineral(
                    k=_val['k'],
                    mu=_val['mu'],
                    rho=_val['rho'],
                    name=_name) for _name, _val in def_minerals.items()
                ]
            self.keys = ['name', 'k', 'mu', 'rho']

    @property
    def cds(self) -> ColumnDataSource:
        _dict = {_x:[] for _x in self.keys}
        for _mineral in self.minerals:
            for _key in self.keys:
                if _key == 'name':
                    _dict[_key].append(_mineral.__dict__['_name'])
                elif _key in ['calculation_method', 'cutoffs', 'volume_fraction', 'status']:
                    _dict[_key].append(_mineral.__dict__[_key])
                else:
                    _dict[_key].append(_mineral.__dict__[_key].magnitude)
        return ColumnDataSource(_dict)

    @property
    def mineral_names(self):
        _mineral_names = []
        for _mineral in self.minerals:
            _this_name = _mineral.name
            if _this_name not in _mineral_names:
                _mineral_names.append(_this_name)
        return _mineral_names

    def table_columns(self):
        from bokeh.models import (SelectEditor, StringEditor, TableColumn, CheckboxEditor)
        _calc_methods = ['Interval average', 'User specified']
        table_columns = [
            TableColumn(field='name', title='Name'),
            TableColumn(field='k', title='K [GPa]'),
            TableColumn(field='mu', title='G [GPa]'),
            TableColumn(field='rho', title='Den. [g/cm3]'),
            TableColumn(field='calculation_method', title='Calc. meth.',
                        editor=SelectEditor(options=_calc_methods)),
            TableColumn(field='volume_fraction', title='Vol. frac.'),
            TableColumn(field='status', title='Status')
        ]

        if self.default_minerals_only:
            table_columns = table_columns[:4]

        return table_columns

    def draw(self,
             cds: ColumnDataSource):
        from bokeh.models import DataTable, Button, CheckboxGroup, Div

        def add_row_function():
            new_data = dict(cds.data)
            for _key in list(new_data.keys()):
                if _key == 'name':
                    new_data[_key].append('')
                else:
                    new_data[_key].append(None)
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
            _minerals = []
            for _i in range(len(new_data['name'])):
                # Avoid empty rows
                if new_data['name'][_i] is None or new_data['name'][_i] == '':
                    continue
                if self.default_minerals_only:
                    this_mineral = Mineral(
                        k=Q_(new_data['k'][_i], 'GPa'),
                        mu=Q_(new_data['mu'][_i], 'GPa'),
                        rho=Q_(new_data['rho'][_i], 'gram / cm^3'),
                        name=new_data['name'][_i])
                else:
                    this_mineral = Mineral(
                        calculation_method=new_data['calculation_method'][_i],
                        k=Q_(new_data['k'][_i], 'GPa'),
                        mu=Q_(new_data['mu'][_i], 'GPa'),
                        rho=Q_(new_data['rho'][_i], 'gram / cm^3'),
                        volume_fraction=new_data['volume_fraction'][_i],
                        status=new_data['status'][_i],
                        cutoffs=new_data['cutoffs'][_i],
                        name=new_data['name'][_i])
                _minerals.append(this_mineral)

            self.minerals = _minerals

            print(self.mineral_names)
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


class Header(AttribDict):
    """
    Class for mineral set header information
    A ``Header`` object may contain all header information (also known as meta
    data) of a MineralSet object. 
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There can be various default attributes which are
    required 
    
    :param
        header: Dictionary containing meta information of a single
        Mineral set object. 
        Possible keywords are
        summarized in the following `Default Attributes`_ section.
    
    .. rubric:: _`Default Attributes`
    
    """
    defaults = {
        'name': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat(),
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
        Return better readable string representation of Stats object.
        """
        #keys = ['creation_date', 'modification_date', 'temp_gradient', 'temp_ref']
        keys = list(self.keys())
        return self._pretty_str(keys)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Mineral(object):
    """
    Small class handling a single mineral
    """
    def __init__(self,
                 calculation_method=None,  # 'Interval average' or 'User specified'
                 k: pint.Quantity | None = None,  # Bulk moduli [GPa]
                 mu: pint.Quantity | None = None,  # Shear moduli [GPa]
                 rho: pint.Quantity | None = None,  # Density [g/cm3]
                 name: str | None = None,  # str
                 volume_fraction: str = None,  # str 'complement', volume log name, or float
                 cutoffs: Cutoffs | None = None,   # cutoffs used when calculating the average mineral properties within an interval
                 status: str | None = None,
                 header: dict | None = None):
        
        if header is None:
            header = {}
        if name is not None:
            header['name'] = name
        elif 'name' not in list(header.keys()):
            header['name'] = None

        self.header = Header(header)
        self.calculation_method = calculation_method
        self.k = k
        self.mu = mu
        self.rho = rho
        self._name = header['name']
        self.volume_fraction = volume_fraction
        self.cutoffs = cutoffs
        self.status = status

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name
        self.header['name'] = new_name

    def __str__(self):
        keys = list(self.__dict__.keys())
        
        pattern = "%%%ds: %%s" % len(keys)
        
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def keys(self):
        """
        Return dict key list of content
        """
        return self.__dict__.keys()

    def from_default(self, mineral_name: str | None = None):
        if mineral_name is None:
            mineral_name = 'quartz'
        self.k = def_minerals[mineral_name]['k']
        self.mu = def_minerals[mineral_name]['mu']
        self.rho = def_minerals[mineral_name]['rho']
        self.calculation_method = 'User specified'
        self.name = mineral_name

    def calc_k(self, dummy_bd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_bd:
        :return:
        """
        pass

    def calc_mu(self, dummy_bd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_bd:
        :return:
        """
        pass

    def calc_rho(self, dummy_bd):
        """
        Dummy function that makes Mineral objects behave same way as Fluid objects
        :param dummy_bd:
        :return:
        """
        pass

    def vp(self) -> pint.Quantity:
        """
        Calculates Vp for the mineral
        :return:
            pint.Quantity with vp
        """
        from blixt_rp.rp.rp_core import v_p
        vp = v_p(self.k, self.mu, self.rho)
        return vp.to('m/s')

    def plot(self, phic=0.4):
        fig, ax = plt.subplots()
        style = {'lw': 0.5, 'c': 'k'}
        phi = np.arange(0., 1, 0.02)
        for k in [0.01, 0.05, 0.1, 0.3, 0.5]:
            plt.plot(phi, 1./(1. + phi/k), **style)

        plt.plot(phi, 1. -phi, **style)
        plt.plot(phi, 1. -phi/phic, **style)
        plt.grid(True)

        ax.set_ylim(0., 1.05)


class MineralMix(object):
    """
    Class handling a set of minerals
    """
    
    def __init__(self,
                 name=None,
                 minerals=None,
                 header=None):
        """
        :param minerals:
            dict
            a Dictionary of Mineral objects
        """
        self.header = Header(header)

        if minerals is None:
            minerals = {}
        self.minerals = minerals

        if name is None:
            name = 'MyMinerals'
        self.name = name

    def __str__(self):
        return self.print_all_minerals()

    def print_all_minerals(self):
        out = ''
        for w in list(self.minerals.keys()):
            for wi in list(self.minerals[w].keys()):
                out += self.print_minerals(w, wi)
        return out

    def print_minerals(self, well_name, wi_name):
        out = 'Mineral mixture: {}, {}, {}\n'.format(well_name, wi_name, self.name)
        this_min = self.minerals[well_name][wi_name]
        for m in list(this_min.keys()):
            out += "    {}\n".format(m)
            out += "      K: {}, Mu: {}, Rho {}\n".format(this_min[m].k,
                                                          this_min[m].mu,
                                                          this_min[m].rho)
            out += "      " \
                       "Calculation method: {}, cutoff: {}\n".format(
                this_min[m].calculation_method, this_min[m].cutoffs)
            out += "      Status: {}\n".format(this_min[m].status)
            out += "      " \
                   "Volume fraction: {}\n".format(this_min[m].volume_fraction)
        return out

    def read_excel(self, filename,
                   min_sheet='Minerals', min_header=1,
                   mix_sheet='Mineral mixtures', mix_header=1):
        """ 
        Read the mineral mixtures from the project table
        """
        # first read in all minerals
        all_minerals = {}
        min_table = pd.read_excel(filename, sheet_name=min_sheet, header=min_header, engine='openpyxl')
        for i, name in enumerate(min_table['Name']):
            if isnan(name):
                continue
            this_min = Mineral(
                calculation_method=min_table['Calculation method'][i] if 'Calculation method' in list(min_table.keys()) else 'User specified',
                k=Q_(float(min_table['Bulk moduli [GPa]'][i]), 'GPa'),
                mu=Q_(float(min_table['Shear moduli [GPa]'][i]), 'GPa'),
                rho=Q_(float(min_table['Density [g/cm3]'][i]), 'G/cc'),
                name=name.lower(),
                cutoffs=Cutoffs(from_string=min_table['Cutoffs'][i]) if 'Cutoffs' in list(min_table.keys()) else None,
                status='from excel'
            )
            all_minerals[name.lower()] = this_min

        # Then read in the mineral mixtures
        min_mixes = {}
        mix_table = pd.read_excel(filename, sheet_name=mix_sheet, header=mix_header, engine='openpyxl')
        for i, name in enumerate(mix_table['Mineral name']):
            vf = mix_table['Volume fraction'][i]
            if isnan(vf):
                continue  # Avoid minerals where the volume fraction is not set
            this_well = mix_table['Well name'][i].upper()
            this_wi = mix_table['Interval name'][i].upper()
            this_mineral = deepcopy(all_minerals[name.lower()])
            this_mineral.volume_fraction = float(vf) if (not isinstance(vf, str)) else vf.lower()
            if this_well in list(min_mixes.keys()):
                if this_wi in list(min_mixes[this_well].keys()):
                    min_mixes[this_well][this_wi][name.lower()] = this_mineral
                else:
                    min_mixes[this_well][this_wi] = {name.lower(): this_mineral}
            else:
                min_mixes[this_well] = {this_wi: {name.lower(): this_mineral}}

        self.minerals = min_mixes
        self.header['orig_file'] = filename

    def calc_elastics(self,
                      wells: list,
                      log_table: LogTable,
                      wis:Intervals,
                      calculation_method: str | None = None,
                      debug: bool = False):
        """
        Calculates k, mu, and rho for each well and working interval they are defined in, AND where the
        mineral calculation method is set to interval average. The other minerals are left untouched
        :param wells:
            list
            List of core.well_new.Well objects
        :param log_table:
            LogTable
            Creates the mapping between a log type (e.g. 'P velocity') and a specific log (e.g. 'vp_oil')
        :param wis:
            Intervals
            Object that holds a number of working intervals / tops for many wells
        :param calculation_method:
            str
            Name of the calculation method.
        :param debug:
            bool
            if True, generate verbose information and create some plots
        :return:
        """
        if calculation_method is None:
            calculation_method = 'Interval average'
        if calculation_method != 'Interval average':
            raise NotImplementedError('{} calculation method not implemented yet')
        lstyle = {'lw': 0.5, 'color': 'k', 'alpha': 0.2}

        # Keep track of number of minerals the elastic properties should be calculated from
        figs = {}
        axes = {}

        warn_txt = None
        for well in wells:
            well_name = well.name
            # test if this well is listed in the mineral mixture
            if well_name not in list(self.minerals.keys()):
                warn_txt = 'Well {} not present in the given mineral mix {}'.format(well_name, self.name)
                if debug:
                    print('WARNING: {}'.format(warn_txt))
                continue

            # test if this well is listed in the working intervals
            if well_name not in wis.well_names():
                warn_txt = 'Well {} not present among the working intervals'.format(well_name)
                if debug:
                    print('WARNING: {}'.format(warn_txt))
                continue

            # test if the required logs exists in this well
            reqd_log_types = ['P velocity', 'S velocity', 'Density']
            if not all([log_table[xx] in well.get_log_names for xx in reqd_log_types]):
                warn_txt = 'The necessary logs ({}) are missing in well {}: {}'.format(
                    ', '.join([log_table[xx] for xx in reqd_log_types]), well_name, ', '.join(well.log_names())
                )
                print_info(warn_txt, 'warning', logger)
                if debug:
                    print('WARNING: {}'.format(warn_txt))
                continue

            figs[well_name] = {}
            axes[well_name] = {}
            # count number of minerals we should plot and create figures and axes
            _n = 0  # number of minerals
            if debug:
                for jj, wi_name in enumerate(list(self.minerals[well_name].keys())):
                    for kk, mineral in enumerate(list(self.minerals[well_name][wi_name].keys())):
                        if self.minerals[well_name][wi_name][mineral].calculation_method == calculation_method:
                            _n += 1
                    if _n != 0:
                        figs[well_name][wi_name] = plt.figure(figsize=(12, 4*_n))
                        axes[well_name][wi_name] = figs[well_name][wi_name].subplots(nrows=_n, ncols=3)

            _n = 0  # number of minerals
            for jj, wi_name in enumerate(list(self.minerals[well_name].keys())):
                for kk, mineral in enumerate(list(self.minerals[well_name][wi_name].keys())):
                    if self.minerals[well_name][wi_name][mineral].calculation_method == calculation_method:
                        _n += 1
                        # Start calculating the interval averages
                        if debug:
                            print('Start calculating the interval average of k, mu and rho for '
                                  '{} in interval {} in well {}, using the cutoff: {}'.format(
                                    mineral, wi_name, well_name, self.minerals[well_name][wi_name][mineral].cutoffs)
                            )

                        # Calculate the necessary cutoffs
                        # First the ones defined in the mineral mix table
                        cutoffs = self.minerals[well_name][wi_name][mineral].cutoffs
                        # Then for the given working interval
                        cutoffs.cutoffs.append(
                            wis.get_cutoff_rule(wi_name, well_name)
                        )

                        # To make sure alle the logs we're using are of the same length, we convert the well to
                        # a ("harmonized") dictionary, with a 'mask' key that meets the requirements of the cutoffs
                        well_dict = well.dict(cutoffs=cutoffs)
                        vp = well_dict[log_table['P velocity']]
                        vs = well_dict[log_table['S velocity']]
                        rho = well_dict[log_table['Density']]
                        mask = well_dict['mask']

                        this_rho = rho[mask].to('G/cc')
                        this_mu = (rho[mask] * vs[mask]**2).to('GPa')
                        this_k = (rho[mask] * (vp[mask]**2 - (4./3.) * vs[mask]**2)).to('GPa')
                        this_rho_med = np.nanmedian(this_rho)
                        this_mu_med = np.nanmedian(this_mu)
                        this_k_med = np.nanmedian(this_k)
                        if debug:
                            print(this_k_med, this_mu_med, this_rho_med)
                            this_depth = well_dict['md'][mask]
                            # TODO the axes indices will only work when plotting one mineral (_n = 1)
                            axes[well_name][wi_name][0].plot(rho[mask], this_depth)
                            axes[well_name][wi_name][0].set_xlabel('Density [{}]'.format(
                                well_dict[log_table['Density']].units))
                            axes[well_name][wi_name][0].axvline(this_rho_med.magnitude, **lstyle)
                            axes[well_name][wi_name][1].plot(vp[mask], this_depth)
                            axes[well_name][wi_name][1].plot(vs[mask], this_depth)
                            axes[well_name][wi_name][1].set_xlabel('Vp & Vs [m/s]')
                            axes[well_name][wi_name][1].legend(['Vp', 'Vs'])
                            axes[well_name][wi_name][2].plot(this_mu, this_depth)
                            axes[well_name][wi_name][2].plot(this_k, this_depth)
                            axes[well_name][wi_name][2].set_xlabel('$\mu$ & K [GPa]')
                            axes[well_name][wi_name][2].legend(['$\mu$', 'K'])
                            axes[well_name][wi_name][2].axvline(this_mu_med.magnitude, **lstyle)
                            axes[well_name][wi_name][2].axvline(this_k_med.magnitude, **lstyle)
                            for an, ax in enumerate(axes[well_name][wi_name]):
                                if an == 0:
                                    ax.set_ylabel('MD [m]')
                                ax.set_ylim(ax.get_ylim()[::-1])
                            figs[well_name][wi_name].suptitle('{}, {}, {}'.format(well_name, wi_name, mineral))

                        # store the calculated elastic properties in the minerals object
                        if not np.isnan(this_k_med):
                            self.minerals[well_name][wi_name][mineral].k = this_k_med
                        if not np.isnan(this_mu_med):
                            self.minerals[well_name][wi_name][mineral].mu = this_mu_med
                        if not np.isnan(this_rho_med):
                            self.minerals[well_name][wi_name][mineral].rho = this_rho_med
                        # update the status of the mineral
                        self.minerals[well_name][wi_name][mineral].status = 'calculated from interval average'
                    else:
                        continue
        if debug:
            plt.show()

class TestCases(unittest.TestCase):
    def test_data(self):
        from blixt_rp.core.project_new import Project
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        project = Project(name='MyProject', project_table=project_table)
        project.load_all_wells()
        wis = project.load_all_wis()
        return project, wis

    def test_mineral(self):
        m = Mineral()
        m.from_default('shale')
        print(m)

        print(m.vp())

    def test_read_mineral_mixes(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        my_mineral_mix = MineralMix()
        my_mineral_mix.read_excel(project_table)
        print(my_mineral_mix)
        print(my_mineral_mix.print_all_minerals())

    def test_calc_elastics(self):
        from blixt_rp.core.core import LogTable, Intervals
        project, wis = self.test_data()
        log_table = LogTable({
            'P velocity': 'vp_dry',
            'S velocity': 'vs_dry',
            'Density': 'rho_dry',
            'Porosity': 'phie',
            'Volume': 'vcl'
        } )
        my_mineral_mix = MineralMix()
        my_mineral_mix.read_excel(project.project_table)
        my_mineral_mix.calc_elastics(project.wells, log_table, wis, debug=True)

    def test_mineral_table(self, unit_test=True):
        ft = MineralsTable(default_minerals_only=True)
        cds = ft.cds
        table, add_row, delete_row, update = ft.draw(cds)
        if unit_test:
            from bokeh.io import output_file
            from bokeh.plotting import show, row, column
            output_file(os.path.join(project_dir, 'blixt_rp\\results_folder\\plot.html'))
            show(column(table, row(add_row, delete_row, update)))
            return None, None, None, None
        else:
            return table, add_row, delete_row, update


# -*- coding: utf-8 -*-
"""
Created on Sun 19th July 2026
Module for handling rockphysics fluid and mineral models

#
# The coding are inspired from CoPilot suggestions
#

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
from scipy.interpolate import interp1d
import pandas as pd
import unittest
import os, sys
from typing import Literal, List, Any, Dict
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
from blixt_rp.core.core import Interval, Intervals, Cutoffs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumeFraction:
    """
    A VolumeFraction object reflects how the volume fraction is defined in a single row of the 'FluidMixture' sheet
    of the project_table Excel file.
    For project-table definitions like fluids, minerals and substitutions, "frozen=True" is generally better because these
    objects represent configuration rather than something that changes during calculations.
    """
    mode: Literal["constant", "log", "complement"]
    value: float | str | None
    excel_value: float | str | None

@dataclass()
class FluidSpec:
    """
    Represents one row of the "Fluids" sheet in the project_table Excel file, formerly known as the "mother fluid"
    """
    name: str
    calculation_method: Literal["batzle and wang", "user specified", "interpolation"]
    bulk_gpa: float | None = None
    shear_gpa: float | None = 0.0
    density_gcc: float | None = None
    params: dict[str, Any] | None = None
    bd: Q_ | None = None

    def __str__(self, indent=4):
        """
        Return better readable string representation of the FluidSpec object.
        """
        keys = list(self.__dict__.keys())
        values = list(self.__dict__.values())
        out = ''
        indent = ' ' * indent
        for _key, _value in zip(keys, values):
            out += '{}{}: {}\n'.format(indent, _key, _value)
        return out

    def calc_elastics(self, fluid_type: str, burial_depth: Q_, pvt_table: dict | None = None) -> str:
        """
        Calculates the elastic (density and bulk modulus) for different types of fluids

        :param fluid_type:
        :param burial_depth:
        :param pvt_table:
            dict
            Dictionary that contains pressure, density and bulk modulus values from typically a PVT simulation.
            It can be used to calculate the elastic properties as a function of pressure for fluids (e.g. gas condensates)
            which can't be modelled accurately from Batzle & Wang
            E.G.
                pvt_table = dict(
                    pressure = Q_([10., 15., 20., 25., 30., 35., 40., 45., 50.], 'MPa'),
                    rho = Q_([0.12, 0.18, 0.25, 0.31, 0.38, 0.45, 0.52, 0.58, 0.64], 'g/cc'),
                    k = Q_([0.020, 0.035, 0.055, 0.080, 0.110, 0.145, 0.185, 0.230, 0.28], 'GPa')
                )
        :return:
        """
        calc_success = False
        _this_density = None
        _this_bulk = None
        if self.calculation_method.lower() == 'batzle and wang':
            _s = self.params['Salinity [ppm]']
            _p = self.params['P ref [MPa]'] + self.params['P gradient [MPa/m]'] * burial_depth.to('meter').magnitude
            _t = self.params['T ref [C]'] + self.params['T gradient [deg C/m]'] * burial_depth.to('meter').magnitude
            if fluid_type.lower() == 'brine':
                _this_density = rp.rho_b(_s, _p,  _t).value
                v_p_b = rp.v_p_b(_s, _p, _t).value
                _this_bulk = v_p_b**2 * _this_density * 1.E-6
                calc_success = True
            elif fluid_type.lower() == 'oil':
                _this_bulk, _this_density = rp.k_and_rho_o(
                    self.params['Oil API'],
                    self.params['Gas gravity'],
                    self.params['GOR'],
                    _p,
                    _t
                )
                _this_bulk = _this_bulk.value
                _this_density = _this_density.value
                calc_success = True
            elif fluid_type.lower() == 'gas':
                _this_bulk, _this_density = rp.k_and_rho_g(self.params['Gas gravity'], _p, _t)
                _this_bulk = _this_bulk.value
                _this_density = _this_density.value
                calc_success = True
            else:
                calc_success = False
                print_info('No calculation done for fluid type {}'.format(fluid_type),
                           'warning', logger)
        elif self.calculation_method.lower() == 'interpolation':
            _p = self.params['P ref [MPa]'] + self.params['P gradient [MPa/m]'] * burial_depth.to('meter').magnitude
            print('XXX: Pressure as calculated from parameters [MPa]: ', _p)
            if pvt_table is None:
                print_info('A table of pressure, rho and k is needed for the calculation of an Interpolation fluid',
                           'error', logger, 'IOError')
            _fluid = InterpolatedFluid(**pvt_table)
            if fluid_type.lower() == 'gas condensate':
                _this_density = _fluid.rho(Q_(_p, 'MPa')).to('g/cc').magnitude
                _this_bulk = _fluid.k(Q_(_p, 'MPa')).to('GPa').magnitude
                calc_success = True
            else:
                calc_success = False
                print_info('No calculation done for fluid type {}'.format(fluid_type),
                           'warning', logger)
        if calc_success:
            self.bd = burial_depth
            self.bulk_gpa = _this_bulk
            self.shear_gpa = 0.0
            self.density_gcc = _this_density

        return 'FluidSpec elastics calculated through {}, to: Bulk; {:.2f}, Shear; {:.2f}, Density; {:.2f}'.format(
                self.calculation_method, self.bulk_gpa, self.shear_gpa, self.density_gcc
            )

@dataclass()
class MineralSpec:
    """
    Represents one row of the "Fluids" sheet in the project_table Excel file, formerly known as the "mother fluid"
    """
    name: str
    calculation_method: Literal["interval average"] | None = None
    bulk_gpa: float | None = None
    shear_gpa: float | None = None
    density_gcc: float | None = None
    cutoffs: Cutoffs | None = None
    params: dict[str, Any] | None = None

    def __str__(self, indent=4):
        """
        Return better readable string representation of the FluidSpec object.
        """
        keys = list(self.__dict__.keys())
        values = list(self.__dict__.values())
        out = ''
        indent = ' ' * indent
        for _key, _value in zip(keys, values):
            out += '{}{}: {}\n'.format(indent, _key, _value)
        return out

    def calc_elastics(self, logs: dict[str, np.ndarray]) -> str:
        """
        Most minerals have the elastic properties fixed, and given in the project table Excel file.
        But some minerals are allowed to be calculated by an interval average

        So this function will only operate on minerals which have 'calculation_method' == 'interval average'

        :param logs:
            dict
            Dictionary with log values for the necessary parameters needed to calculate bulk and shear moduli, and
            density.
            E.G.
                logs = {'vp': np.ndarray, # in m/s
                        'vs': np.ndarray, # in m/s
                        'rho': np.ndarray, # in g/cc}

            If there is a Cutoff specified, the log values for the logs specified in the Cutoffs string must also be
            present.
            E.G if Cutoffs is given by e.g. 'VCL>0.8[], PHIE<0.1[]' the logs also need to contain
                {'VCL': np.ndarray,
                 'PHIE: np.ndarray}

            All of which must have the same length
        :return:
            str
        """
        make_logs_lowercase(logs)
        if self.calculation_method.lower() == 'interval average':
            # Check that the necessary logs are present
            for _param in ['vp', 'vs', 'rho']:
                if _param not in list(logs.keys()):
                    raise IOError('Log {} is missing among the input logs'.format(_param))

            # Check that the logs defined in any cutoff are present
            if self.cutoffs is not None:
                for _param in self.cutoffs.cutoff_params:
                    if _param.lower() not in list(logs.keys()):
                        raise IOError('Log {} is missing among the input logs'.format(_param))


            _this_density = logs['rho']
            _this_bulk = (logs['vp']**2 - 4. * logs['vs']**2 / 3.) * _this_density * 1.E-6
            _this_shear = logs['vs']**2 * _this_density * 1.E-6

            # Calculate the mask
            if self.cutoffs is not None:
                mask = self.cutoffs.create_mask(logs, verbose=False)
                # print('Cutoffs are defined ({}) -> '.format(self.cutoffs), mask[:5])
            else:
                n = len(logs[list(logs.keys())[0]])
                mask =  np.array(np.ones(n), dtype=bool)

            # Check that there any data left after masking
            n_data_left =  len(_this_bulk[mask])
            if n_data_left < 3:
                warn_txt = 'Little or no ({} data points) data to base the mineral calculation on for {}'.format(
                    n_data_left, self.name)
                print_info(warn_txt, 'warning', logger)

            self.bulk_gpa = float(np.nanmedian(_this_bulk[mask]))
            self.shear_gpa = float(np.nanmedian(_this_shear[mask]))
            self.density_gcc = float(np.nanmedian(_this_density[mask]))
        else:
            pass

        return 'MineralSpec elastics calculated through {}, to: Bulk; {:.2f}, Shear; {:.2f}, Density; {:.2f}'.format(
                self.calculation_method, self.bulk_gpa, self.shear_gpa, self.density_gcc
            )

@dataclass(frozen=True)
class MixtureComponent:
    """
    One object per row in the 'FluidMixture' (or 'MineralMixture') sheet of the project_table Excel file.
    A combination of MixtureComponent's are used to specify one SubstitutionCase

    "frozen=True" is generally better because this object shouldn't change during calculation
    """
    mother_name: str
    type: Literal["gas", "oil", "brine", "user specified", "mineral"]
    volume: VolumeFraction
    source: FluidSpec | MineralSpec

    def __str__(self, indent=4):
        """
        Return better readable string representation of the FluidSpec object.
        """
        keys = list(self.__dict__.keys())
        values = list(self.__dict__.values())
        out = ''
        indent = ' ' * indent
        for _key, _value in zip(keys, values):
            out += '{}{}: {}\n'.format(indent, _key, _value)
        return out

@dataclass()
class SubstitutionCase:
    """
    One substitution case object per well, interval and tag
    """
    well: str
    interval: str
    tag: str
    initial: List[MixtureComponent]
    final: List[MixtureComponent]
    initial_bulk_gpa: np.ndarray | float | None = None
    initial_shear_gpa: np.ndarray | float | None = 0.0
    initial_density_gcc: np.ndarray | float | None = None
    final_bulk_gpa: np.ndarray | float | None = None
    final_shear_gpa: np.ndarray | float | None = 0.0
    final_density_gcc: np.ndarray | float | None = None

    def calc_elastics(self, logs: dict[str, np.ndarray], burial_depth: pint.Quantity | None = None,
                      pvt_table: dict | None = None,
                      verbose: bool = False) -> str:
        """
        Calculates the elastic properties of the Voigt-Reuss-Hill averaged fluid for the initial and final fluids
        for this specific SubstitutionCase

        :param burial_depth:
            pint.Quantity
            The burial depth representative for this SubstitutionCase
        :param logs:
            dict
            Dictionary with log values that represents the volume fraction for a fluid (e.g. SW = water saturation)
        :param pvt_table:
            dict
            Dictionary that contains pressure, density and bulk modulus values from typically a PVT simulation.
            It can be used to calculate the elastic properties as a function of pressure for fluids (e.g. gas condensates)
            which can't be modelled accurately from Batzle & Wang
            E.G.
                pvt_table = dict(
                    pressure = Q_([10., 15., 20., 25., 30., 35., 40., 45., 50.], 'MPa'),
                    rho = Q_([0.12, 0.18, 0.25, 0.31, 0.38, 0.45, 0.52, 0.58, 0.64], 'g/cc'),
                    k = Q_([0.020, 0.035, 0.055, 0.080, 0.110, 0.145, 0.185, 0.230, 0.28], 'GPa')
                )
        :param verbose:
            bool
        :return:
            str
            String useful for logging the progress of the fluid substitution case
        """
        def print_nicely(_param_name, _param_value, indent=4):
            _str = '\n{}In {} | {} | {}, {} was calculated to: '.format(' '*indent, self.well, self.interval, self.tag, _param_name)
            _str += 'Min: {:.2f}, Mean: {:.2f}, Max {:.2f} '.format(np.nanmin(_param_value), np.nanmean(_param_value), np.nanmax(_param_value))
            try:
                _n = len(_param_value)
            except TypeError:
                _n = 1
            _str += 'with length {}'.format(_n)
            return _str

        info_txt = ''

        if len(self.initial) == 0 or len(self.final) == 0:
            raise IOError('Need a list of initial and final MixtureComponents to calculate the elastics')

        # Try to separate if the input MixtureComponents are based on FluidSpec or MineralSpec
        from_fluids = isinstance(self.initial[0].source, FluidSpec)
        type_txt = 'FLUIDS' if from_fluids else 'MINERALS'

        if from_fluids and burial_depth is None:
            err_txt = 'Burial depth is necessary for calculating the elastic properties of fluid mixtures'
            print_info(err_txt, 'error', logger, 'IOError')


        # Check the logs if they are of the same length, and extract that length
        n = None
        last_n = 0
        for i, _key in enumerate(list(logs.keys())):
            if i == 0:
                last_n = len(logs[_key])
            if last_n != len(logs[_key]):
                raise IOError('Each log must have the same length ({}). Not {}'.format(last_n, len(logs[_key])))
            last_n = len(logs[_key])
        n = last_n

        # Start with the initial MixtureComponents
        vol_fractions = resolve_volume_fractions(self.initial, logs, n)
        _f = []
        _bulk = []
        _shear = []
        _density = []
        info_txt += '----INITIAL {}----'.format(type_txt)
        for _x in self.initial:
            # _key = f'{_x.mother_name}:User specified' if _x.type == 'User specified' else f'{_x.mother_name}:{_x.type}'
            if from_fluids:
                _key = f'{_x.mother_name}:User specified' if _x.type == 'User specified' else f'{_x.mother_name}:{_x.type}'
                _info_txt = _x.source.calc_elastics(_x.type, burial_depth, pvt_table=pvt_table)
            else:
                _key = f'{_x.mother_name}:Mineral'
                _info_txt = _x.source.calc_elastics(logs)

            info_txt += '\n -Calculating initial elastics for {} in {} | {} | {}:\n   {}\n'.format(
                        _key, self.well, self.interval, self.tag, _info_txt)
            info_txt += '   Using volume mode {}, with value {}, which yields volume fractions from {:.2f} to {:.2f}'.format(
                    _x.volume.mode, _x.volume.value, np.nanmin(vol_fractions[_key]), np.nanmax(vol_fractions[_key]))

            _f.append(vol_fractions[_key])
            _bulk.append([_x.source.bulk_gpa] * n)
            _shear.append([_x.source.shear_gpa] * n)
            _density.append([_x.source.density_gcc] * n)
        # Bulk modulus
        self.initial_bulk_gpa = rp.vrh_bounds(_f, _bulk)[2]
        # Shear modulus
        self.initial_shear_gpa = rp.vrh_bounds(_f, _shear)[2]
        # Density
        self.initial_density_gcc = rp.vrh_bounds(_f, _density)[0]
        info_txt += print_nicely('initial_bulk_gpa', self.initial_bulk_gpa)
        info_txt += print_nicely('initial_shear_gpa', self.initial_shear_gpa)
        info_txt += print_nicely('initial_density_gcc', self.initial_density_gcc)
        info_txt += '\n'

        # Then take the final fluids
        vol_fractions = resolve_volume_fractions(self.final, logs, n)
        _f = []
        _bulk = []
        _shear = []
        _density = []
        if from_fluids:  # The mineral composition will stay the same, so we only print the final fluids
            info_txt += '\n----FINAL {}----'.format(type_txt)
        for _x in self.final:
            # _key = f'{_x.mother_name}:User specified' if _x.type == 'User specified' else f'{_x.mother_name}:{_x.type}'
            if from_fluids:
                _key = f'{_x.mother_name}:User specified' if _x.type == 'User specified' else f'{_x.mother_name}:{_x.type}'
                _info_txt = _x.source.calc_elastics(_x.type, burial_depth, pvt_table=pvt_table)
                info_txt += '\n -Calculating final elastics for {} in {} | {} | {}:\n   {}\n'.format(
                    _key, self.well, self.interval, self.tag, _info_txt)
                info_txt += '   Using volume mode {}, with value {}, which yields volume fractions from {:.2f} to {:.2f}\n'.format(
                    _x.volume.mode, _x.volume.value, np.nanmin(vol_fractions[_key]), np.nanmax(vol_fractions[_key]))
            else:
                _key = f'{_x.mother_name}:Mineral'
                _info_txt = _x.source.calc_elastics(logs)

            _f.append(vol_fractions[_key])
            _bulk.append([_x.source.bulk_gpa] * n)
            _shear.append([_x.source.shear_gpa] * n)
            _density.append([_x.source.density_gcc] * n)
        # Bulk modulus
        self.final_bulk_gpa = rp.vrh_bounds(_f, _bulk)[2]
        # Shear modulus
        self.final_shear_gpa = rp.vrh_bounds(_f, _shear)[2]
        # Density
        self.final_density_gcc = rp.vrh_bounds(_f, _density)[0]
        if from_fluids:
            info_txt += print_nicely('final_bulk_gpa', self.final_bulk_gpa)
            info_txt += print_nicely('final_shear_gpa', self.final_shear_gpa)
            info_txt += print_nicely('final_density_gcc', self.final_density_gcc)
            info_txt += '\n'

        return info_txt

class InterpolatedFluid:
    """
    Class that takes an input table of simulated / observed pressure, density and bulk modulus values
    and allows the density and bulk modulus to be extracted at ~any pressure.
    These are useful for e.g. gas condensates which are not well covered by the Batzle & Wang model
    """
    def __init__(self, pressure: pint.Quantity , rho: pint.Quantity, k: pint.Quantity):

        self.rho_interp = interp1d(
            pressure.magnitude,
            rho.magnitude,
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.k_interp = interp1d(
            pressure.magnitude,
            k.magnitude,
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.p_min = min(pressure)
        self.p_max = max(pressure)
        self.pressure_units = pressure.units
        self.rho_units = rho.units
        self.k_units = k.units

    def rho(self, pressure: pint.Quantity):
        if pressure < self.p_min or pressure > self.p_max:
            warn_txt = 'Pressure {} is outside the initial bounds ({} - {}) and extrapolation is used'.format(
                pressure, self.p_min, self.p_max)
            print_info(warn_txt, 'warning', logger)
        return Q_(self.rho_interp(pressure.to(self.pressure_units)), self.rho_units)

    def k(self, pressure: pint.Quantity):
        return Q_(self.k_interp(pressure.to(self.pressure_units)), self.k_units)

def read_sheet_table(xlsx: str | Path, sheet_name: str, required_columns: set[str]) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name=sheet_name, header=None, engine='openpyxl')
    header_row = None
    for i, row in raw.iterrows():
        vals = {str(v).strip() for v in row.dropna().tolist()}
        if required_columns.issubset(vals):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f'Could not find header row in sheet {sheet_name!r}')
    df = pd.read_excel(xlsx, sheet_name=sheet_name, header=header_row, engine='openpyxl')
    df = df.dropna(how='all')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_str(x: Any) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None

def parse_volume_fraction(x: Any) -> VolumeFraction:
    s = clean_str(x)
    if s is None:
        raise ValueError('Missing Volume fraction')
    if s.lower() == 'complement':
        return VolumeFraction('complement', None, x)
    # Accept comma decimal input from Excel/users, e.g. '0,2'
    try:
        val = float(s.replace(',', '.'))
        if not 0 <= val <= 1:
            raise ValueError(f'Constant volume fraction outside [0, 1]: {x!r}')
        return VolumeFraction('constant', val, x)
    except ValueError:
        # Not numeric: interpret as well-log name, e.g. SW
        return VolumeFraction('log', s, x)

def as_float_or_none(x: Any) -> float | None:
    if pd.isna(x):
        return None
    return float(x)

def load_fluid_specs(xlsx: str | Path) -> dict[str, FluidSpec]:
    fluids = read_sheet_table(xlsx, 'Fluids', {'Name', 'Calculation method'})
    fluids = fluids[fluids['Name'].notna()].copy()
    if fluids['Name'].duplicated().any():
        dups = fluids.loc[fluids['Name'].duplicated(), 'Name'].tolist()
        raise ValueError(f'Duplicate fluid names in Fluids sheet: {dups}')
    out = {}
    for _, r in fluids.iterrows():
        name = clean_str(r['Name'])
        method = clean_str(r['Calculation method']) or ''
        if method.lower() not in ['batzle and wang', 'user specified', 'interpolation']:
            raise IOError('Fluid calculation method must be either "Batzle and Wang" or "User specified". Not: {}'.format(
                method
            ))
        params = {c: None if pd.isna(r[c]) else r[c] for c in fluids.columns if c not in {'Name'}}
        shear = as_float_or_none(r.get('Shear moduli [GPa]'))
        # For pore fluids, missing shear modulus is normally treated as zero.
        if shear is None and method.lower() == 'user specified':
            shear = 0.0
        out[name] = FluidSpec(
            name=name,
            calculation_method=method.lower(),
            bulk_gpa=as_float_or_none(r.get('Bulk moduli [GPa]')),
            shear_gpa=shear,
            density_gcc=as_float_or_none(r.get('Density [g/cm3]')),
            params=params,
        )
    return out

def load_mineral_specs(xlsx: str | Path) -> dict[str, FluidSpec]:
    minerals = read_sheet_table(xlsx, 'Minerals', {'Name', 'Calculation method', 'Cutoffs'})
    minerals = minerals[minerals['Name'].notna()].copy()
    if minerals['Name'].duplicated().any():
        dups = minerals.loc[minerals['Name'].duplicated(), 'Name'].tolist()
        raise ValueError(f'Duplicate fluid names in Fluids sheet: {dups}')
    out = {}
    for _, r in minerals.iterrows():
        name = clean_str(r['Name'])
        method = clean_str(r['Calculation method']) or ''
        cutoffs = clean_str(r['Cutoffs']) or None
        if cutoffs is not None:
            cutoffs = Cutoffs(from_string=cutoffs)
        if method.lower() not in ['interval average', '']:
            raise IOError('Mineral calculation method must be either "Interval average" or empty. Not: {}'.format(
                method
            ))
        params = {c: None if pd.isna(r[c]) else r[c] for c in minerals.columns if c not in {'Name'}}

        out[name] = MineralSpec(
            name=name,
            calculation_method=method.lower(),
            bulk_gpa=as_float_or_none(r.get('Bulk moduli [GPa]')),
            shear_gpa=as_float_or_none(r.get('Shear moduli [GPa]')),
            density_gcc=as_float_or_none(r.get('Density [g/cm3]')),
            cutoffs=cutoffs,
            params=params,
        )
    return out

def component_from_row(r: pd.Series, sources: dict[str, FluidSpec | MineralSpec]) -> MixtureComponent:
    """
    Returns the Mixture component from one row in the project_table Excel fil

    :param r:
    :param sources:
    :return:
    """

    # Discriminate between the case where the sources are a dictionary of FluidSpec's or MineralSpec's
    if isinstance(list(sources.values())[0], FluidSpec):
        fluid = True
    else:
        fluid = False

    if fluid:
        source_name = clean_str(r['Fluid name'])
        _type = clean_str(r['Fluid type']) or ''
    else:
        source_name = clean_str(r['Mineral name'])
        _type = 'mineral'

    if source_name not in sources:
        raise ValueError(f'Fluid / Mineral mixture references unknown source {source_name!r}')
    return MixtureComponent(
        mother_name=source_name,
        type=_type,
        volume=parse_volume_fraction(r['Volume fraction']),
        source=sources[source_name],
    )

def validate_component_set(components: list[MixtureComponent], label: str) -> None:
    n_comp = sum(c.volume.mode == 'complement' for c in components)
    # for c in components:
    #     print('-x-')
    #     print(c.volume.mode)
    #     print(c.volume.value)
    if n_comp > 1:
        raise ValueError(f'{label}: more than one complement fraction')
    const_sum = sum(c.volume.value for c in components if c.volume.mode == 'constant')
    if const_sum > 1 + 1e-9:
        raise ValueError(f'{label}: constant fractions sum to {const_sum:.3f} > 1')
    if n_comp == 0 and all(c.volume.mode == 'constant' for c in components):
        if abs(const_sum - 1) > 1e-6:
            raise ValueError(f'{label}: constant fractions sum to {const_sum:.3f}, expected 1')

def load_substitution_cases(xlsx: str | Path, from_fluid_mixture: bool = True) -> list[SubstitutionCase]:
    """
    By including "Well name", "Interval name", "Tag" in the groupby method of a panda table we can run several
    substitution scenarios for the same well and interval
    :param xlsx:

    :param from_fluid_mixture:
        bool
        Setting this to False tells the script to load the substitution cases from a mineral mixture table.
        Now in reality, The mineral mixture will be the same before and after fluid substitution, so substitution order
        doesn't make sense.
    :return:
    """
    # Load the fluid or mineral mixtures
    if from_fluid_mixture:
        fluids = load_fluid_specs(xlsx)
        minerals = None
        mix = read_sheet_table(xlsx, 'Fluid mixtures', {'Use', 'Substitution order', 'Well name', 'Interval name',
                                                        'Fluid name', 'Fluid type', 'Volume fraction'})
    else:
        fluids = None
        minerals = load_mineral_specs(xlsx)
        mix = read_sheet_table(xlsx, 'Mineral mixtures', {'Use', 'Well name', 'Interval name',
                                                          'Mineral name', 'Volume fraction'})

    # Only keep the mixtures which have a 'Yes' in the 'Use' column
    mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()

    # For fluid mixtures, clean up and check the 'Substitution order' column
    if from_fluid_mixture:
        mix['Substitution order'] = mix['Substitution order'].map(lambda x: (clean_str(x) or '').title())
        # Checks if there are any other substitution orders that "Initial" and "Final"
        bad_orders = sorted(set(mix['Substitution order']) - {'Initial', 'Final'})
        if bad_orders:
            raise ValueError(f'Unsupported substitution order values: {bad_orders}')

    # Make sure the loaded table is consistent with respect to our requirements
    for col in ['Well name', 'Interval name']:
        if mix[col].isna().any():
            raise ValueError(f'Missing {col} in enabled mixtures rows')
    if 'Tag' not in mix.columns:
        mix['Tag'] = None

    # Create an empty results container to begin with
    cases: list[SubstitutionCase] = []

    # Group the table and start iterating
    group_cols = ['Well name', 'Interval name', 'Tag']
    for key, g in mix.groupby(group_cols, dropna=False, sort=False):
        well, interval, tag = key
        well, interval, tag = clean_str(well), clean_str(interval), clean_str(tag)
        if from_fluid_mixture:
            initial = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Initial')].iterrows()]
            final = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Final')].iterrows()]
        else:
            # For minerals we have the same initial and final minerals
            initial = [component_from_row(r, minerals) for _, r in g.iterrows()]
            final = [component_from_row(r, minerals) for _, r in g.iterrows()]

        if not initial or not final:
            raise ValueError(f'{well}/{interval}/{tag}: need both Initial and Final mixture rows')
        validate_component_set(initial, f'{well}/{interval}/{tag}/Initial')
        validate_component_set(final, f'{well}/{interval}/{tag}/Final')
        cases.append(SubstitutionCase(well=well, interval=interval, tag=tag, initial=initial, final=final))
    return cases

def resolve_volume_fractions(components: list[MixtureComponent], logs: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """
    Calculates the volume fractions of each of the fluids-, mineral-, mixtures listed in components

    :param components:
        list
        List of MixtureComponent's
    :param logs:
        dict
        Dictionary with log values that represents the volume fraction for a fluid (e.g. SW = water saturation)
    :param n:
        int
        Length of the input log and/or output
    :return:
        dict
        Dictionary with the volume fraction for each fluid in the SubstitionCase which is made up from the input list
        of MixtureComponent's
    """
    make_logs_lowercase(logs)
    # Try to separate if the input MixtureComponents are based on FluidSpec or MineralSpec
    from_fluids = isinstance(components[0].source, FluidSpec)

    known_sum = np.zeros(n, dtype=float)
    out: dict[str, np.ndarray] = {}
    complement_name = None
    for c in components:
        if from_fluids:
            key = f'{c.mother_name}:User specified' if c.type == 'User specified' else f'{c.mother_name}:{c.type}'
        else:
            key = f'{c.mother_name}:Mineral'

        if c.volume.mode == 'constant':
            arr = np.full(n, float(c.volume.value))
            out[key] = arr
            known_sum += arr
        elif c.volume.mode == 'log':
            log_name = str(c.volume.value).lower()
            if log_name not in [_l.lower() for _l in logs]:
                raise KeyError(f'Missing log curve {log_name!r}')
            arr = np.asarray(logs[log_name], dtype=float)
            if arr.shape[0] != n:
                raise ValueError(f'Log {log_name!r} length {arr.shape[0]} != expected {n}')
            out[key] = arr
            known_sum += arr
        else:
            complement_name = key
    if complement_name is not None:
        out[complement_name] = 1.0 - known_sum
    if np.nanmin(np.vstack(list(out.values()))) < -1e-6:
        raise ValueError('One or more resolved volume fractions are negative')
    total = np.sum(np.vstack(list(out.values())), axis=0)
    if not np.allclose(total, 1.0, atol=1e-6, equal_nan=True):
        raise ValueError(f'Resolved volume fractions do not sum to 1; range={np.nanmin(total):.3f}-{np.nanmax(total):.3f}')
    return out

def make_logs_lowercase(x:dict):
    for _key in list(x.keys()):
        x[_key.lower()] = x.pop(_key)

class TestCases(unittest.TestCase):
    def test_lowercase(self):
        a = {'A':1, 'B':2}
        print(a)
        make_logs_lowercase(a)
        print(a)

    def test_print_fluid_spec(self):
        fs = FluidSpec(
            name='AnyThing',
            calculation_method='user specified',
            bulk_gpa=30,
            shear_gpa=16,
            density_gcc=3
        )
        print(fs)

    def test_load_fluid_spec(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        for _key, _value in list(load_fluid_specs(project_table).items()):
            for _fluid in ['brine', 'oil', 'gas']:
                info_txt = _value.calc_elastics(_fluid, Q_(3000., 'm'))
                print(info_txt, _key, _fluid , _value)

    def test_load_mineral_spec(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        for _key, _value in list(load_mineral_specs(project_table).items()):
            print(_key, _value)

    def test_read_sheet_table(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        mix = read_sheet_table(project_table, 'Fluid mixtures',
                               {'Use', 'Substitution order',
                                'Well name', 'Interval name',
                                'Fluid name', 'Fluid type', 'Volume fraction'})
        # The following code line should only keep the lines in the table which have "Use == Yes"
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        print(mix)


    def test_component_from_row(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')

        # First test the case with fluids
        fluids = load_fluid_specs(project_table)
        mix = read_sheet_table(project_table, 'Fluid mixtures',
                               {'Use', 'Substitution order',
                                'Well name', 'Interval name',
                                'Fluid name', 'Fluid type', 'Volume fraction'})
        # The following code line should only keep the lines in the table which have "Use == Yes"
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        group_cols = ['Well name', 'Interval name', 'Tag']
        for key, g in mix.groupby(group_cols, dropna=False, sort=False):
            initial = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Initial')].iterrows()]
            final = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Final')].iterrows()]
            print('Initial:\n', '\n'.join([str(_i) for _i in initial]))
            print('Final:\n', '\n'.join([str(_f) for _f in final]))

        # Next, try loading the mineral components
        minerals = load_mineral_specs(project_table)
        mix = read_sheet_table(project_table, 'Mineral mixtures',
                               {'Use',
                                'Well name', 'Interval name',
                                'Mineral name', 'Volume fraction'})
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        group_cols = ['Well name', 'Interval name']
        for key, g in mix.groupby(group_cols, dropna=False, sort=False):
            _minerals = [component_from_row(r, minerals) for _, r in g.iterrows()]
            print('Minerals:\n', '\n'.join([str(_i) for _i in _minerals]))

    def test_validate_component_set(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        fluids = load_fluid_specs(project_table)
        mix = read_sheet_table(project_table, 'Fluid mixtures',
                               {'Use', 'Substitution order',
                                'Well name', 'Interval name',
                                'Fluid name', 'Fluid type', 'Volume fraction'})
        # The following code line should only keep the lines in the table which have "Use == Yes"
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        group_cols = ['Well name', 'Interval name', 'Tag']
        for key, g in mix.groupby(group_cols, dropna=False, sort=False):
            initial = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Initial')].iterrows()]
            final = [component_from_row(r, fluids) for _, r in g[g['Substitution order'].eq('Final')].iterrows()]
            print('Initial is valid?:\n', validate_component_set(initial, 'TEST'))
            print('Final is valid?:\n', validate_component_set(final, 'TEST FINAL'))

        # Next, try loading the mineral components
        minerals = load_mineral_specs(project_table)
        mix = read_sheet_table(project_table, 'Mineral mixtures',
                               {'Use',
                                'Well name', 'Interval name',
                                'Mineral name', 'Volume fraction'})
        mix = mix[mix['Use'].astype(str).str.strip().str.lower().eq('yes')].copy()
        group_cols = ['Well name', 'Interval name']
        for key, g in mix.groupby(group_cols, dropna=False, sort=False):
            _minerals = [component_from_row(r, minerals) for _, r in g.iterrows()]
            print('Minerals are valid:\n', validate_component_set(_minerals, 'TEST MINERAL'))


    def test_load_substitution_cases(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        # First the fluids
        cases = load_substitution_cases(project_table, from_fluid_mixture=True)
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            print('  Initial:', [(x.mother_name, x.type, x.volume.mode, x.volume.value) for x in c.initial])
            print('  Final:  ', [(x.mother_name, x.type, x.volume.mode, x.volume.value) for x in c.final])

        # Then the minerals
        cases = load_substitution_cases(project_table, from_fluid_mixture=False)
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            print('  Initial:', [(x.mother_name, x.type, x.volume.mode, x.volume.value) for x in c.initial])
            print('  Final:  ', [(x.mother_name, x.type, x.volume.mode, x.volume.value) for x in c.final])

    def test_resolve_volume_fraction(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        n = 10

        # First for fluids
        cases = load_substitution_cases(project_table)
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            _out = resolve_volume_fractions(c.initial, {'SW': np.full(n, 0.33)}, n)
            print('  Initial: ', _out)
            _out = resolve_volume_fractions(c.final, {'SW': np.full(n, 0.33)}, n)
            print('  Final: ', _out)

        # Then for minerals
        # And in this case, the initial and final results will be the same
        cases = load_substitution_cases(project_table, from_fluid_mixture=False)
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            _out = resolve_volume_fractions(c.initial, {'VCL': np.full(n, 0.33)}, n)
            print('  Initial: ', _out)
            _out = resolve_volume_fractions(c.final, {'VCL': np.full(n, 0.33)}, n)
            print('  Final: ', _out)

    def test_calc_elastics_for_substitution_cases(self):
        project_table = os.path.join(project_dir, 'blixt_rp\\excels\\project_table_new.xlsx')
        n = 10
        logs = {
            'vp': np.linspace(2900., 3100, n),
            'vs': np.linspace(1400., 1600, n),
            'rho': np.linspace(2.3, 2.5, n),
            'VCL': np.linspace(1., 0.6, n),
            'PHIE': np.linspace(0., 0.2, n),
            'SW': np.linspace(0.05, 1., n)
        }

        # First for fluids
        cases = load_substitution_cases(project_table)
        print('FLUIDS')
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            _str = c.calc_elastics(logs, Q_(3000., 'meter'), verbose=True)
            print(_str)

        # Then for minerals
        cases = load_substitution_cases(project_table, from_fluid_mixture=False)
        print('\nMINERALS')
        for c in cases:
            print(f'{c.well} | {c.interval} | tag={c.tag}')
            _str = c.calc_elastics(logs, Q_(3000., 'meter'), verbose=True)
            print(_str)

    def test_calc_elastics_for_substitution_cases_detailed(self):
        fluid_params = {'Bulk moduli [GPa]': None, 'Shear moduli [GPa]': None, 'Density [g/cm3]': None,
                        'Calculation method': 'Batzle and Wang',
                        'T gradient [deg C/m]': 0.03, 'T ref [C]': 5.0,
                        'P gradient [MPa/m]': 0.0107, 'P ref [MPa]': 0.0,
                        'Salinity [ppm]': 70000.0, 'GOR': 1.0, 'Oil API': 30.0,
                        'Gas gravity': 0.6, 'Gas mixing': 'Brie', 'Brie exponent': 2.0}
        fs = FluidSpec(
            name='MyFluid',
            calculation_method='batzle and wang',
            params=fluid_params
        )
        mc1 = MixtureComponent(
            mother_name='MyFluid',
            type='brine',
            volume=VolumeFraction(
                mode='constant',
                value=1,
                excel_value=1
            ),
            source=fs
        )
        mc2 = MixtureComponent(
            mother_name='MyFluid',
            type='brine',
            volume=VolumeFraction(
                mode='log',
                value='SW',
                excel_value='SW'
            ),
            source=fs
        )
        mc3 = MixtureComponent(
            mother_name='MyFluid',
            type='gas',
            volume=VolumeFraction(
                mode='complement',
                value=None,
                excel_value='complement'
            ),
            source=fs
        )
        sc = SubstitutionCase(
            well='Well 1',
            interval='Cook FM',
            tag='GAS',
            initial=[mc1],
            final=[mc2, mc3]
        )

        info_txt = fs.calc_elastics('brine', Q_(2000., 'm'))
        brine_bulk = fs.bulk_gpa
        brine_density = fs.density_gcc

        info_txt = fs.calc_elastics('gas', Q_(2000., 'm'))
        gas_bulk = fs.bulk_gpa
        gas_density = fs.density_gcc

        # The water saturation goes from 0 to 1
        info_txt = sc.calc_elastics({'SW': np.linspace(0.05, 1., 10)}, Q_(2000., 'm'), verbose=False)

        # print(brine_bulk, brine_density)
        # print(gas_bulk, gas_density)
        # print(sc.final_bulk_gpa)
        # print(sc.final_density_gcc)

        # The last sample of the VRH averaged value of the final bulk should equal the brine bulk moduli
        self.assertAlmostEqual(sc.final_bulk_gpa[-1], brine_bulk)

        # The last sample of the VRH averaged value of the final density should equal the brine density
        self.assertAlmostEqual(sc.final_density_gcc[-1], brine_density)

        # The first sample of the VRH averaged value of the final bulk should equal the gas bulk moduli
        self.assertAlmostEqual(sc.final_bulk_gpa[0], gas_bulk)

        # The first sample of the VRH averaged value of the final density should equal the gas density
        self.assertAlmostEqual(sc.final_density_gcc[0], gas_density)

    def test_calc_elastics_for_mineral_detailed(self):
        cutoffs = Cutoffs(from_string='VCL>0.8[], PHIE<0.1[]')
        ms = MineralSpec(
            name='MyMineral',
            calculation_method='interval average',
            cutoffs=cutoffs
        )
        n=20
        logs = {
            'vp': np.linspace(2900., 3100, n),
            'vs': np.linspace(1400., 1600, n),
            'rho': np.linspace(2.3, 2.5, n),
            'VCL': np.linspace(1., 0.6, n),
            'PHIE': np.linspace(0., 0.2, n),
            'SW': np.linspace(0.05, 1., n)
        }
        info_txt = ms.calc_elastics(logs)
        print(ms)

        # if we shift the order of data in VCL and PHIE, we should get a higher result, as the mask
        # then should filter out the lower values of vp, vs, and rho
        logs['VCL'] = np.linspace(0.6, 1, 20)
        logs['PHIE'] = np.linspace(0.2, 0., 20)
        info_txt = ms.calc_elastics(logs)
        print(ms)

    def test_interpolated_fluid(self):
        pvt_table = dict(
            pressure = Q_([10., 15., 20., 25., 30., 35., 40., 45., 50.], 'MPa'),
            rho = Q_([0.12, 0.18, 0.25, 0.31, 0.38, 0.45, 0.52, 0.58, 0.64], 'g/cc'),
            k = Q_([0.020, 0.035, 0.055, 0.080, 0.110, 0.145, 0.185, 0.230, 0.28], 'GPa')
        )

        eos_fluid = InterpolatedFluid(**pvt_table)

        for p in [Q_(_p, 'Pa') for _p in [12.5E+6, 15.E+6, 17.5E+6, 60.E+6]]:
            print(p, eos_fluid.rho(p), eos_fluid.k(p))
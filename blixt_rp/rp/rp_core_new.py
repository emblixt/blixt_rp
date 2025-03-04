import numpy as np
import matplotlib.pyplot as plt
import logging
# from dataclasses import dataclass
from copy import deepcopy
import sys
import bruges.rockphysics.rockphysicsmodels as brr

from .. import ureg, Q_
import pint

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.basename(__file__).replace('blixt_rp\\blixt_rp\\rp', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.rp_utils.definitions as ud

logger = logging.getLogger(__name__)

class RockType:
    """
    Class containing the elastic properties, and the statistics describing it, for one single rock type
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
                 vs_rho_cc: float | pint.Quantity | None = None
                 ):
        self.name = name
        self.vp = vp
        self.vs = vs
        self.rho= rho
        self.vp_std_dev = vp_std_dev
        self.vs_std_dev = vs_std_dev
        self.rho_std_dev= rho_std_dev
        self.vp_vs_cc = vp_vs_cc
        self.vp_rho_cc = vp_rho_cc
        self.vs_rho_cc= vs_rho_cc

    def from_excel(self, excel_file, name):


class RockTypes:
    """
    Class containing the properties of the different rock types that can be modelled
    """
    def __init__(self,
                 rock_types: list | None = None
                 ):
        if rock_types is None:
            rock_types = []
        self._rock_types = rock_types

    @property
    def rock_types(self):
        return self._rock_types

    @rock_types.setter
    def rock_types(self, value: list):
       self._rock_types = value

    def __add__(self, other: RockType):
        self._rock_types.append(other)


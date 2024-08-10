""" Init file. """
import os
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
dim_file = os.path.join(os.path.dirname(__file__), 'units_to_pint.txt')
ureg.load_definitions(dim_file)

from .blixt_rp import *
__path__ = [os.path.join(os.path.dirname(__file__), 'blixt_rp')]

import os

from pint import UnitRegistry
ureg = UnitRegistry()
# ureg.formatter.default_format = 'C'
Q_ = ureg.Quantity
dim_file = os.path.join(os.path.dirname(__file__), 'units_to_pint.txt')
print('Loaded: {}'.format(dim_file))
ureg.load_definitions(dim_file)

__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass

"""
Class to handle (working) intervals and tops
"""
import sys
import os

# Add to path to avoid having to install libraries, useful in development
project_dir = os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core','')
sys.path.append(os.path.join(str(project_dir), 'blixt_utils'))

from blixt_utils.misc.attribdict import AttribDict


class Intervals(AttribDict):
    pass


class Well(object):
    """
    Class for handling the top and base values of an interval for one well
    """
    def __init__(self, name, top, base, coord_units='meter', coord_type='md'):
        pass

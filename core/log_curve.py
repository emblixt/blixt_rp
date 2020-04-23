# -*- coding: utf-8 -*-
"""
Module for handling LogCurve objects
:copyright:
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from utils.attribdict import AttribDict
from utils.utils import arrange_logging, info
from datetime import datetime
import numpy as np

class Header(AttribDict):
    """
    A container for header information for a log curve object.
    A "Header" object may contain all header information necessary to describe a well log
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required
    """
    defaults = {
        'name': None,
        'well': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat(),
        'orig_filename': None,  
        'modification_date': None,
        'modification_history': None,
        'log_type': None
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
        if key in ['creation_date', 'modification_date', 'modification_history']:
            if key == 'modification_history':  # append instead of overwrite
                this_history = self.__getitem__(key)
                if this_history is None:
                    super(Header, self).__setitem__(key, value)
                else:
                    super(Header, self).__setitem__(key, '{}, {}'.format(this_history, value))
            pass
        else:

            # all other keys
            if isinstance(value, dict):
                super(Header, self).__setitem__(key, AttribDict(value))
            else:
                super(Header, self).__setitem__(key, value)

        super(Header, self).__setitem__('modification_date', datetime.now().isoformat())

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


class LogCurve(object):
    """
    Class handling logs of a well
    """
    
    def __init__(self,
                 name=None,
                 block=None,
                 well=None,
                 data=np.array([]),
                 header=None):
        _data_sanity_checks(data)
        self.name = name
        self.block = block
        self.well = well
        if header is None:
            header = {}
        if 'name' not in list(header.keys()) or header['name'] is None:
            header['name'] = name
        if 'well' not in list(header.keys()) or header['well'] is None:
            header['well'] = well

        if isinstance(header, dict):
            self.header = Header(header)
        elif isinstance(header, Header):
            self.header = header
        else:
            raise IOError('Input header is neither dictionary nor Header')
        #super(LogCurve, self).__setattr__('data', data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_log_type(self):
        log_type = None
        if self.header.log_type is not None:
            log_type = self.header.log_type
        return log_type

    log_type = property(get_log_type)

    #def get_log_name(self):
    #    log_name = None
    #    if self.header.name is not None:
    #        log_type = self.header.name
    #    return log_name
    #
    #name = property(get_log_name)
    
    
def _data_sanity_checks(value):
    """
    Check if a given input is suitable to be used for LogCurve.data. Raises the
    corresponding exception if it is not, otherwise silently passes.
    """
    if not isinstance(value, np.ndarray):
        msg = "LogCurve.data must be a NumPy array."
        raise ValueError(msg)
    if value.ndim != 1:
        msg = ("NumPy array for LogCurve.data has bad shape ('%s'). Only 1-d "
               "arrays are allowed for initialization.") % str(value.shape)
        raise ValueError(msg)
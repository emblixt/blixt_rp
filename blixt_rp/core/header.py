# -*- coding: utf-8 -*-

from datetime import datetime
import logging

from blixt_rp.rp_utils.version import info
from blixt_utils.misc.param import Param
from blixt_utils.misc.attribdict import AttribDict

logger = logging.getLogger(__name__)


class Header(AttribDict):
    """
    Class for well header information
    A ``Header`` object may contain all header information (also known as meta
    data) of a Well object.
    Those headers may be
    accessed or modified either in the dictionary style or directly via a
    corresponding attribute. There are various default attributes which are
    required by every variable import and export modules
    :param
        header: Dictionary containing meta information of a single
        Well object.
    """
    defaults = {
        'name': None,
        'well': None,
        'creation_info': info(),
        'creation_date': datetime.now().isoformat(),
        'orig_filename': None,
        'modification_date': None,
        'modification_history': '',
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
        if key in ['creation_date', 'modification_date']:
            pass
        super(Header, self).__setitem__(
            'modification_date',  datetime.now().isoformat())

        # all other keys
        if isinstance(value, float) or isinstance(value, int):
            super(Header, self).__setitem__(key, Param(name=key, value=value))
        else:
            super(Header, self).__setitem__(key, value)

    __setattr__ = __setitem__

    def __str__(self):
        """
        Return better readable string representation of Header object.
        """
        # keys = ['creation_date', 'modification_date', 'temp_gradient', 'temp_ref']
        keys = list(self.keys())
        try:
            i = max([len(k) for k in keys])
        except ValueError:
            # no keys
            return ''
        pattern = "%%%ds: %%s" % (i)
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


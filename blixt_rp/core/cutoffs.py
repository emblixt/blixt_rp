# Object class that contains masks for masking out data
# Name
# Description
# Parameter: operator
#
import os, sys
from .. import ureg, Q_
import pint
import logging

from .. import ureg, Q_
import pint

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.basename(__file__).replace('blixt_rp\\blixt_rp\\rp', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_rp.core.log_tables import LogTable
from blixt_utils.utils import print_info

logger = logging.getLogger(__name__)

class CutoffRule:
    """
    Class for rules for cutoffs
    NOTE, a parameter (param) is masked OUT if its value is NOT within the given rules and limits
    This behaviour is guaranteed through the function create_mask in blixt_utils.misc.masks.py
    EG:
    > t = np.arange(10)
    > mask = create_mask(t,'>=',8)
    > print(t[mask])
    >      array([8, 9])
    """
    def __init__(self,
                 param: str,
                 operator: str | None,
                 limit: pint.Quantity | list | str ):
        """

        :param param:
            name of the parameter
        :param operator:
            string or None
            representing the masking operation
            '<':  masked_less
            '<=': masked_less_equal
            '>':  masked_greater
            '>=': masked_greater_equal
            '><': masked_inside
            '==': masked_equal
            '!=': masked_not_equal
        :param limit:
            pint.Quantity | list | str
            If limit is a string, the CutoffRule becomes an interval cutoff, where the string is the name of
            the interval we limit the data to.
        """
        raise_unit_error = False
        self.interval_cutoff = False

        # instantiate 'param'
        self.__name__ = param
        self.param = param

        # instantiate 'operator'
        if operator is None and not isinstance(limit, str):
            error_txt = 'If no operator is provided, the limit must be the name of an interval, not {}'.format(limit)
            print_info(error_txt, 'error', logger=logger, raiser='IOError')
        if isinstance(operator, str):
            if operator not in [ '<', '<=', '>', '>=', '><', '==', '!=']:
                error_txt = 'Could not match "{}" with any valid operator'.format(operator)
                print_info(error_txt, 'error', logger=logger, raiser='IOError')

        self.operator = operator

        # instantiate 'limit'
        if isinstance(limit, list):
            for _item in limit:
                if not isinstance(_item, pint.Quantity):
                    raise_unit_error = True
        elif isinstance(limit, str):
            self.interval_cutoff = True
        elif not isinstance(limit, pint.Quantity):
            raise_unit_error = True
        if raise_unit_error:
            error_txt = 'Limits must be provided as a pint.Quantity (has units)'
            print_info(error_txt, 'error', logger=logger, raiser='IOError')

        self.limit = limit

    def __str__(self):
        _param = '' if self.param is None else self.param
        _operator = '' if self.operator is None else self.operator
        if isinstance(self.limit, list):
            _limits = '[{}]'.format(' ,'.join([str(m.magnitude) for m in self.limit]))
        elif isinstance(self.limit, pint.Quantity):
            _limits = str(self.limit.magnitude)
        else:
            _limits = self.limit
        return '{}: {} {}'.format( _param, _operator, _limits)

class Cutoffs:
    def __init__(self,
                 name: str | None = None,
                 log_table: LogTable | None = None,
                 cutoffs: list | None = None
                 ):
        self.info_keys = ['name', 'desc', 'log_table', 'info_keys']
        self.name = name
        self.log_table = log_table
        if cutoffs is None:
            cutoffs = []
        for _key in cutoffs:
            if not isinstance(_key, CutoffRule):
                error_txt = 'Cutoffs must be provided as a CutoffRule'
                print_info(error_txt, 'error', logger=logger, raiser='IOError')
        self.cutoffs = cutoffs

    @property
    def cutoff_names(self):
        return [_x.param for _x in self.cutoffs]

    def __len__(self):
        return len(self.cutoffs)

    def __str__(self):
        return ', '.join([str(m) for m in self.cutoffs])

    def append(self, new_cutoffs):
        if isinstance(new_cutoffs, list):
            self.cutoffs = self.cutoffs + new_cutoffs
        elif isinstance(new_cutoffs, CutoffRule):
            self.cutoffs.append(new_cutoffs)

def cutoffs_string(cutoffs_list):
    msk_str = ''
    for key in cutoffs_list:
        msk_str += '{}: {} [{}]'.format(
            key.param, key.operator, ', '.join([str(m) for m in key.limit])) if \
            isinstance(key.limit, list) else \
            '{}: {} {}, '.format(
                key.param, key.operator, key.limit)
    if (len(msk_str) > 2) and (msk_str[-2:] == ', '):
        msk_str = msk_str.rstrip(', ')
    return msk_str





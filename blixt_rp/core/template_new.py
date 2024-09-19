"""
Class to handle templates
"""
import pandas as pd
import logging
import sys
import os

# Add to path to avoid having to install libraries, useful in development
project_dir = os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core','')
sys.path.append(os.path.join(str(project_dir), 'blixt_utils'))

from blixt_utils.utils import isnan
from blixt_utils.misc.attribdict import AttribDict


class Template(AttribDict):
    """
    Template class
    """
    defaults = {
        'full_name': None,
        'units': None,
        'min': None, 'max': None,
        'colormap': None,
        'center': None,
        'bounds': None,
        'scale': None,
        'line color': None, 'line style': None, 'line width': None,
        'marker': None}

    def __init__(self, template=None):
        if template is None:
            template = {}
        # super(Template, self).__init__(template)
        super().__init__(template)

    def __str__(self):
        """
        Return better readable string representation of template object.
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

    def get_from_project(self,
                         project_file: str,
                         log_type: str):
        """

        :param project_file:
            str
            Full pathname of project xlsx file
        :param log_type:
            str
            Name of the log type, e.g. "S Velocity"
        :return:
        """
        table = pd.read_excel(project_file, header=1, sheet_name='Templates', engine='openpyxl')
        for i, ans in enumerate(table['Log type']):
            if not isinstance(ans, str):
                continue
            if ans.lower() != log_type.lower():
                continue
            for _key in list(self.defaults.keys()):
                if _key == 'full_name':
                    self.__setattr__(_key, ans)
                    self.__setitem__(_key, ans)
                elif _key == 'units':
                    # After starting to use Pint, we shifted from using 'unit' to 'units' to be more
                    # similar to xarray in terminology
                    self.__setattr__(_key, None if isnan(table['unit'][i]) else table['unit'][i])
                    self.__setitem__(_key, None if isnan(table['unit'][i]) else table['unit'][i])
                else:
                    self.__setattr__(_key, None if isnan(table[_key][i]) else table[_key][i])
                    self.__setitem__(_key, None if isnan(table[_key][i]) else table[_key][i])

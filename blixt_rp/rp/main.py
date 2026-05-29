# This can be invoked by calling:
# T:\Python\EMB\blixt_rp\blixt_rp> T:\Python\EMB\blixt_rp\.p310\Scripts\python.exe -m bokeh serve --show .\rp

import sys
import os
from bokeh.plotting import column, figure, show, curdoc, row

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\rp', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

test = 'rp_wrapper_table'


if test == 'rp_wrapper_table':
    import rp_wrapper_new
    test = rp_wrapper_new.TestCases()
    table, add_row, delete_row, update_table = test.test_rpt_table()
    curdoc().add_root(
        column(
            table,
            row(add_row, delete_row, update_table)
        )
    )

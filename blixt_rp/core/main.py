# This can be invoked by calling:
# T:\Python\EMB\blixt_rp\blixt_rp> T:\Python\EMB\blixt_rp\.p310\Scripts\python.exe -m bokeh serve --show core

import sys
import os
from bokeh.plotting import column, figure, show, curdoc, row
from dask.array import delete

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

# test = 'seismic__test_avo_qc'
test = 'fluid_table'
# test = 'mineral_table'


if test == 'seismic__test_avo_qc':
    import seismic
    test = seismic.TestCases()
    p, b1, b2, b3, b4, b5, b6, b7, b8 = test.test_avo_qc()
    curdoc().add_root(row(column(row(b1, b2, b3, b4), p, b5), column(b7, b8, b6)))

elif test == 'mineral_table':
    import minerals_new
    test = minerals_new.TestCases()
    table, add_row, delete_row, update = test.test_mineral_table(unit_test=False)
    curdoc().add_root(column(table, row(add_row, delete_row, update)))

elif test == 'fluid_table':
    import fluids_new
    test = fluids_new.TestCases()
    table, add_row, delete_row, update = test.test_fluid_table(unit_test=False)
    curdoc().add_root(column(table, row(add_row, delete_row, update)))

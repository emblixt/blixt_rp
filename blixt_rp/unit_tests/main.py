# This can be invoked by calling:
# C:\Users\emb\Documents\PycharmProjects\blixt_rp\blixt_rp>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show unit_tests

import sys
from bokeh.plotting import column, figure, show, curdoc, row

# To test blixt_rp and blixt_utils libraries directly, without installation:
sys.path.append('C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp')
sys.path.append('C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_utils')

test = 'rp_core_new__test_litho_fluid_table'


if test == 'rp_core_new__test_litho_fluid_table':
    import test_rp_core_new
    test = test_rp_core_new.SomeTests()
    table, add_row, delete_row, update = test.test_litho_fluid_table(unit_test=False)
    curdoc().add_root(column(table, row(add_row, delete_row, update)))

# This can be invoked by calling:
# C:\Users\emb\Documents\PycharmProjects\blixt_rp\blixt_rp>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show unit_tests

import sys
from bokeh.plotting import column, figure, show, curdoc, row

# To test blixt_rp and blixt_utils libraries directly, without installation:
sys.path.append('C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp')
sys.path.append('C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_utils')

# test = 'rp_core_new__test_litho_fluid_table'
# test = 'models__test_model_table'
test = 'models__test_laminar_model'


if test == 'rp_core_new__test_litho_fluid_table':
    import test_rp_core_new
    test = test_rp_core_new.SomeTests()
    table, add_row, delete_row, update = test.test_litho_fluid_table(unit_test=False)
    curdoc().add_root(column(table, row(add_row, delete_row, update)))

elif test == 'models__test_model_table':
    import test_models

    test = test_models.TestCase()
    model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update = test.test_model_table(
        unit_test=False)
    curdoc().add_root(row(
        column(model_table, row(add_row_m, delete_row_m, update_m)),
        column(lf_table, row(add_row, delete_row, update))
    ))

elif test == 'models__test_laminar_model':
    import test_models
    test = test_models.TestCase()
    model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update, grid = test.test_laminar_model(unit_test=False)
    curdoc().add_root(column(
        grid,
        row(
            column(model_table, row(add_row_m, delete_row_m, update_m)),
            column(lf_table, row(add_row, delete_row, update))
    )))

# This can be invoked by calling:
# C:\Users\emb\Documents\PycharmProjects\blixt_rp\blixt_rp>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show unit_tests
# C:\Users\marten.blixt\PycharmProjects\blixt_rp\blixt_rp>C:\Users\marten.blixt\PycharmProjects\blixt_rp\.p310\Scripts\bokeh serve --show unit_tests
# T:\Python\EMB\blixt_rp\blixt_rp> T:\Python\EMB\blixt_rp\.p310\Scripts\python.exe -m bokeh serve --show unit_tests
import sys
import os
from bokeh.plotting import column, figure, show, curdoc, row

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))
# sys.path.append('C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_rp')
# sys.path.append('C:\\Users\\emb\\Documents\\PycharmProjects\\blixt_utils')

# test = 'core__test_litho_fluid_table'
# test = 'models__test_model_table'
test = 'models__test_laminar_model'
# test = 'models__test_wedge'


if test == 'core__test_litho_fluid_table':
    import test_core
    test = test_core.LithoFluidTests()
    # table, add_row, delete_row, update = test.test_table(unit_test=False)
    table, add_row, delete_row, update = test.test_quasi_2D_table(unit_test=False)
    curdoc().add_root(column(table, row(add_row, delete_row, update)))

elif test == 'models__test_model_table':
    import test_models

    test = test_models.TestCase()
    # model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update = test.test_model_table(
    model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update = test.test_quasi2d(
        unit_test=False)
    curdoc().add_root(row(
        column(model_table, row(add_row_m, delete_row_m, update_m)),
        column(lf_table, row(add_row, delete_row, update))
    ))

elif test == 'models__test_laminar_model':
    import test_models
    test = test_models.TestCase()
    # model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update, grid, controls = test.test_laminar_model(unit_test=False)
    model_table, add_row_m, delete_row_m, update_m, lf_table, add_row, delete_row, update, grid, controls = test.test_laminar_q2d_model(unit_test=False)
    curdoc().add_root(column(
        grid, controls,
        row(
            column(model_table, row(add_row_m, delete_row_m, update_m)),
            column(lf_table, row(add_row, delete_row, update))
        )))

elif test == 'models__test_wedge':
    import test_models
    test = test_models.TestCase()
    (model_table, model_controls, lf_table, lf_controls, grid, controls, new_grid, points_table, update_avo, avo_figure,
     ixg_figure, eei_figure, chi_input) = test.test_wedge(unit_test=False)
    curdoc().add_root(
        row(
            column(grid, controls, new_grid,
                row(
                    column(model_table, model_controls),
                    column(lf_table, lf_controls)
                )
            ),
            column(points_table, row(update_avo, chi_input), avo_figure, ixg_figure, eei_figure)
        )
    )

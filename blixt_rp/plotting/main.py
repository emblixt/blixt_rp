# This can be invoked by calling:
# C:\Users\emb\Documents\PycharmProjects\blixt_rp\blixt_rp>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show plotting

import sys
import os
from bokeh.plotting import column, figure, show, curdoc, row

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

# test = 'log_plotter__test_cutoffs'
test = 'log_plotter__test_well_plotter_with_cutoffs'


if test == 'log_plotter__test_cutoffs':
    import log_plotter
    test = log_plotter.TestCases()
    grid, table, add_row, delete_row, update, apply_mask, reset_mask = test.test_cutoffs(unit_test=False)
    curdoc().add_root(
        row(
            grid, column(
                table,
                row(add_row, delete_row, update, apply_mask, reset_mask)
            )
        )
    )
elif test == 'log_plotter__test_well_plotter_with_cutoffs':
    import log_plotter
    test = log_plotter.TestCases()
    grid, table, add_row, delete_row, update, apply_mask, reset_mask = (
        test.test_well_plotter_with_cutoffs(unit_test=False))
    curdoc().add_root(
        row(
            grid, column(
                table,
                row(add_row, delete_row, update, apply_mask, reset_mask)
            )
        )
    )


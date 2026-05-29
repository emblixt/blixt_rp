# This can be invoked by calling:
# T:\Python\EMB\blixt_rp\blixt_rp> T:\Python\EMB\blixt_rp\.p310\Scripts\python.exe -m bokeh serve --show .\plotting

import sys
import os
from bokeh.plotting import column, figure, show, curdoc, row

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\plotting', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

# test = 'log_plotter__test_cutoffs'
# test = 'log_plotter__test_well_plotter_with_cutoffs'
# test = 'cross_plotter__test_both'
# test = 'rp_plotter__test_well'
test = 'rp_plotter__test_well_with_rpt'


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
elif test == 'cross_plotter__test_both':
    import cross_plotter
    test = cross_plotter.TestCases()
    xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis = test.test_both()
    curdoc().add_root(
       row(
          column(
              xplot,
              row(x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask),
          ),
          column(ct_guis[0],
                 row(ct_guis[1], ct_guis[2], ct_guis[3]),  # , ct_guis[4]),
                 wis_guis[0]  # ,
                 #  wis_guis[1]
                 )
       )
    )

elif test == 'rp_plotter__test_well':
    import rp_plotter
    test = rp_plotter.TestCases()
    xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table\
        = test.test_well(unit_test=False)
    curdoc().add_root(
        row(
            column(
                xplot,
                row(x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask)
            ),
            column(style_table,
                   column(ct_guis[0], row(ct_guis[1], ct_guis[2], ct_guis[3])),  #,  ct_guis[4])),
                   column(wis_guis[0])  #, wis_guis[1])
                   )
        )
    )

elif test == 'rp_plotter__test_well_with_rpt':
    import rp_plotter
    test = rp_plotter.TestCases()
    (xplot, x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask, ct_guis, wis_guis, style_table,
     rpt_dt, add_row, var_select, delete_row, update_table) = test.test_well_with_rpt(unit_test=False)
    curdoc().add_root(
        row(
            column(
                xplot,
                row(x_menu, y_menu, size_menu, color_menu, apply_mask, reset_mask)
            ),
            column(style_table,
                   column(ct_guis[0], row(ct_guis[1], ct_guis[2], ct_guis[3])),  #,  ct_guis[4])),
                   column(wis_guis[0]),  #, wis_guis[1]),
                   column(
                       rpt_dt, row(add_row, var_select, delete_row, update_table)
                   )
           )
        )
    )



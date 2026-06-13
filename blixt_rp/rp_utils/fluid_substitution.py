"""
Outline:
    This is to be a new home for fluid substitution, but with all the "new" functions and methods that allows
    pint Quantities

    It should have two main functionalities:
    1. One 'advanced' function
        Copy and rewrite the run_fluid_sub() from rp_core.py, AND run_fluid_substitution.py,
        OR, at least use them as inspiration

    2. One 'naiv' method that is interactive and uses bokeh as front-end
           Use test_fluidsub() in fluids_new as inspiration
           There is one problem with using the CrossPlotter for visualizing the results, as the parameters both
           before and after fluid substitution need to have the same name for them to be plotted simultaneously.
           So we need a new 'Well' (or DataSource) to store the substituted results.
"""
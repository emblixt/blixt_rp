# This can be invoked by calling:
# C:\Users\emb\Documents\PycharmProjects\blixt_rp\blixt_rp>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show core
# C:\Users\marten.blixt\PycharmProjects\blixt_rp\blixt_rp>C:\Users\marten.blixt\PycharmProjects\blixt_rp\.p310\Scripts\bokeh serve --show core

import sys
import os
from bokeh.plotting import column, figure, show, curdoc, row

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\core', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

test = 'seismic__test_avo_qc'


if test == 'seismic__test_avo_qc':
    import seismic
    test = seismic.TestCases()
    p, b1, b2, b3, b4, b5, b6, b7, b8 = test.test_avo_qc()
    curdoc().add_root(row(column(row(b1, b2, b3, b4), p, b5), column(b7, b8, b6)))

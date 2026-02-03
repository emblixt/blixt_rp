**ToDo**

###### 2022-10-10

Add the following usages to the different introduction notebooks
 - Depth trends, and how to use them
 - Checkshots
 - well paths
 - EEI
 - models

##### 2022-12-15

~~Modify and generalize the import of data to a Block and to a Well so that we don't need to have both read_las() and read_log_data()~~ 


##### 2023-03-15
~~Modify the Param data class rp_core, so that it follows the "standard" used in the LogCurve object (e.g. "data" vs, "value").~~

~~We could think about adding a "style" element to both LogCurve, Well & Param classes, so that we are less dependent on using the "template" dictionary everywhere.~~

Look into the possibility of using calculated masks, using calc_mask(), and their name & description, instead of sending around cutoffs and cutoff strings. 
Many functions could use a mask, and its description, as input. See blixt_rp.rp_utils.calc_toc.calc_toc() as example

##### 2023-03-20
~~Move LogCurve class in to well.py~~

Get rid of the "AttribDict" thing~~ 

~~Make "start", "stop", and "step" mandatory items of the LogCurve object too~~

~~Add a "write_las" function to both LogCurves, Blocks, and Wells (one lasfile per block)~~

#### 2025-11-10 
Move rp_core.py functions to rp_core_new and let them use Pint instead of params

Figure out why the LithoFluidsTable doesnt work in avo_chi_plotter.py

Merge the WorkingIntervalsTable of core.py into the add_strat_table() in log_plotter.py

~~Try to replace the python callback that is used for line adding / removal in the different Tables to a CustomJS callback~~

### 2025-12-30
Update models.py so that we don't need both the older Model and Layer class AND the newer ModelTable and ModelLayer
(Merge)

### 2026-02-03
Use the lasio package to deal with reading and writing .las files instead of self-built programs. It can read wrapped 
files and will be updated to deal with las 3 format that is coming.
Use the Agile Geoscience bruges library for wavelets, filters and rock physics instead of self-built programs. It is no
longer developed from Agile Geoscience, but it is maintained by volunteers (seldomly). Write a wrapper so that it takes 
pint units into use. 
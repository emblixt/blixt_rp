# -*- coding: utf-8 -*-
"""
Erik Mårten Blixt
2019-09-10

This is a simple script returns a list of well names in different forms, that can be used to translate well names
"""
import pandas

npd_wellbore_file = 'H:\MBlixt_PL902_Rein_AVO_modeling\\wellbore_exploration_all.xlsx'


def main(npd_wellbore_file=npd_wellbore_file):
    
    well_names = pandas.read_excel(npd_wellbore_file, usecols=['Wellbore name', 'Well name','NPDID wellbore'])

    # create (hopefully) unique short well names without spaces and slashes
    unique_names = []
    for name in well_names['Wellbore name']:
        unique_names.append(
                name.replace('/', '_').replace(' ','')
                )
        well_names['Shortname'] = unique_names

    return well_names

if __name__ == '__main__':
    main()
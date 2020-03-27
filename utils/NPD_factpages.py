# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:17:42 2019

@author: mblixt
"""

import pandas as pd
import os
from pathlib import Path

import utils.io as uio


def fix_well_name(well_name):
    if isinstance(well_name, str):
        return well_name.replace('/', '_').replace('-', '_').replace(' ', '').upper()
    else:
        return


def get_table_from_npd(
        test=False,
        save_to=None,
        stratigraphy=False
):
    """
    This function returns "all" Exploration wells from NPD as a panda table.
    
    :param test:
        bool
        if True, only first 300 lines are downloaded from NPD
    
    :param save_to:
        str
        Fullname of file where the .xlsx file is saved to
        if None, it is not saved

    :param stratigraphy:
        bool
        if True it returns NPD well tops
    """
    if test:
        mystr = 'true'
    else:
        mystr = 'false'

    #url = r'https://factpages.npd.no/ReportServer_npdpublic?/FactPages/TableView/wellbore_exploration_all&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&rs:Format=EXCEL&Top100={}&IpAddress=77.241.96.18&CultureCode=en'.format(mystr)
    url = r'https://factpages.npd.no/ReportServer_npdpublic?/FactPages/TableView/wellbore_exploration_all&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&rs:Format=EXCEL&Top100={}&IpAddress=not_used&CultureCode=en'.format(mystr)

    if stratigraphy:
        url = r'https://factpages.npd.no/ReportServer_npdpublic?/FactPages/TableView/strat_litho_wellbore&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&rs:Format=EXCEL&Top100={}&IpAddress=not_used&CultureCode=en'.format(mystr)

    table = pd.read_excel(url)
    
    if save_to is not None:
        table.to_excel(save_to)
    
    return table


def get_table(
        filename,
        new=False, 
        test=False,
        stratigraphy=False
):
    """
    Reads an excel file downloaded from npd.no which contains information about wells, (tops if stratigraphy is True)
    and returns a dictionary of wells, where each item is a dictionary of well information.

    :param filename:
        str
        Name of file to read, if it doesn't exist, you are prompted to download a new from npd
    :param new:
        bool
        If true, a new well table is downloaded and saved to 'filename'
    :param test:
        bool
        if True, only first 300 lines of the wells table is downloaded from npd.no
    :param stratigraphy:
        bool
        if True it returns NPD well tops
    """
    
    if (not os.path.isfile(filename)) or new:
        if not new:  # case we haven't specifically asked for downloading a new file
            ans = input('Do you want to download new well table from NPD? [Y]es:')    
            if not 'Y' in ans:
                return None
        print('Now we will start downloading a new file')
        # First see if we can save the content to the specific file
        try:
            Path(filename).touch()
        except:
            print('Not allowed to write to {}'.format(filename))
            return None
        table = get_table_from_npd(test=test, save_to=filename, stratigraphy=stratigraphy)
        if test:
            # the last two rows are rubbish
            table.drop(table.tail(2).index, inplace=True)
    
    else:
        table = pd.read_excel(filename)

    if stratigraphy:
        output = uio.return_dict_from_tops(table, 'Wellbore name', 'Lithostrat. unit', 'Top depth [m]')
    else:
        output = {}
        for i, well_name in enumerate(table['Wellbore name']):
            output[fix_well_name(well_name)] = {
                    key: table[key][i] for key in table.keys() if key not in ['Unnamed: 0', 'Wellbore name']}
    
    return output

def get_well_info(well_id):
    """
    Reads the npd table for the well specified by NPDID unique well id, and returns XXX
    :param well_id:
        int
        unique npd.no well id number (NPDID)
    """
    output = {}
    try:
        well_id = int(well_id)
    except:
        raise IOError('Well id has to be an integer, or at least a floating number')

    url = r'https://factpages.npd.no/ReportServer_npdpublic?/FactPages/PageView/wellbore_exploration&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&NpdId={}&IpAddress=77.241.96.18&CultureCode=en'.format(well_id)
    tables = pd.read_html(url)
    for table in tables:
        this_len = len(table[0])
        #print(this_len)  # tables in above web page need to have some length to be of interest
        if this_len > 3:
            if table[0][1] == 'Wellbore name':
                this_dict = {}
                for i in range(this_len):
                    if i == 0: continue  # skip first row which is a nan
                    this_dict[table[0][i]] = table[1][i]
                    #print('{}:\t{}'.format(table[0][i], table[1][i]))
                output['General information'] = this_dict

            elif table[0][1] == 'Top depth [m]':
                this_dict = {}
                for i in range(this_len):
                    if i < 2: continue  # skip first row which is a nan
                    this_dict[table[1][i]] = table[0][i]
                output['Lithostratigraphy'] = this_dict

    return output

if __name__ == '__main__':
#    get_well_table_from_npd(test=True, save_to='C:/Users/mblixt/Downloads/test.xlsx')
#    wells = get_well_table('C:/Users/mblixt/Downloads/test.xlsx', new=False)
#    wells = get_well_table('C:/Users/MårtenBlixt/Downloads/test.xlsx', new=False, test=True)
#    print(wells.keys())
#    print(wells[list(wells.keys())[3]]['NPDID wellbore'])
    get_well_info(165)
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: harmonize_logs.py
#  Purpose: Fixing well logs so that they have same length and sampling distance
#   Author: Erik Marten Blixt
#   Date: 2020-03-19
#   Email: marten.blixt@gmail.com
#
# --------------------------------------------------------------------
import numpy as np
import logging

logger = logging.getLogger(__name__)


def harmonize_logs(well_dict, start, stop, step):
    """
    Takes the input well_dict and arranges the logs inside to fit the given start, stop and step values.

    :param well_dict:
        dict
        as returned from
        null_val, generated_keys, well_dict = core.well._read_las(filename)
    :param start:
        float
        The desired start (in MD [m]) of the well
    :param stop:
        float
        The desired stop (in MD [m]) of the well
    :param step:
        float
        The desired stop (in MD [m]) of the well
    :return:
    """
    # Extract the input start, stop and step values
    input_start = well_dict['well_info']['strt']['value']
    input_stop = well_dict['well_info']['stop']['value']
    input_step = well_dict['well_info']['step']['value']

    # Extract the depth (MD) log from the input logs
    if 'depth' in list(well_dict['data'].keys()):
        input_md = np.array(well_dict['data']['depth'])
    else:
        input_md = np.arange(input_start, input_stop+input_step, input_step)

    # Create a true MD array from the given start, stop, step
    true_md = np.arange(start, stop+step, step)

    # First make sure the step is the same
    if input_step != step:
        warn_txt = 'Step lengths are not equal. {:.4f} vs {:.4f}'.format(input_step, step)
        logger.warning(warn_txt)
        print('WARNING: {}'.format(warn_txt))
        raise NotImplementedError(warn_txt)

    # Test the start value
    if input_start < start:  # input logs starts above given start value
        # Find index in input depth which corresponds to start
        ind = np.nanargmin((input_md-start)**2)
        for key in list(well_dict['data'].keys()):
            this_data = well_dict['data'][key]
            # Cut the input logs
            this_data = this_data[ind:]
            # re-insert them
            well_dict['data'][key] = this_data
        # Update well_info
        well_dict['well_info']['strt']['value'] = start

    elif input_start > start:  # input logs starts below given stop value
        # Find index in the true depth which corresponds to input_start
        ind = np.nanargmin((true_md-input_start)**2)
        for key in list(well_dict['data'].keys()):
            this_data = well_dict['data'][key]
            nan_pad = list(np.ones(ind)*np.nan)
            # pad the log with nans
            nan_pad.extend(this_data)
            # re-insert it
            well_dict['data'][key] = nan_pad
        # Update well_info
        well_dict['well_info']['strt']['value'] = start

    # Test the stop value
    if input_stop > stop:  # input logs ends below the given stop value
        # Find index in input depth which corresponds to stop
        ind = np.nanargmin((input_md - stop) ** 2)
        for key in list(well_dict['data'].keys()):
            this_data = well_dict['data'][key]
            # Cut the input logs
            this_data = this_data[:ind+1]
            # re-insert them
            well_dict['data'][key] = this_data
        # Update well_info
        well_dict['well_info']['stop']['value'] = stop

    elif input_stop < stop:  # input logs ends above given stop value
        # Find index in the true depth which corresponds to input_stop
        ind = np.nanargmin((true_md-input_stop)**2)
        for key in list(well_dict['data'].keys()):
            this_data = well_dict['data'][key]
            nan_pad = list(np.ones(len(true_md)-ind-1)*np.nan)
            # pad the log with nans
            this_data.extend(nan_pad)
            # re-insert it
            well_dict['data'][key] = this_data
        # Update well_info
        well_dict['well_info']['stop']['value'] = stop


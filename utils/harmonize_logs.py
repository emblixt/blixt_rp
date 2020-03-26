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
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def harmonize_logs(well_dict, start, stop, step, orig_len):
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
    :param orig_len:
        int
        The desired length of the data array
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
    if len(true_md) > orig_len:
        true_md = np.arange(start, stop, step)
    if len(true_md) < orig_len:
        raise Warning('Length of true data ({}) does not match')

    info_txt = 'Actual length versus desired length: {} - {}'.format(len(input_md), len(true_md))
    #print(info_txt)
    #logger.info(info_txt)

    # First make sure the step is the same
    if input_step != step:
        warn_txt = 'Step lengths are not equal. {:.4f} vs {:.4f} in well {}'.format(
            input_step, step, well_dict['well_info']['well']['value'])
        logger.warning(warn_txt)
        print('WARNING: {}'.format(warn_txt))
        logger.info('Re-sampling the log with step {} to use {} instead.'.format(
            input_step, step
        ))
        dd = None
        for key in list(well_dict['data'].keys()):
            # re-sample the data
            dd, this_data = interpolate(input_md, well_dict['data'][key], step)
            # re-insert them
            well_dict['data'][key] = list(this_data)
        # Update well_info and the input_md
        well_dict['well_info']['step']['value'] = step
        input_md = dd

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
        # Update well_info and the input_md
        well_dict['well_info']['strt']['value'] = start
        input_md = input_md[ind:]

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
        # Update well_info and the input_md
        well_dict['well_info']['strt']['value'] = start
        nan_pad = list(np.ones(ind) * np.nan)
        nan_pad.extend(list(input_md))
        input_md = np.array(nan_pad)

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

    info_txt = 'New length versus desired length: {} - {}'.format(len(well_dict['data']['depth']), len(true_md))
    #print(info_txt)
    #logger.info(info_txt)


def interpolate(MD, log, step, kind='linear'):
    """
    Takes an array of log values ("log") with corresponding MD values and creates a regularly sampled log with step = "step"
    """
    outMD = np.arange(MD[0], MD[-1], step)

    return outMD, interp1d(MD, log, kind=kind)(outMD)


def info(wd):
    input_start = wd['well_info']['strt']['value']
    input_stop = wd['well_info']['stop']['value']
    input_step = wd['well_info']['step']['value']
    length = np.arange(input_start, input_stop + input_step, input_step)
    txt = 'Start: {}, Stop: {}, Step: {}\n'.format(input_start, input_stop, input_step)
    txt += 'Actual length vs estimated length: {} vs {}'.format(len(wd['data']['depth']), len(length))
    return txt

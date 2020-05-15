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
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def harmonize_logs(well_dict, start, stop, step, orig_len, debug=False):
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
    :param debug:
        bool
        if True, plots and extra information is generated
    :return:
    """
    # Extract the input start, stop and step values
    input_start = well_dict['well_info']['strt']['value']
    input_stop = well_dict['well_info']['stop']['value']
    input_step = well_dict['well_info']['step']['value']

    # Extract the depth (MD) log from the input logs
    if 'depth' not in list(well_dict['data'].keys()):
        raise IOError('Depth log is missing in input')
    input_md = np.array(well_dict['data']['depth'])

    # Create a true MD array
    #true_md = np.arange(start, stop+step, step)
    true_md = np.linspace(start, stop, orig_len)

    info_txt = 'Actual length versus desired length: {} - {}'.format(len(input_md), len(true_md))
    if debug:
        #key = list(well_dict['data'].keys())[0]  # select the first inp
        lname = 'tvd'
        fig, axes = plt.subplots(2, 1, figsize=(8, 14))
        print(info_txt)
        axes[0].plot(input_md, well_dict['data'][lname], 'o', lw=0, c='y')
        axes[0].axvline(start)
        axes[0].axvline(stop)
    #logger.info(info_txt)

    # First make sure the step is the same
    if input_step != step:
        warn_txt = 'Step lengths are not equal. {:.4f} vs {:.4f} in well {}'.format(
            input_step, step, well_dict['well_info']['well']['value'])
        logger.warning(warn_txt)
        # print('WARNING: {}'.format(warn_txt))
        logger.info('Re-sampling the log with step {} to use {} instead.'.format(
            input_step, step
        ))
        dd = None
        for key in list(well_dict['data'].keys()):
            # re-sample the data
            #dd, this_data = interpolate(input_md, well_dict['data'][key], 0., length=orig_len)
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
        if debug:
            axes[0].plot(input_md, well_dict['data'][lname], 'o', lw=0, c='b')

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
        if debug:
            axes[0].plot(input_md, well_dict['data'][lname], 'o', lw=0, c='b')

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

    # Now test the result if it has the same length as the desired length
    if len(well_dict['data']['depth']) != orig_len:
        warn_txt = 'Lengths does not match. Input versus desired length: {} - {}\n Interpolation started'.format(
            len(well_dict['data']['depth']), orig_len)
        logger.warning(warn_txt)
        print('WARNING: {}'.format(warn_txt))
        this_md =  well_dict['data']['depth']
        for key in list(well_dict['data'].keys()):
            # re-sample the data
            dd, this_data = interpolate(this_md, well_dict['data'][key], 0., length=orig_len)
            # re-insert them
            well_dict['data'][key] = list(this_data)


    info_txt = 'New length versus desired length: {} - {}'.format(len(well_dict['data']['depth']), len(true_md))
    #print(info_txt)
    #logger.info(info_txt)


def interpolate(MD, log, step, length=None, kind='linear'):
    """
    Takes an array of log values ("log") with corresponding MD values and creates a regularly sampled log with step = "step"
    """
    if length is not None:
        outMD = np.linspace(MD[0], MD[-1], length)
    else:
        outMD = np.arange(MD[0], MD[-1], step)

    #print('In interpolation: length = {}'.format(length))

    return outMD, interp1d(MD, log, kind=kind)(outMD)


def info(wd):
    input_start = wd['well_info']['strt']['value']
    input_stop = wd['well_info']['stop']['value']
    input_step = wd['well_info']['step']['value']
    txt = 'Start: {}, Stop: {}, Step: {}\n'.format(input_start, input_stop, input_step)
    return txt

def check_lengths(wd):
    for key in list(wd['data'].keys()):
        print(' - Length of {}: {}'.format(key, len(wd['data'][key])))

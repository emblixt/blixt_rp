# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:45:27 2020

@author: mblixt
"""

import logging
import sys
import os
import subprocess
import getpass
import socket
from datetime import datetime
import numpy as np

import utils.masks as msks

def log_header_to_template(log_header):
    """
    Returns a template dictionary from the infromation in the log header.
    :param log_header:
        core.log_curve.Header
    :return:
        dict
        Dictionary contain template information used by crossplot.py
    """
    tdict = {
        'id': log_header.name,
        'description': log_header.desc,
        'full_name': log_header.log_type,
        'type': 'float',
        'unit': log_header.unit
    }
    return tdict


def arrange_logging(log_to_stdout, log_to_this_file, level=logging.INFO):
    """
    :param log_to_stdout:
        bool
        If True, the logging is output to stdout
    :param log_to_this_file:
    :param level:
    :return:
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    frmt = '%(filename)s - %(funcName)s: %(levelname)s:%(message)s'
    if log_to_stdout:
        logging.basicConfig(stream=sys.stdout,
                            format=frmt,
                            level=level)
    else:
        logfile = log_to_this_file
        logging.basicConfig(filename=logfile,
                            format=frmt,
                            level=level)

def gitversion():
    thisgit =  os.path.realpath(__file__).replace('utils/returnVersion.py','.git')
    tagstring = ''
    hashstring = ''

    def cleanstring(text):
        text = text.decode("utf-8")
        text = text.rstrip('\\n')
        text = text.rstrip('\n')
        return text


    # extract the git tag
    try:
        tagstring = subprocess.check_output(['git', '--git-dir=%s' % thisgit,  'describe',
                                             '--abbrev=0'])
        #                                    '--exact-match', '--abbrev=0'])
        tagstring = cleanstring(tagstring)
    except:
        tagstring = 'Unknown tag'

    # extract the git hash
    try:
        hashstring = subprocess.check_output(['git', '--git-dir=%s' % thisgit,  'rev-parse', '--short', 'HEAD'])
        hashstring = cleanstring(hashstring)
    except:
        hashstring = 'Unknown hash'

    return 'Git: %s, %s' % (tagstring, hashstring)

def svnversion():
    try:
        svn_info = subprocess.check_output(['svn', 'info']).decode("utf-8")
    except:
        return '??'
    t = svn_info.split('\n')
    for row in t:
        if 'Revision:' in row:
            return row


def version():
    # TODO
    # Fix the return of version
    return 'XXX'        


def info():
    t0 = datetime.now().isoformat()
    return 'Created by {}, at {}, on {}, using version {}'.format(
        getpass.getuser(), socket.gethostname(), t0, version())


def nan_corrcoef(x,y):
    maskx = ~np.ma.masked_invalid(x).mask
    masky = ~np.ma.masked_invalid(y).mask
    mask = msks.combine_masks([maskx, masky])
    return np.corrcoef(x[mask], y[mask])


def isnan(val):
    """
    Can test both numbers and strings if they are nan's.

    :param val:
    :return:
        bool
    """
    if isinstance(val, str):
        return False
    try:
        return np.isnan(val)
    except:
        raise NotImplementedError('Cant handle this input properly')


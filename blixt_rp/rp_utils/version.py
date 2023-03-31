# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:45:27 2020

@author: mblixt
"""
__all__ = ('get_version')

import re
import os
import getpass
import socket
from datetime import datetime


def version():
    VERSIONFILE = os.path.dirname(__file__).replace('rp_utils', '_version.py')
    with open(VERSIONFILE, "r") as f:
        verstrline = f.read().strip()
        pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
        mo = pattern.search(verstrline)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


def info():
    t0 = datetime.now().isoformat()
    return 'Created by {}, at {}, on {}, using version {}'.format(
        getpass.getuser(), socket.gethostname(), t0, version())

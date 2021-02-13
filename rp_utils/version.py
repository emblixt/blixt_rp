# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:45:27 2020

@author: mblixt
"""

import getpass
import socket
from datetime import datetime

def version():
    # TODO
    # Fix the return of version
    return '0.1.1'


def info():
    t0 = datetime.now().isoformat()
    return 'Created by {}, at {}, on {}, using version {}'.format(
        getpass.getuser(), socket.gethostname(), t0, version())



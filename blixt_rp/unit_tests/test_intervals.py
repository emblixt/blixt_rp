import unittest
import os


import pint
from .. import ureg, Q_

from blixt_rp.core.intervals import WellInterval

class UnitsTestCase(unittest.TestCase):

    def test_create_well(self):
        wi = WellInterval('my_well', 100., 200.0)



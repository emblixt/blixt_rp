import unittest
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from pint import Quantity as Q_

# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_rp\\blixt_rp\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

import blixt_rp.core.cutoffs as brcc

rule1 = brcc.CutoffRule('name1', '>', Q_(100, 'm'))
rule2 = brcc.CutoffRule('name2', '<', Q_(10, 'm'))
rule3 = brcc.CutoffRule('name3', '==', Q_(1000, 'm'))
rule4 = brcc.CutoffRule('name4', None, 'my_interval')
rule5 = brcc.CutoffRule('name5', '><', [Q_(10, 'm'), Q_(1000, 'm')])


class SomeTests(unittest.TestCase):
    def test_rule_init(self):
        limits = [Q_(4,'m'), [Q_(4,'m'), Q_(10,'m')], 'interval_name']
        for limit in limits:
            print(limit)
            rule = brcc.CutoffRule('test', '>', limit)
            self.assertIsInstance(rule, brcc.CutoffRule)
        rule = brcc.CutoffRule('', None, 'test_interval')
        self.assertIsInstance(rule, brcc.CutoffRule)

    def test_failed_rule_init(self):
        limits = [4.0, [4.0, 4.0], 11]
        for limit in limits:
            print(limit)
            self.assertRaises(IOError, brcc.CutoffRule, param='test', operator='>', limit=limit)
        self.assertRaises(IOError, brcc.CutoffRule, param='', operator=None, limit=4)
        self.assertRaises(IOError, brcc.CutoffRule, param='', operator='XXX', limit=4)

    def test_print_rules(self):
        for rule in [rule1, rule2, rule3, rule4, rule5]:
            print(rule)

    def test_failed_init(self):
        cutoffs = brcc.Cutoffs
        self.assertRaises(IOError, cutoffs, name='test', alpha='JA', beta='Nej' )

    def test_init(self):
        cutoffs = brcc.Cutoffs(name='test', cutoffs=[rule1, rule2])
        print(len(cutoffs))
        cutoffs.append([rule3, rule4])
        cutoffs.append(rule1)
        print(len(cutoffs))
        print(cutoffs.cutoff_names)
        print(cutoffs)
        self.assertTrue(True)


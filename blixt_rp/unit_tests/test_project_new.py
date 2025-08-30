import unittest
from blixt_rp.core.project_new import Project

def_lb_name = 'Logs'  # default Block name
def_msk_name = 'Mask'  # default mask name


class ProjectTestCase(unittest.TestCase):

    def test_create_project(self):
        wp = Project(name='MyProject',   # uses the new project table
                     log_to_stdout=True)
        with self.subTest():
            print(type(wp))
            self.assertTrue(isinstance(wp, Project))

    def test_load_wells(self):
        from blixt_rp.core.core import LogTable
        log_table = LogTable({'Density': 'rho_dry', 'P velocity': 'vp_dry', 'S velocity': 'vs_dry',
                              'Porosity': 'PHIE', 'Volume': 'VCL'})
        for _log_table in (None, log_table):
            wp = Project(name='MyProject', log_to_stdout=True)
            if _log_table is None:
                print('- No LogTable')
            else:
                print('- With LogTable')
            wp.load_all_wells(log_table=_log_table)
            for _w in wp.wells:
                print('-', _w.name, '\n', _w.header, '\n', _w.style)
                print(_w.calc_press_ref(None))
                for _l in _w.logs:
                    if _l.style is None:
                        style_txt = 'STYLE IS LACKING'
                    else:
                        style_txt = 'True: {} '.format(_l.style.units)
                    print('  - log name: {}, unit: {}, depth units: {}, evenly spaced? {}, len: {}, style? {}'.format(_l.name,
                           _l.units, _l.depth_units, _l.is_evenly_spaced, len(_l), style_txt ))

    def test_load_selected_wells(self):
        """
        For this test to work, the listed wells below need to have "use = Yes" in the project table
        """
        these_wells = ['WELL_A', 'WELL_B']
        wp = Project(name='MyProject', log_to_stdout=True)
        wells = wp.load_all_wells(include_these_wells=these_wells)
        for wname, _ in wells.items():
            with self.subTest():
                self.assertTrue(wname in these_wells)
            with self.subTest():
                self.assertFalse(wname == 'WELL_C')

    def test_load_selected_intervals(self):
        these_intervals = ['Sand E', 'Sand F']
        #these_intervals = ['Sand F']
        wp = Project(name='MyProject', log_to_stdout=True)
        wells = wp.load_all_wells(include_these_intervals=these_intervals)
        for _, well in wells.items():
            md = well.block[def_lb_name].get_md()
            print(well.well, md.min(), md.max())
        with self.subTest():
            self.assertTrue(True)

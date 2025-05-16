import unittest
from unittest.mock import MagicMock

from src.section_design.detailing_minimums import (BeamDetailsDM76,
                                                   BeamDetailsRD39,
                                                   ColumnDetailsDM76,
                                                   ColumnDetailsRD39,
                                                   RebarsType)

# Mock RectangularSection and RectangularSectionElement for testing
RectangularSection = MagicMock()
RectangularSectionElement = MagicMock()

# --- BeamDetailsRD39 ---
class TestBeamDetailsRD39(unittest.TestCase):

    def test_shear_strirrups_share(self):
        # input
        # compute
        result = BeamDetailsRD39.get_shear_stirrups_share()
        # expected
        expected = 0.5
        # assert
        self.assertEqual(result, expected)

    def test_check_section_true(self):
        # input
        section = MagicMock()
        section.bot_reinf_d = 10
        section.top_reinf_d = 10
        # compute
        result = BeamDetailsRD39.check_section(section)
        # expected
        expected = True
        # assert
        self.assertIs(result, expected)


# --- BeamDetailsDM76 ---
class TestBeamDetailsDM76(unittest.TestCase):
    def test_min_rebar_diameter(self):
        # input
        # compute
        result = BeamDetailsDM76.get_min_rebar_diameter()
        # expected
        expected = 12
        # assert
        self.assertEqual(result, expected)

    def test_shear_strirrups_share(self):
        # input
        # compute
        result = BeamDetailsDM76.get_shear_stirrups_share()
        # expected
        expected = 0.4
        # assert
        self.assertEqual(result, expected)

    def test_min_long_rebar_area_deformed(self):
        # input
        section_area = 1.0
        rebar_type = RebarsType.DEFORMED
        # compute
        result = BeamDetailsDM76.get_min_long_rebar_area(section_area, rebar_type)
        # expected
        expected = 0.0015
        # assert
        self.assertEqual(result, expected)

    def test_min_long_rebar_area_plain(self):
        # input
        section_area = 2.0
        rebar_type = RebarsType.PLAIN
        # compute
        result = BeamDetailsDM76.get_min_long_rebar_area(section_area, rebar_type)
        # expected
        expected = 0.005
        # assert
        self.assertEqual(result, expected)

    def test_max_stirrups_spacing_1(self):
        # input
        section_depth = 1.0
        # compute
        result = BeamDetailsDM76.get_max_stirrups_spacing(section_depth)
        # expected
        expected = min(.8 * section_depth, .33)
        # assert
        self.assertAlmostEqual(result, expected, places=2)

    def test_max_stirrups_spacing_2(self):
        # input
        section_depth = 2.0
        # compute
        result = BeamDetailsDM76.get_max_stirrups_spacing(section_depth)
        # expected
        expected = min(.8 * section_depth, .33)
        # assert
        self.assertAlmostEqual(result, expected, places=2)

    def test_min_stirrups_area(self):
        # input
        # compute
        result = BeamDetailsDM76.get_min_stirrups_area()
        # expected
        expected = 0.0003
        # assert
        self.assertEqual(result, expected)

    def test_check_section_true(self):
        # input
        section = MagicMock()
        section.b = 0.3
        section.h = 0.5
        section.area = 0.15
        section.bot_reinf_d = 16
        section.top_reinf_d = 16
        section.bot_reinf_area = 0.002
        section.top_reinf_area = 0.002
        section.stirrups_spacing = 0.2
        section.stirrups_reinf_area = 0.0001
        section.cop = 0.05
        rebar_type = RebarsType.DEFORMED
        # compute
        result = BeamDetailsDM76.check_section(section, rebar_type)
        # expected
        expected = True
        # assert
        self.assertIs(result, expected)

    def test_check_section_failures(self):
        test_cases = [
            (10, 10, 0.002, 0.002, 0.2, 0.0001, False),  # Fail diameter
            (16, 16, 0.0001, 0.0001, 0.2, 0.0001, False), # Fail area
            (16, 16, 0.002, 0.002, 2.0, 0.0001, False),   # Fail stirrups spacing
            (16, 16, 0.002, 0.002, 0.2, 0.00001, False),  # Fail stirrups area
        ]
        for bot_reinf_d, top_reinf_d, bot_reinf_area, top_reinf_area, stirrups_spacing, stirrups_reinf_area, expected in test_cases:
            # input
            section = MagicMock()
            section.b = 0.3
            section.h = 0.5
            section.area = 0.15
            section.bot_reinf_d = bot_reinf_d
            section.top_reinf_d = top_reinf_d
            section.bot_reinf_area = bot_reinf_area
            section.top_reinf_area = top_reinf_area
            section.stirrups_spacing = stirrups_spacing
            section.stirrups_reinf_area = stirrups_reinf_area
            section.cop = 0.05
            rebar_type = RebarsType.DEFORMED
            # compute
            result = BeamDetailsDM76.check_section(section, rebar_type)
            # expected/assert
            self.assertIs(result, expected)

# --- ColumnDetailsRD39 ---
class TestColumnDetailsRD39(unittest.TestCase):
    def test_min_rebar_diameter(self):
        # input
        # compute
        result = ColumnDetailsRD39.get_min_rebar_diameter()
        # expected
        expected = 12
        # assert
        self.assertEqual(result, expected)

    def test_compute_min_long_reinf_area_lower(self):
        # input
        area = 0.1
        # compute
        result = ColumnDetailsRD39.compute_min_long_reinf_area(area)
        # expected
        expected = 0.1 * 0.008
        # assert
        self.assertAlmostEqual(result, expected, places=4)

    def test_compute_min_long_reinf_area_upper(self):
        # input
        area = 1.
        # compute
        result = ColumnDetailsRD39.compute_min_long_reinf_area(area)
        # expected
        expected = 1. * 0.005
        # assert
        self.assertAlmostEqual(result, expected, places=4)

    def test_compute_min_long_reinf_area_between(self):
        # input
        area = 0.5
        # compute
        result = ColumnDetailsRD39.compute_min_long_reinf_area(area)
        # expected
        expected_min = 0.005 * 0.5
        expected_max = 0.008 * 0.5
        # assert
        self.assertTrue(expected_min <= result <= expected_max)

    def test_get_max_stirrups_spacing(self):
        # input
        long_bar_d = 0.01
        min_dim = 0.3
        # compute
        result = ColumnDetailsRD39.get_max_stirrups_spacing(long_bar_d, min_dim)
        # expected
        expected = min(0.5 * min_dim, 10 * long_bar_d)
        # assert
        self.assertEqual(result, expected)

    def test_check_section_true(self):
        # input
        section = MagicMock()
        section.b = 0.3
        section.h = 0.5
        section.area = 0.15
        section.bot_reinf_d = 16
        section.top_reinf_d = 16
        section.bot_reinf_area = 0.001
        section.top_reinf_area = 0.001
        section.stirrups_spacing = 0.15
        section.stirrups_reinf_area = 0.0001
        min_cls_area = 0.15
        # compute
        result = ColumnDetailsRD39.check_section(section, min_cls_area)
        # expected
        expected = True
        # assert
        self.assertIs(result, expected)

    def test_check_section_failures(self):
        test_cases = [
            (10, 10, 0.001, 0.001, 0.2, False),      # Fail diameter
            (16, 16, 0.00001, 0.00001, 0.2, False),  # Fail area
            (16, 16, 0.001, 0.001, 1, False),     # Fail stirrups spacing (too big)
        ]
        for bot_reinf_d, top_reinf_d, bot_reinf_area, top_reinf_area, stirrups_spacing, expected in test_cases:
            # input
            section = MagicMock()
            section.b = 0.3
            section.h = 0.5
            section.area = 0.15
            section.bot_reinf_d = bot_reinf_d
            section.top_reinf_d = top_reinf_d
            section.bot_reinf_area = bot_reinf_area
            section.top_reinf_area = top_reinf_area
            section.stirrups_spacing = stirrups_spacing
            section.stirrups_reinf_area = 0.0001
            min_cls_area = 0.15
            # compute
            result = ColumnDetailsRD39.check_section(section, min_cls_area)
            # expected/assert
            self.assertIs(result, expected)

# --- ColumnDetailsDM76 ---
class TestColumnDetailsDM76(unittest.TestCase):
    def test_min_rebar_diameter(self):
        # input
        # compute
        result = ColumnDetailsDM76.get_min_rebar_diameter()
        # expected
        expected = 12
        # assert
        self.assertEqual(result, expected)

    def test_compute_min_long_reinf_area(self):
        # input
        column_area = 1.0
        column_min_cls_area = 0.5
        # compute
        result = ColumnDetailsDM76.compute_min_long_reinf_area(column_area, column_min_cls_area)
        # expected
        expected = max(0.006 * 0.5, 0.003 * 1.0)
        # assert
        self.assertEqual(result, expected)

    def test_compute_max_long_reinf_area(self):
        # input
        column_area = 1.0
        # compute
        result = ColumnDetailsDM76.compute_max_long_reinf_area(column_area)
        # expected
        expected = 0.05
        # assert
        self.assertEqual(result, expected)

    def test_get_rebar_adm_stress_deformed(self):
        # input
        rebar_type = RebarsType.DEFORMED
        # compute
        result = ColumnDetailsDM76.get_rebar_adm_stress(rebar_type)
        # expected
        expected = 180_000
        # assert
        self.assertEqual(result, expected)

    def test_get_rebar_adm_stress_plain(self):
        # input
        rebar_type = RebarsType.PLAIN
        # compute
        result = ColumnDetailsDM76.get_rebar_adm_stress(rebar_type)
        # expected
        expected = 120_000
        # assert
        self.assertEqual(result, expected)

    def test_get_max_stirrups_spacing(self):
        # input
        long_bar_d = 0.01
        # compute
        result = ColumnDetailsDM76.get_max_stirrups_spacing(long_bar_d)
        # expected
        expected = min(15 * 0.01, 0.25)
        # assert
        self.assertEqual(result, expected)

    def test_check_section_true(self):
        # input
        section = MagicMock()
        section.b = 0.3
        section.h = 0.5
        section.area = 0.15
        section.bot_reinf_d = 16
        section.top_reinf_d = 16
        section.bot_reinf_area = 0.002
        section.top_reinf_area = 0.002
        section.stirrups_spacing = 0.1
        section.stirrups_reinf_area = 0.0001
        min_cls_area = 0.1
        # compute
        result = ColumnDetailsDM76.check_section(section, min_cls_area)
        # expected
        expected = True
        # assert
        self.assertIs(result, expected)

    def test_check_section_failures(self):
        test_cases = [
            (10, 10, 0.002, 0.002, 0.1, False),      # Fail diameter
            (16, 16, 0.00001, 0.00001, 0.1, False),  # Fail min area
            (16, 16, 0.1, 0.1, 0.1, False),          # Fail max area
            (16, 16, 0.002, 0.002, 1.0, False),      # Fail stirrups spacing (too large)
        ]
        for bot_reinf_d, top_reinf_d, bot_reinf_area, top_reinf_area, stirrups_spacing, expected in test_cases:
            # input
            section = MagicMock()
            section.b = 0.3
            section.h = 0.5
            section.area = 0.15
            section.bot_reinf_d = bot_reinf_d
            section.top_reinf_d = top_reinf_d
            section.bot_reinf_area = bot_reinf_area
            section.top_reinf_area = top_reinf_area
            section.stirrups_spacing = stirrups_spacing
            section.stirrups_reinf_area = 0.0001
            min_cls_area = 0.1
            # compute
            result = ColumnDetailsDM76.check_section(section, min_cls_area)
            # expected/assert
            self.assertIs(result, expected)


if __name__ == '__main__':
    unittest.main()

import math
import unittest

from src.section_design.section_utils import (ReinforcementCombination,
                                              reinforcement_area)


class TestReinforcementArea(unittest.TestCase):

    def test_reinforcement_area_multiple_bars(self):
        """Test reinforcement area for multiple bars."""
        count = 4
        diameter = 12  # mm

        # Expected area calculation
        expected_area = count * math.pi * (diameter / 2) ** 2

        # Assert
        self.assertAlmostEqual(reinforcement_area(count, diameter), expected_area)


class TestReinforcementCombination(unittest.TestCase):

    def test_generate_reinforcements(self):
        """Test generation of reinforcement combinations."""
        min_bars = 2
        max_bars = 4
        min_d = 6
        max_d = 10

        expected_combinations = {
            (bars, diameter): reinforcement_area(bars, diameter)
            for bars in range(min_bars, max_bars + 1)
            for diameter in range(min_d, max_d + 1, 2)
        }

        result = ReinforcementCombination._generate_reinforcements(
            min_bars=min_bars, max_bars=max_bars, min_d=min_d, max_d=max_d
        )

        self.assertEqual(result, expected_combinations)

    def test_minimum_section_width(self):
        """Test calculation of minimum section width."""
        n_bars = 4
        diameter = 0.012  # 12 mm in meters
        cover = 0.03  # 30 mm

        expected_width = 2 * cover + n_bars * diameter + (n_bars - 1) * max(0.02, diameter)
        result = ReinforcementCombination.minimum_section_width(n_bars, diameter, cover)

        self.assertAlmostEqual(result, expected_width)

    def test_find_combination_valid(self):
        """Test finding a valid reinforcement combination."""
        section_width = 0.3  # 300 mm
        min_area = 0.0005  # 500 mm^2 in m^2
        cover = 0.03  # 30 mm

        rc = ReinforcementCombination(section_width=section_width)
        result = rc.find_combination(min_area, cover)

        # Assert there is a vali combination
        self.assertIsNotNone(result)

        n_bars, diameter = result     # type: ignore[reportgeneralTypeIssues]
        area_result = rc.available_combinations.get(result) # type: ignore[reportgeneralTypeIssues]type: ignore[reportgeneralTypeIssues]

        # Check if the vaild combination is within the section width
        self.assertGreaterEqual(area_result, min_area)  # type: ignore[reportCallIssue]
        self.assertLessEqual(
            ReinforcementCombination.minimum_section_width(n_bars, diameter * 1e-3, cover),
            section_width
        )

    def test_find_combination_invalid(self):
        """Test finding a combination when no valid option exists."""
        section_width = 0.1  # 100 mm
        min_area = 0.01  # 10,000 mm^2 in m^2
        cover = 0.03  # 30 mm

        rc = ReinforcementCombination(section_width=section_width)
        result = rc.find_combination(min_area, cover)

        self.assertIsNone(result)

    def test_combinations_sorted(self):
        """Test if combinations are sorted by area."""
        rc = ReinforcementCombination(section_width=0.3)

        available_combinations = rc.available_combinations
        sorted_combinations = rc.sorted_combinations

        sorted_areas = [available_combinations[comb] for comb in sorted_combinations]

        self.assertEqual(sorted_areas, sorted(sorted_areas))

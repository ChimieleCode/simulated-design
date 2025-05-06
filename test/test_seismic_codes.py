import unittest
from typing import List

from src.seismic_codes import (BuildingCode, SeismicCat, SeismicForces,
                               SeismicWeight)


class TestSeismicCalculations(unittest.TestCase):

    def setUp(self):
        # Create fresh instances of each class before each test
        self.forces = SeismicForces()
        self.weights = SeismicWeight()

    # === SeismicWeight Tests ===

    def test_rdl_573_weight(self):
        # Test standard calculation for RDL_573
        # W = 1.5 * (G + Q)
        self.assertAlmostEqual(self.weights.compute_weight(BuildingCode.RDL_573, 10, 5), 22.5)

    def test_rdl_431_weight(self):
        # Test standard calculation for RDL_431
        # W = 4/3 * (G + Q)
        self.assertAlmostEqual(self.weights.compute_weight(BuildingCode.RDL_431, 10, 5), 20.0)

    def test_rdl_640_weight_catI(self):
        # Test RDL_640 with CatI, checking proper use of amplification factor
        weight = self.weights.compute_weight(BuildingCode.RDL_640, 10, 5, SeismicCat.CatI)
        self.assertAlmostEqual(weight, max(1/3 * 5 + 10, 2/3 * 15) * 1.4)

    def test_rdl_640_weight_missing_category(self):
        # Edge case: missing required seismic category for RDL_640
        with self.assertRaises(ValueError):
            self.weights.compute_weight(BuildingCode.RDL_640, 10, 5)

    def test_new_code_not_implemented(self):
        # Edge case: NewCode raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.weights.compute_weight(BuildingCode.NewCode, 10, 5)

    # === SeismicForces Tests ===

    def test_rdl_573_forces(self):
        # Test RDL_573 force coefficients (1/8 for first floor, 1/6 for others)
        weights = [10, 20, 30]
        expected = [10 * 1/8, 20 * 1/6, 30 * 1/6]
        result = self.forces.compute_forces(BuildingCode.RDL_573, weights)
        self.assertListAlmostEqual(result, expected)

    def test_rdl_431_forces_catI(self):
        # Test fallback to RDL_573 when category is CatI
        weights = [10, 20]
        expected = self.forces._RDL_573_forces(weights)
        result = self.forces.compute_forces(BuildingCode.RDL_431, weights, SeismicCat.CatI)
        self.assertListAlmostEqual(result, expected)

    def test_rdl_431_forces_catII(self):
        # Test all floors equal at 1/10 of weight for CatII
        weights = [10, 20]
        expected = [w * 0.1 for w in weights]
        result = self.forces.compute_forces(BuildingCode.RDL_431, weights, SeismicCat.CatII)
        self.assertListAlmostEqual(result, expected)

    def test_rdl_431_forces_catIII(self):
        # Edge case: CatIII means zero forces
        weights = [10, 20]
        result = self.forces.compute_forces(BuildingCode.RDL_431, weights, SeismicCat.CatIII)
        self.assertEqual(result, [0, 0])

    def test_rdl_640_forces_catII(self):
        # Test correct multiplication by amplification factor for CatII
        weights = [10, 20]
        expected = [w * 0.07 for w in weights]
        result = self.forces.compute_forces(BuildingCode.RDL_640, weights, SeismicCat.CatII)
        self.assertListAlmostEqual(result, expected)

    def test_dm_40_75_forces(self):
        # Test distribution proportional to height and weight
        weights = [10, 20]
        heights = [3, 6]
        result = self.forces.compute_forces(BuildingCode.DM_40_75, weights, SeismicCat.CatI, heights)

        # Validate structure: length matches and all values are floats
        self.assertEqual(len(result), len(weights))
        self.assertTrue(all(isinstance(f, float) for f in result))

    def test_dm_1984_catIII_custom_coeff(self):
        # CatIII uses custom coeff (0.04) for DM_1984
        weights = [10, 20]
        heights = [3, 6]
        expected = self.forces._DM_40_75_forces(weights, heights, SeismicCat.CatIII, 0.04)
        result = self.forces.compute_forces(BuildingCode.DM_1984, weights, SeismicCat.CatIII, heights)
        self.assertListAlmostEqual(result, expected)

    def test_missing_heights_raises(self):
        # Edge case: height-required code without heights should raise error
        with self.assertRaises(ValueError):
            self.forces.compute_forces(BuildingCode.DM_40_75, [10, 20], SeismicCat.CatI)

    def test_unsupported_code_raises(self):
        # Edge case: unsupported building code
        with self.assertRaises(ValueError):
            self.forces.compute_forces(BuildingCode.NewCode, [10], SeismicCat.CatI)

    # === Utility ===

    def assertListAlmostEqual(self, list1: List[float], list2: List[float], places=6):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, places=places)

import unittest
from unittest.mock import MagicMock

from src.structure_loads import (GravityLoadDesignFullSpan, SeismicLoadDesign,
                                 compute_beam_moment_end,
                                 compute_beam_shear_end, compute_total_load)


class TestComputeTotalLoad(unittest.TestCase):

    def test_all_loads(self):
        area = [(10, 2)]  # 10 m² * 2 kN/m² = 20 kN
        line = [(5, 1.5)]  # 5 m * 1.5 kN/m = 7.5 kN
        punctual = [(3, 4)]  # 3 * 4 kN = 12 kN
        expected = 20 + 7.5 + 12
        self.assertAlmostEqual(compute_total_load(area, line, punctual), expected)

    # Test case for only area loads
    def test_only_area_loads(self):
        # Area load: 2 m² * 3 kN/m² = 6 kN
        self.assertAlmostEqual(compute_total_load(area_loads=[(2, 3)]), 6)

    # Test case for only line loads
    def test_only_line_loads(self):
        # Line load: 4 m * 2 kN/m = 8 kN
        self.assertAlmostEqual(compute_total_load(line_loads=[(4, 2)]), 8)

    # Test case for only punctual loads
    def test_only_punctual_loads(self):
        # Punctual load: 1 * 5 kN = 5 kN
        self.assertAlmostEqual(compute_total_load(punctual_loads=[(1, 5)]), 5)

    # Test case for empty inputs
    def test_empty_inputs(self):
        # No loads provided, total load should be 0
        self.assertEqual(compute_total_load(), 0)

    # Test case for mixed inputs with zeros
    def test_mixed_with_zeros(self):
        # All load inputs are empty, total load should be 0
        self.assertEqual(compute_total_load(area_loads=[], line_loads=[], punctual_loads=[]), 0)


class TestBeamCalculations(unittest.TestCase):

    def test_compute_beam_moment_end(self):
        # load = 10 kN/m, length = 6 m → expected moment = 10 * 6² / 12 = 30 kNm
        self.assertAlmostEqual(compute_beam_moment_end(10, 6), 30.0)

    def test_compute_beam_shear_end(self):
        # load = 10 kN/m, length = 6 m → expected shear = 10 * 6 / 2 = 30 kN
        self.assertAlmostEqual(compute_beam_shear_end(10, 6), 30.0)


class TestSeismicLoadDesign(unittest.TestCase):

    def setUp(self):
        # Setup a mock building geometry for testing
        self.building_geometry = MagicMock()
        self.building_geometry.floors = 2  # Two floors in the building
        self.building_geometry.floor_height = 3  # Each floor is 3 meters high
        self.building_geometry.area = 80  # Total floor area is 80 m²
        self.building_geometry.perimeter = 36  # Building perimeter is 36 meters
        self.building_geometry.get_total_beam_length.return_value = 54  # Total beam length is 54 meters
        self.building_geometry.column_count = 9  # Building has 9 columns
        self.building_geometry.n_cross_spans = 2  # Two cross spans
        self.building_geometry.n_main_spans = 2  # Two main spans
        self.building_geometry.comulative_floor_height = [3, 6]  # Heights of floors are 3m and 6m cumulatively
        self.building_geometry.span_main = 4  # Main span length is 4 meters
        self.building_geometry.span_cross = 5  # Cross span length is 5 meters

        # Create a SeismicLoadDesign instance for testing
        self.seismic_load_design = SeismicLoadDesign(
            seismic_cat=MagicMock(),  # Mock seismic category
            building_code=MagicMock(),  # Mock building code
            floaring_load=2.0,  # Floor load is 2.0 kN/m²
            infill_load=1.5,  # Infill load is 1.5 kN/m²
            beam_load=3.0,  # Beam load is 3.0 kN/m
            column_load=0.5,  # Column load is 0.5 kN/m
            overload=4.0,  # Overload is 4.0 kN/m²
            roof_overload=2.5  # Roof overload is 2.5 kN/m²
        )

    def assertListAlmostEqual(self, list1, list2, places=7):
        # Assert that the computed axial loads match the expected values
        for v, e in zip(list1, list2):
            self.assertAlmostEqual(v, e, 1)

    def test_compute_floor_weights(self):
        # Call the private method to test
        permanent_weight_floor, overloads_floor = self.seismic_load_design._compute_floor_weights(self.building_geometry)

        # Expected values based on the mock geometry and loads
        expected_permanent = 389.5  # Computed as (80 * (2.0 + 1.5)) + (54 * 3.0) + (9 * 0.5)
        expected_live = 320  # Computed as (80 * 4.0)

        # Check that the expected values are returned
        self.assertAlmostEqual(permanent_weight_floor, expected_permanent, 1)
        self.assertAlmostEqual(overloads_floor, expected_live, 1)

    def test_compute_roof_weights(self):
        # Call the private method to test
        permanent_weight_roof, overloads_roof = self.seismic_load_design._compute_roof_weights(self.building_geometry)

        # Expected values based on the mock geometry and loads
        expected_permanent = 355.75  # Computed as (80 * (2.0 + 1.5)) + (54 * 3.0) + (9 * 0.5) - roof adjustments
        expected_live = 200  # Computed as (80 * 2.5)

        # Check that the expected values are returned
        self.assertAlmostEqual(permanent_weight_roof, expected_permanent, 1)
        self.assertAlmostEqual(overloads_roof, expected_live, 1)

    def test_compute_frame_forces(self):
        # Test the computation of frame forces
        seismic_forces = [3000.0, 3000.0]  # Total seismic forces for two floors

        main_frame_forces, cross_frame_forces = self.seismic_load_design._compute_frame_forces(seismic_forces, self.building_geometry)

        # Expected values based on the mock geometry and loads
        expected_main_frame_forces = [1000.0, 1000.0]  # Divided equally among main spans
        expected_cross_frame_forces = [1000.0, 1000.0]  # Divided equally among cross spans

        # Check that the forces are divided correctly based on spans
        self.assertListAlmostEqual(main_frame_forces, expected_main_frame_forces, 1)
        self.assertListAlmostEqual(cross_frame_forces, expected_cross_frame_forces, 1)


class TestGravityLoadDesignFullSpan(unittest.TestCase):

    def setUp(self):
        # Basic mock building geometry
        self.bg = MagicMock()
        self.bg.floors = 3
        self.bg.span_main = 6.0
        self.bg.span_cross = 5.0
        self.bg.floor_height = 3.0

        # Create an instance with example loads
        self.design = GravityLoadDesignFullSpan(
            floaring_load=2.0,
            overload=1.0,
            roof_overload=0.5,
            infill_load=0.8,
            beam_load=1.2,
            column_load=1.5
        )

    def assertListAlmostEqual(self, list1, list2, places=7):
        # Assert that the computed axial loads match the expected values
        for v, e in zip(list1, list2):
            self.assertAlmostEqual(v, e, 1)

    def test_get_beam_loads(self):
        # Compute the beam loads
        floor_loads, roof_loads = self.design._get_beam_loads(self.bg)

        # Expected values based on the mock geometry and loads
        expected_floor_loads = [
            (9.5, 6.),
            (2., 5.),
            (16.2, 6.),
            (1.2, 5)
        ]
        expected_roof_loads = [
            (7.45, 6.),
            (1.2, 5.),
            (13.7, 6.),
            (1.2, 5)
        ]

        # Compare the computed values with expected values
        for (p, ll), (ep, el) in zip(floor_loads, expected_floor_loads):
            self.assertAlmostEqual(p, ep, 2)
            self.assertAlmostEqual(ll, el, 2)

        for (p, ll), (ep, el) in zip(roof_loads, expected_roof_loads):
            self.assertAlmostEqual(p, ep, 2)
            self.assertAlmostEqual(ll, el, 2)

    def test_get_column_loads(self):
        # Compute the column loads
        roof_cols, floor_cols = self.design._get_column_loads(self.bg)

        # Expected values based on the mock geometry and loads
        expectd_roof_cols = [
            92.7,
            52.2,
            51.6,
            29.88
        ]

        expected_floor_cols = [
            107.7,
            64.5,
            63.1,
            38
        ]

        # Compare the computed values with expected values
        self.assertListAlmostEqual(roof_cols, expectd_roof_cols, 2)
        self.assertListAlmostEqual(floor_cols, expected_floor_cols, 2)

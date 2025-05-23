import unittest

from src.portal_frame_method import (compute_beam_moments, compute_beam_shears,
                                     compute_column_moments,
                                     compute_columns_shear,
                                     compute_floors_shear,
                                     compute_seismic_axial_load)


class TestStructuralCalculations(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, places=7):
        # Assert that the computed axial loads match the expected values
        for v, e in zip(list1, list2):
            self.assertAlmostEqual(v, e, 1)

    def test_compute_floors_shear(self):
        force_pattern = [100, 200]
        floor_shear = compute_floors_shear(force_pattern)

        # Expected shear forces for each floor
        expected_floor_shear = [300, 200]

        # Assert that the computed shear forces match the expected values
        for v, e in zip(floor_shear, expected_floor_shear):
            self.assertAlmostEqual(v, e, 1)

    def test_compute_columns_shear(self):
        floor_shear = [300, 200]
        columns = 3
        column_shear = compute_columns_shear(floor_shear, column_count=columns)

        # Expected shear forces for each column
        expected_column_shear = [150, 100]

        # Assert that the computed shear forces match the expected values
        self.assertListAlmostEqual(column_shear, expected_column_shear)

    def comute_column_moments(self):
        column_shear = [150, 100]
        heights = [3., 6.]
        column_moments = compute_column_moments(column_shear, heights=heights)

        # Expected moments for each column
        expected_column_moments = [100, 66.7]

        # Assert that the computed moments match the expected values
        self.assertListAlmostEqual(column_moments, expected_column_moments)

    def test_compute_beam_moments(self):
        column_moments = [100, 66.7]
        beam_moments = compute_beam_moments(column_moments)

        # Expected moments for each beam
        expected_beam_moments = [83.35, 33.35]

        # Assert that the computed moments match the expected values
        self.assertListAlmostEqual(beam_moments, expected_beam_moments)

    def test_compute_beam_shears(self):
        beam_moments = [83.35, 33.35]
        span_length = 5.
        beam_shear = compute_beam_shears(beam_moments, span_length=span_length)

        # Expected shear forces for each beam
        expected_beam_shear = [33.34, 13.34]

        # Assert that the computed shears match the expected values
        self.assertListAlmostEqual(beam_shear, expected_beam_shear)


    def test_column_axial(self):
        beam_shears = [33.34, 13.34]
        axial_load = compute_seismic_axial_load(beam_shears)

        # Expected axial loads for each column
        expected_axial = [46.68, 13.34]

        # Assert that the computed axial loads match the expected values
        self.assertListAlmostEqual(axial_load, expected_axial)

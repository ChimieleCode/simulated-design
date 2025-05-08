import unittest
from math import ceil

from src.frame_definition import (BuildingBox, BuildingGeometry,
                                  BuildingGeometryFactory)


class TestBuildingGeometryFactory(unittest.TestCase):

    def setUp(self):
        # Setup the necessary data for testing
        self.factory = BuildingGeometryFactory(
            main_target_span=4.0,
            cross_target_span=4.5,
            floor_height=3.0
        )

        self.bbox = BuildingBox(length=40.0, width=30.0, height=12.0, floors=3)

    def test_main_spans(self):
        # Test for _main_spans private method
        length = 40.0
        expected_spans = round(length / self.factory.main_target_span)

        n_spans = self.factory._main_spans(length)
        self.assertEqual(n_spans, expected_spans)

    def test_cross_spans(self):
        # Test for _cross_spans private method
        length = 30.0
        expected_spans = ceil(length / self.factory.cross_target_span)

        n_spans = self.factory._cross_spans(length)
        self.assertEqual(n_spans, expected_spans)

    def test_create_building_geometry(self):
        # Test the create method of BuildingGeometryFactory
        building_geometry = self.factory.create(self.bbox)

        # Exected values
        expected_main_span_length = 4.0
        expected_cross_spans_length = 4.29
        expected_n_main_spans = round(self.bbox.length / expected_main_span_length)
        expected_n_cross_spans = round(self.bbox.width / expected_cross_spans_length)

        # Verify if the returned BuildingGeometry object is correctly created
        self.assertEqual(building_geometry.floors, self.bbox.floors)
        self.assertAlmostEqual(building_geometry.span_main, expected_main_span_length, 2)
        self.assertAlmostEqual(building_geometry.span_cross, expected_cross_spans_length, 2)
        self.assertAlmostEqual(building_geometry.floor_height, self.factory.floor_height, 2)
        self.assertEqual(building_geometry.n_main_spans, expected_n_main_spans)
        self.assertEqual(building_geometry.n_cross_spans, expected_n_cross_spans)

    def test_create_with_different_span_target(self):
        # Test for creating a building geometry with adjusted span targets
        factory = BuildingGeometryFactory(
            main_target_span=5.0,
            cross_target_span=5.0,
            floor_height=3.0
        )
        bbox = BuildingBox(length=50.0, width=40.0, height=15.0, floors=4)
        building_geometry = factory.create(bbox)

        self.assertEqual(building_geometry.n_main_spans, 10)
        self.assertEqual(building_geometry.n_cross_spans, 8)


class TestBuildingGeometry(unittest.TestCase):

    def setUp(self):
        self.geometry = BuildingGeometry(
            floors=4,
            span_main=6.0,
            span_cross=5.0,
            floor_height=3.2,
            n_main_spans=4,
            n_cross_spans=3,
            is_fully_braced=True
        )

    def assertListAlmostEqual(self, list1, list2, places=7):
        # Assert that the computed axial loads match the expected values
        for v, e in zip(list1, list2):
            self.assertAlmostEqual(v, e, 1)

    def test_bay_area(self):
        expected = 6.0 * 5.0
        self.assertEqual(self.geometry.bay_area, expected)

    def test_area(self):
        expected = 4 * 3 * (6.0 * 5.0)
        self.assertEqual(self.geometry.area, expected)

    def test_column_count(self):
        expected = (4 + 1) * (3 + 1)
        self.assertEqual(self.geometry.column_count, expected)

    def test_perimeter(self):
        expected = 2 * (4 * 6.0 + 3 * 5.0)
        self.assertEqual(self.geometry.perimeter, expected)

    def test_comulative_floor_height(self):
        expected = [3.2, 6.4, 9.6, 12.8]
        self.assertListAlmostEqual(self.geometry.comulative_floor_height, expected)

    def test_n_main_frames(self):
        self.assertEqual(self.geometry.n_main_frames, 5)

    def test_n_cross_frames_fully_braced(self):
        self.assertEqual(self.geometry.n_cross_frames, 4)

    def test_n_cross_frames_not_fully_braced(self):
        self.geometry.is_fully_braced = False
        self.assertEqual(self.geometry.n_cross_frames, 2)

    def test_get_total_beam_length_all(self):
        # main beams: 4 spans * (3 + 1 frames) * 6m
        # cross beams: 3 spans * (4 + 1 frames) * 5m
        expected = 4 * 4 * 6.0 + 3 * 5 * 5.0
        self.assertEqual(self.geometry.get_total_beam_length(True, True), expected)

    def test_get_total_beam_length_main_only(self):
        expected = 4 * 4 * 6.0
        self.assertEqual(self.geometry.get_total_beam_length(True, False), expected)

    def test_get_total_beam_length_cross_only(self):
        expected = 3 * 5 * 5.0
        self.assertEqual(self.geometry.get_total_beam_length(False, True), expected)

    def test_get_total_beam_length_none(self):
        self.assertEqual(self.geometry.get_total_beam_length(False, False), 0.0)

import unittest
from unittest.mock import MagicMock

from src.section_design.column_design import (ColumnSectionDesign,
                                              VerifyRectangularColumn)

# Unit conversion constants
mq_cmq = 1e4  # Conversion factor from m^2 to cm^2


class TestColumnSectionDesign(unittest.TestCase):

    def setUp(self):
        # Mock SectionGeometry and MemberSollicitation
        self.mock_geometry = MagicMock()
        self.mock_geometry.h = 0.8  # Height of the section in meters
        self.mock_geometry.b = 0.5  # Width of the section in meters
        self.mock_geometry.cop = 0.05  # Cover to reinforcement in meters

        self.mock_sollicitations = MagicMock()
        self.mock_sollicitations.M = 300  # Moment in kNm
        self.mock_sollicitations.N = 600  # Axial force in kN

        # Instantiate the ColumnSectionDesign class
        self.column_design = ColumnSectionDesign(self.mock_geometry, self.mock_sollicitations)

    def test_compute_minimum_steel_area(self):
        # Input
        sigma_adm_cls = 7000  # Allowable compressive stress of concrete in kPa
        sigma_adm_steel = 170000  # Allowable tensile stress of steel in kPa
        n = 15  # Modular ratio
        As_pred = 1e-4  # Initial estimate of steel area in m^2

        # Mock the optimization result
        result, success = self.column_design.compute_minimum_steel_area(
            sigma_adm_cls,
            sigma_adm_steel,
            n,
            As_pred
        )

        # Expected
        expected_As = 17.3  # Expected steel area in cm^2 (0.2 difference form actual from excel sheet)

        # Assertion
        self.assertAlmostEqual(result * mq_cmq, expected_As, places=1)
        self.assertTrue(success, 'Optimization did not converge.')


class TestVerifyRectangularColumn(unittest.TestCase):

    def setUp(self):
        As = 0.00174  # m^2
        height = 0.8  # Height of the section in meters
        width = 0.5  # Width of the section in meters
        cover = 0.05  # Cover to reinforcement in meters
        As_top = As  # Steel area at the top
        As_bot = As  # Steel area at the bottom

        self.column = VerifyRectangularColumn(
            h=height,
            b=width,
            As_top=As_top,
            As_bot=As_bot,
            cop=cover
        )

    def test_verify_section(self):
        # Input
        sigma_cls_adm = 7000.0  # kPa
        sigma_s_adm = 170000.0  # kPa
        N = 600.0              # kN
        M = 300.0              # kNm

        # Compute the verification of the section
        check_cls, check_steel = self.column.verify_section(
            sigma_cls_adm,
            sigma_s_adm,
            N,
            M
        )

        # Expected values (these should be calculated based on the design criteria)
        cls_f = 1.0  # Concrete check factor
        steel_f = 0.656


        self.assertAlmostEqual(check_cls, cls_f, places=2)
        self.assertAlmostEqual(check_steel, steel_f, places=2)

    def test_verify_section_false_high_eccentricity(self):
        # Input
        sigma_cls_adm = 7000.0  # kPa
        sigma_s_adm = 170000.0  # kPa
        N = 1500.0              # kN
        M = 200.0              # kNm

        # Compute the verification of the section
        check_cls, check_steel = self.column.verify_section(
            sigma_cls_adm,
            sigma_s_adm,
            N,
            M
        )

        # Expected values (these should be calculated based on the design criteria)
        cls_f = 0.886  # Concrete check factor
        steel_f = 0.515

        self.assertAlmostEqual(check_cls, cls_f, places=2)
        self.assertAlmostEqual(check_steel, steel_f, places=2)

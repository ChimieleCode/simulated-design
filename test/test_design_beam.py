import unittest
from unittest.mock import MagicMock

from src.section_design.beam_design import (SkwBeamSectionDesign,
                                            VerifyRectangularBeam,
                                            design_beam_element,
                                            unconditioned_design_bottom_reinf)
from src.section_design.detailing_minimums import DetailingCode, RebarsType
from src.section_design.section_utils import reinforcement_area

# Unit conversion constants
mq_cmq = 1e4  # Conversion factor from m^2 to cm^2
mmq_mq: float = 1e-6  # Conversion factor from mm^2 to m^2


class TestColumnSectionDesign(unittest.TestCase):

    def setUp(self):
        # Mock SectionGeometry and MemberSollicitation
        self.mock_geometry = MagicMock()
        self.mock_geometry.h = 0.5  # Height of the section in meters
        self.mock_geometry.b = 0.3  # Width of the section in meters
        self.mock_geometry.cop = 0.03  # Cover to reinforcement in meters
        # Mocking sollicitations positive
        self.mock_sollicitations_pos = MagicMock()
        self.mock_sollicitations_pos.M = 85  # Moment in kNm
        # Mocking sollicitations negative
        self.mock_sollicitations_neg = MagicMock()
        self.mock_sollicitations_neg.M = 30  # Moment in kNm

        # Instantiate the ColumnSectionDesign class
        self.beam_design = SkwBeamSectionDesign(
            self.mock_geometry,
            self.mock_sollicitations_pos,
            self.mock_sollicitations_neg
        )

    def test_compute_minimum_steel_area(self):
        # Input
        sigma_adm_cls = 9000  # Allowable compressive stress of concrete in kPa
        sigma_adm_steel = 255000  # Allowable tensile stress of steel in kPa
        n = 15  # Modular ratio
        As_pred = 1e-4  # Initial estimate of steel area in m^2

        # Mock the optimization result
        As, As_ratio, success = self.beam_design.compute_minimum_steel_area(
            sigma_adm_cls,
            sigma_adm_steel,
            n,
            As_pred
        )

        # Expected
        expected_As = 7.8833  # Expected steel area in cm^2
        expected_As_ratio = 0.337

        # Assertion
        self.assertAlmostEqual(As * mq_cmq, expected_As, places=1)
        self.assertAlmostEqual(As_ratio, expected_As_ratio, places=2)
        self.assertTrue(success, 'Optimization did not converge.')


class TestVerifyRectangularColumn(unittest.TestCase):

    def test_verify_section_pos(self):
        # Input
        sigma_cls_adm = 9000.0  # kPa
        sigma_s_adm = 255000.0  # kPa
        M = 85.0                # kNm

        As = 0.0007883  # m^2
        As_ratio = 0.3376
        height = 0.5  # Height of the section in meters
        width = 0.3  # Width of the section in meters
        cover = 0.03  # Cover to reinforcement in meters

        beam = VerifyRectangularBeam(
            h=height,
            b=width,
            As_top=As * As_ratio,
            As_bot=As,
            cop=cover
        )

        # Compute the verification of the section
        check_cls, check_steel = beam.verify_section(
            sigma_cls_adm,
            sigma_s_adm,
            M
        )

        # Expected values (these should be calculated based on the design criteria)
        cls_f = 0.8754 # Concrete check factor
        steel_f = 1.0


        self.assertAlmostEqual(check_cls, cls_f, places=2)
        self.assertAlmostEqual(check_steel, steel_f, places=2)

    def test_verify_section_neg(self):
        # Input
        sigma_cls_adm = 9000.0  # kPa
        sigma_s_adm = 255000.0  # kPa
        M = 30.0                # kNm

        As = 0.0007883  # m^2
        As_ratio = 0.3376
        height = 0.5  # Height of the section in meters
        width = 0.3  # Width of the section in meters
        cover = 0.03  # Cover to reinforcement in meters

        beam = VerifyRectangularBeam(
            h=height,
            b=width,
            As_top=As,
            As_bot=As * As_ratio,
            cop=cover
        )

        # Compute the verification of the section
        check_cls, check_steel = beam.verify_section(
            sigma_cls_adm,
            sigma_s_adm,
            M
        )

        # Expected values (these should be calculated based on the design criteria)
        cls_f = 0.388 # Concrete check factor
        steel_f = 1.0


        self.assertAlmostEqual(check_cls, cls_f, places=2)
        self.assertAlmostEqual(check_steel, steel_f, places=2)

class TestUnconditionedDesignBottomReinf(unittest.TestCase):

    def test_unconditioned_design_bottom_reinf(self):
        # Input parameters
        M = 85.0  # Applied bending moment in kNm
        b = 0.3   # Width of the beam section in meters
        sigma_cls_adm = 9000.0  # Allowable stress in concrete in kPa
        sigma_s_adm = 255000.0  # Allowable stress in steel in kPa
        n = 15  # Modular ratio

        # Expected values (these should be calculated based on the design criteria)
        expected_d = 0.4534  # Effective depth in meters
        expected_As = 0.000831  # Area of reinforcement in m^2

        # Call the function
        d, As = unconditioned_design_bottom_reinf(M, b, sigma_cls_adm, sigma_s_adm, n)

        # Assertions
        self.assertAlmostEqual(d, expected_d, places=3)
        self.assertAlmostEqual(As, expected_As, places=6)

    def test_unconditioned_design_bottom_reinf_small_moment(self):
        # Input parameters for a smaller moment
        M = 30.0  # Applied bending moment in kNm
        b = 0.3   # Width of the beam section in meters
        sigma_cls_adm = 9000.0  # Allowable stress in concrete in kPa
        sigma_s_adm = 255000.0  # Allowable stress in steel in kPa
        n = 15  # Modular ratio

        # Expected values (these should be calculated based on the design criteria)
        expected_d = 0.2694  # Effective depth in meters (example value)
        expected_As = 0.0004937  # Area of reinforcement in m^2 (example value)

        # Call the function
        d, As = unconditioned_design_bottom_reinf(M, b, sigma_cls_adm, sigma_s_adm, n)

        # Assertions
        self.assertAlmostEqual(d, expected_d, places=3)
        self.assertAlmostEqual(As, expected_As, places=6)

    def test_design_beam_section(self):
        # Input parameters
        Mpos = 85.
        Mneg = 30.
        Vmax = 30.
        sigma_cls_adm = 9000.
        sigma_s_adm = 255000
        detailing_code = DetailingCode.DM_76
        rebar_type = RebarsType.DEFORMED

        # Compute
        beam_element = design_beam_element(
            Mpos=Mpos,
            Mneg=Mneg,
            Vmax=Vmax,
            sigma_cls_adm=sigma_cls_adm,
            sigma_s_adm=sigma_s_adm,
            detailing_code=detailing_code,
            rebar_type=rebar_type
        )

        # Expected
        b = .3
        h = .5
        cop = .03
        As_top = reinforcement_area(2, 14) * mmq_mq
        As_bot = reinforcement_area(4, 16) * mmq_mq
        Asw = reinforcement_area(2, 6) * mmq_mq
        s = 0.188

        # Assertions
        self.assertAlmostEqual(beam_element.h, h, places=2)
        self.assertAlmostEqual(beam_element.b, b, places=2)
        self.assertAlmostEqual(beam_element.cop, cop, places=2)
        self.assertAlmostEqual(beam_element.top_reinf_area, As_top, places=2)
        self.assertAlmostEqual(beam_element.bot_reinf_area, As_bot, places=2)
        self.assertAlmostEqual(beam_element.stirrups_reinf_area, Asw, places=2)
        self.assertAlmostEqual(beam_element.stirrups_spacing, s, places=2)

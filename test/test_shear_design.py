import unittest

from src.section_design.shear_design import (ShearSectionDesignStirrups,
                                             compute_tangential_stress,
                                             compute_transversal_reinf_pitch)


class TestShearDesign(unittest.TestCase):

    def test_compute_transversal_reinf_pitch(self):
        # Input
        V = 100  # kN
        stirrup_area = 0.0001  # m²
        d = 0.5  # m
        sigma_s_adm = 250000  # kPa (updated to match the new implementation)

        # Computation
        result = compute_transversal_reinf_pitch(V, stirrup_area, d, sigma_s_adm)

        # Expected
        expected = (0.9 * stirrup_area * sigma_s_adm * d) / V

        # Assertion
        self.assertAlmostEqual(result, expected, places=6)

    def test_compute_tangential_stress(self):
        # Input
        V = 100  # kN
        d = 0.5  # m
        b = 0.3  # m

        # Computation
        result = compute_tangential_stress(V, d, b)

        # Expected
        expected = V / (0.9 * b * d)

        # Assertion
        self.assertAlmostEqual(result, expected, places=6)

    def test_design_reinforcement(self):
        # Input
        h = 0.6  # m
        b = 0.3  # m
        cop = 0.05  # m
        V_striups = 100  # kN
        sigma_s_adm = 250000  # kPa (updated to match the new implementation)
        diameter = 8  # mm
        min_spacing = 0.33  # m
        Ast = 1e-4

        section = ShearSectionDesignStirrups(h, b, cop)

        # Computation
        result = section.design_reinforcement(V_striups, sigma_s_adm, diameter, min_spacing)

        # Expected
        d = h - cop
        expected = compute_transversal_reinf_pitch(V_striups, Ast, d, sigma_s_adm)

        # Assertion
        self.assertAlmostEqual(result, expected, places=2)

    def test_verify_tau(self):
        # Input
        h = 0.6  # m
        b = 0.3  # m
        cop = 0.05  # m
        V = 100  # kN
        tau_adm = 2500  # kPa (updated to match the new implementation)

        section = ShearSectionDesignStirrups(h, b, cop)

        # Computation
        result = section.verify_tau(V, tau_adm)

        # Expected
        d = h - cop
        tau = compute_tangential_stress(V, d, b)
        expected = tau / tau_adm

        # Assertion
        self.assertAlmostEqual(result, expected, places=6)


if __name__ == '__main__':
    unittest.main()

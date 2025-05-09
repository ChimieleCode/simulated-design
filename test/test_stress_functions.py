import unittest

from src.section_design.stress_functions import (
    compute_alpha_bottom_reinf, compute_beta_bottom_reinf,
    compute_maximum_concrete_stress, compute_neutral_axis,
    compute_neutral_axis_ratio, compute_static_moment)

# Unit test class for stress functions
cmq_mq = 1e-4  # cm^2 to m^2


class TestStressFunctions(unittest.TestCase):

    def test_compute_neutral_axis_ratio(self):
        # Input data
        sigma_cls_adm = 9000.0  # Allowable stress in concrete (kPa)
        sigma_s_adm = 255000.0  # Allowable stress in steel (kPa)
        n = 15  # Modular ratio

        # Computation
        result = compute_neutral_axis_ratio(sigma_cls_adm, sigma_s_adm, n)

        # Expected data
        expected = 0.346153

        # Assertion
        self.assertAlmostEqual(result, expected, places=4, msg='Neutral axis ratio computation failed.')

    def test_compute_alpha_bottom_reinf(self):
        # Input data
        k = 0.346153  # Neutral axis ratio
        sigma_cls_adm = 9000.0  # Allowable stress in concrete (kPa)

        # Computation
        result = compute_alpha_bottom_reinf(k, sigma_cls_adm)

        # Expected data
        expected = .026939

        # Assertion
        self.assertAlmostEqual(result, expected, places=4, msg='Alpha coefficient computation failed.')

    def test_compute_beta_bottom_reinf(self):
        # Input data
        alpha = .026939  # Alpha coefficient
        k = 0.346153  # Neutral axis ratio
        sigma_s_adm = 255000.0  # Allowable stress in steel (MPa)

        # Computation
        result = compute_beta_bottom_reinf(alpha, k, sigma_s_adm)

        # Expected data
        expected = .00016456

        # Assertion
        self.assertAlmostEqual(result, expected, places=8, msg='Beta coefficient computation failed.')

    def test_compute_maximum_concrete_stress(self):
        # Input data
        N = 300  # Axial force (kN)
        Sx = 0.012981  # Section modulus (m^3)
        y = .2596  # Neutral axis position (m)

        # Computation
        result = compute_maximum_concrete_stress(N, Sx, y)

        # Expected data
        expected = 6000

        # Assertion
        self.assertAlmostEqual(result, expected, places=0, msg='Maximum concrete stress computation failed.')

    def test_compute_maximum_steel_stress(self):
        # Input data
        N = 300  # Axial force (kN)
        Sx = 0.012981  # Section modulus (m^3)
        y = .2596  # Neutral axis position (m)
        n = 15.0  # Modular ratio
        d = 0.75  # Distance from neutral axis to steel location (m)

        # Computation
        result = 170000 # kPa


        # Expected data
        expected = n * N / Sx * (d - y)

        # Assertion
        self.assertAlmostEqual(result, expected, places=-2, msg='Maximum steel stress computation failed.')

    def test_compute_static_moment(self):
        # Input data
        b = 0.5  # Beam width (m)
        y = .2596  # Neutral axis position (m)
        n = 15  # Modular ratio
        As_s = [0.000918, 0.000918]  # Areas of steel reinforcement (m^2)
        d_s = [0.05, 0.75]  # Depths of steel reinforcement (m)

        # Computation
        result = compute_static_moment(b, y, n, As_s, d_s)

        # Expected data
        expected = 0.012981
        # Assertion
        self.assertAlmostEqual(result, expected, places=6, msg='Static moment computation failed.')

    def test_compute_neutral_axis(self):
        # Input data
        n = 15.0  # Modular ratio
        b = 0.5  # Beam width (m)
        u = 0.2667  # Top flange eccentricity (m)
        As_s = [0.000918, 0.000918]  # Areas of steel reinforcement (m^2)
        d_s = [0.05, 0.75]  # Depths of steel reinforcement (m)

        # Computation
        result = compute_neutral_axis(n, b, u, As_s, d_s)

        # Expected data
        expected = 0.2596

        # Assertion
        self.assertAlmostEqual(result, expected, places=4, msg='Neutral axis computation failed.')

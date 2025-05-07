import unittest

from src.sollicitations import MemberSollicitation


class TestMemberSollicitation(unittest.TestCase):
    def test_add(self):
        # Create two MemberSollicitation instances
        m1 = MemberSollicitation(M=10.0, V=5.0, N=3.0)
        m2 = MemberSollicitation(M=2.0, V=4.0, N=7.0)

        # Perform addition using the overloaded __add__ method
        result = m1 + m2

        # Check that each value matches the expected sum
        self.assertEqual(result.M, 12.0, 'Incorrect bending moment after addition')
        self.assertEqual(result.V, 9.0, 'Incorrect shear force after addition')
        self.assertEqual(result.N, 10.0, 'Incorrect axial force after addition')

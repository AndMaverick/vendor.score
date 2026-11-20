import unittest
from fractions import Fraction as F

from vendor_score.lp import Unbounded, exact, maximize


class TestLp(unittest.TestCase):
    def test_textbook_problem(self):
        # max 3x + 2y  s.t. x + y <= 4, x + 3y <= 6, x <= 3   -> x=3, y=1, value 11
        value, x = maximize([F(3), F(2)], [[F(1), F(1)], [F(1), F(3)], [F(1), F(0)]], [F(4), F(6), F(3)])
        self.assertEqual(value, 11)
        self.assertEqual(x, [F(3), F(1)])

    def test_degenerate_problem_terminates(self):
        # Beale-style degeneracy: several rows tight at the origin. Bland's rule must not cycle.
        a = [[F(1, 4), F(-8), F(-1), F(9)], [F(1, 2), F(-12), F(-1, 2), F(3)], [F(0), F(0), F(1), F(0)]]
        value, _ = maximize([F(3, 4), F(-20), F(1, 2), F(-6)], a, [F(0), F(0), F(1)])
        self.assertEqual(value, F(5, 4))

    def test_unbounded(self):
        with self.assertRaises(Unbounded):
            maximize([F(1)], [[F(-1)]], [F(1)])

    def test_input_checks(self):
        with self.assertRaises(ValueError):
            maximize([F(1)], [[F(1)]], [F(-1)])
        with self.assertRaises(ValueError):
            maximize([F(1), F(1)], [[F(1)]], [F(1)])

    def test_exact_reads_decimals_as_written(self):
        self.assertEqual(exact(0.1), F(1, 10))
        self.assertEqual(exact(7), 7)
        self.assertEqual(exact(F(1, 3)), F(1, 3))
        for bad in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                exact(bad)


if __name__ == "__main__":
    unittest.main()

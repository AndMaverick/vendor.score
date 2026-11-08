import unittest

from vendor_score import margin, normalize, rank, sensitivity

WEIGHTS = {"cost": 0.5, "quality": 0.5}
SCORES = {"A": {"cost": 9, "quality": 6}, "B": {"cost": 5, "quality": 9}}


class VendorTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize({"a": 1, "b": 3}), {"a": 0.25, "b": 0.75})

    def test_rank_orders_by_weighted_total(self):
        ordered = rank(WEIGHTS, SCORES)
        self.assertEqual([v for v, _ in ordered], ["A", "B"])
        self.assertAlmostEqual(ordered[0][1], 7.5)

    def test_ties_break_alphabetically(self):
        tied = {"Z": {"cost": 5, "quality": 5}, "M": {"cost": 5, "quality": 5}}
        self.assertEqual([v for v, _ in rank(WEIGHTS, tied)], ["M", "Z"])

    def test_small_swing_is_stable(self):
        result = sensitivity(WEIGHTS, SCORES, swing=0.2)
        self.assertEqual(result["leader"], "A")
        self.assertTrue(result["stable"])

    def test_large_swing_flips_the_leader(self):
        result = sensitivity(WEIGHTS, SCORES, swing=0.5)
        self.assertFalse(result["stable"])
        self.assertIn(("cost", "down", "B"), result["flips"])

    def test_margin(self):
        self.assertAlmostEqual(margin(WEIGHTS, SCORES), 0.5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            rank(WEIGHTS, {"A": {"cost": 9}})
        with self.assertRaises(ValueError):
            rank(WEIGHTS, {"A": {"cost": 11, "quality": 5}})
        with self.assertRaises(ValueError):
            normalize({"a": 0})


if __name__ == "__main__":
    unittest.main()

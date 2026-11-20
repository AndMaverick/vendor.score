import random
import unittest

from vendor_score.problem import parse_problem
from vendor_score.values import SpecError, parse_estimate, parse_value


def base(**over):
    d = {
        "criteria": {"cost": {"weight": 1}, "quality": {"weight": 1}},
        "vendors": {"A": {"cost": 5, "quality": 5}, "B": {"cost": 6, "quality": 4}},
    }
    d.update(over)
    return d


class TestValueFunctions(unittest.TestCase):
    def test_linear_both_directions_and_clamping(self):
        up = parse_value({"worst": 0, "best": 100}, "v")
        down = parse_value({"worst": 100, "best": 0}, "v")
        self.assertEqual(up(50), 5.0)
        self.assertEqual(down(25), 7.5)
        self.assertEqual(up(-10), 0.0)
        self.assertEqual(up(1000), 10.0)
        self.assertEqual(down(-5), 10.0)

    def test_points_interpolate_and_clamp(self):
        f = parse_value({"points": [[99.9, 8], [99.0, 0], [99.99, 10]]}, "v")
        self.assertEqual(f(99.0), 0)
        self.assertAlmostEqual(f(99.45), 4.0)
        self.assertEqual(f(98), 0)
        self.assertEqual(f(100), 10)
        self.assertAlmostEqual(f(99.945), 9.0)

    def test_categories(self):
        f = parse_value({"categories": {"a": 10, "b": 2}}, "v")
        self.assertEqual(f("b"), 2)
        with self.assertRaises(SpecError):
            f("c")

    def test_identity_default(self):
        self.assertEqual(parse_value(None, "v")(7.5), 7.5)

    def test_bad_specs(self):
        bad = [
            {"worst": 1, "best": 1}, {"worst": 1}, {"points": [[1, 2]]}, {"points": [[1, 2], [1, 3]]},
            {"points": [[1, 11], [2, 3]]}, {"categories": {}}, {"categories": {"a": 11}}, {"worst": 1, "best": 2, "extra": 3},
            {"worst": "x", "best": 2}, "linear",
        ]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(SpecError):
                parse_value(spec, "v")


class TestEstimates(unittest.TestCase):
    def test_kinds_and_means(self):
        self.assertEqual(parse_estimate(7, "e").mean(), 7)
        self.assertEqual(parse_estimate({"low": 2, "high": 6}, "e").mean(), 4)
        self.assertEqual(parse_estimate({"low": 0, "mode": 3, "high": 6}, "e").mean(), 3)
        self.assertEqual(parse_estimate([4, 6, 8], "e").mean(), 6)

    def test_sampling_stays_in_range_and_matches_the_mean(self):
        rng = random.Random(3)
        for spec in ({"low": 2, "high": 6}, {"low": 0, "mode": 1, "high": 9}, [3, 4, 9]):
            e = parse_estimate(spec, "e")
            xs = [e.sample(rng) for _ in range(40000)]
            lo, hi = e.low_high()
            self.assertTrue(all(lo <= x <= hi for x in xs))
            self.assertAlmostEqual(sum(xs) / len(xs), e.mean(), delta=0.06)

    def test_uncertain_flag(self):
        self.assertFalse(parse_estimate(5, "e").uncertain)
        self.assertFalse(parse_estimate({"low": 5, "high": 5}, "e").uncertain)
        self.assertTrue(parse_estimate([4, 5], "e").uncertain)

    def test_bad_estimates(self):
        for spec in ("x", True, [], {"low": 5}, {"low": 6, "high": 5}, {"low": 1, "mode": 9, "high": 5}, {"low": 1, "high": 2, "z": 3}, float("nan"), None):
            with self.subTest(spec=spec), self.assertRaises(SpecError):
                parse_estimate(spec, "e")


class TestProblem(unittest.TestCase):
    def test_parses_and_scores(self):
        p = parse_problem(base())
        self.assertEqual(p.criterion_names, ["cost", "quality"])
        self.assertEqual(p.point_scores()["B"], {"cost": 6.0, "quality": 4.0})

    def test_raw_measures_through_value_functions(self):
        d = base()
        d["criteria"]["cost"] = {"weight": 1, "value": {"worst": 1000, "best": 0}}
        d["vendors"]["A"]["cost"] = {"low": 200, "high": 400}
        d["vendors"]["B"]["cost"] = 700
        p = parse_problem(d)
        self.assertAlmostEqual(p.score_of("A", "cost"), 7.0)
        self.assertAlmostEqual(p.score_of("B", "cost"), 3.0)

    def test_must_have_elimination_uses_best_estimate(self):
        d = base()
        d["criteria"]["quality"]["must_have"] = 4.5
        p = parse_problem(d)
        self.assertEqual(p.eliminated(), {"B": ["quality"]})

    def test_strict_validation(self):
        cases = {
            "unknown top-level key": base(extra=1),
            "one vendor": base(vendors={"A": {"cost": 5, "quality": 5}}),
            "missing score": base(vendors={"A": {"cost": 5}, "B": {"cost": 5, "quality": 5}}),
            "unknown criterion": base(vendors={"A": {"cost": 5, "quality": 5, "z": 1}, "B": {"cost": 5, "quality": 5}}),
            "score out of range": base(vendors={"A": {"cost": 11, "quality": 5}, "B": {"cost": 5, "quality": 5}}),
            "negative weight": base(criteria={"cost": {"weight": -1}, "quality": {"weight": 1}}),
            "zero total weight": base(criteria={"cost": {"weight": 0}, "quality": {"weight": 0}}),
            "unknown criterion key": base(criteria={"cost": {"weight": 1, "colour": 2}, "quality": {"weight": 1}}),
            "bad must_have": base(criteria={"cost": {"weight": 1, "must_have": 11}, "quality": {"weight": 1}}),
            "stakeholder missing criterion": base(stakeholders={"cfo": {"cost": 1}}),
            "stakeholder zero": base(stakeholders={"cfo": {"cost": 0, "quality": 0}}),
            "not an object": [1, 2],
        }
        for label, data in cases.items():
            with self.subTest(label), self.assertRaises(SpecError):
                parse_problem(data)

    def test_category_criterion_needs_a_known_category(self):
        d = {
            "criteria": {"exit": {"weight": 1, "value": {"categories": {"open": 9, "closed": 2}}}, "q": {"weight": 1}},
            "vendors": {"A": {"exit": "open", "q": 5}, "B": {"exit": "closed", "q": 5}},
        }
        p = parse_problem(d)
        self.assertEqual(p.point_scores()["A"]["exit"], 9)
        d["vendors"]["A"]["exit"] = "ajar"
        with self.assertRaises(SpecError):
            parse_problem(d)
        d["vendors"]["A"]["exit"] = 5
        with self.assertRaises(SpecError):
            parse_problem(d)

    def test_raw_measure_without_value_function_is_refused(self):
        with self.assertRaises(SpecError):
            parse_problem(base(vendors={"A": {"cost": 5000, "quality": 5}, "B": {"cost": 5, "quality": 5}}))


if __name__ == "__main__":
    unittest.main()

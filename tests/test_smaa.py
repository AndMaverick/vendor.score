import random
import unittest

from vendor_score import robust
from vendor_score.problem import parse_problem
from vendor_score.smaa import dirichlet, simulate, wilson
from vendor_score.synthetic import scenario


def problem(**kw):
    d = {
        "criteria": {"cost": {"weight": 0.5}, "quality": {"weight": 0.3}, "risk": {"weight": 0.2}},
        "vendors": {
            "A": {"cost": 9, "quality": 6, "risk": 4},
            "B": {"cost": 5, "quality": 9, "risk": 7},
            "C": {"cost": 6, "quality": 5, "risk": 5},
        },
    }
    d.update(kw)
    return parse_problem(d)


class TestWilson(unittest.TestCase):
    def test_known_value(self):
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_edges_are_exact(self):
        self.assertEqual(wilson(0, 200)[0], 0.0)
        self.assertEqual(wilson(200, 200)[1], 1.0)
        self.assertGreater(wilson(0, 200)[1], 0.0)

    def test_coverage(self):
        rng = random.Random(0)
        hits, runs = 0, 2000
        for _ in range(runs):
            k = sum(rng.random() < 0.3 for _ in range(200))
            lo, hi = wilson(k, 200)
            hits += lo <= 0.3 <= hi
        self.assertGreater(hits / runs, 0.93)

    def test_needs_trials(self):
        with self.assertRaises(ValueError):
            wilson(0, 0)


class TestDirichlet(unittest.TestCase):
    def test_mean_and_sum(self):
        rng = random.Random(1)
        alphas = [5.0, 3.0, 2.0]
        draws = [dirichlet(rng, alphas) for _ in range(30000)]
        for d in draws[:50]:
            self.assertAlmostEqual(sum(d), 1.0, places=12)
        for i, a in enumerate(alphas):
            self.assertAlmostEqual(sum(d[i] for d in draws) / len(draws), a / sum(alphas), delta=0.004)

    def test_zero_alpha_stays_zero(self):
        rng = random.Random(1)
        self.assertTrue(all(dirichlet(rng, [1.0, 0.0, 2.0])[1] == 0.0 for _ in range(100)))


class TestSimulate(unittest.TestCase):
    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        a = simulate(problem(), trials=3000, seed=4)
        b = simulate(problem(), trials=3000, seed=4)
        c = simulate(problem(), trials=3000, seed=5)
        self.assertEqual(a.rank_counts, b.rank_counts)
        self.assertNotEqual(a.rank_counts, c.rank_counts)

    def test_fixed_weights_and_point_scores_reproduce_the_ranking(self):
        r = simulate(problem(), trials=500, weights="fixed")
        self.assertEqual(r.rank_counts["A"], [500, 0, 0])
        self.assertEqual(r.rank_counts["B"], [0, 500, 0])
        self.assertEqual(r.expected_rank("C"), 3.0)

    def test_every_trial_places_every_vendor_once(self):
        r = simulate(problem(), trials=4000, seed=2)
        for k in range(3):
            self.assertEqual(sum(r.rank_counts[v][k] for v in r.vendors), 4000)
        for v in r.vendors:
            self.assertAlmostEqual(sum(r.acceptability(v)), 1.0, places=12)

    def test_uniform_weights_agree_with_the_closed_form(self):
        p = problem()
        r = simulate(p, trials=60000, seed=9, weights="uniform")
        exact_ab = robust.win_probability_uniform(p.point_scores(), "A", "B")
        # with only A and B in play A comes first exactly when it beats B
        two = parse_problem({
            "criteria": {c.name: {"weight": c.weight} for c in p.criteria},
            "vendors": {v: {c: p.point_scores()[v][c] for c in p.criterion_names} for v in ("A", "B")},
        })
        r2 = simulate(two, trials=60000, seed=9, weights="uniform")
        self.assertAlmostEqual(r2.p_best("A"), exact_ab, delta=0.006)
        self.assertGreater(r.p_best("A"), r.p_best("C"))

    def test_must_have_veto_probability(self):
        p = parse_problem({
            "criteria": {"q": {"weight": 1, "must_have": 6}, "c": {"weight": 1}},
            "vendors": {"A": {"q": {"low": 4, "high": 8}, "c": 5}, "B": {"q": 7, "c": 5}},
        })
        r = simulate(p, trials=40000, seed=3, weights="fixed")
        self.assertAlmostEqual(r.p_disqualified("A"), 0.5, delta=0.01)
        self.assertEqual(r.p_disqualified("B"), 0.0)
        self.assertEqual(sum(r.rank_counts["A"]) + r.disqualified["A"], 40000)
        # B is alone in some trials and is then first
        self.assertEqual(r.rank_counts["B"][0] + r.rank_counts["A"][0], 40000)

    def test_a_dominated_vendor_is_never_first(self):
        p = parse_problem({
            "criteria": {"x": {"weight": 1}, "y": {"weight": 1}, "z": {"weight": 1}},
            "vendors": {"A": {"x": 5, "y": 5, "z": 5}, "B": {"x": 6, "y": 6, "z": 6}, "C": {"x": 9, "y": 1, "z": 1}},
        })
        self.assertFalse(robust.best_case(p.point_scores(), "A").possibly_best)
        r = simulate(p, trials=30000, seed=1, weights="uniform")
        self.assertEqual(r.rank_counts["A"][0], 0)
        self.assertGreater(r.p_best("C"), 0.0)

    def test_possibly_best_is_necessary_for_winning_in_samples(self):
        for seed in range(6):
            sc = scenario(5, 3, seed)
            p = parse_problem(sc)
            r = simulate(p, trials=6000, seed=seed, weights="uniform")
            for v, bc in robust.possibly_best(p.point_scores()).items():
                if not bc.possibly_best:
                    self.assertEqual(r.rank_counts[v][0], 0, msg=f"seed {seed} {v}")

    def test_central_weights_win_for_their_vendor(self):
        p = problem()
        r = simulate(p, trials=20000, seed=6, weights="uniform")
        scores = p.point_scores()
        for v in ("A", "B"):
            cw = r.central_weights[v]
            t = robust.totals(cw, scores)
            self.assertEqual(max(t, key=lambda x: t[x]), v)
        self.assertIsNone(r.central_weights["C"])

    def test_tighter_concentration_makes_the_stated_leader_more_certain(self):
        loose = simulate(problem(), trials=15000, seed=1, concentration=5)
        tight = simulate(problem(), trials=15000, seed=1, concentration=500)
        self.assertGreater(tight.p_best("A"), loose.p_best("A"))

    def test_zero_weight_criterion_never_matters(self):
        d = {
            "criteria": {"x": {"weight": 1}, "y": {"weight": 0}},
            "vendors": {"A": {"x": 6, "y": 0}, "B": {"x": 5, "y": 10}},
        }
        r = simulate(parse_problem(d), trials=3000, seed=1, weights="uniform")
        self.assertEqual(r.p_best("A"), 1.0)

    def test_argument_checks(self):
        for kw in ({"trials": 0}, {"weights": "x"}, {"concentration": 0}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                simulate(problem(), **kw)


if __name__ == "__main__":
    unittest.main()

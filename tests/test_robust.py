import itertools
import random
import unittest
from fractions import Fraction as F

from tests import oracles
from vendor_score import robust
from vendor_score.synthetic import scorecard

W = {"cost": 0.5, "quality": 0.3, "risk": 0.2}
S = {
    "A": {"cost": 9, "quality": 6, "risk": 4},
    "B": {"cost": 5, "quality": 9, "risk": 7},
    "C": {"cost": 6, "quality": 5, "risk": 5},
}


class TestTotalsAndDominance(unittest.TestCase):
    def test_totals_match_exact_oracle(self):
        for seed in range(30):
            w, s = scorecard(5, 4, seed)
            ref = oracles.totals_exact(w, s)
            got = robust.totals(w, s)
            for v in s:
                self.assertAlmostEqual(got[v], float(ref[v]), places=12)

    def test_known_values(self):
        t = robust.totals(W, S)
        self.assertEqual({k: round(v, 9) for k, v in t.items()}, {"A": 7.1, "B": 6.6, "C": 5.5})
        self.assertEqual(robust.leader(W, S), "A")

    def test_weights_are_normalised(self):
        self.assertEqual(robust.totals({k: 10 * v for k, v in W.items()}, S), robust.totals(W, S))

    def test_dominance(self):
        s = {"A": {"x": 5, "y": 5}, "B": {"x": 6, "y": 5}, "C": {"x": 6, "y": 5}}
        self.assertTrue(robust.dominates(s, "B", "A"))
        self.assertFalse(robust.dominates(s, "B", "C"))          # equal, not strict
        self.assertTrue(robust.dominates(s, "B", "C", strict=False))
        self.assertEqual(robust.dominated(s), {"A": "B"})

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            robust.totals({}, S)
        with self.assertRaises(ValueError):
            robust.totals({"cost": 1, "quality": 1, "risk": 1}, {"A": {"cost": 1, "quality": 1}})
        with self.assertRaises(ValueError):
            robust.totals({"cost": 1}, {"A": {"cost": 11}})
        with self.assertRaises(ValueError):
            robust.totals({"cost": -1, "quality": 2}, {"A": {"cost": 1, "quality": 1}})


class TestPossiblyBest(unittest.TestCase):
    def test_matches_vertex_enumeration_oracle(self):
        for seed in range(60):
            rng = random.Random(seed)
            nv, nc = rng.randint(2, 5), rng.randint(2, 4)
            _, s = scorecard(nv, nc, seed, integer_scores=seed % 2 == 0)
            for v in s:
                got = robust.best_case(s, v).margin
                want = float(oracles.best_lead_by_vertices(s, v))
                self.assertAlmostEqual(got, want, places=9, msg=f"seed {seed} vendor {v}")

    def test_witness_weights_achieve_the_margin(self):
        for seed in range(40):
            _, s = scorecard(4, 3, 100 + seed)
            for v in s:
                bc = robust.best_case(s, v)
                self.assertAlmostEqual(sum(bc.weights.values()), 1.0, places=12)
                self.assertTrue(all(x >= 0 for x in bc.weights.values()))
                lead = min(
                    sum(bc.weights[c] * (s[v][c] - s[u][c]) for c in bc.weights) for u in s if u != v
                )
                self.assertAlmostEqual(lead, bc.margin, places=9)

    def test_dominated_vendor_cannot_win(self):
        s = {"A": {"x": 5, "y": 5, "z": 5}, "B": {"x": 6, "y": 6, "z": 6}, "C": {"x": 9, "y": 1, "z": 1}}
        self.assertFalse(robust.best_case(s, "A").possibly_best)
        self.assertTrue(robust.best_case(s, "B").possibly_best)

    def test_a_tie_for_first_counts(self):
        s = {"A": {"x": 4, "y": 6}, "B": {"x": 6, "y": 4}}
        self.assertEqual(robust.best_case(s, "A").margin, 2.0)
        s2 = {"A": {"x": 5, "y": 5}, "B": {"x": 5, "y": 5}}
        self.assertEqual(robust.best_case(s2, "A").margin, 0.0)
        self.assertTrue(robust.best_case(s2, "A").possibly_best)

    def test_one_criterion(self):
        s = {"A": {"x": 4}, "B": {"x": 7}}
        self.assertEqual(robust.best_case(s, "B").margin, 3.0)
        self.assertEqual(robust.best_case(s, "A").margin, -3.0)

    def test_agrees_with_dense_weight_search_in_one_direction(self):
        # no weighting on a fine grid may beat the exact answer
        _, s = scorecard(4, 3, 5)
        step = 40
        for v in s:
            exact_margin = robust.best_case(s, v).margin
            grid_best = max(
                min(
                    (i * (s[v]["c0"] - s[u]["c0"]) + j * (s[v]["c1"] - s[u]["c1"]) + (step - i - j) * (s[v]["c2"] - s[u]["c2"])) / step
                    for u in s if u != v
                )
                for i in range(step + 1) for j in range(step + 1 - i)
            )
            self.assertGreaterEqual(exact_margin + 1e-9, grid_best)


class TestWindows(unittest.TestCase):
    def _leader_at(self, criterion, x):
        """Leader when ``criterion`` has share x and the others keep their proportions (independent re-implementation)."""
        tot = sum(W.values())
        rest = {c: W[c] / (tot - W[criterion]) * (1 - x) for c in W if c != criterion}
        rest[criterion] = x
        t = oracles.totals_exact(rest, S)
        return sorted(t, key=lambda v: (-t[v], v))[0]

    def test_window_matches_scan(self):
        for w in robust.weight_windows(W, S):
            for k in range(0, 401):
                x = F(k, 400)
                inside = F(w.low).limit_denominator(10**12) <= x <= F(w.high).limit_denominator(10**12)
                if abs(float(x) - w.low) < 1e-9 or abs(float(x) - w.high) < 1e-9:
                    continue
                self.assertEqual(self._leader_at(w.criterion, x) == "A", inside, msg=f"{w.criterion} {float(x)}")

    def test_rival_draws_level_at_the_edge(self):
        for w in robust.weight_windows(W, S):
            for edge, rival in ((w.low, w.low_rival), (w.high, w.high_rival)):
                if rival is None:
                    continue
                tot = sum(W.values())
                shares = {c: W[c] / (tot - W[w.criterion]) * (1 - edge) for c in W if c != w.criterion}
                shares[w.criterion] = edge
                t = robust.totals(shares, S)
                self.assertAlmostEqual(t["A"], t[rival], places=9)

    def test_known_values(self):
        by = {w.criterion: w for w in robust.weight_windows(W, S)}
        self.assertAlmostEqual(by["cost"].low, 0.428571428571, places=9)
        self.assertEqual(by["cost"].low_rival, "B")
        self.assertEqual(by["cost"].high, 1.0)
        self.assertAlmostEqual(by["quality"].high, 0.4, places=9)

    def test_sorted_narrowest_first(self):
        rooms = [min(w.room_down, w.room_up) for w in robust.weight_windows(W, S)]
        self.assertEqual(rooms, sorted(rooms))

    def test_unknown_criterion(self):
        with self.assertRaises(ValueError):
            robust.weight_window(W, S, "nope")

    def test_random_windows_hold(self):
        for seed in range(40):
            w, s = scorecard(4, 4, 200 + seed)
            top = robust.leader(w, s)
            for win in robust.weight_windows(w, s):
                for x in (win.low, win.high, win.weight):
                    if win.low_rival is None and x == win.low and x == 0.0 or win.high_rival is None and x == win.high and x == 1.0:
                        pass
                    others = {c: v / (sum(w.values()) - w[win.criterion]) * (1 - x) for c, v in w.items() if c != win.criterion}
                    others[win.criterion] = x
                    t = oracles.totals_exact(others, s)
                    # inside or on the edge the leader is at least tied for first
                    self.assertGreaterEqual(t[top] + F(1, 10**9), max(t.values()), msg=f"seed {seed} {win.criterion} {x}")


class TestFlips(unittest.TestCase):
    def test_flip_matches_grid_search(self):
        step = 100
        for seed in range(25):
            w, s = scorecard(3, 3, 300 + seed)
            top = robust.leader(w, s)
            tot = sum(w.values())
            w0 = [w[c] / tot for c in sorted(w)]
            for f in robust.flips(w, s):
                ch = f.challenger
                crit = sorted(w)
                d = [s[ch][c] - s[top][c] for c in crit]
                best = None
                for i in range(step + 1):
                    for j in range(step + 1 - i):
                        p = [i / step, j / step, (step - i - j) / step]
                        if sum(a * b for a, b in zip(p, d, strict=True)) >= -1e-12:
                            dist = sum(abs(a - b) for a, b in zip(p, w0, strict=True)) / 2
                            best = dist if best is None else min(best, dist)
                self.assertIsNotNone(best)
                self.assertLessEqual(f.moved, best + 1e-9)               # exact answer is no worse than the grid
                self.assertGreaterEqual(f.moved, best - 2 / step)         # and the grid is within its own resolution

    def test_flip_weights_tie_and_distance_is_honest(self):
        for seed in range(40):
            w, s = scorecard(4, 4, 400 + seed)
            tot = sum(w.values())
            top = robust.leader(w, s)
            for f in robust.flips(w, s):
                self.assertAlmostEqual(sum(f.weights.values()), 1.0, places=12)
                self.assertTrue(all(x >= -1e-12 for x in f.weights.values()))
                t = robust.totals(f.weights, s)
                self.assertAlmostEqual(t[f.challenger], t[top], places=9)
                moved = sum(abs(f.weights[c] - w[c] / tot) for c in w) / 2
                self.assertAlmostEqual(moved, f.moved, places=9)

    def test_known_value(self):
        f = robust.flip_for(W, S, "B")
        self.assertAlmostEqual(f.moved, 0.5 / 7, places=12)
        self.assertEqual(robust.flip_for(W, S, "C").moved, 0.4)

    def test_impossible_flip_is_none(self):
        s = {"A": {"x": 9, "y": 9}, "B": {"x": 3, "y": 3}}
        self.assertIsNone(robust.flip_for({"x": 1, "y": 1}, s, "B"))
        self.assertEqual(robust.flips({"x": 1, "y": 1}, s), [])

    def test_leader_cannot_be_challenger(self):
        with self.assertRaises(ValueError):
            robust.flip_for(W, S, "A")

    def test_possibly_best_implies_a_flip_exists(self):
        # moving the weights continuously from the stated ones to the best-case ones must pass through a tie with the leader
        for seed in range(60):
            w, s = scorecard(4, 3, 500 + seed)
            top = robust.leader(w, s)
            have = {f.challenger for f in robust.flips(w, s)}
            for v in s:
                if v != top and robust.best_case(s, v).possibly_best:
                    self.assertIn(v, have, msg=f"seed {seed} {v}")

    def test_nearest_flip_is_where_the_leader_loses_first_place(self):
        for seed in range(40):
            w, s = scorecard(4, 3, 800 + seed)
            top = robust.leader(w, s)
            fl = robust.flips(w, s)
            if not fl:
                continue
            t = robust.totals(fl[0].weights, s)
            self.assertAlmostEqual(t[top], max(t.values()), places=9)         # the leader is level with someone...
            self.assertAlmostEqual(t[top], t[fl[0].challenger], places=9)     # ...namely the nearest challenger


class TestLevers(unittest.TestCase):
    def test_each_lever_ties_and_a_bit_more_flips(self):
        for seed in range(25):
            w, s = scorecard(3, 3, 600 + seed)
            top = robust.leader(w, s)
            for lv in robust.levers(w, s):
                moved = {v: dict(r) for v, r in s.items()}
                moved[lv.vendor][lv.criterion] = lv.to
                t = robust.totals(w, moved)
                self.assertAlmostEqual(t[top], t[lv.against], places=9, msg=str(lv))
                sign = -1 if lv.direction == "falls" else 1
                moved[lv.vendor][lv.criterion] = lv.to + sign * 1e-4
                if 0 <= moved[lv.vendor][lv.criterion] <= 10:
                    t = robust.totals(w, moved)
                    self.assertGreater(t[lv.against], t[top], msg=str(lv))

    def test_known_value_and_bounds(self):
        lv = robust.levers(W, S)
        self.assertEqual((lv[0].vendor, lv[0].criterion, lv[0].direction), ("A", "cost", "falls"))
        self.assertAlmostEqual(lv[0].change, 1.0)
        self.assertTrue(all(0 <= x.to <= 10 for x in lv))
        self.assertEqual([x.change for x in lv], sorted(x.change for x in lv))


class TestUniformProbability(unittest.TestCase):
    def test_matches_exact_area_for_three_criteria(self):
        for seed in range(60):
            _, s = scorecard(2, 3, 700 + seed, integer_scores=seed % 3 == 0)
            d = [s["v0"][c] - s["v1"][c] for c in sorted(s["v0"])]
            # P(v0 beats v1) is the share of the triangle where the lead is positive
            want = 1 - float(oracles.triangle_area_fraction(d)) - (0 if len(set(d)) > 1 else 0)
            got = robust.win_probability_uniform(s, "v0", "v1")
            self.assertAlmostEqual(got, want, places=6, msg=f"seed {seed}")

    def test_complements_and_bounds(self):
        _, s = scorecard(3, 4, 11)
        a = robust.win_probability_uniform(s, "v0", "v1")
        b = robust.win_probability_uniform(s, "v1", "v0")
        self.assertAlmostEqual(a + b, 1.0, places=6)
        self.assertTrue(0 <= a <= 1)

    def test_agrees_with_sampling_for_five_criteria(self):
        _, s = scorecard(2, 5, 3)
        rng = random.Random(1)
        n, wins = 200_000, 0
        crit = sorted(s["v0"])
        for _ in range(n):
            g = [rng.expovariate(1.0) for _ in crit]
            tot = sum(g)
            lead = sum(gi / tot * (s["v0"][c] - s["v1"][c]) for gi, c in zip(g, crit, strict=True))
            wins += lead > 0
        self.assertAlmostEqual(robust.win_probability_uniform(s, "v0", "v1"), wins / n, delta=0.006)

    def test_certain_outcomes(self):
        s = {"A": {"x": 9, "y": 8, "z": 7}, "B": {"x": 1, "y": 2, "z": 3}}
        self.assertAlmostEqual(robust.win_probability_uniform(s, "A", "B"), 1.0, places=9)
        self.assertAlmostEqual(robust.win_probability_uniform(s, "B", "A"), 0.0, places=9)

    def test_ties_among_differences_are_handled(self):
        s = {"A": {"x": 7, "y": 7, "z": 3}, "B": {"x": 5, "y": 5, "z": 5}}      # differences 2, 2, -2
        p = robust.win_probability_uniform(s, "A", "B")
        d = [2, 2, -2]
        self.assertAlmostEqual(p, 1 - float(oracles.triangle_area_fraction(d)), places=6)

    def test_single_criterion(self):
        self.assertEqual(robust.win_probability_uniform({"A": {"x": 5}, "B": {"x": 4}}, "A", "B"), 1.0)

    def test_subset_of_criteria(self):
        s = {"A": {"x": 9, "y": 1, "z": 9}, "B": {"x": 1, "y": 9, "z": 1}}
        self.assertAlmostEqual(robust.win_probability_uniform(s, "A", "B", ["x", "z"]), 1.0, places=9)


class TestKendall(unittest.TestCase):
    def test_against_brute_force(self):
        rng = random.Random(0)
        for _ in range(200):
            n = rng.randint(1, 9)
            a = list(range(n))
            b = a[:]
            rng.shuffle(b)
            self.assertAlmostEqual(robust.kendall_tau(a, b), oracles.kendall_brute(a, b), places=12)

    def test_extremes_and_errors(self):
        self.assertEqual(robust.kendall_tau(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertEqual(robust.kendall_tau(["a", "b", "c"], ["c", "b", "a"]), -1.0)
        with self.assertRaises(ValueError):
            robust.kendall_tau(["a", "b"], ["a", "c"])
        with self.assertRaises(ValueError):
            robust.kendall_tau(["a", "a"], ["a", "a"])


class TestInvariances(unittest.TestCase):
    def test_scaling_weights_changes_nothing(self):
        w, s = scorecard(4, 4, 1)
        a = robust.flips(w, s)
        b = robust.flips({k: 37 * v for k, v in w.items()}, s)
        for x, y in zip(a, b, strict=True):
            self.assertEqual(x.challenger, y.challenger)
            self.assertAlmostEqual(x.moved, y.moved, places=12)

    def test_criterion_order_does_not_matter(self):
        w, s = scorecard(4, 5, 2)
        keys = list(w)
        for perm in itertools.islice(itertools.permutations(keys), 6):
            w2 = {k: w[k] for k in perm}
            s2 = {v: {k: row[k] for k in perm} for v, row in s.items()}
            self.assertEqual(robust.totals(w2, s2), robust.totals(w, s))
            self.assertEqual([f.challenger for f in robust.flips(w2, s2)], [f.challenger for f in robust.flips(w, s)])

    def test_constant_shift_on_one_criterion_changes_no_margin(self):
        # adding 1 to every vendor on one criterion lifts every total by the same amount
        w, s = scorecard(4, 3, 9)
        s = {v: {c: min(8.0, max(2.0, x)) for c, x in row.items()} for v, row in s.items()}
        shifted = {v: {c: x + (1 if c == "c1" else 0) for c, x in row.items()} for v, row in s.items()}
        for v in s:
            self.assertAlmostEqual(robust.best_case(s, v).margin, robust.best_case(shifted, v).margin, places=9)
        self.assertEqual(robust.leader(w, s), robust.leader(w, shifted))
        for a, b in zip(robust.flips(w, s), robust.flips(w, shifted), strict=True):
            self.assertAlmostEqual(a.moved, b.moved, places=9)


if __name__ == "__main__":
    unittest.main()

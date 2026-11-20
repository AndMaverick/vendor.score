"""Measure vendor.score against independent implementations and print the counts quoted in the README.

Run from the repository root:  python examples/evaluate.py
Every number printed is computed here, from fixed seeds, so the output is the same on every machine."""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import oracles  # noqa: E402  (independent reference implementations)
from vendor_score import robust  # noqa: E402
from vendor_score.problem import parse_problem  # noqa: E402
from vendor_score.scenario import analyze  # noqa: E402
from vendor_score.smaa import simulate  # noqa: E402
from vendor_score.synthetic import scenario, scorecard  # noqa: E402


def possibly_best() -> None:
    cards = vendors = worst = 0
    wrong_verdicts = 0
    for seed in range(250):
        rng = random.Random(seed)
        _, s = scorecard(rng.randint(2, 6), rng.randint(2, 4), seed, integer_scores=seed % 2 == 0)
        cards += 1
        for v in s:
            vendors += 1
            got = robust.best_case(s, v)
            want = oracles.best_lead_by_vertices(s, v)
            worst = max(worst, abs(got.margin - float(want)))
            wrong_verdicts += got.possibly_best != (want >= 0)
    print(f"possibly-best vs exact vertex enumeration: {vendors} vendors on {cards} scorecards, "
          f"verdicts differing {wrong_verdicts}, largest gap in lead {worst:.1e}")


def flip_distance() -> None:
    step, cards, flips, bad, worst = 100, 0, 0, 0, 0.0
    for seed in range(120):
        w, s = scorecard(3, 3, 1000 + seed)
        crit = sorted(w)
        tot = sum(w.values())
        w0 = [w[c] / tot for c in crit]
        top = robust.leader(w, s)
        cards += 1
        for f in robust.flips(w, s):
            flips += 1
            d = [s[f.challenger][c] - s[top][c] for c in crit]
            best = min(
                sum(abs(a - b) for a, b in zip(p, w0, strict=True)) / 2
                for i in range(step + 1) for j in range(step + 1 - i)
                for p in [[i / step, j / step, (step - i - j) / step]]
                if sum(a * b for a, b in zip(p, d, strict=True)) >= -1e-12
            )
            worst = max(worst, f.moved - best)
            bad += not (f.moved <= best + 1e-9 and f.moved >= best - 2 / step)
    print(f"flip distance vs {step}-step grid search over all weightings: {flips} flips on {cards} scorecards, "
          f"disagreements {bad}, exact answer ever worse than the grid {max(0.0, worst):.1e}")


def uniform_probability() -> None:
    n, worst = 0, 0.0
    for seed in range(300):
        _, s = scorecard(2, 3, 2000 + seed, integer_scores=seed % 3 == 0)
        d = [s["v0"][c] - s["v1"][c] for c in sorted(s["v0"])]
        want = 1 - float(oracles.triangle_area_fraction(d))
        worst = max(worst, abs(robust.win_probability_uniform(s, "v0", "v1") - want))
        n += 1
    print(f"unknown-weights win probability vs exact polygon area: {n} pairs, largest gap {worst:.1e}")


def never_first() -> None:
    impossible = wins = 0
    for seed in range(40):
        p = parse_problem(scenario(6, 3, seed))
        r = simulate(p, trials=5000, seed=seed, weights="uniform")
        for v, bc in robust.possibly_best(p.point_scores()).items():
            if not bc.possibly_best:
                impossible += 1
                wins += r.rank_counts[v][0]
    print(f"vendors the exact solver says can never win: {impossible}; times they came first in 5000 random weightings each: {wins}")


def windows() -> None:
    checked = bad = 0
    for seed in range(60):
        w, s = scorecard(4, 4, 3000 + seed)
        top = robust.leader(w, s)
        tot = sum(w.values())
        for win in robust.weight_windows(w, s):
            for k in range(0, 101):
                x = k / 100
                if abs(x - win.low) < 1e-9 or abs(x - win.high) < 1e-9:
                    continue
                shares = {c: v / (tot - w[win.criterion]) * (1 - x) for c, v in w.items() if c != win.criterion}
                shares[win.criterion] = x
                t = oracles.totals_exact(shares, s)
                first = sorted(t, key=lambda v: (-t[v], v))[0]
                checked += 1
                bad += (first == top) != (win.low <= x <= win.high)
    print(f"weight windows vs a 1% scan of each weight: {checked} points checked, disagreements {bad}")


def speed() -> None:
    d = scenario(10, 8, 1)
    d["simulation"] = {"trials": 20000, "seed": 1}
    start = time.perf_counter()
    r = analyze(d)
    secs = time.perf_counter() - start
    print(f"full analysis of 10 vendors x 8 criteria with 20000 trials: {secs:.1f} s (leader {r['leader']})")


if __name__ == "__main__":
    possibly_best()
    flip_distance()
    uniform_probability()
    never_first()
    windows()
    speed()

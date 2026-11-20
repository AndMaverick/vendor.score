"""Stochastic multicriteria acceptability analysis: how often does each vendor come first when the weights and the scores are uncertain.

Each trial draws a weighting (completely unknown, or scattered around the stated weights), draws each uncertain score from its range,
applies the must-haves, and ranks the vendors that survive. The tallies are rank acceptabilities: the share of trials in which a
vendor took each place. Everything is driven by one seeded generator, so a run is reproducible to the last digit."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .problem import Problem


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Exact 0 and 1 at the edges."""
    if n <= 0:
        raise ValueError("need at least one trial")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = centre - half, centre + half
    if successes == 0:
        lo = 0.0
    if successes == n:
        hi = 1.0
    return max(0.0, lo), min(1.0, hi)


def dirichlet(rng: random.Random, alphas: list[float]) -> list[float]:
    """A draw from a Dirichlet distribution. A zero alpha gives an exact zero, so a criterion weighted 0 stays at 0."""
    draws = [rng.gammavariate(a, 1.0) if a > 0 else 0.0 for a in alphas]
    total = sum(draws)
    return [d / total for d in draws]


@dataclass(frozen=True)
class Acceptability:
    trials: int
    vendors: list[str]
    rank_counts: dict[str, list[int]]          # rank_counts[v][k] = trials where v was placed k+1 among the vendors still standing
    disqualified: dict[str, int]               # trials where v missed a must-have
    central_weights: dict[str, dict[str, float] | None]
    mode: str

    def p_best(self, vendor: str) -> float:
        return self.rank_counts[vendor][0] / self.trials

    def p_best_interval(self, vendor: str) -> tuple[float, float]:
        return wilson(self.rank_counts[vendor][0], self.trials)

    def p_disqualified(self, vendor: str) -> float:
        return self.disqualified[vendor] / self.trials

    def expected_rank(self, vendor: str) -> float:
        """Mean place, counting a disqualified trial as last place."""
        n = len(self.vendors)
        placed = sum((k + 1) * c for k, c in enumerate(self.rank_counts[vendor]))
        return (placed + n * self.disqualified[vendor]) / self.trials

    def acceptability(self, vendor: str) -> list[float]:
        return [c / self.trials for c in self.rank_counts[vendor]]


def simulate(
    problem: Problem,
    *,
    trials: int = 20000,
    seed: int = 1,
    weights: str = "stated",
    concentration: float = 50.0,
) -> Acceptability:
    """Run the analysis. ``weights`` is "stated" (scattered around the stated weights with the given concentration, higher is tighter),
    "uniform" (any weighting equally likely) or "fixed" (the stated weights exactly, so only the scores vary)."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if weights not in {"stated", "uniform", "fixed"}:
        raise ValueError("weights must be 'stated', 'uniform' or 'fixed'")
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    rng = random.Random(seed)
    names = problem.criterion_names
    vendors = sorted(problem.vendors)
    stated = [c.weight for c in problem.criteria]
    total = sum(stated)
    fixed = [w / total for w in stated]
    alphas = [concentration * w for w in fixed]
    if weights == "uniform":
        alphas = [1.0 if w > 0 else 0.0 for w in fixed]
    crit = problem.criteria
    n = len(vendors)
    counts = {v: [0] * n for v in vendors}
    out = dict.fromkeys(vendors, 0)
    sums: dict[str, list[float]] = {v: [0.0] * len(names) for v in vendors}
    wins = dict.fromkeys(vendors, 0)
    for _ in range(trials):
        w = fixed if weights == "fixed" else dirichlet(rng, alphas)
        totals: dict[str, float] = {}
        for v in vendors:
            row = problem.vendors[v]
            t, ok = 0.0, True
            for i, c in enumerate(crit):
                e = row[c.name]
                raw = e.category if e.kind == "category" else e.sample(rng)
                s = c.value(raw)
                if c.must_have is not None and s < c.must_have:
                    ok = False
                t += w[i] * s
            if ok:
                totals[v] = t
            else:
                out[v] += 1
        for k, v in enumerate(sorted(totals, key=lambda x: (-totals[x], x))):
            counts[v][k] += 1
            if k == 0:
                wins[v] += 1
                for i in range(len(names)):
                    sums[v][i] += w[i]
    central = {v: ({names[i]: sums[v][i] / wins[v] for i in range(len(names))} if wins[v] else None) for v in vendors}
    return Acceptability(trials, vendors, counts, out, central, weights)

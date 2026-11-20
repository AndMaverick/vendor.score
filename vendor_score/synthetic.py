"""Random but reproducible scorecards, for tests and benchmarks."""
from __future__ import annotations

import random
from typing import Any


def scorecard(n_vendors: int, n_criteria: int, seed: int = 0, *, integer_scores: bool = False) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """A weights dict and a scores dict with random weights and scores."""
    rng = random.Random(seed)
    criteria = [f"c{i}" for i in range(n_criteria)]
    weights = {c: round(rng.uniform(0.05, 1.0), 3) for c in criteria}
    scores = {
        f"v{j}": {c: float(rng.randint(0, 10)) if integer_scores else round(rng.uniform(0, 10), 2) for c in criteria}
        for j in range(n_vendors)
    }
    return weights, scores


def scenario(n_vendors: int, n_criteria: int, seed: int = 0) -> dict[str, Any]:
    """A scenario dict built from :func:`scorecard`."""
    w, s = scorecard(n_vendors, n_criteria, seed)
    return {
        "name": f"synthetic-{n_vendors}x{n_criteria}-{seed}",
        "criteria": {c: {"weight": v} for c, v in w.items()},
        "vendors": s,
        "simulation": {"trials": 2000, "seed": seed},
    }

"""vendor.score: weighted decision matrices that tell you whether the winner is actually robust."""

from __future__ import annotations

__version__ = "0.1.0"


def normalize(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0 or any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative and sum above zero")
    return {k: w / total for k, w in weights.items()}


def _validate(weights: dict, scores: dict) -> None:
    for vendor, row in scores.items():
        missing = set(weights) - set(row)
        if missing:
            raise ValueError(f"{vendor} is missing scores for: {sorted(missing)}")
        if any(not 0 <= v <= 10 for v in row.values()):
            raise ValueError(f"{vendor} has a score outside 0 to 10")


def rank(weights: dict, scores: dict) -> list:
    """Vendors by weighted score, best first. Ties break alphabetically."""
    _validate(weights, scores)
    w = normalize(weights)
    totals = {v: sum(w[c] * row[c] for c in w) for v, row in scores.items()}
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def sensitivity(weights: dict, scores: dict, swing: float = 0.2) -> dict:
    """Move each weight up and down by swing (0.2 means 20 percent) and see who leads.

    stable is True only if the leader never changes. flips lists every move that changes it.
    """
    leader = rank(weights, scores)[0][0]
    flips = []
    for criterion in weights:
        for label, factor in (("up", 1 + swing), ("down", 1 - swing)):
            changed = {**weights, criterion: weights[criterion] * factor}
            new_leader = rank(changed, scores)[0][0]
            if new_leader != leader:
                flips.append((criterion, label, new_leader))
    return {"leader": leader, "stable": not flips, "flips": flips}


def margin(weights: dict, scores: dict) -> float:
    """Gap between first and second place, in score points."""
    ordered = rank(weights, scores)
    return ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]

"""Exact robustness analysis of a weighted scorecard.

Everything here is computed in rational arithmetic on the numbers as written, so there is no sampling error and no tolerance to tune.
The inputs are the same dicts the 0.1 functions use: ``weights`` maps criterion to weight, ``scores`` maps vendor to criterion to a
score from 0 to 10."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from .lp import exact, maximize

Weights = Mapping[str, float]
Scores = Mapping[str, Mapping[str, float]]


def _setup(weights: Weights, scores: Scores) -> tuple[list[str], list[str], dict[str, Fraction], dict[str, dict[str, Fraction]]]:
    if not weights:
        raise ValueError("need at least one criterion")
    if len(scores) < 1:
        raise ValueError("need at least one vendor")
    criteria = sorted(weights)
    w = {c: exact(weights[c]) for c in criteria}
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("weights must be non-negative and sum above zero")
    total = sum(w.values())
    w = {c: v / total for c, v in w.items()}
    vendors = sorted(scores)
    s: dict[str, dict[str, Fraction]] = {}
    for v in vendors:
        missing = sorted(set(criteria) - set(scores[v]))
        if missing:
            raise ValueError(f"{v} is missing scores for: {missing}")
        row = {c: exact(scores[v][c]) for c in criteria}
        if any(not 0 <= x <= 10 for x in row.values()):
            raise ValueError(f"{v} has a score outside 0 to 10")
        s[v] = row
    return criteria, vendors, w, s


def _totals(criteria: list[str], w: dict[str, Fraction], s: dict[str, dict[str, Fraction]]) -> dict[str, Fraction]:
    return {v: sum((w[c] * s[v][c] for c in criteria), Fraction(0)) for v in s}


def _order(totals: dict[str, Fraction]) -> list[str]:
    return sorted(totals, key=lambda v: (-totals[v], v))


def totals(weights: Weights, scores: Scores) -> dict[str, float]:
    """Weighted score of every vendor (weights are normalised), computed exactly and then rounded to a float."""
    criteria, _, w, s = _setup(weights, scores)
    return {v: float(t) for v, t in _totals(criteria, w, s).items()}


def leader(weights: Weights, scores: Scores) -> str:
    """The vendor on top; exact ties go to the alphabetically first."""
    criteria, _, w, s = _setup(weights, scores)
    return _order(_totals(criteria, w, s))[0]


# dominance -----------------------------------------------------------------------------------------------------------------------
def dominates(scores: Scores, a: str, b: str, *, strict: bool = True) -> bool:
    """``a`` dominates ``b`` when it scores at least as high on every criterion (and, if ``strict``, higher on at least one). A dominated
    vendor cannot win under any weighting, so it should not take up room in the discussion."""
    ra, rb = scores[a], scores[b]
    if set(ra) != set(rb):
        raise ValueError("vendors must be scored on the same criteria")
    ge = all(exact(ra[c]) >= exact(rb[c]) for c in ra)
    gt = any(exact(ra[c]) > exact(rb[c]) for c in ra)
    return ge and (gt if strict else True)


def dominated(scores: Scores) -> dict[str, str]:
    """Vendors that another vendor dominates, mapped to one dominator (the alphabetically first)."""
    out = {}
    for b in sorted(scores):
        for a in sorted(scores):
            if a != b and dominates(scores, a, b):
                out[b] = a
                break
    return out


# can this vendor ever win ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BestCase:
    vendor: str
    margin: float
    weights: dict[str, float]

    @property
    def possibly_best(self) -> bool:
        """True if some weighting puts this vendor first (a tie for first counts)."""
        return self.margin >= 0


def best_case(scores: Scores, vendor: str) -> BestCase:
    """The weighting that treats ``vendor`` most kindly: it maximises the smallest lead over every other vendor. A non-negative margin
    means the vendor can win under some weights; the weights returned are a witness anyone can check by hand. Solved exactly."""
    criteria = sorted(next(iter(scores.values())))
    others = [u for u in sorted(scores) if u != vendor]
    s = {v: {c: exact(scores[v][c]) for c in criteria} for v in scores}
    for v in s:
        if any(not 0 <= x <= 10 for x in s[v].values()):
            raise ValueError(f"{v} has a score outside 0 to 10")
    n = len(criteria)
    if n == 1 or not others:
        weights = dict.fromkeys(criteria, 1.0 / n)
        lead = min((min(s[vendor][c] - s[u][c] for c in criteria) for u in others), default=Fraction(10))
        return BestCase(vendor, float(lead), weights if n > 1 else {criteria[0]: 1.0})
    # Variables: w_1..w_{n-1} (w_n = 1 - their sum) and tau = t + 10, with t the smallest lead. Lead over u at weights w is
    # sum_i w_i d_ui, with d_ui = s_vendor,i - s_u,i. Shift by 10 (scores are within 0..10, so d + 10 >= 0) to keep everything non-negative:
    #   tau - sum_{i<n} w_i (a_ui - a_un) <= a_un   for each u,   sum_{i<n} w_i <= 1,   maximise tau.
    a_rows, rhs = [], []
    for u in others:
        a = [s[vendor][c] - s[u][c] + 10 for c in criteria]
        a_rows.append([-(a[i] - a[n - 1]) for i in range(n - 1)] + [Fraction(1)])
        rhs.append(a[n - 1])
    a_rows.append([Fraction(1)] * (n - 1) + [Fraction(0)])
    rhs.append(Fraction(1))
    cost = [Fraction(0)] * (n - 1) + [Fraction(1)]
    value, x = maximize(cost, a_rows, rhs)
    w_last = 1 - sum(x[: n - 1], Fraction(0))
    weights = {c: float(x[i]) for i, c in enumerate(criteria[:-1])}
    weights[criteria[-1]] = float(w_last)
    return BestCase(vendor, float(value - 10), weights)


def possibly_best(scores: Scores) -> dict[str, BestCase]:
    """For every vendor, whether any weighting makes it win and the weighting that does it by the widest margin."""
    return {v: best_case(scores, v) for v in sorted(scores)}


# weight windows -------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    criterion: str
    weight: float
    low: float
    high: float
    low_rival: str | None
    high_rival: str | None

    @property
    def room_down(self) -> float:
        return self.weight - self.low

    @property
    def room_up(self) -> float:
        return self.high - self.weight


def weight_window(weights: Weights, scores: Scores, criterion: str) -> Window:
    """The exact range of ``criterion``'s weight (as a share of the total) over which the current leader stays on top, with the other
    weights scaled in proportion. At each end of the range a named rival draws level. ``low_rival`` or ``high_rival`` is None when the
    leader stays on top all the way to that end (0 or 1)."""
    criteria, _, w, s = _setup(weights, scores)
    if criterion not in w:
        raise ValueError(f"unknown criterion {criterion!r}")
    top = _order(_totals(criteria, w, s))[0]
    x0 = w[criterion]
    rest = [c for c in criteria if c != criterion]
    rest_total = sum((w[c] for c in rest), Fraction(0))
    share = {c: (w[c] / rest_total if rest_total > 0 else Fraction(1, len(rest))) for c in rest} if rest else {}

    def other(v: str) -> Fraction:
        return sum((share[c] * s[v][c] for c in rest), Fraction(0))

    lo, hi = Fraction(0), Fraction(1)
    lo_rival: str | None = None
    hi_rival: str | None = None
    for u in sorted(s):
        if u == top:
            continue
        a = other(top) - other(u)                       # leader's lead over u when this criterion has no weight
        b = s[top][criterion] - s[u][criterion]         # ... and when it has all of it
        if a < 0 and b > a:
            bound = -a / (b - a)
            if bound > lo:
                lo, lo_rival = bound, u
        elif b < 0 and a > b:
            bound = a / (a - b)
            if bound < hi:
                hi, hi_rival = bound, u
        # when a >= 0 and b >= 0 the leader never trails u; when both are negative u already leads (only possible in an exact tie)
    return Window(criterion, float(x0), float(lo), float(hi), lo_rival, hi_rival)


def weight_windows(weights: Weights, scores: Scores) -> list[Window]:
    """The window for every criterion, narrowest total room first (the criteria the decision is most sensitive to)."""
    out = [weight_window(weights, scores, c) for c in sorted(weights)]
    return sorted(out, key=lambda x: (min(x.room_down, x.room_up), x.criterion))


# smallest change of weights that changes the winner -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Flip:
    challenger: str
    moved: float
    weights: dict[str, float]

    def __lt__(self, other: Flip) -> bool:
        return (self.moved, self.challenger) < (other.moved, other.challenger)


def flip_for(weights: Weights, scores: Scores, challenger: str) -> Flip | None:
    """The smallest shift of weight, from some criteria to others, that lets ``challenger`` draw level with the leader. (Another vendor
    may be ahead of both at that point; the smallest ``moved`` over all challengers is the distance at which the leader loses first place.) ``moved`` is the
    share of the total weight that has to change hands (0.12 means 12 percentage points). Exact; None when no reweighting can do it.

    A unit of weight moved from criterion i to criterion j changes the challenger's lead by (d_j - d_i), where d is the challenger's
    score minus the leader's. The best recipient is the criterion with the largest d, and the best donors are the criteria with the
    smallest d, taken in order until the gap is closed."""
    criteria, _, w, s = _setup(weights, scores)
    top = _order(_totals(criteria, w, s))[0]
    if challenger == top:
        raise ValueError("the challenger is already the leader")
    if challenger not in s:
        raise ValueError(f"unknown vendor {challenger!r}")
    d = {c: s[challenger][c] - s[top][c] for c in criteria}
    gap = -sum((w[c] * d[c] for c in criteria), Fraction(0))
    if gap <= 0:
        return Flip(challenger, 0.0, {c: float(w[c]) for c in criteria})
    recipient = max(criteria, key=lambda c: (d[c], c))
    new = dict(w)
    moved = Fraction(0)
    for donor in sorted((c for c in criteria if c != recipient), key=lambda c: (d[c], c)):
        gain = d[recipient] - d[donor]
        if gain <= 0 or w[donor] == 0:
            continue
        take = min(w[donor], gap / gain)
        new[donor] -= take
        new[recipient] += take
        moved += take
        gap -= take * gain
        if gap == 0:
            return Flip(challenger, float(moved), {c: float(v) for c, v in new.items()})
    return None


def flips(weights: Weights, scores: Scores) -> list[Flip]:
    """Every challenger that some reweighting can lift level with the leader, easiest first."""
    top = leader(weights, scores)
    out = [f for u in sorted(scores) if u != top and (f := flip_for(weights, scores, u)) is not None]
    return sorted(out)


# which scores to check first ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Lever:
    vendor: str
    criterion: str
    direction: str
    change: float
    to: float
    against: str


def levers(weights: Weights, scores: Scores) -> list[Lever]:
    """The single-score corrections that would change the winner, smallest first. Each one is the exact change to one score (the
    leader's falling, or a challenger's rising, by the amount shown) that draws ``against`` level with the leader, so these are the
    scores that deserve evidence before the decision is signed. Changes that would take a score outside 0 to 10 are left out."""
    criteria, _, w, s = _setup(weights, scores)
    tot = _totals(criteria, w, s)
    top = _order(tot)[0]
    out = []
    for u in sorted(s):
        if u == top:
            continue
        gap = tot[top] - tot[u]
        for c in criteria:
            if w[c] == 0:
                continue
            delta = gap / w[c]
            if s[top][c] - delta >= 0:
                out.append(Lever(top, c, "falls", float(delta), float(s[top][c] - delta), u))
            if s[u][c] + delta <= 10:
                out.append(Lever(u, c, "rises", float(delta), float(s[u][c] + delta), u))
    return sorted(out, key=lambda x: (x.change, x.vendor, x.criterion, x.against))


# know-nothing probability, in closed form -----------------------------------------------------------------------------------------
def win_probability_uniform(scores: Scores, a: str, b: str, criteria: Sequence[str] | None = None) -> float:
    """The probability that ``a`` scores strictly higher than ``b`` if the weights are completely unknown, meaning drawn uniformly from
    all possible weightings of ``criteria`` (default all). Closed form: the lead is a weighted average of the n per-criterion differences d_i with uniform weights, whose
    distribution is the B-spline P(sum d_i w_i <= x) = sum_i (x - d_i)_+^(n-1) / prod_{j != i} (d_j - d_i). Needs distinct d_i; equal ones
    are separated by 1e-9, which changes the answer by less than 1e-6."""
    names = sorted(scores[a]) if criteria is None else sorted(criteria)
    d = [exact(scores[a][c]) - exact(scores[b][c]) for c in names]
    n = len(d)
    if n == 1:
        return 1.0 if d[0] > 0 else 0.0
    d = [x + Fraction(i, 10 ** 9 * n) for i, x in enumerate(sorted(d))] if len(set(d)) < n else sorted(d)
    zero = Fraction(0)
    cdf = Fraction(0)
    for i in range(n):
        if zero > d[i]:
            denom = Fraction(1)
            for j in range(n):
                if j != i:
                    denom *= d[j] - d[i]
            cdf += (zero - d[i]) ** (n - 1) / denom
    return float(1 - cdf)


def kendall_tau(first: Sequence[str], second: Sequence[str]) -> float:
    """Rank agreement between two orderings of the same items: 1 identical, -1 reversed."""
    if set(first) != set(second) or len(first) != len(set(first)):
        raise ValueError("both orderings must contain the same distinct items")
    n = len(first)
    if n < 2:
        return 1.0
    pos = {item: i for i, item in enumerate(second)}
    seq = [pos[item] for item in first]
    inversions = _count_inversions(seq)
    pairs = n * (n - 1) // 2
    return 1 - 2 * inversions / pairs


def _count_inversions(seq: list[int]) -> int:
    if len(seq) < 2:
        return 0
    mid = len(seq) // 2
    left, right = seq[:mid], seq[mid:]
    count = _count_inversions(left) + _count_inversions(right)
    i = j = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            count += len(left) - i
            j += 1
    merged += left[i:] + right[j:]
    seq[:] = merged
    return count


def share_of(weights: Weights) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0 or any(v < 0 for v in weights.values()):
        raise ValueError("weights must be non-negative and sum above zero")
    return {c: v / total for c, v in weights.items()}


def finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))

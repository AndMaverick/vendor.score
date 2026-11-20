"""Turning raw measures into 0 to 10 scores, and describing how sure we are of a score.

A criterion can be scored directly (a number from 0 to 10) or measured in its own units (dollars, days, percent uptime) and converted by
a value function. A vendor's figure can be a single number or a range, so that "we are not sure about their support" is part of the model
and not hidden in a decimal."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


class SpecError(ValueError):
    """The scenario file contains something the model cannot use."""


def _num(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise SpecError(f"{where}: expected a number, got {x!r}")
    if math.isnan(x) or math.isinf(x):
        raise SpecError(f"{where}: must be finite")
    return float(x)


def _only(d: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(d) - allowed)
    if extra:
        raise SpecError(f"{where}: unknown keys {extra}")


@dataclass(frozen=True)
class ValueFunction:
    """Maps a raw measure to a score from 0 to 10."""

    kind: str                              # "identity", "linear", "points" or "categories"
    worst: float = 0.0
    best: float = 10.0
    points: tuple[tuple[float, float], ...] = ()
    categories: tuple[tuple[str, float], ...] = ()

    def __call__(self, raw: float | str) -> float:
        if self.kind == "categories":
            table = dict(self.categories)
            if raw not in table:
                raise SpecError(f"unknown category {raw!r}; known: {sorted(table)}")
            return table[raw]
        if isinstance(raw, str):
            raise SpecError(f"expected a number, got {raw!r}")
        if self.kind == "identity":
            return float(raw)
        if self.kind == "linear":
            return min(10.0, max(0.0, 10.0 * (raw - self.worst) / (self.best - self.worst)))
        xs = self.points
        if raw <= xs[0][0]:
            return xs[0][1]
        if raw >= xs[-1][0]:
            return xs[-1][1]
        for (x0, y0), (x1, y1) in zip(xs, xs[1:], strict=False):
            if x0 <= raw <= x1:
                return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
        raise AssertionError("unreachable")  # pragma: no cover


IDENTITY = ValueFunction("identity")


def parse_value(spec: Any, where: str) -> ValueFunction:
    if spec is None:
        return IDENTITY
    if not isinstance(spec, dict):
        raise SpecError(f"{where}: value must be an object")
    if "categories" in spec:
        _only(spec, {"categories"}, where)
        cats = spec["categories"]
        if not isinstance(cats, dict) or not cats:
            raise SpecError(f"{where}.categories: need at least one category")
        items = []
        for k, v in sorted(cats.items()):
            s = _num(v, f"{where}.categories.{k}")
            if not 0 <= s <= 10:
                raise SpecError(f"{where}.categories.{k}: score must be within 0 to 10")
            items.append((k, s))
        return ValueFunction("categories", categories=tuple(items))
    if "points" in spec:
        _only(spec, {"points"}, where)
        pts = spec["points"]
        if not isinstance(pts, list) or len(pts) < 2:
            raise SpecError(f"{where}.points: need at least two [measure, score] pairs")
        out = []
        for i, p in enumerate(pts):
            if not isinstance(p, list) or len(p) != 2:
                raise SpecError(f"{where}.points[{i}]: expected [measure, score]")
            x, y = _num(p[0], f"{where}.points[{i}][0]"), _num(p[1], f"{where}.points[{i}][1]")
            if not 0 <= y <= 10:
                raise SpecError(f"{where}.points[{i}]: score must be within 0 to 10")
            out.append((x, y))
        out.sort()
        if any(a[0] == b[0] for a, b in zip(out, out[1:], strict=False)):
            raise SpecError(f"{where}.points: measures must be distinct")
        return ValueFunction("points", points=tuple(out))
    _only(spec, {"worst", "best"}, where)
    if "worst" not in spec or "best" not in spec:
        raise SpecError(f"{where}: give 'worst' and 'best', or 'points', or 'categories'")
    worst, best = _num(spec["worst"], f"{where}.worst"), _num(spec["best"], f"{where}.best")
    if worst == best:
        raise SpecError(f"{where}: worst and best must differ")
    return ValueFunction("linear", worst=worst, best=best)


@dataclass(frozen=True)
class Estimate:
    """What we believe about one vendor on one criterion."""

    kind: str                                  # "point", "uniform", "triangular", "raters", "category"
    values: tuple[float, ...] = ()
    category: str = ""

    def mean(self) -> float:
        if self.kind == "uniform":
            return (self.values[0] + self.values[1]) / 2
        if self.kind == "triangular":
            return sum(self.values) / 3
        return sum(self.values) / len(self.values)

    def sample(self, rng: random.Random) -> float:
        if self.kind == "point":
            return self.values[0]
        if self.kind == "uniform":
            return rng.uniform(self.values[0], self.values[1])
        if self.kind == "triangular":
            lo, mode, hi = self.values
            return rng.triangular(lo, hi, mode)
        return rng.choice(self.values)

    @property
    def uncertain(self) -> bool:
        return self.kind in {"uniform", "triangular", "raters"} and len(set(self.values)) > 1

    def low_high(self) -> tuple[float, float]:
        return min(self.values), max(self.values)


def parse_estimate(spec: Any, where: str, *, categorical: bool = False) -> Estimate:
    if categorical:
        if not isinstance(spec, str):
            raise SpecError(f"{where}: this criterion uses categories, so give one of its category names")
        return Estimate("category", category=spec)
    if isinstance(spec, (bool, str)):
        raise SpecError(f"{where}: expected a number or a range, got {spec!r}")
    if isinstance(spec, (int, float)):
        return Estimate("point", (_num(spec, where),))
    if isinstance(spec, list):
        if not spec:
            raise SpecError(f"{where}: raters list is empty")
        return Estimate("raters", tuple(_num(x, f"{where}[{i}]") for i, x in enumerate(spec)))
    if isinstance(spec, dict):
        _only(spec, {"low", "mode", "high"}, where)
        if "low" not in spec or "high" not in spec:
            raise SpecError(f"{where}: a range needs 'low' and 'high'")
        lo, hi = _num(spec["low"], f"{where}.low"), _num(spec["high"], f"{where}.high")
        if lo > hi:
            raise SpecError(f"{where}: low is above high")
        if "mode" in spec:
            mode = _num(spec["mode"], f"{where}.mode")
            if not lo <= mode <= hi:
                raise SpecError(f"{where}: mode must sit between low and high")
            return Estimate("triangular", (lo, mode, hi))
        return Estimate("uniform", (lo, hi))
    raise SpecError(f"{where}: expected a number, a range or a list of rater scores")

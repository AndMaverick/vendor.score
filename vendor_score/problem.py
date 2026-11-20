"""A scorecard decision as a validated object: criteria, weights, vendors with estimates, must-haves and stakeholders."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .values import IDENTITY, Estimate, SpecError, ValueFunction, _num, _only, parse_estimate, parse_value


@dataclass(frozen=True)
class Criterion:
    name: str
    weight: float
    value: ValueFunction = IDENTITY
    must_have: float | None = None      # smallest acceptable score; a vendor below it is out

    @property
    def categorical(self) -> bool:
        return self.value.kind == "categories"


@dataclass(frozen=True)
class Stakeholder:
    name: str
    weights: dict[str, float]


@dataclass(frozen=True)
class Problem:
    name: str
    criteria: tuple[Criterion, ...]
    vendors: dict[str, dict[str, Estimate]]
    stakeholders: tuple[Stakeholder, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def criterion_names(self) -> list[str]:
        return [c.name for c in self.criteria]

    @property
    def weights(self) -> dict[str, float]:
        return {c.name: c.weight for c in self.criteria}

    def criterion(self, name: str) -> Criterion:
        return next(c for c in self.criteria if c.name == name)

    def score_of(self, vendor: str, criterion: str, raw: float | str | None = None) -> float:
        """The 0 to 10 score for a measure; the vendor's best estimate unless ``raw`` is given."""
        c = self.criterion(criterion)
        est = self.vendors[vendor][criterion]
        if raw is None:
            raw = est.category if est.kind == "category" else est.mean()
        return c.value(raw)

    def point_scores(self) -> dict[str, dict[str, float]]:
        """Every vendor's score on every criterion at its best estimate."""
        return {v: {c: self.score_of(v, c) for c in self.criterion_names} for v in self.vendors}

    def eliminated(self) -> dict[str, list[str]]:
        """Vendors that miss a must-have at their best estimate, with the criteria they miss."""
        out: dict[str, list[str]] = {}
        scores = self.point_scores()
        for v in sorted(self.vendors):
            missed = [c.name for c in self.criteria if c.must_have is not None and scores[v][c.name] < c.must_have]
            if missed:
                out[v] = missed
        return out


def parse_problem(data: Any) -> Problem:
    if not isinstance(data, dict):
        raise SpecError("the scenario must be a JSON object")
    _only(data, {"name", "criteria", "vendors", "stakeholders", "simulation", "gates", "notes"}, "scenario")
    name = data.get("name", "scorecard")
    if not isinstance(name, str) or not name:
        raise SpecError("name must be a non-empty string")
    raw_criteria = data.get("criteria")
    if not isinstance(raw_criteria, dict) or not raw_criteria:
        raise SpecError("criteria: need at least one criterion")
    criteria = []
    for cname, spec in sorted(raw_criteria.items()):
        where = f"criteria.{cname}"
        if not isinstance(spec, dict):
            raise SpecError(f"{where}: must be an object with a weight")
        _only(spec, {"weight", "value", "must_have"}, where)
        if "weight" not in spec:
            raise SpecError(f"{where}: missing weight")
        weight = _num(spec["weight"], f"{where}.weight")
        if weight < 0:
            raise SpecError(f"{where}.weight: must not be negative")
        must = spec.get("must_have")
        if must is not None:
            must = _num(must, f"{where}.must_have")
            if not 0 <= must <= 10:
                raise SpecError(f"{where}.must_have: a score threshold from 0 to 10")
        criteria.append(Criterion(cname, weight, parse_value(spec.get("value"), f"{where}.value"), must))
    if sum(c.weight for c in criteria) <= 0:
        raise SpecError("criteria: weights must sum above zero")
    raw_vendors = data.get("vendors")
    if not isinstance(raw_vendors, dict) or len(raw_vendors) < 2:
        raise SpecError("vendors: need at least two vendors to compare")
    vendors: dict[str, dict[str, Estimate]] = {}
    for vname, row in sorted(raw_vendors.items()):
        if not isinstance(row, dict):
            raise SpecError(f"vendors.{vname}: must map each criterion to a score")
        extra = sorted(set(row) - set(raw_criteria))
        missing = sorted(set(raw_criteria) - set(row))
        if extra:
            raise SpecError(f"vendors.{vname}: unknown criteria {extra}")
        if missing:
            raise SpecError(f"vendors.{vname}: missing scores for {missing}")
        est = {}
        for c in criteria:
            e = parse_estimate(row[c.name], f"vendors.{vname}.{c.name}", categorical=c.categorical)
            if c.value.kind == "categories":
                c.value(e.category)                 # raises on unknown category
            else:
                lo, hi = (e.low_high() if e.kind != "category" else (0.0, 0.0))
                if c.value.kind == "identity" and not (lo >= 0 and hi <= 10):
                    raise SpecError(f"vendors.{vname}.{c.name}: scores are 0 to 10; give a 'value' mapping for raw measures")
            est[c.name] = e
        vendors[vname] = est
    stakeholders = []
    raw_st = data.get("stakeholders", {})
    if not isinstance(raw_st, dict):
        raise SpecError("stakeholders: must be an object")
    for sname, sw in sorted(raw_st.items()):
        if not isinstance(sw, dict) or set(sw) != set(raw_criteria):
            raise SpecError(f"stakeholders.{sname}: give a weight for every criterion, and no others")
        ws = {k: _num(v, f"stakeholders.{sname}.{k}") for k, v in sw.items()}
        if any(v < 0 for v in ws.values()) or sum(ws.values()) <= 0:
            raise SpecError(f"stakeholders.{sname}: weights must be non-negative and sum above zero")
        stakeholders.append(Stakeholder(sname, ws))
    notes = data.get("notes", {})
    if not isinstance(notes, dict):
        raise SpecError("notes: must be an object")
    return Problem(name, tuple(criteria), vendors, tuple(stakeholders), notes)

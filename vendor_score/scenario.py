"""Scenario files: strict loading, a fingerprint that pins the result, the full analysis, and the pass/fail gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from . import robust
from ._version import __version__
from .problem import Problem, parse_problem
from .smaa import Acceptability, simulate
from .values import SpecError, _num, _only

METHOD = "exact-lp+smaa-1"


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise SpecError(f"duplicate key {k!r} in the scenario file")
        out[k] = v
    return out


def loads(text: str) -> dict[str, Any]:
    """Parse scenario JSON. A repeated key is an error, because the second silently overriding the first is how a number goes wrong."""
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise SpecError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError("the scenario must be a JSON object")
    return data


def _reject_constant(name: str) -> Any:
    raise SpecError(f"{name} is not allowed in a scenario")


def fingerprint(data: dict[str, Any]) -> str:
    """SHA-256 over the canonical form of the scenario plus the library version and method. Same input, same fingerprint, on any machine."""
    canon = json.dumps({"scenario": data, "version": __version__, "method": METHOD}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Settings:
    trials: int = 20000
    seed: int = 1
    weights: str = "stated"
    concentration: float = 50.0


def parse_settings(raw: Any) -> Settings:
    if raw is None:
        return Settings()
    if not isinstance(raw, dict):
        raise SpecError("simulation: must be an object")
    _only(raw, {"trials", "seed", "weights", "concentration"}, "simulation")
    trials = raw.get("trials", 20000)
    seed = raw.get("seed", 1)
    if isinstance(trials, bool) or not isinstance(trials, int) or not 100 <= trials <= 2_000_000:
        raise SpecError("simulation.trials: an integer from 100 to 2000000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SpecError("simulation.seed: must be an integer")
    mode = raw.get("weights", "stated")
    if mode not in {"stated", "uniform", "fixed"}:
        raise SpecError("simulation.weights: 'stated', 'uniform' or 'fixed'")
    conc = _num(raw.get("concentration", 50.0), "simulation.concentration")
    if conc <= 0:
        raise SpecError("simulation.concentration: must be positive")
    return Settings(trials, seed, mode, conc)


GATE_KEYS = {"min_margin", "min_prob_leader_best", "min_flip_distance", "leader_must_be", "unanimous_stakeholders", "max_leader_disqualified"}


def parse_gates(raw: Any, problem: Problem) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SpecError("gates: must be an object")
    _only(raw, GATE_KEYS, "gates")
    out: dict[str, Any] = {}
    for k in ("min_margin", "min_prob_leader_best", "min_flip_distance", "max_leader_disqualified"):
        if k in raw:
            out[k] = _num(raw[k], f"gates.{k}")
            if out[k] < 0 or (k != "min_margin" and out[k] > 1 and k != "min_flip_distance"):
                raise SpecError(f"gates.{k}: out of range")
    if "min_flip_distance" in out and out["min_flip_distance"] > 1:
        raise SpecError("gates.min_flip_distance: a share of the total weight, from 0 to 1")
    if "leader_must_be" in raw:
        if raw["leader_must_be"] not in problem.vendors:
            raise SpecError(f"gates.leader_must_be: unknown vendor {raw['leader_must_be']!r}")
        out["leader_must_be"] = raw["leader_must_be"]
    if "unanimous_stakeholders" in raw:
        if not isinstance(raw["unanimous_stakeholders"], bool):
            raise SpecError("gates.unanimous_stakeholders: true or false")
        if raw["unanimous_stakeholders"] and not problem.stakeholders:
            raise SpecError("gates.unanimous_stakeholders: the scenario has no stakeholders")
        out["unanimous_stakeholders"] = raw["unanimous_stakeholders"]
    return out


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    """Run the whole analysis on a scenario dict and return a JSON-ready result with the gate verdicts."""
    problem = parse_problem(data)
    settings = parse_settings(data.get("simulation"))
    gates = parse_gates(data.get("gates"), problem)
    eliminated = problem.eliminated()
    alive = [v for v in sorted(problem.vendors) if v not in eliminated]
    if not alive:
        raise SpecError("every vendor misses a must-have; there is nothing to rank")
    scores = {v: s for v, s in problem.point_scores().items() if v in alive}
    weights = problem.weights
    crit_used = [c for c in problem.criterion_names if weights[c] > 0]
    totals = robust.totals(weights, scores)
    order = sorted(alive, key=lambda v: (-totals[v], v))
    top = order[0]
    margin = totals[top] - totals[order[1]] if len(order) > 1 else None
    result: dict[str, Any] = {
        "name": problem.name,
        "version": __version__,
        "method": METHOD,
        "fingerprint": fingerprint(data),
        "criteria": {c.name: {"weight": c.weight, "share": c.weight / sum(weights.values())} for c in problem.criteria},
        "eliminated": eliminated,
        "ranking": [{"vendor": v, "total": totals[v], "place": i + 1} for i, v in enumerate(order)],
        "leader": top,
        "margin": margin,
        "dominated": {},
        "possible": {},
        "windows": [],
        "flips": [],
        "levers": [],
        "uniform_win_probability": {},
        "acceptability": {},
        "stakeholders": None,
    }
    if len(alive) > 1:
        result["dominated"] = robust.dominated(scores)
        result["possible"] = {
            v: {"possibly_best": b.possibly_best, "margin": b.margin, "weights": b.weights}
            for v, b in robust.possibly_best(scores).items()
        }
        result["windows"] = [
            {"criterion": w.criterion, "weight_share": w.weight, "low": w.low, "high": w.high, "low_rival": w.low_rival, "high_rival": w.high_rival}
            for w in robust.weight_windows(weights, scores)
        ]
        result["flips"] = [{"challenger": f.challenger, "moved": f.moved, "weights": f.weights} for f in robust.flips(weights, scores)]
        result["levers"] = [
            {"vendor": lv.vendor, "criterion": lv.criterion, "direction": lv.direction, "change": lv.change, "to": lv.to, "against": lv.against}
            for lv in robust.levers(weights, scores)[:10]
        ]
        result["uniform_win_probability"] = {
            v: robust.win_probability_uniform(scores, top, v, crit_used) for v in alive if v != top
        }
    acc = simulate(problem, trials=settings.trials, seed=settings.seed, weights=settings.weights, concentration=settings.concentration)
    result["simulation"] = {"trials": settings.trials, "seed": settings.seed, "weights": settings.weights, "concentration": settings.concentration}
    result["acceptability"] = _acceptability(acc)
    if problem.stakeholders:
        result["stakeholders"] = _stakeholders(problem, scores)
    result["gates"] = _gates(gates, result, acc)
    result["passed"] = all(g["passed"] for g in result["gates"])
    return result


def _acceptability(acc: Acceptability) -> dict[str, Any]:
    out = {}
    for v in acc.vendors:
        lo, hi = acc.p_best_interval(v)
        out[v] = {
            "p_best": acc.p_best(v),
            "p_best_low": lo,
            "p_best_high": hi,
            "p_disqualified": acc.p_disqualified(v),
            "expected_rank": acc.expected_rank(v),
            "rank_probabilities": acc.acceptability(v),
            "central_weights": acc.central_weights[v],
        }
    return out


def _stakeholders(problem: Problem, scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    leaders, orders = {}, {}
    for s in problem.stakeholders:
        t = robust.totals(s.weights, scores)
        orders[s.name] = sorted(t, key=lambda v: (-t[v], v))
        leaders[s.name] = orders[s.name][0]
    names = sorted(orders)
    taus = [robust.kendall_tau(orders[a], orders[b]) for i, a in enumerate(names) for b in names[i + 1:]]
    return {
        "leaders": leaders,
        "unanimous": len(set(leaders.values())) == 1,
        "min_rank_agreement": min(taus) if taus else 1.0,
        "orderings": orders,
    }


def _gates(gates: dict[str, Any], r: dict[str, Any], acc: Acceptability) -> list[dict[str, Any]]:
    out = []
    top = r["leader"]

    def add(name: str, passed: bool, detail: str) -> None:
        out.append({"gate": name, "passed": passed, "detail": detail})

    if "leader_must_be" in gates:
        add("leader_must_be", top == gates["leader_must_be"], f"leader is {top}, required {gates['leader_must_be']}")
    if "min_margin" in gates:
        m = r["margin"]
        add("min_margin", m is None or m >= gates["min_margin"], "single vendor left" if m is None else f"margin {m:.4f}, required {gates['min_margin']}")
    if "min_prob_leader_best" in gates:
        p = acc.p_best(top)
        add("min_prob_leader_best", p >= gates["min_prob_leader_best"], f"{top} is first in {p:.1%} of trials, required {gates['min_prob_leader_best']:.1%}")
    if "min_flip_distance" in gates:
        flips = r["flips"]
        nearest = min((f["moved"] for f in flips), default=None)
        if nearest is None:
            add("min_flip_distance", True, "no reweighting can dethrone the leader")
        else:
            add("min_flip_distance", nearest >= gates["min_flip_distance"], f"nearest flip needs {nearest:.1%} of the weight to move, required {gates['min_flip_distance']:.1%}")
    if "max_leader_disqualified" in gates:
        p = acc.p_disqualified(top)
        add("max_leader_disqualified", p <= gates["max_leader_disqualified"], f"{top} misses a must-have in {p:.1%} of trials, allowed {gates['max_leader_disqualified']:.1%}")
    if gates.get("unanimous_stakeholders"):
        st = r["stakeholders"]
        add("unanimous_stakeholders", st["unanimous"], "stakeholders back " + ", ".join(sorted(set(st["leaders"].values()))))
    return out

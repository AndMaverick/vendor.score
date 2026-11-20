"""Plain-text and JSON rendering of an analysis."""
from __future__ import annotations

import json
from typing import Any


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)


def to_text(r: dict[str, Any]) -> str:
    out = [f"{r['name']}   (vendor.score {r['version']}, fingerprint {r['fingerprint'][:16]})", ""]
    out.append("Ranking at the stated weights and best estimates")
    for row in r["ranking"]:
        out.append(f"  {row['place']}. {row['vendor']:<18} {row['total']:.3f}")
    for v, missed in sorted(r["eliminated"].items()):
        out.append(f"  out  {v:<18} misses must-have: {', '.join(missed)}")
    if r["margin"] is not None:
        out.append(f"  lead of {r['leader']} over second place: {r['margin']:.3f}")
    if r["dominated"]:
        out.append("")
        out.append("Dominated (cannot win under any weighting)")
        for v, by in sorted(r["dominated"].items()):
            out.append(f"  {v} is beaten or matched on every criterion by {by}")
    if r["possible"]:
        out.append("")
        out.append("Can it ever win? (best possible weighting for each vendor)")
        for v in sorted(r["possible"]):
            p = r["possible"][v]
            verdict = "yes" if p["possibly_best"] else "no"
            out.append(f"  {v:<18} {verdict:<3} best-case lead {p['margin']:+.3f}")
    if r["windows"]:
        out.append("")
        out.append(f"How far can each weight move before {r['leader']} loses the top place")
        for w in r["windows"]:
            lo = "0.0%" if w["low_rival"] is None else f"{_pct(w['low'])} ({w['low_rival']} draws level)"
            hi = "100.0%" if w["high_rival"] is None else f"{_pct(w['high'])} ({w['high_rival']} draws level)"
            out.append(f"  {w['criterion']:<18} now {_pct(w['weight_share'])}, safe from {lo} to {hi}")
    if r["flips"]:
        out.append("")
        out.append("Smallest reweighting that lets each rival draw level with the leader")
        for f in r["flips"]:
            out.append(f"  {f['challenger']:<18} {_pct(f['moved'])} of the weight moves")
    if r["levers"]:
        out.append("")
        out.append("Scores to check first (one correction that would change the winner)")
        for lv in r["levers"][:5]:
            out.append(f"  {lv['vendor']} {lv['criterion']} {lv['direction']} by {lv['change']:.3f} to {lv['to']:.3f} (draws level with {lv['against']})")
    if r["uniform_win_probability"]:
        out.append("")
        out.append(f"If the weights were unknown, chance {r['leader']} beats each rival")
        for v, p in sorted(r["uniform_win_probability"].items()):
            out.append(f"  {v:<18} {_pct(p)}")
    sim = r["simulation"]
    out.append("")
    out.append(f"Uncertainty ({sim['trials']} trials, seed {sim['seed']}, weights {sim['weights']})")
    out.append(f"  {'vendor':<18} {'first':>7}  {'95% interval':<15} {'expected place':>14} {'out':>6}")
    for v, a in sorted(r["acceptability"].items(), key=lambda kv: (-kv[1]["p_best"], kv[0])):
        interval = f"{_pct(a['p_best_low'])} to {_pct(a['p_best_high'])}"
        out.append(f"  {v:<18} {_pct(a['p_best']):>7}  {interval:<15} {a['expected_rank']:>14.2f} {_pct(a['p_disqualified']):>6}")
    if r["stakeholders"]:
        st = r["stakeholders"]
        out.append("")
        out.append("Stakeholders")
        for name, lead in sorted(st["leaders"].items()):
            out.append(f"  {name:<18} picks {lead}")
        out.append(f"  unanimous: {'yes' if st['unanimous'] else 'no'}; weakest agreement between two orderings (Kendall tau): {st['min_rank_agreement']:.2f}")
    if r["gates"]:
        out.append("")
        out.append("Gates")
        for g in r["gates"]:
            out.append(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['gate']}: {g['detail']}")
        out.append("")
        out.append("RESULT: " + ("PASS" if r["passed"] else "FAIL"))
    return "\n".join(out) + "\n"

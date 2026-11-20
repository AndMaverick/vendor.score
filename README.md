# vendor.score

Every vendor decision comes down to a weighted scorecard, and every scorecard hides the same question: would the answer change if our priorities shifted a little? vendor.score ranks vendors and then stress tests the result, so you learn whether the winner is robust or an artifact of the weights. Standard library only.

Version 0.2 turns the ranking into a defensible exhibit. It answers, with exact arithmetic where that is possible and a seeded simulation where it is not, the questions a board asks of a scorecard:

- Can the second place vendor ever win, under any weighting, and with which weights? (solved exactly)
- How far can each weight move before the leader loses first place, and who takes over? (exact)
- What is the smallest reweighting that lets each rival draw level? (exact)
- Which single score, if wrong, would change the winner? Check that one first. (exact)
- If nobody knows the weights, how likely is it that the leader beats each rival? (closed form)
- If the weights and the scores are uncertain, how often does each vendor come first, with what interval? (simulation)
- Do the stakeholders agree? Does anyone's weighting pick a different winner?
- Does the decision clear the bar you set in advance? The command exits 1 when it does not, so a pipeline or a review checklist can enforce it.

## Use

```python
from vendor_score import margin, rank, sensitivity

weights = {"cost": 0.5, "quality": 0.5}
scores = {"A": {"cost": 9, "quality": 6}, "B": {"cost": 5, "quality": 9}}

rank(weights, scores)                      # [('A', 7.5), ('B', 7.0)]
margin(weights, scores)                    # 0.5 points between first and second
sensitivity(weights, scores, swing=0.5)    # {'leader': 'A', 'stable': False, 'flips': [('cost', 'down', 'B'), ...]}
```

The three functions above are the 0.1 interface and behave exactly as before. The 0.2 analysis works on the same two dicts:

```python
from vendor_score import best_case, flips, levers, weight_windows, win_probability_uniform

best_case(scores, "B").possibly_best        # can B win under some weighting? exact, with a witness
best_case(scores, "B").weights              # the weighting that treats B most kindly
flips(weights, scores)                      # smallest reweighting for each rival to draw level
weight_windows(weights, scores)             # exact safe range of every weight, narrowest first
levers(weights, scores)                     # single score corrections that would change the winner
win_probability_uniform(scores, "A", "B")   # chance A beats B if the weights are completely unknown
```

## Scenario files

A scorecard with measured criteria, uncertain scores, minimum scores, stakeholders and pass/fail gates lives in one JSON file. Loading is strict: an unknown key, a repeated key, a missing score, a number outside its range or a NaN is an error, never a silent default.

```json
{
  "criteria": {
    "annual_cost": {"weight": 0.25, "value": {"worst": 1200000, "best": 600000}},
    "uptime":      {"weight": 0.20, "value": {"points": [[99.0, 0], [99.5, 4], [99.9, 8], [99.99, 10]]}},
    "security":    {"weight": 0.20, "must_have": 5},
    "exit_terms":  {"weight": 0.10, "value": {"categories": {"portable": 10, "standard": 6, "locked_in": 1}}}
  },
  "vendors": {
    "Aster": {"annual_cost": {"low": 720000, "mode": 780000, "high": 900000}, "uptime": 99.95,
              "security": 8.5, "exit_terms": "portable"}
  },
  "stakeholders": {"finance": {"annual_cost": 0.5, "uptime": 0.2, "security": 0.2, "exit_terms": 0.1}},
  "simulation": {"trials": 20000, "seed": 7, "weights": "stated", "concentration": 60},
  "gates": {"leader_must_be": "Aster", "min_margin": 0.3, "min_prob_leader_best": 0.6, "min_flip_distance": 0.05}
}
```

- **Scores** are a number, a range (`low`/`high`, or `low`/`mode`/`high`), or a list of rater scores. Raw measures such as dollars, days and percent uptime are converted to 0 to 10 by a linear, piecewise or category value function.
- **Minimum scores** (`must_have` in the file) remove a vendor that scores below the threshold. At the best estimate it is dropped from the ranking; inside the simulation the check is made on every draw, so a vendor that only sometimes misses is penalized by exactly how often.
- **Weights** do not have to sum to one. The simulation scatters them around the stated weights (`concentration`, higher is tighter), draws them uniformly (`uniform`, know nothing), or holds them fixed.
- **Gates** are `leader_must_be`, `min_margin`, `min_prob_leader_best`, `min_flip_distance`, `max_leader_disqualified` and `unanimous_stakeholders`.

```
python -m vendor_score analyze scenario.json            # report; exit 0 pass, 1 a gate failed, 2 unusable input
python -m vendor_score analyze scenario.json --json     # the same, machine readable
python -m vendor_score fingerprint scenario.json        # SHA-256 of scenario + version + method
```

Two scenarios ship in `examples/`. `cloud_platform.json` clears every gate. `board_mandate.json` does not, and this is what it prints:

```
Payments processor, board mandate   (vendor.score 0.2.0, fingerprint 7c296014899740d9)

Ranking at the stated weights and best estimates
  1. Marlin             7.461
  2. Halyard            7.320
  3. Tidewater          6.921
  lead of Marlin over second place: 0.140

Can it ever win? (best possible weighting for each vendor)
  Halyard            yes best-case lead +0.417
  Marlin             yes best-case lead +2.500
  Tidewater          yes best-case lead +1.364

How far can each weight move before Marlin loses the top place
  fees               now 30.0%, safe from 0.0% to 33.4% (Halyard draws level)
  settlement         now 15.0%, safe from 10.0% (Halyard draws level) to 100.0%
  fraud_tools        now 20.0%, safe from 11.8% (Halyard draws level) to 100.0%
  roadmap            now 10.0%, safe from 0.0% to 100.0%
  reliability        now 25.0%, safe from 2.1% (Halyard draws level) to 100.0%

Smallest reweighting that lets each rival draw level with the leader
  Halyard            2.7% of the weight moves
  Tidewater          5.9% of the weight moves

Scores to check first (one correction that would change the winner)
  Halyard fees rises by 0.467 to 7.285 (draws level with Halyard)
  Marlin fees falls by 0.467 to 3.624 (draws level with Halyard)
  Halyard reliability rises by 0.561 to 8.761 (draws level with Halyard)
  Marlin reliability falls by 0.561 to 8.239 (draws level with Halyard)
  Halyard fraud_tools rises by 0.701 to 7.701 (draws level with Halyard)

If the weights were unknown, chance Marlin beats each rival
  Halyard            81.5%
  Tidewater          86.2%

Uncertainty (20000 trials, seed 11, weights stated)
  vendor               first  95% interval    expected place    out
  Marlin               61.0%  60.3% to 61.7%            1.50   0.0%
  Halyard              32.3%  31.7% to 33.0%            1.95  25.1%
  Tidewater             6.7%  6.3% to 7.0%              2.55   0.0%

Stakeholders
  engineering        picks Marlin
  finance            picks Halyard
  risk               picks Marlin
  unanimous: no; weakest agreement between two orderings (Kendall tau): -0.33

Gates
  FAIL  min_margin: margin 0.1402, required 0.25
  FAIL  min_prob_leader_best: Marlin is first in 61.0% of trials, required 70.0%
  FAIL  min_flip_distance: nearest flip needs 2.7% of the weight to move, required 5.0%
  PASS  max_leader_disqualified: Marlin misses a must-have in 0.0% of trials, allowed 5.0%
  FAIL  unanimous_stakeholders: stakeholders back Halyard, Marlin

RESULT: FAIL
```

The leader comes first in only 61 percent of trials, a second place vendor that finance prefers draws level after just 2.7 percent of the weight moves, and the three stakeholders back two different vendors. The ranking looked decisive; the stress test says the board should not sign this yet.

## Design

- **Exact where it can be.** The best case weighting, the flip distance, the weight windows and the score levers are solved in rational arithmetic (a small simplex with Bland's rule, `vendor_score/lp.py`). A tie is a tie and a zero is a zero; there is no tolerance to tune, and the weighting returned as proof is one anyone can check by hand.
- **Simulation only where it must be.** Uncertain scores and uncertain weights are sampled with one seeded generator, and the report prints trial count, seed and a Wilson interval.
- **Pinned.** Every result carries a SHA-256 fingerprint of the scenario, the library version and the method. Two people running the same file get the same fingerprint and the same numbers.
- **Bad input fails loudly.** Missing scores and out of range values raise, instead of quietly skewing a seven figure decision.

## Evidence

The exact routines are tested against reference implementations that share no code with them (`tests/oracles.py`): brute force vertex enumeration of the linear program, a grid search over every weighting, exact polygon clipping for probabilities, a quadratic Kendall tau. `python examples/evaluate.py` reruns the comparison from fixed seeds:

```
possibly-best vs exact vertex enumeration: 1002 vendors on 250 scorecards, verdicts differing 0, largest gap in lead 0.0e+00
flip distance vs 100-step grid search over all weightings: 176 flips on 120 scorecards, disagreements 0, exact answer ever worse than the grid 1.1e-16
unknown-weights win probability vs exact polygon area: 300 pairs, largest gap 1.4e-10
vendors the exact solver says can never win: 106; times they came first in 5000 random weightings each: 0
weight windows vs a 1% scan of each weight: 23974 points checked, disagreements 0
full analysis of 10 vendors x 8 criteria with 20000 trials: 0.5 s (leader v8)
```

The suite has 100+ tests: oracles, invariances (scaling the weights, reordering criteria, shifting a criterion by a constant leave the answers unchanged), strict input rejection, gate behavior, exit codes, and identical output under different `PYTHONHASHSEED` values.

## Honest limits

- The tool checks the decision, not the inputs. If a score is wrong, the analysis is exactly right about a wrong scorecard. The score levers list which numbers deserve evidence first.
- The model is a weighted sum. It assumes criteria trade off linearly and are independent. Where a failure on one criterion cannot be bought back, set a minimum score.
- "Distance" is the share of total weight that has to move (half the L1 change). Weight windows scale the other weights in proportion. Both are conventions, stated in the output, and not the only reasonable ones.
- Estimates for different vendors and criteria are drawn independently. Correlated errors (one analyst scoring everything high) are not modeled.
- With uncertain scores, the deterministic sections use each estimate's mean. The simulation is where the ranges matter.
- The closed form unknown weights probability separates exactly tied score differences by 1e-9, which moves the answer by less than 1e-6.
- Exact results are identical everywhere. Simulation results are reproducible for a given seed on a given Python version; a different Python's random generator may differ in the last digits.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**If this is your problem.** If you are deciding what to build, buy, and fund, I help leadership teams make that call with evidence. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=vendor.score).

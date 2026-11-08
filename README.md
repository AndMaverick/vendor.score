# vendor.score

Every vendor decision comes down to a weighted scorecard, and every scorecard hides the same question: would the answer change if our priorities shifted a little? vendor.score ranks vendors and then stress-tests the result, so you learn whether the winner is robust or an artifact of the weights. Standard library only.

## Use

```python
from vendor_score import margin, rank, sensitivity

weights = {"cost": 0.5, "quality": 0.5}
scores = {"A": {"cost": 9, "quality": 6}, "B": {"cost": 5, "quality": 9}}

rank(weights, scores)                      # [('A', 7.5), ('B', 7.0)]
margin(weights, scores)                    # 0.5 points between first and second
sensitivity(weights, scores, swing=0.5)    # {'leader': 'A', 'stable': False, 'flips': [('cost', 'down', 'B'), ...]}
```

## Design

- **The ranking is the easy part.** Scores are 0 to 10, weights are normalized, ties break alphabetically so output is deterministic.
- **Sensitivity is the point.** Each weight moves up and down by the swing you choose. If any move changes the leader, the result says which criterion, which direction, and who takes over.
- **Margin shows how close it is.** A leader by half a point on a fragile weighting is a different decision from a leader by three.
- **Bad input fails loudly.** Missing scores and out-of-range values raise, instead of quietly skewing a seven-figure decision.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**If this is your problem.** If you are deciding what to build, buy, and fund, I help leadership teams make that call with evidence. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=vendor.score).

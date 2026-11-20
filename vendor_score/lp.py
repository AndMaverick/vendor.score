"""A small exact linear-programming solver (simplex with Bland's rule over rational numbers).

Vendor decisions are small: a handful of criteria and a few dozen vendors. At that size exact arithmetic costs nothing, and it means a
tie is a tie and a zero is a zero, with no tolerance to tune."""
from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction


def exact(x: float | int | Fraction) -> Fraction:
    """A number as an exact fraction. Floats go through their shortest decimal form, so 0.1 is exactly one tenth, as written."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise ValueError("expected a number")
    if isinstance(x, int):
        return Fraction(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"cannot use {x} in an exact calculation")
    return Fraction(repr(x))


class Unbounded(Exception):
    """The objective can grow without limit."""


def maximize(c: Sequence[Fraction], a: Sequence[Sequence[Fraction]], b: Sequence[Fraction]) -> tuple[Fraction, list[Fraction]]:
    """Maximise ``c . x`` subject to ``a x <= b`` and ``x >= 0``, where every ``b`` is non-negative (so ``x = 0`` is feasible).

    Returns the optimal value and one optimal ``x``. Bland's rule (always pick the lowest-numbered improving column and the
    lowest-numbered tied row) guarantees termination even on degenerate problems."""
    m, n = len(a), len(c)
    if any(len(row) != n for row in a) or len(b) != m:
        raise ValueError("dimension mismatch")
    if any(v < 0 for v in b):
        raise ValueError("right-hand sides must be non-negative")
    # tableau: m constraint rows [a | I | b], then the objective row [-c | 0 | 0]
    t = [[*row, *[Fraction(int(i == j)) for j in range(m)], b[i]] for i, row in enumerate(a)]
    t.append([-v for v in c] + [Fraction(0)] * (m + 1))
    basis = [n + i for i in range(m)]
    while True:
        col = next((j for j in range(n + m) if t[m][j] < 0), None)
        if col is None:
            break
        best_row, best_ratio = None, None
        for i in range(m):
            if t[i][col] > 0:
                ratio = t[i][-1] / t[i][col]
                if best_ratio is None or ratio < best_ratio or (ratio == best_ratio and basis[i] < basis[best_row]):  # type: ignore[index]
                    best_row, best_ratio = i, ratio
        if best_row is None:
            raise Unbounded
        pivot = t[best_row][col]
        t[best_row] = [v / pivot for v in t[best_row]]
        for i in range(m + 1):
            if i != best_row and t[i][col] != 0:
                factor = t[i][col]
                t[i] = [v - factor * w for v, w in zip(t[i], t[best_row], strict=True)]
        basis[best_row] = col
    x = [Fraction(0)] * n
    for i, var in enumerate(basis):
        if var < n:
            x[var] = t[i][-1]
    return t[m][-1], x

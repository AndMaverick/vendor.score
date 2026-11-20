"""Independent implementations used to check the library: they share no code with it."""
from __future__ import annotations

import itertools
from fractions import Fraction


def fr(x):
    return Fraction(repr(x)) if isinstance(x, float) else Fraction(x)


def solve(matrix, rhs):
    """Solve a square linear system exactly; None if singular."""
    n = len(matrix)
    m = [list(map(fr, row)) + [fr(r)] for row, r in zip(matrix, rhs, strict=True)]
    for col in range(n):
        piv = next((r for r in range(col, n) if m[r][col] != 0), None)
        if piv is None:
            return None
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col], strict=True)]
    return [m[r][n] for r in range(n)]


def best_lead_by_vertices(scores, vendor):
    """max over weights of the smallest lead of ``vendor`` over the others, by enumerating every vertex of the feasible region.

    Unknowns are the n weights and the lead t. Constraints: sum w = 1, w_i >= 0, lead_u(w) >= t for each rival u. An optimum of this
    linear program sits where n+1 independent constraints are tight, so try every choice of n tight ones besides the sum."""
    criteria = sorted(scores[vendor])
    others = [u for u in sorted(scores) if u != vendor]
    n = len(criteria)
    d = {u: [fr(scores[vendor][c]) - fr(scores[u][c]) for c in criteria] for u in others}
    # candidate tight constraints: ("w", i) means w_i = 0; ("u", u) means lead_u = t
    cands = [("w", i) for i in range(n)] + [("u", u) for u in others]
    best = None
    for combo in itertools.combinations(cands, n):
        rows, rhs = [[Fraction(1)] * n + [Fraction(0)]], [Fraction(1)]
        for kind, key in combo:
            if kind == "w":
                rows.append([Fraction(int(j == key)) for j in range(n)] + [Fraction(0)])
                rhs.append(Fraction(0))
            else:
                rows.append(list(d[key]) + [Fraction(-1)])
                rhs.append(Fraction(0))
        sol = solve(rows, rhs)
        if sol is None:
            continue
        w, t = sol[:n], sol[n]
        if any(x < 0 for x in w):
            continue
        if any(sum(wi * di for wi, di in zip(w, d[u], strict=True)) < t for u in others):
            continue
        if best is None or t > best:
            best = t
    return best


def triangle_area_fraction(d):
    """Share of the 3-criterion weight triangle where sum d_i w_i < 0, by clipping the triangle with a half-plane (exact)."""
    d = [fr(x) for x in d]
    poly = [(Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1))]

    def f(p):
        return sum(a * b for a, b in zip(p, d, strict=True))

    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % 3]
        fp, fq = f(p), f(q)
        if fp < 0:
            out.append(p)
        if (fp < 0) != (fq < 0) and fp != fq:
            t = fp / (fp - fq)
            out.append(tuple(a + t * (b - a) for a, b in zip(p, q, strict=True)))

    def area(pts):
        # area in the (w1, w2) plane; the full triangle has area 1/2
        s = Fraction(0)
        for i in range(len(pts)):
            x0, y0 = pts[i][0], pts[i][1]
            x1, y1 = pts[(i + 1) % len(pts)][0], pts[(i + 1) % len(pts)][1]
            s += x0 * y1 - x1 * y0
        return abs(s) / 2

    return area(out) / Fraction(1, 2) if len(out) >= 3 else Fraction(0)


def kendall_brute(a, b):
    pos = {x: i for i, x in enumerate(b)}
    s = [pos[x] for x in a]
    n = len(s)
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if s[i] > s[j])
    return 1 - 2 * inv / (n * (n - 1) / 2) if n > 1 else 1.0


def totals_exact(weights, scores):
    tot = sum(fr(v) for v in weights.values())
    return {v: sum(fr(weights[c]) / tot * fr(scores[v][c]) for c in weights) for v in scores}

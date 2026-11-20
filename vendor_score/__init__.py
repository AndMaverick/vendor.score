"""vendor.score: stress-test a weighted vendor scorecard.

The 0.1 functions (``normalize``, ``rank``, ``sensitivity``, ``margin``) are unchanged. The 0.2 additions answer the questions a board
asks of a scorecard: could another vendor win, how far do the weights have to move, which scores matter, and how often does the leader
come first when we are unsure."""
from ._version import __version__
from .robust import (
    BestCase,
    Flip,
    Lever,
    Window,
    best_case,
    dominated,
    dominates,
    flip_for,
    flips,
    kendall_tau,
    leader,
    levers,
    possibly_best,
    totals,
    weight_window,
    weight_windows,
    win_probability_uniform,
)
from .scenario import analyze, fingerprint, loads
from .simple import margin, normalize, rank, sensitivity
from .smaa import Acceptability, simulate
from .values import SpecError

__all__ = [
    "Acceptability", "BestCase", "Flip", "Lever", "SpecError", "Window", "__version__", "analyze", "best_case", "dominated",
    "dominates", "fingerprint", "flip_for", "flips", "kendall_tau", "leader", "levers", "loads", "margin", "normalize",
    "possibly_best", "rank", "sensitivity", "simulate", "totals", "weight_window", "weight_windows", "win_probability_uniform",
]

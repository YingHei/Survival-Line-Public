"""生死線 — a dynamic support/resistance ladder from 升/跌段 轉角位.

Rules R1-R12 are specified in
``docs/superpowers/specs/2026-07-31-生死線-indicator-design.md``.
"""

from .classify import classify_bar, classify_series, inside_runs
from .engine import EngineResult, run
from .legs import find_legs
# from .lines import LineLadder  # dead: LineLadder itself is commented out
from .pivots import find_pivots
from .types import BarClass, Direction, Leg, Line, LineKind, Pivot, PivotKind

__all__ = [
    "BarClass",
    "Direction",
    "EngineResult",
    "Leg",
    "Line",
    "LineKind",
    # "LineLadder",  # dead: LineLadder itself is commented out
    "Pivot",
    "PivotKind",
    "classify_bar",
    "classify_series",
    "find_legs",
    "find_pivots",
    "inside_runs",
    "run",
]

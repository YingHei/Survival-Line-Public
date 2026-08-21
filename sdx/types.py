"""Shared types for the 生死線 engine.

Rule references (R1-R12) point at
``docs/superpowers/specs/2026-07-31-生死線-indicator-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BarClass(str, Enum):
    """R1 — the four 陰陽燭 classes. 陰陽 (bullish/bearish) is ignored throughout."""

    UP_BAR = "上移"  # high and low both above the reference bar
    DOWN_BAR = "下移"  # high and low both below the reference bar
    INSIDE_BAR = "內困"  # contained by the reference bar — 無方向
    OUTSIDE_BAR = "外擴"  # breaks the reference bar on both sides — 或為轉角位


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


#: R1/R2 — the direction each class contributes.
#: 內困 is 無方向 and defers forward; 外擴 is a pivot candidate resolved by R3.
DIRECTION_OF: dict[BarClass, Optional[Direction]] = {
    BarClass.UP_BAR: Direction.UP,
    BarClass.DOWN_BAR: Direction.DOWN,
    BarClass.INSIDE_BAR: None,
    BarClass.OUTSIDE_BAR: None,
}


class PivotKind(str, Enum):
    TOP = "頂"
    BOTTOM = "底"


class LineKind(str, Enum):
    FUSU = "復甦線"  # support at a 跌→升 轉段
    SIWANG = "死亡線"  # resistance at a 升→跌 轉段
    SUPPORT = "支持線"  # ordinary support
    RESISTANCE = "阻力線"  # ordinary resistance


@dataclass(frozen=True)
class Pivot:
    """A 轉角位.

    R12 — three distinct indices, easily conflated:

    ``bar``          the bar the pivot sits on; the chart draws here
    ``confirmed_at`` when it became *knowable*; the backtest may only read here
    """

    bar: int
    price: float
    kind: PivotKind
    confirmed_at: int
    rule: str  # "R3" (外擴 fractal) or "R4" (內困 run resolution)

    def __post_init__(self) -> None:
        if self.confirmed_at < self.bar:
            raise ValueError(
                f"R12 violation: confirmed_at={self.confirmed_at} < bar={self.bar}"
            )


@dataclass(frozen=True)
class Line:
    """A level in the 生死線 ladder (R6).

    ``valid_from`` implements R7 — 劃線 ≠ 即時有效 — in ``lines.LineLadder``: a
    line is inert until price breaks the prior 轉角位, and only then becomes
    an active support / 止賺位, with ``None`` meaning it never became valid
    within the series. But the ladder ``engine.run()`` actually returns is
    built by ``engine._display_lines()`` instead, which draws every line
    already active: there ``valid_from`` always equals ``confirmed_at`` and
    is never ``None``. See ``lines.py``'s module docstring.
    """

    bar: int
    price: float
    kind: LineKind
    confirmed_at: int
    valid_from: Optional[int]

    @property
    def is_support(self) -> bool:
        return self.kind in (LineKind.FUSU, LineKind.SUPPORT)


@dataclass(frozen=True)
class Leg:
    """A 段 (R5). Legs alternate direction and are derived from price action only.

    ``start_bar`` is retroactive — a leg begins at the previous leg's extreme,
    which is in the past by the time the reversal is detected. ``confirmed_at``
    is the bar where the range break actually occurred, and is the only index a
    backtest may act on (R12).
    """

    direction: Direction
    start_bar: int
    start_price: float
    confirmed_at: int
    end_bar: Optional[int] = None
    end_price: Optional[float] = None

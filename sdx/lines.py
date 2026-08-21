"""R6-R9 — the 生死線 ladder (superseded implementation).

``LineLadder`` below is NOT the path ``engine.run()`` uses for the ladder it
returns. That now goes through ``engine._display_lines()``, a from-scratch
alternating regime state machine — see the comment above its call site in
``engine.py`` for why (the short version: this class's R7 queue and volume
rule could seat a stop that did not match what was actually drawn, e.g. TSLA
2026-03-02). ``LineLadder`` is still instantiated and driven bar-by-bar
inside ``engine.run()``'s loop, but nothing reads its output back — it is
kept here as a record of the original per-rule design. In particular R7
below does NOT hold for the ladder the system actually returns:
``_display_lines()`` draws every line already active
(``valid_from == confirmed_at``, never inert, never ``None``).

生死線 is not two lines but a ladder of levels (R6):

    復甦線   support at a 跌→升 轉段
    死亡線   resistance at a 升→跌 轉段
    支持線   ordinary support — volume-doubling pivots
    阻力線   ordinary resistance

R7 — 劃線 ≠ 即時有效. A line is inert until price 升穿 the previous 轉角位, at which
point it becomes an active support / 止賺位.

R8 — ``current_stop`` is the latest valid support, held monotonically
non-decreasing within an 升段 (「最新支持位就係止賺位並會一直向上褪」).

R9 — price trading below ``current_stop`` is 清貨, the system's ONLY full
liquidation trigger. The test is on the bar's LOW, not its close: a 支持線 the
day's range dipped through has been broken, and the position is closed there.
死亡線 is overhead resistance formed at 升轉跌; it is not a level that gets
broken downward.
"""

from __future__ import annotations

from typing import Optional

from .types import Line, LineKind, Pivot, PivotKind


# DEAD CODE — commented out, not removed. Not the path engine.run() uses (see
# module docstring above); kept as a record of the original per-rule design.
# class LineLadder:
#     """Maintains the support/resistance stacks and exposes ``current_stop``.
#
#     Driven bar by bar by the engine. Every mutation is causal — nothing is added
#     before its ``confirmed_at``.
#     """
#
#     def __init__(self) -> None:
#         self.supports: list[Line] = []
#         self.resistances: list[Line] = []
#         self._pending: list[Line] = []  # drawn but not yet valid (R7)
#         self._last_pivot_top: Optional[float] = None
#         self._stop: Optional[float] = None
#         self._flat: bool = True  # no position until a support goes valid
#
#     # ---------------------------------------------------------------- drawing
#
#     def draw(self, bar: int, price: float, kind: LineKind, confirmed_at: int) -> Line:
#         """R6 — add a level. It starts inert; R7 decides when it goes live."""
#         line = Line(
#             bar=bar,
#             price=price,
#             kind=kind,
#             confirmed_at=confirmed_at,
#             valid_from=None,
#         )
#         if line.is_support:
#             self._pending.append(line)
#         else:
#             self.resistances.append(line)
#         return line
#
#     def draw_active_support(
#         self, bar: int, price: float, kind: LineKind, confirmed_at: int
#     ) -> Line:
#         """A 支持線 whose 突破 has already happened — drawn and live in one step.
#
#         Restated R6: a 支持線 exists only after 創新高 → 拉回 → 突破 that 新高.
#         That 突破 is precisely the event R7 waits for, so for this kind of line
#         drawing and activation collapse into a single step; there is no inert
#         phase. A 拉回 that never reclaims the 新高 produces no line at all, which
#         is why the ordinary 底-pivot feed into ``draw`` no longer exists.
#
#         復甦線 still goes through ``draw`` + ``update_validity``: it is placed
#         retroactively at a 轉段 extreme and genuinely is inert until price
#         confirms the turn.
#         """
#         line = Line(
#             bar=bar,
#             price=price,
#             kind=kind,
#             confirmed_at=confirmed_at,
#             valid_from=confirmed_at,
#         )
#         # Anything drawn at or before this bar is superseded, not queued — a
#         # lower old support can never become the current 止賺位.
#         self._pending = [ln for ln in self._pending if ln.bar > bar]
#         self.supports.append(line)
#
#         self._flat = False
#         # R8 — the stop trails up and never down while in a position.
#         self._stop = price if self._stop is None else max(self._stop, price)
#         return line
#
#     def note_pivot(self, pivot: Pivot) -> None:
#         """Track the most recent 頂, which R7 uses as the breakout threshold."""
#         if pivot.kind is PivotKind.TOP:
#             self._last_pivot_top = pivot.price
#
#     # ------------------------------------------------------------- validation
#
#     def update_validity(self, bar: int, high: float) -> list[Line]:
#         """R7 — a higher high activates the LATEST pending line.
#
#         「劃線 ≠ 即時有效」. Two kinds still queue here:
#
#         復甦線, placed retroactively at the extreme that ended the previous 段,
#         genuinely inert until price confirms the turn; and the volume-doubling
#         支持線, which is seated on 「成交量放大逾一倍」 alone and so has no 突破
#         of its own to activate it.
#
#         The gated 支持線 do NOT come through here — under the restated R6 they
#         are born at their 突破, which is the very event R7 waits for, so they go
#         straight to ``draw_active_support``.
#
#         Only the latest is promoted. Promoting every pending line at once let
#         the stop re-seed from the *lowest* of a batch — on 388.HK it collapsed
#         from ~400 to 360 in a single bar, which R8 forbids.
#         """
#         threshold = self._last_pivot_top
#         if threshold is not None and high <= threshold:
#             return []
#
#         ready = [ln for ln in self._pending if bar >= ln.confirmed_at]
#         if not ready:
#             return []
#
#         latest = max(ready, key=lambda ln: (ln.bar, ln.confirmed_at))
#         promoted = Line(
#             bar=latest.bar,
#             price=latest.price,
#             kind=latest.kind,
#             confirmed_at=latest.confirmed_at,
#             valid_from=bar,
#         )
#
#         # Anything drawn at or before the activated line is superseded, not
#         # queued — a lower old support can never become the current 止賺位.
#         self._pending = [ln for ln in self._pending if ln.bar > latest.bar]
#         self.supports.append(promoted)
#
#         self._flat = False
#         # R8 — the stop trails up and never down while in a position
#         self._stop = (
#             promoted.price if self._stop is None else max(self._stop, promoted.price)
#         )
#
#         return [promoted]
#
#     # ------------------------------------------------------------------ state
#
#     @property
#     def current_stop(self) -> Optional[float]:
#         """R8 — the operative 生死線. ``None`` while flat."""
#         return None if self._flat else self._stop
#
#     def check_liquidation(self, low: float) -> bool:
#         """R9 — 跌破生死線 ⇒ 清貨. Returns True on the bar the exit fires.
#
#         Tested against the bar's LOW. Using the close let a long lower shadow
#         pierce the 支持線 and recover by the bell without stopping out, which
#         then allowed the NEXT 支持線 to be drawn below the current one — a stop
#         that loosens, contradicting R8. On the low the break is taken when it
#         happens, so a 支持線 can never step down inside a held position.
#         """
#         stop = self.current_stop
#         if stop is None or low >= stop:
#             return False
#
#         self._flat = True
#         self._stop = None
#         return True

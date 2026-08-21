"""End-to-end 生死線 engine — a single causal forward pass.

Produces ``current_stop`` per bar, plus the intermediate artifacts the chart
needs (classes, pivots, legs, lines).

Causality (R12) is the load-bearing property. Legs and pivots are located over
the whole series for convenience, but each carries ``confirmed_at`` and the loop
applies it only when the bar index reaches that value. A state machine that
peeks by one bar shifts every subsequent 段, so the error cascades rather than
staying local — ``tests/test_no_future_leak.py`` is what proves it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .candles import (
    PINE_TREND_BARS,
    PatternHit,
    find_five_bar_patterns,
    find_patterns,
    find_rally_attacks,
    find_three_bar_patterns,
    find_two_bar_patterns,
)
from .classify import classify_series
from .legs import find_legs
from .pivots import find_pivots
from .types import BarClass, Direction, Leg, Line, LineKind, Pivot, PivotKind

#: R6 — 「今日成交量放大逾一倍」, measured against yesterday's volume.
VOLUME_SPIKE_MULTIPLE = 2.0

#: R10 — 量增即攻 is only valid on days 1-3 counted from the swing low its
#: 支持線 sits on. 「第 4 日起等拉回」.
BUY_WINDOW_BARS = 3

#: When a lone 外擴K has to carry a level itself, its close must clear a
#: fraction of the PREVIOUS bar's high-low range — from that bar's low for a
#: 支持線 test, from its high for a 阻力線 test. 0.5 is the previous bar's own
#: midpoint. A smaller fraction loosens the gate on BOTH sides symmetrically
#: (a smaller zone to clear); a larger fraction tightens both. Anchoring to
#: the previous bar rather than the 外擴K's own (outside-bar-inflated) range
#: is what lets a close that's genuinely weak/strong relative to where the
#: market actually traded the day before qualify even when a shadow pushes
#: it into the "wrong" half of its own wider range — and what filters a
#: 鎚頭-shaped 外擴K, whose close typically doesn't clear this threshold even
#: though it broke both of the previous bar's extremes.
OUTSIDE_BAR_CLOSE_FRACTION = 0.6

@dataclass
class EngineResult:
    classes: list[Optional[BarClass]]
    pivots: list[Pivot]
    legs: list[Leg]
    lines: list[Line] = field(default_factory=list)
    current_stop: list[Optional[float]] = field(default_factory=list)
    liquidations: list[int] = field(default_factory=list)
    #: (bar, direction) where a new extreme broke the previous 轉角位 — the
    #: chart arrows. ↑ only in 升市, ↓ only in 跌市.
    breakouts: list[tuple[int, Direction]] = field(default_factory=list)
    #: (bar, price, is_support) of levels an arrow has activated. A level that
    #: never activates is still drawn — dashed — because 劃線 ≠ 即時有效.
    activated_lines: set[tuple[int, float, bool]] = field(default_factory=set)
    #: Bars firing 量增即攻 — the entry signal (R10).
    buy_signals: list[int] = field(default_factory=list)
    #: Bars firing 好友反攻 — 錘頭 shape carrying 大量 (vs the 50-bar volume
    #: average). Keyed on the shape, not the 錘頭/吊頸 label, so it fires in an
    #: 升市 too. See :func:`sdx.candles.find_rally_attacks`.
    rally_signals: list[int] = field(default_factory=list)
    #: 陰陽燭形態 — the ported Pine Script pattern set (單日/雙日/三日), gated by
    #: whichever trend mode :func:`run` was called with. See `sdx.candles`.
    patterns: list[PatternHit] = field(default_factory=list)


def _gated_supports(
    highs: Sequence[float],
    pivots: Sequence[Pivot],
    trend: Sequence[Direction],
    n: int,
) -> list[tuple[Pivot, int]]:
    """Restated R6 — 支持線 require 創新高 → 拉回 → 突破 that 新高.

    Not every 底 in an 升市 is a 支持線. The sequence must complete: price makes
    a 新高, pulls back (a 底 轉角位 confirms), and then takes out that 新高. Only
    then does a level exist, and it sits on the pivot the winning leg started
    from — the LAST 底 before the 突破, not the deepest of the 拉回.

    A 拉回 that never reclaims the 新高 draws nothing and moves nothing: the
    prevailing 支持線 stays exactly where it is.

    Returns ``(pivot, breakout_bar)``. The 突破 bar is both ``confirmed_at`` and
    ``valid_from`` — the break IS the R7 activation, so there is no inert phase.

    Causal: a 底 enters via ``confirmed_at`` and the 突破 is judged on the bar it
    happens, so the caller may pass either regime.
    """
    bottoms_by_conf: dict[int, list[Pivot]] = {}
    for p in pivots:
        if p.kind is PivotKind.BOTTOM:
            bottoms_by_conf.setdefault(p.confirmed_at, []).append(p)

    out: list[tuple[Pivot, int]] = []
    ref_high: Optional[float] = None  # the 新高 that must be 突破'd
    pullback: Optional[Pivot] = None  # last 底 since that 新高

    for i in range(n):
        for p in bottoms_by_conf.get(i, []):
            pullback = p  # a later 底 supersedes — the leg starts at the last one

        if trend[i] is not Direction.UP:
            ref_high, pullback = None, None
            continue

        if ref_high is None:
            ref_high = highs[i]
            continue

        if highs[i] > ref_high:
            if pullback is not None:
                out.append((pullback, i))
                pullback = None
            ref_high = highs[i]

    return out


# DEAD CODE — commented out, not removed. Superseded by the alternating regime
# machine in _display_lines(); see the comment above its call site in run().
# def _relocate_to_retest(
#     lines: Sequence[Line], pivots: Sequence[Pivot], tol: float = 1e-6
# ) -> list[Line]:
#     """Move a 轉段 line forward to the retest that shares its price.
#
#     A double bottom puts the 復甦線 on the second touch — the one that actually
#     launched the advance. At the moment 跌轉升 confirms, that later bar may not
#     exist yet, so the line is first drawn on the first touch and moved when the
#     retest confirms. 388.HK bottomed twice at 379.0 (2026-03-23, 2026-03-30) and
#     the line belongs on 03-30.
#
#     Causal: the relocation is driven by a pivot that has itself been confirmed,
#     and ``confirmed_at`` is advanced to match.
#     """
#     out: list[Line] = []
#     for ln in lines:
#         if ln.kind not in (LineKind.FUSU, LineKind.SIWANG):
#             out.append(ln)
#             continue
#
#         want = (
#             PivotKind.BOTTOM if ln.kind is LineKind.FUSU else PivotKind.TOP
#         )
#         later = [
#             p
#             for p in pivots
#             if p.kind is want
#             and p.bar > ln.bar
#             and abs(p.price - ln.price) <= tol
#         ]
#         if not later:
#             out.append(ln)
#             continue
#
#         retest = max(later, key=lambda p: p.bar)
#         out.append(
#             Line(
#                 bar=retest.bar,
#                 price=ln.price,
#                 kind=ln.kind,
#                 confirmed_at=max(ln.confirmed_at, retest.confirmed_at),
#                 valid_from=ln.valid_from,
#             )
#         )
#     return out


def _display_lines(
    pivots: Sequence[Pivot],
    highs: Sequence[float],
    lows: Sequence[float],
    n: int,
    outside_bar_bearish: bool = True,
    outside_bar_bullish: bool = False,
    opens: Optional[Sequence[float]] = None,
    closes: Optional[Sequence[float]] = None,
    outside_bar_body: bool = True,
    outside_bar_close_fraction: Optional[float] = None,
    volumes: Optional[Sequence[float]] = None,
) -> tuple[list[Line], list[Direction], list[Optional[float]], list[int]]:
    """The drawn ladder — a strictly alternating 升市 / 跌市 state machine.

    The regime turns on a break of the prevailing LINE, not of the last 轉角位::

        復甦線 → 支持線* → 死亡線 → 阻力線* → 復甦線 → ...

    - 死亡線 is drawn when price breaks DOWN the latest 支持線, or the 復甦線
      when no 支持線 has been seated yet. It sits on the latest 頂 since the
      升市 began.
    - 阻力線 exist only inside the 跌市 a 死亡線 opened.
    - 復甦線 is the mirror: price breaking UP the latest 阻力線, or the 死亡線
      when no 阻力線 has been drawn yet, placed on the latest 底.
    - 支持線 exist only inside the 升市 a 復甦線 opened, and still require the
      full 創新高 → 拉回 → 突破 sequence of R6a.

    This replaces a regime taken from ``_trend_per_bar``, which turned on a break
    of the last 轉角位 and so could open a 跌市 while the 支持線 underneath was
    untouched. XLF drew a 死亡線 on 2026-06-17 with the prevailing 支持線 at
    51.86 and the lowest low of the following three weeks at 53.20 — a decline
    that never happened, plus the three lines that followed from it. The engine
    already knew: 清貨 did not fire between 06-03 and 07-23. Breaking the line
    IS the 清貨 test of R9, so the drawn ladder and the backtest now turn on the
    same event instead of two different ones.

    Directions follow R9: downward breaks are judged on the LOW, upward breaks
    on the HIGH.

    Causal — a pivot enters at ``confirmed_at``, and every break is judged on the
    bar it happens. Lines are born active, so ``valid_from == confirmed_at``.

    R6 — a bar whose volume is at least ``VOLUME_SPIKE_MULTIPLE`` the previous
    bar's overrides the pending R6a 支持線 anchor with that bar's own low, 支持線
    side only. See the loop body for the mechanism (modeled as a synthetic 底
    pivot so it flows through the same ``pullback`` variable R6a already reads).
    """
    by_conf: dict[int, list[Pivot]] = {}
    for p in pivots:
        by_conf.setdefault(p.confirmed_at, []).append(p)

    def body_ok(i: int, want_up: bool) -> bool:
        """Does this 外擴K's own close let its own extreme carry a level on its own?

        Two conditions, both about the close. 陽燭/陰燭 says which side won the
        session; the fraction test says it won by enough against the PREVIOUS
        bar's range, not this bar's own — a 外擴K's own range is inflated by
        the very breakout that makes it 外擴, so a lower/upper shadow can push
        the close into the "wrong" half of its own range while it is still
        genuinely weak/strong relative to where the market actually traded
        the day before. Reading against bar i-1 instead is what also filters
        a 鎚頭-shaped 外擴K (the opposite side's participants visibly stepped
        in intraday), whose close typically doesn't clear this threshold even
        though it broke both of bar i-1's extremes. Off, any 外擴K qualifies,
        which fires on nearly every breakout.

        R1 ignores 陰陽 when classifying bars. This is not classification: it is
        whether one bar's extreme can carry the 止賺位. Only ever called once
        the caller has established the bar is a 外擴K, so ``i - 1`` is always
        in range.

        ``outside_bar_close_fraction`` (the enclosing function's parameter)
        overrides the module default per call — the UI's 收市比例 control
        threads a caller-chosen value through here without mutating shared
        module state, which a concurrent server handling other requests at
        the same fraction must not see change mid-request.
        """
        if not outside_bar_body:
            return True
        if opens is None or closes is None:
            return False
        fraction = (
            OUTSIDE_BAR_CLOSE_FRACTION
            if outside_bar_close_fraction is None
            else outside_bar_close_fraction
        )
        prev_span = highs[i - 1] - lows[i - 1]
        if want_up:
            return (closes[i] > opens[i]
                    and closes[i] > lows[i - 1] + fraction * prev_span)
        return (closes[i] < opens[i]
                and closes[i] < highs[i - 1] - fraction * prev_span)

    out: list[Line] = []

    # A same-bar exception can draw a 支持線/阻力線 immediately from a bar's
    # own extreme; the general fractal-pivot mechanism can later,
    # independently, recognize that SAME extreme as a confirmed 頂/底 and try
    # to draw the identical level again once a further break activates it.
    # 3690.HK 2026-06-18 is a real instance: its close clears the
    # previous-bar-fraction body_ok test, firing the same-bar 阻力線
    # exception immediately — but 06-18's high also independently holds up
    # as an ordinary fractal 頂 (06-22 never exceeds it), so when 06-22
    # breaks the floor the bounce-based branch tries to draw (06-18, 76.25)
    # again. Track what's already out to skip the redundant repeat.
    seen_levels: set[tuple[int, float, LineKind]] = set()

    def add_line(bar_at: int, price_at: float, kind: LineKind, i: int) -> None:
        key = (bar_at, round(price_at, 6), kind)
        if key in seen_levels:
            return
        seen_levels.add(key)
        out.append(Line(bar_at, price_at, kind, i, i))

    # Opens 升市 with no level: nothing can break, so the first line drawn is
    # necessarily a 支持線 off the R6a gate. 阻力線 cannot precede a 死亡線,
    # which is the invariant the whole machine exists to hold.
    rising = True
    level: Optional[float] = None

    # The regime in force at each bar. This is the ONE definition of 升市/跌市
    # the system uses — the arrows, 量增即攻 and 陰陽燭形態 all read it, so a bar
    # can never be 跌市 for the lines and 升市 for the signals drawn on them.
    regime = [Direction.UP] * n

    # R8/R9 — the prevailing 支持線 per bar, and the bars where the low broke it.
    # 清貨 IS the 死亡線 event: both are 「price breaks down the latest 支持線」,
    # so they are emitted from the same branch and can never disagree.
    stop: list[Optional[float]] = []
    liquidations: list[int] = []

    tops: list[Pivot] = []      # 轉角位 confirmed since the regime opened
    bottoms: list[Pivot] = []

    # R6a and its mirror. A 支持線 needs 創新高 → 拉回 → 突破 of that same 新高;
    # a 阻力線 needs 創新低 → 反彈 → 跌破 of that same 新低. Same shape, opposite
    # sign — 阻力線 is not simply "every 頂 in a 跌市".
    ref_high: Optional[float] = None   # the 新高 awaiting its 突破
    pullback: Optional[Pivot] = None   # last 底 since that 新高
    ref_low: Optional[float] = None    # the 新低 awaiting its 跌破
    ref_low_bar = 0                    # and the bar that made it
    bounce: Optional[Pivot] = None     # last 頂 since that 新低

    for i in range(n):
        for p in by_conf.get(i, []):
            if p.kind is PivotKind.TOP:
                tops.append(p)
                # A later 頂 supersedes: the leg starts at the last one.
                bounce = p
            else:
                bottoms.append(p)
                pullback = p

        # R6 — a bar whose volume at least doubles the previous bar's overrides
        # the pending R6a pullback anchor with its own low, regardless of
        # whether it is itself pivot-shaped. Modeled as a synthetic 底 pivot
        # (confirmed_at == bar — a volume ratio needs no lookahead, unlike a
        # fractal 底) so it flows through the same `pullback` variable R6a
        # already reads. This MUST run after the pivot-dispatch block above:
        # a same-bar confirmed 底 pivot is always dated at some earlier bar
        # (p.bar < i), while a same-bar volume spike is dated exactly at i, so
        # running this second is what makes "whichever happened most
        # recently wins" hold even when both fire on the same bar.
        if (
            volumes is not None
            and i > 0
            and volumes[i] >= VOLUME_SPIKE_MULTIPLE * volumes[i - 1]
        ):
            pullback = Pivot(
                bar=i, price=lows[i], kind=PivotKind.BOTTOM, confirmed_at=i, rule="R6"
            )

        # Whether bar i is a 外擴K — breaks both bar i-1's high and low. Used
        # both by the existing same-bar exceptions below (only reachable when
        # a leg-wide ref_high/ref_low is ALSO broken) and by the unconditional
        # re-anchor branches, which fire on any 外擴K regardless of whether it
        # is the leg's overall extreme.
        outside = i > 0 and highs[i] > highs[i - 1] and lows[i] < lows[i - 1]

        if rising:
            # 升轉跌 — the prevailing support gave way (R9, on the low).
            #
            # `tops` is required, not optional: a regime may not open without the
            # line that opens it, or the ladder silently enters a 跌市 with no
            # 死亡線 and the alternation breaks. With no 頂 to anchor on the break
            # is held over — the level stands, and the next break once a 轉角位
            # has confirmed turns the 段.
            if level is not None and lows[i] < level:
                # The 死亡線 marks the top of the 升市 that just ended. Usually
                # that is the latest confirmed 頂, but the breaking bar can BE
                # the top: an 外擴K makes a new high and breaks support on the
                # same bar, and its 頂 does not confirm until the bar after, so
                # the level would land on a stale, lower high. VOO 2025-10-10
                # ran to 614.12 and crashed to 594.92 in one session; anchoring
                # on the 10-03 頂 put the 死亡線 at 612.95, below the very bar
                # that drew it.
                top = max(tops, key=lambda p: p.bar) if tops else None
                if top is None or highs[i] > top.price:
                    bar_at, price_at = i, highs[i]
                else:
                    bar_at, price_at = top.bar, top.price
                out.append(Line(bar_at, price_at, LineKind.SIWANG, i, i))
                liquidations.append(i)
                level = price_at
                rising = False
                tops, bottoms = [], []
                ref_low, ref_low_bar, bounce = lows[i], i, None
                ref_high, pullback = None, None
                # The 轉段 line confirms here, so this bar is already 跌市.
                regime[i] = Direction.DOWN
                stop.append(None)          # flat the moment the stop is hit
                continue

            # R6a — 支持線 need 創新高 → 拉回 → 突破 of that same 新高.
            if ref_high is None:
                ref_high = highs[i]
            elif highs[i] > ref_high:
                # The 突破 bar can itself hold the low of the leg: it dips
                # under the 拉回 底 and then rallies to complete the breakout in
                # one session, so seating on the 底 puts the 止賺位 above the bar
                # that drew it — a stop already broken. VOO 2024-06-11 seated
                # 475.96 with a low of 475.54.
                #
                # No 外擴K fallback on this side, deliberately. Symmetry would
                # say an 外擴K's own low can seat a 支持線 when no 底 has yet
                # confirmed, and that was measured: it fires on nearly every
                # breakout bar, adds 25-50% more lines, and shifts 388.HK's
                # 2026-07-07 復甦線 and XLF's 2026-07-17 死亡線 off the bars both
                # were verified on. The 阻力線 side stays selective because a
                # 跌破 also demands a new low; a 突破 alone is far more common.
                if pullback is None:
                    # No 底 has confirmed, so the bar must speak for itself: an
                    # 外擴K that closed UP. 外擴 gives it a genuine dip below the
                    # previous bar; 陽燭 says buyers took the bar back, which is
                    # what makes that dip a support rather than the start of a
                    # decline.
                    #
                    # R1 ignores 陰陽 when classifying bars, and rightly — but
                    # this is not classification, it is deciding whether one
                    # bar's low can carry the 止賺位. XLF 2026-07-17 is exactly
                    # the case: 外擴K making a new high, but 陰燭 (56.43 → 56.26),
                    # and its low of 56.12 broke two sessions later. Seating
                    # there moved the 07-23 死亡線 to 07-20.
                    bar_at = None
                    if outside_bar_bullish and outside and body_ok(i, True):
                        bar_at, price_at = i, lows[i]
                elif lows[i] < pullback.price:
                    bar_at, price_at = i, lows[i]
                else:
                    bar_at, price_at = pullback.bar, pullback.price
                if bar_at is not None:
                    add_line(bar_at, price_at, LineKind.SUPPORT, i)
                    level = price_at
                    pullback = None
                ref_high = highs[i]
            elif outside_bar_bullish and outside:
                # 外擴K re-anchors ref_high/pullback to its own extremes even
                # when it is NOT the leg's overall highest point yet. Without
                # this, ref_high only ratchets up on a genuine new leg-high,
                # so a 外擴K that undercuts the running high never gets its
                # own low registered as anything — a later break of THIS
                # bar's own low is invisible until some much earlier, taller
                # reference high is retaken, which may never happen. Mirrors
                # the 阻力線-side branch below; gated on the same toggle as
                # the existing same-bar exception above, and reachable only
                # when that one is NOT (this bar did not make a new leg-high).
                bar_at = None
                if body_ok(i, True):
                    bar_at, price_at = i, lows[i]
                if bar_at is not None:
                    add_line(bar_at, price_at, LineKind.SUPPORT, i)
                    level = price_at
                pullback = None
                ref_high = highs[i]
        else:
            # 跌轉升 — price reclaimed the prevailing overhead level. `bottoms`
            # is required for the same reason `tops` is above.
            if level is not None and highs[i] > level:
                # Mirror of the 死亡線 rule above. It also guarantees the 復甦線
                # is never seated ABOVE the bar that draws it, which used to
                # create a 止賺位 the opening bar had already traded through.
                low = max(bottoms, key=lambda p: p.bar) if bottoms else None
                if low is None or lows[i] < low.price:
                    bar_at, price_at = i, lows[i]
                else:
                    bar_at, price_at = low.bar, low.price
                out.append(Line(bar_at, price_at, LineKind.FUSU, i, i))
                level = price_at
                rising = True
                tops, bottoms = [], []
                ref_high, pullback = highs[i], None
                ref_low, bounce = None, None
                # Set explicitly rather than left to the default, so the two
                # flip paths read the same way.
                regime[i] = Direction.UP
                stop.append(level)
                continue

            # The mirror of R6a — 創新低 → 反彈 → 跌破 that same 新低. The 阻力線
            # sits on the 反彈 頂, and only the new lower low activates it.
            #
            # Drawing at every confirmed 頂 instead put a 阻力線 on 388.HK's
            # 2026-07-06 高 of 379.6, which was never followed by a lower low —
            # price took it out two days later and turned the 段 升.
            if ref_low is None:
                ref_low, ref_low_bar = lows[i], i
            elif lows[i] < ref_low:
                # Mirror: the 跌破 bar can hold the high of the 反彈, and when no
                # 頂 has confirmed yet it is the only high there is. XLF
                # 2025-11-20 is an 外擴K that rallied to 51.76 and made the new
                # low of 50.47 in one session, four bars into the 跌市 — its own
                # 頂 confirms a bar too late to be reachable here.
                if bounce is not None and highs[i] > bounce.price:
                    bar_at, price_at = i, highs[i]
                elif bounce is not None:
                    bar_at, price_at = bounce.bar, bounce.price
                elif (
                    outside_bar_bearish
                    and outside
                    and highs[i] > highs[ref_low_bar]
                    and body_ok(i, False)
                ):
                    bar_at, price_at = i, highs[i]
                else:
                    bar_at = None
                if bar_at is not None:
                    add_line(bar_at, price_at, LineKind.RESISTANCE, i)
                    level = price_at
                    bounce = None
                ref_low, ref_low_bar = lows[i], i
            elif outside_bar_bearish and outside:
                # Mirror of the 支持線-side branch above. 外擴K re-anchors
                # ref_low/ref_low_bar to its own extremes even when it is NOT
                # the leg's overall lowest point yet. 3690.HK 2026-06-05 is a
                # 陽燭 外擴K whose low (77.70) sits well above the leg's true
                # reference low (72.25, from 2026-05-28) — under the old
                # rule, 2026-06-08 breaking 77.70 did nothing at all, since
                # 75.10 never broke 72.25 too, leaving 06-05's high
                # permanently unreachable as a 阻力線.
                bar_at = None
                if body_ok(i, False):
                    bar_at, price_at = i, highs[i]
                if bar_at is not None:
                    add_line(bar_at, price_at, LineKind.RESISTANCE, i)
                    level = price_at
                bounce = None
                ref_low, ref_low_bar = lows[i], i

        # No flip on this bar, so the regime carries forward unchanged.
        regime[i] = Direction.UP if rising else Direction.DOWN
        stop.append(level if rising else None)

    return out, regime, stop, liquidations


# DEAD CODE — commented out, not removed. Superseded by the alternating regime
# machine in _display_lines(); see the comment above its call site in run().
# def _dedupe_lines(lines: Sequence[Line]) -> list[Line]:
#     """One level per (bar, price), preferring the 轉段 kind.
#
#     A 復甦線 relocated onto its retest can land on a bar that already carries an
#     ordinary 支持線 at the same price — 388.HK drew both at 379.0 on 2026-03-30.
#     The 轉段 label is the more specific of the two and wins.
#     """
#     major = {LineKind.FUSU, LineKind.SIWANG}
#     best: dict[tuple[int, float, bool], Line] = {}
#     for ln in lines:
#         key = (ln.bar, round(ln.price, 6), ln.is_support)
#         held = best.get(key)
#         if held is None or (ln.kind in major and held.kind not in major):
#             best[key] = ln
#     return sorted(best.values(), key=lambda ln: (ln.bar, ln.price))


def _display_arrows(
    lines: Sequence[Line],
) -> tuple[list[tuple[int, Direction]], set[tuple[int, float, bool]]]:
    """Arrows are line ACTIVATIONS — exactly one per line, no more, no less.

    A line's own ``confirmed_at`` bar IS the break that activated it (R7:
    lines are born active, ``valid_from == confirmed_at`` — see
    ``_display_lines``), so an arrow fires there directly. This USED to be
    re-derived independently, by watching for a break of the latest
    fractal-pivot 轉角位 with a "spent" flag to avoid re-firing — a different
    signal from what actually confirms a line in ``_display_lines`` (a break
    of ``ref_low``/``ref_high``, which is not always the same bar as the
    latest fractal pivot's own break). The two signals silently diverge
    whenever a leg's reference low/high moves without a fresh fractal pivot
    confirming in between: 3690.HK 2026-06-18 and 06-22 each confirm a
    distinct 阻力線 off two different breaks of an ever-lower reference low,
    but shared one still-unspent 底, so only 06-18 produced an arrow. The
    same root cause previously left 17 of 121 支持線 looking inert before a
    narrow, SUPPORT-only patch in ``run()`` (now removed, along with the
    ``_gated_supports`` call it existed only to feed).
    """
    out = [(ln.confirmed_at, Direction.UP if ln.is_support else Direction.DOWN)
           for ln in lines]
    out.sort()
    activated = {(ln.bar, round(ln.price, 6), ln.is_support) for ln in lines}

    return out, activated


def _buy_signals(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Optional[Sequence[float]],
    lines: Sequence[Line],
    pivots: Sequence[Pivot],
    regime: Sequence[Direction],
    n: int,
) -> list[int]:
    """量增即攻 (R10) — three conditions, two gates.

        1. 今日 High > 昨日 High          break of yesterday's high
        2. Close in the upper half       close >= (high + low) / 2
        3. 今日 Volume > 昨日 Volume       volume expanding

    Gated on 升市 and on price holding above the prevailing 支持線: the same bar
    pattern below support is a bounce inside a decline, not an attack.

    The day count is anchored on the latest 底 轉角位, NOT on the 支持線. The two
    came apart once 支持線 required a 突破 of the 新高: a 拉回 that never reclaims
    the high draws no line, but it still starts a fresh leg that can 量增即攻.
    Anchoring on the line would have kept counting from a low several 拉回 back
    and silently suppressed every such entry.
    """
    if volumes is None:
        return []

    supports = sorted(
        (ln for ln in lines if ln.is_support), key=lambda ln: ln.confirmed_at
    )
    bottoms_by_conf: dict[int, list[Pivot]] = {}
    for p in pivots:
        if p.kind is PivotKind.BOTTOM:
            bottoms_by_conf.setdefault(p.confirmed_at, []).append(p)

    out: list[int] = []
    idx = 0
    support: Optional[Line] = None
    pivot: Optional[Pivot] = None

    for i in range(1, n):
        while idx < len(supports) and supports[idx].confirmed_at <= i:
            support = supports[idx]
            idx += 1
        for p in bottoms_by_conf.get(i, []):
            pivot = p

        if regime[i] is not Direction.UP:
            continue
        if support is None or closes[i] <= support.price:
            continue

        # 「升穿日起計第 1-3 日可量增即攻」 — the swing low IS day 1, so valid
        # offsets are 0, 1, 2. Offset 0 is unreachable in practice: an R3 pivot
        # confirms at bar+1, so the window a backtest can actually trade is
        # days 2-3. That is R12 doing its job, not a bug. 第 4 日起要等拉回.
        if pivot is None or not 0 <= i - pivot.bar < BUY_WINDOW_BARS:
            continue

        broke_high = highs[i] > highs[i - 1]
        upper_half = closes[i] >= (highs[i] + lows[i]) / 2
        vol_up = volumes[i] > volumes[i - 1]

        if broke_high and upper_half and vol_up:
            out.append(i)

    return out


def run(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Optional[Sequence[float]] = None,
    opens: Optional[Sequence[float]] = None,
    *,
    skip_volume_rule: Optional[set[int]] = None,
    outside_bar_bearish: bool = True,
    outside_bar_bullish: bool = False,
    outside_bar_body: bool = True,
    outside_bar_close_fraction: Optional[float] = None,
    trend_mode: str = "regime",
    trend_bars: int = PINE_TREND_BARS,
) -> EngineResult:
    """Run the engine over one symbol's bars.

    ``skip_volume_rule`` holds bar indices where R6's volume test must not fire
    — half-day sessions and the bar after one, where volume is structurally
    ~50% and the yesterday-comparison misfires in both directions.

    ``trend_mode``/``trend_bars`` select how 陰陽燭形態's trend-gated patterns
    read 升市/跌市: ``"regime"`` (default) reuses this engine's own swing-
    structure regime; ``"5day"`` is the Pine script's own literal open-vs-
    open-N-bars-back comparison, with ``trend_bars`` as N. Unrelated to
    every other rule here (R1-R12), which always reads the one regime this
    function computes.
    """
    n = len(highs)
    if not (n == len(lows) == len(closes)):
        raise ValueError("highs/lows/closes length mismatch")
    if volumes is not None and len(volumes) != n:
        raise ValueError("volumes length mismatch")

    skip = skip_volume_rule or set()

    classes = classify_series(highs, lows)
    pivots = find_pivots(highs, lows, classes)
    legs = find_legs(highs, lows)

    result = EngineResult(classes=classes, pivots=pivots, legs=legs)

    # _display_lines is the whole drawing pipeline, and it now yields the regime
    # as well. Everything downstream — the arrows, 量增即攻, 陰陽燭形態 — reads
    # THAT regime, so the system has a single definition of 升市/跌市 rather than
    # one for the lines and another for the signals drawn on top of them.
    #
    # _drop_breached_resistances is gone: under the alternating machine a 阻力線
    # above the prevailing one cannot occur, since the high that exceeded it
    # would already have reclaimed the level and turned the regime 升.
    #
    # _relocate_to_retest moved a 轉段 line forward onto a later retest at the
    # same price, advancing confirmed_at with it. The alternating machine already
    # anchors 復甦線/死亡線 on the LATEST 轉角位 of the regime that just ended, so
    # the relocation is redundant — and it reordered lines past ones that
    # legitimately followed them, which breaks the alternation outright.
    #
    # _dedupe_lines collapsed a 轉段 line and an ordinary level sharing a bar and
    # price. The machine cannot emit both: a regime flip clears the R6a gate
    # state, so the 底 that carries a 復甦線 is never re-seated as a 支持線.
    # R8/R9 come from the same machine as the lines, so 清貨 fires on exactly the
    # bar the 死亡線 confirms — both are 「the low broke the latest 支持線」.
    #
    # They used to come from the causal LineLadder, which seats supports through
    # its own R7 queue and the volume rule and so held a stop that was never
    # drawn. TSLA 2026-03-02 fired 清貨 with a low of 388.25 against a drawn
    # 復甦線 of 387.53 — no break at all — while current_stop read None.
    result.lines, regime, stop, liqs = _display_lines(
        pivots, highs, lows, n, outside_bar_bearish, outside_bar_bullish, opens, closes,
        outside_bar_body, outside_bar_close_fraction, volumes,
    )
    result.current_stop = stop
    result.liquidations = liqs
    result.breakouts, result.activated_lines = _display_arrows(result.lines)
    result.buy_signals = _buy_signals(
        highs, lows, closes, volumes, result.lines, pivots, regime, n
    )
    if opens is not None:
        result.rally_signals = find_rally_attacks(
            opens, highs, lows, closes, volumes
        )
        result.patterns = sorted(
            find_patterns(
                opens, highs, lows, closes, regime,
                trend_mode=trend_mode, trend_bars=trend_bars,
            )
            + find_two_bar_patterns(
                opens, highs, lows, closes, regime,
                trend_mode=trend_mode, trend_bars=trend_bars, volumes=volumes,
            )
            + find_three_bar_patterns(
                opens, highs, lows, closes, regime,
                trend_mode=trend_mode, trend_bars=trend_bars,
            )
            + find_five_bar_patterns(
                opens, highs, lows, closes, regime,
                trend_mode=trend_mode, trend_bars=trend_bars,
            ),
            key=lambda h: (h.bar, h.pattern.value),
        )
    return result

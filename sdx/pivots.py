"""R3 (外擴K 轉角位) and R4 (內困 run resolution).

R3 — for each bar N classified 外擴K, two INDEPENDENT fractal tests:

    low[N]  < low[N-1]  and low[N]  < low[N+1]   ->  底 (pivot low)
    high[N] > high[N-1] and high[N] > high[N+1]  ->  頂 (pivot high)

A bar may emit 0, 1 or 2 pivots. The 前一枝 clause is automatically satisfied for
any 外擴K (外擴 is *defined* as breaking both sides of bar N-1), so the operative
test is against bar N+1 — which is why ``confirmed_at = N + 1``.

The older formulation (升緊出外擴陽燭 → 底; 跌緊出外擴陰燭 → 頂) is a special case
of this rule, not a separate rule. Dropping its 升緊/跌緊 precondition is what
removes the circular dependency between pivots and legs.

R4 — when a 內困K run resolves at bar M, with R the last bar before the run:

    M is 下移K  ->  pivot high at high[R]
    M is 上移K  ->  pivot low  at low[R]

The pivot sits at R, not inside the run.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .classify import InsideRun, inside_runs
from .types import BarClass, Pivot, PivotKind


def find_fractal_pivots(
    highs: Sequence[float],
    lows: Sequence[float],
) -> list[Pivot]:
    """R3 — the 3-bar fractal test applied to EVERY bar.

    Scoping 轉角位 to 外擴K alone misses the ordinary V-turn: a 下移K followed by
    a 上移K is the commonest shape of a 谷, and on 388.HK that omitted 75 bottoms
    and 67 tops — more turning points than it found. The reference chart
    (388.HK, Mar-Aug 2026) draws a level at each of them, so the test is
    general and 外擴K is simply the case where both halves can fire on one bar.
    """
    out: list[Pivot] = []
    n = len(highs)

    # Strict backward, non-strict forward. A flat extreme spanning two adjacent
    # bars — 388.HK bottomed at 374.2 on both 2026-07-07 and 07-08 — registers
    # as nothing under strict-on-both-sides, yet it is plainly a 谷. This form
    # emits exactly one pivot, on the first bar of the flat.
    for i in range(1, n - 1):
        if lows[i] < lows[i - 1] and lows[i] <= lows[i + 1]:
            out.append(
                Pivot(
                    bar=i,
                    price=lows[i],
                    kind=PivotKind.BOTTOM,
                    confirmed_at=i + 1,
                    rule="R3",
                )
            )
        if highs[i] > highs[i - 1] and highs[i] >= highs[i + 1]:
            out.append(
                Pivot(
                    bar=i,
                    price=highs[i],
                    kind=PivotKind.TOP,
                    confirmed_at=i + 1,
                    rule="R3",
                )
            )

    return out


# DEAD CODE — commented out, not removed. Never called anywhere; the engine
# uses find_fractal_pivots(). Kept only so the two readings stay comparable.
# def find_outside_bar_pivots(
#     highs: Sequence[float],
#     lows: Sequence[float],
#     classes: Sequence[Optional[BarClass]],
# ) -> list[Pivot]:
#     """The narrow reading of R3 — fractal test restricted to 外擴K bars.
#
#     Kept so the two readings stay directly comparable; the engine uses
#     :func:`find_fractal_pivots`.
#     """
#     out: list[Pivot] = []
#     n = len(highs)
#
#     for i in range(1, n - 1):  # needs both i-1 and i+1
#         if classes[i] is not BarClass.OUTSIDE_BAR:
#             continue
#
#         if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
#             out.append(
#                 Pivot(
#                     bar=i,
#                     price=lows[i],
#                     kind=PivotKind.BOTTOM,
#                     confirmed_at=i + 1,
#                     rule="R3",
#                 )
#             )
#
#         if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
#             out.append(
#                 Pivot(
#                     bar=i,
#                     price=highs[i],
#                     kind=PivotKind.TOP,
#                     confirmed_at=i + 1,
#                     rule="R3",
#                 )
#             )
#
#     return out


def find_inside_run_pivots(
    highs: Sequence[float],
    lows: Sequence[float],
    classes: Sequence[Optional[BarClass]],
    runs: Optional[Sequence[InsideRun]] = None,
) -> list[Pivot]:
    """R4 — pivots produced when a 內困K run resolves.

    A run resolving into 外擴K produces nothing here; that bar is a pivot
    candidate in its own right under R3.
    """
    if runs is None:
        runs = inside_runs(classes)

    out: list[Pivot] = []

    for run in runs:
        m = run.resolve_bar
        if m is None:  # series ended mid-run — unresolved, emit nothing
            continue

        r = run.ref_bar
        if r < 1:  # no 前一枝 to test against
            continue

        resolving = classes[m]

        # The 內困 run guarantees containment on the RIGHT of bar R, so the
        # 後一枝 half of the fractal test is automatic. The 前一枝 half is not —
        # bar R must still be a genuine extreme against bar R-1. Without this
        # check, 38% of R4 pivots on 388.HK were not turning points at all
        # (e.g. bar R itself a 下移K being labelled a 頂).
        if resolving is BarClass.DOWN_BAR and highs[r] > highs[r - 1]:
            out.append(
                Pivot(
                    bar=r,
                    price=highs[r],
                    kind=PivotKind.TOP,
                    confirmed_at=m,
                    rule="R4",
                )
            )
        elif resolving is BarClass.UP_BAR and lows[r] < lows[r - 1]:
            out.append(
                Pivot(
                    bar=r,
                    price=lows[r],
                    kind=PivotKind.BOTTOM,
                    confirmed_at=m,
                    rule="R4",
                )
            )

    return out


def find_pivots(
    highs: Sequence[float],
    lows: Sequence[float],
    classes: Sequence[Optional[BarClass]],
) -> list[Pivot]:
    """All 轉角位 (R3 + R4), ordered by when they became knowable.

    Sorted by ``confirmed_at`` rather than ``bar`` because that is the order a
    live run would learn them in, and the order the backtest must consume them.
    """
    pivots = find_fractal_pivots(highs, lows)
    pivots += find_inside_run_pivots(highs, lows, classes)
    pivots.sort(key=lambda p: (p.confirmed_at, p.bar, p.kind.value))

    # R3 and R4 can both fire on the same bar — an 外擴K that also happens to be
    # the reference bar of a following 內困 run. Keep the earliest confirmation,
    # since that is when the pivot first became knowable.
    seen: dict[tuple[int, PivotKind], Pivot] = {}
    for p in pivots:
        key = (p.bar, p.kind)
        if key not in seen:
            seen[key] = p

    return sorted(seen.values(), key=lambda p: (p.confirmed_at, p.bar, p.kind.value))

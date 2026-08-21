"""R5 — 段 determination.

Per [[段與轉向]]: 打破上一段幅度先算段. 跌轉升 = 升穿最近跌段嘅頂, and its mirror
升轉跌 = 跌穿最近升段嘅底.

A 跌段's 頂 is where it began, so 跌轉升 means price retraces the whole of the last
down leg. Legs therefore alternate strictly, and a reversal is detected at the bar
that breaks the range — no lookahead.

This module depends on price action ONLY. It must never import ``pivots``: the
one-way dependency is what keeps R3/R5 from becoming circular (see spec §3).
"""

from __future__ import annotations

from typing import Sequence

from .types import Direction, Leg


def find_legs(highs: Sequence[float], lows: Sequence[float]) -> list[Leg]:
    """Segment the series into alternating 升段/跌段.

    Seeding: the first leg is assumed UP anchored at bar 0. If price breaks
    below ``low[0]`` before making a new high the state machine flips
    immediately, so the seed self-corrects within the first few bars. The
    opening leg is warm-up and should not be trusted — golden vectors skip it.
    """
    n = len(highs)
    if n == 0:
        return []

    legs: list[Leg] = []

    direction = Direction.UP
    start_bar, start_price = 0, lows[0]
    confirmed_at = 0
    extreme_bar, extreme_price = 0, highs[0]

    for i in range(1, n):
        if direction is Direction.UP:
            if highs[i] > extreme_price:
                extreme_bar, extreme_price = i, highs[i]

            # 升轉跌 — price broke below where this up leg started
            if lows[i] < start_price:
                legs.append(
                    Leg(
                        direction=Direction.UP,
                        start_bar=start_bar,
                        start_price=start_price,
                        confirmed_at=confirmed_at,
                        end_bar=extreme_bar,
                        end_price=extreme_price,
                    )
                )
                direction = Direction.DOWN
                start_bar, start_price = extreme_bar, extreme_price
                confirmed_at = i
                extreme_bar, extreme_price = i, lows[i]

        else:  # DOWN
            if lows[i] < extreme_price:
                extreme_bar, extreme_price = i, lows[i]

            # 跌轉升 — price broke above this down leg's 頂
            if highs[i] > start_price:
                legs.append(
                    Leg(
                        direction=Direction.DOWN,
                        start_bar=start_bar,
                        start_price=start_price,
                        confirmed_at=confirmed_at,
                        end_bar=extreme_bar,
                        end_price=extreme_price,
                    )
                )
                direction = Direction.UP
                start_bar, start_price = extreme_bar, extreme_price
                confirmed_at = i
                extreme_bar, extreme_price = i, highs[i]

    # Trailing leg, still open at the end of the series
    legs.append(
        Leg(
            direction=direction,
            start_bar=start_bar,
            start_price=start_price,
            confirmed_at=confirmed_at,
            end_bar=extreme_bar,
            end_price=extreme_price,
        )
    )

    return legs

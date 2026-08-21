"""R1 (bar classification) and R2 (內困 deferral).

R1 — compare bar N against bar N-1, ignoring 陰陽:

    外擴K   high[N] >  high[N-1]  and  low[N] <  low[N-1]
    上移K   high[N] >  high[N-1]  and  low[N] >= low[N-1]
    下移K   high[N] <= high[N-1]  and  low[N] <  low[N-1]
    內困K   high[N] <= high[N-1]  and  low[N] >= low[N-1]

The four fall out of two independent booleans, which makes them mutually
exclusive and exhaustive by construction. The >/<= and </>= split fixes
behaviour on equal highs or lows, which the source rules leave open.

R2 — the comparison baseline is ALWAYS the immediately preceding bar, including
when that bar is 內困K. A 內困K yields no direction; resolution defers forward to
the next non-內困 bar. 內困 bars never change the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .types import BarClass


def classify_bar(
    high: float, low: float, ref_high: float, ref_low: float
) -> BarClass:
    """R1 — classify one bar against its reference bar."""
    higher_high = high > ref_high
    lower_low = low < ref_low

    if higher_high and lower_low:
        return BarClass.OUTSIDE_BAR
    if higher_high:  # low >= ref_low
        return BarClass.UP_BAR
    if lower_low:  # high <= ref_high
        return BarClass.DOWN_BAR
    return BarClass.INSIDE_BAR


def classify_series(
    highs: Sequence[float], lows: Sequence[float]
) -> list[Optional[BarClass]]:
    """R1 + R2 — classify every bar against its immediate predecessor.

    Bar 0 has no predecessor and is returned as ``None``.

    R2 is expressed by the ``i - 1`` indexing: the baseline never skips over a
    內困 bar. Comparing against the last *non*-內困 bar instead would be a
    different rule and would shift classifications downstream.
    """
    n = len(highs)
    if n != len(lows):
        raise ValueError(f"highs/lows length mismatch: {n} vs {len(lows)}")

    out: list[Optional[BarClass]] = [None] * n
    for i in range(1, n):
        out[i] = classify_bar(highs[i], lows[i], highs[i - 1], lows[i - 1])
    return out


@dataclass(frozen=True)
class InsideRun:
    """A maximal run of consecutive 內困K bars.

    R4 needs, for each run:

    ``ref_bar``      bar R, the last bar *before* the run. Since 內困 bars are by
                     definition contained within their predecessor, R holds the
                     run's extreme in both directions.
    ``resolve_bar``  bar M, the first non-內困 bar after the run. ``None`` if the
                     series ends while still inside the run.
    """

    ref_bar: int
    start: int  # first 內困 bar
    end: int  # last 內困 bar (inclusive)
    resolve_bar: Optional[int]

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def inside_runs(classes: Sequence[Optional[BarClass]]) -> list[InsideRun]:
    """Locate every 內困K run and the bar that resolves it (R2 → R4)."""
    runs: list[InsideRun] = []
    i = 1
    n = len(classes)

    while i < n:
        if classes[i] is not BarClass.INSIDE_BAR:
            i += 1
            continue

        start = i
        while i < n and classes[i] is BarClass.INSIDE_BAR:
            i += 1
        end = i - 1

        runs.append(
            InsideRun(
                ref_bar=start - 1,
                start=start,
                end=end,
                resolve_bar=i if i < n else None,
            )
        )

    return runs

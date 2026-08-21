"""Render an engine run to a standalone HTML chart.

Self-contained by design: the Lightweight Charts UMD bundle is inlined from
``vendor/``, so the output is one file that opens straight in a browser with no
server, no build step, and no network. Regenerate it after every rule change —
visual review is the point (「visual review 係最重要步驟」).

    python -m sdx.viz 0388.HK

Lightweight Charts only *draws*. Every value plotted here was computed in Python
by the engine; the chart has no idea what 生死線 is.
"""

from __future__ import annotations

import json
import webbrowser
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .candles import (
    PINE_TREND_BARS,
    RALLY_VOLUME_MA,
    Pattern,
    find_five_bar_patterns,
    find_patterns,
    find_three_bar_patterns,
    find_two_bar_patterns,
)
from .engine import EngineResult, run
from .indicators import dmi, macd, rsi
from .types import BarClass, Direction

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "vendor" / "lightweight-charts.standalone.production.js"

#: Default history window. 生死線 is a path-dependent state machine, so a longer
#: run is not just more chart — it is the only way to see the ladder survive
#: several 轉段. Shared with ``sdx.serve`` so both entrypoints agree.
DEFAULT_START = "1999-01-01"

#: How far off the bar the 陰陽燭形態 labels sit, as a fraction of price.
#: Markers anchor to a series value, and SeriesMarkersOptions has no offset —
#: so the labels hang on their own invisible series set this far from the high
#: or low. Enlarging the marker instead only adds visual mass.
PATTERN_OFFSET = 0.015

#: Every pattern's identity/grouping, injected into the page so the 畫圖
#: menu can build its 陰陽燭形態 → 單日/雙日/三日/五日 → per-pattern checkbox
#: tree in JS rather than this file hand-writing 28 checkboxes — stays in
#: sync automatically if patterns are ever added or removed. `value` is the
#: same stable per-pattern key `_pattern_markers` puts on each marker.
#:
#: `zh` here is a MENU-only label, not `zh_name` itself — the two diverge
#: exactly where a name is shared across directions (身懷六甲/穿頭破腳/十字胎,
#: see `zh_name`'s own docstring on why) and that pattern has a direction to
#: disambiguate with (`Pattern.direction`): those get `（看好）`/`（看淡）`
#: appended so their two checkboxes read differently, computed generically
#: from whichever names are ACTUALLY duplicated rather than a hardcoded
#: list, so this self-updates if a name collision is ever added or removed.
#: 陀螺 (Spinning Top White/Black) shares a name too but has no direction
#: (`Pattern.direction` is None — see there) — left unsuffixed on purpose;
#: the JS menu collapses same-label patterns within a kind into one shared
#: checkbox (see buildPatternMenu()), which is what actually resolves the
#: duplicate for a direction-neutral pair like this one.
_zh_name_counts = Counter(p.zh_name for p in Pattern)
PATTERN_CATALOG = [
    {
        "value": p.value,
        "zh": (
            f"{p.zh_name}（{p.direction}）"
            if _zh_name_counts[p.zh_name] > 1 and p.direction
            else p.zh_name
        ),
        "kind": p.kind,
    }
    for p in Pattern
]

CLASS_COLOR = {
    BarClass.OUTSIDE_BAR: "#f59e0b",  # 外擴 — pivot candidates under R3
    BarClass.INSIDE_BAR: "#8b5cf6",  # 內困 — 無方向, defers under R2
}


#: Payload keys that depend on the ladder, and so differ between the two
#: 外擴K settings. Everything else — candles, classes, pivots, indicators — is
#: computed before the ladder and is identical either way.
LADDER_KEYS = (
    "levels", "markers", "stop", "signals", "patterns", "patternAnchor",
    "patterns5day", "patternAnchor5day", "stats",
)


#: The four ladders the 外擴K 轉角位 toggle and its two sub-toggles (陰燭外擴K,
#: 陽燭外擴K) switch between. Each sub-toggle independently gates the 阻力線
#: side (陰燭, bearish) or 支持線 side (陽燭, bullish) — both the existing
#: same-bar exception and the re-anchor branch for that side, together. The
#: master toggle off is equivalent to both subs off, so that combination
#: reuses "noPivot" rather than needing its own run. Default (陰燭 on, 陽燭
#: off) matches OUTSIDE_BAR_BEARISH_DEFAULT/OUTSIDE_BAR_BULLISH_DEFAULT in
#: sdx.engine — 陽燭 defaults off because applying it symmetrically produced a
#: false 死亡線 on XLF 2026-06-17 (an ordinary uptrend pullback bar wrongly
#: seating a premature 支持線).
def _ladders(df):
    """Run the engine four ways over one symbol's bars."""
    args = (
        df["high"].tolist(), df["low"].tolist(), df["close"].tolist(),
        df["volume"].tolist(), df["open"].tolist(),
    )
    return (
        run(*args, outside_bar_bearish=True, outside_bar_bullish=False, outside_bar_body=True),
        {
            "bothOn": run(*args, outside_bar_bearish=True, outside_bar_bullish=True),
            "bullishOnly": run(*args, outside_bar_bearish=False, outside_bar_bullish=True),
            "noPivot": run(*args, outside_bar_bearish=False, outside_bar_bullish=False),
        },
    )


def _pattern_markers(df: pd.DataFrame, dates: list, hits: list) -> tuple[list, list]:
    """陰陽燭形態 marker + anchor pairs for one set of pattern hits.

    Position and color follow the source Pine script's own per-pattern
    location=/color= (or, for the patterns ported from the second, larger
    "*All Candlestick Patterns*" script, its label position/color), not a
    bull/bear binary: Hammer and Inverted Hammer sit below the bar like the
    bullish set, but in the same neutral white/circle as Doji. Marker text
    is `Pattern.marker_text` — the Chinese `zh_name`, shortened for
    Hammer/Inverted Hammer ("鎚"/"倒鎚").

    Each marker also carries `pattern` (`.value`, English) as a stable
    per-pattern key — `text`/`kind` alone can't drive the 畫圖 menu's
    individual-pattern toggles, since several patterns intentionally share
    one `zh_name` (十字胎, 陀螺, 身懷六甲, 穿頭破腳) but must still be
    toggleable independently.
    """
    patterns, pattern_anchor = [], []
    for h in hits:
        above = h.pattern.above_bar
        anchor = (
            float(df["high"].iloc[h.bar]) * (1 + PATTERN_OFFSET)
            if above
            else float(df["low"].iloc[h.bar]) * (1 - PATTERN_OFFSET)
        )
        pattern_anchor.append(
            {"time": dates[h.bar], "value": anchor, "kind": h.pattern.kind}
        )
        patterns.append(
            {
                "time": dates[h.bar],
                "position": "aboveBar" if above else "belowBar",
                "color": h.pattern.color,
                "shape": h.pattern.shape,
                "text": h.pattern.marker_text,
                "size": 0.6,
                "kind": h.pattern.kind,
                "pattern": h.pattern.value,
            }
        )
    return patterns, pattern_anchor


def patterns_for_trend_bars(df: pd.DataFrame, dates: list, trend_bars: int) -> dict:
    """陰陽燭形態 recomputed under 5-day trend mode for a given ``trend_bars``.

    Cheap by design: unlike a Trend-mode switch between Regime and 5-day at
    the *default* N (folded into the main payload's ``patterns5day``, no
    extra cost since the ladder already ran), an arbitrary N has to be
    recomputed — but only the pattern finders, not the whole ladder, since
    5-day mode needs no ``regime`` input at all.
    """
    opens = df["open"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    volumes = df["volume"].tolist()
    hits = sorted(
        find_patterns(
            opens, highs, lows, closes,
            trend_mode="5day", trend_bars=trend_bars,
        )
        + find_two_bar_patterns(
            opens, highs, lows, closes,
            trend_mode="5day", trend_bars=trend_bars, volumes=volumes,
        )
        + find_three_bar_patterns(
            opens, highs, lows, closes,
            trend_mode="5day", trend_bars=trend_bars,
        )
        + find_five_bar_patterns(
            opens, highs, lows, closes,
            trend_mode="5day", trend_bars=trend_bars,
        ),
        key=lambda h: (h.bar, h.pattern.value),
    )
    patterns, pattern_anchor = _pattern_markers(df, dates, hits)
    return {"patterns": patterns, "patternAnchor": pattern_anchor}


def _ladder_payload_fields(
    df: pd.DataFrame, dates: list, result: EngineResult, down_arrows: bool
) -> dict:
    """Everything in a payload that depends on an :class:`EngineResult` —
    split out of :func:`build_payload` so bars the ladder engine never ran
    over (a non-daily Webull interval) can skip this block entirely rather
    than needing a fake/empty result threaded through it."""
    classes = [c.value if c else None for c in result.classes]

    # Highlight the two classes that drive the rules, so miscounts are visible.
    class_overlay = [
        {"time": dates[i], "color": CLASS_COLOR[c]}
        for i, c in enumerate(result.classes)
        if c in CLASS_COLOR
    ]

    # Arrows mark BREAKOUTS — a new extreme taking out the previous 轉角位 —
    # not every pivot. Marking all 36 pivots buried the handful of moments that
    # actually activate a line.
    # ↓ 跌穿 arrows are the exact mirror of ↑ 升穿 and are drawn belowBar in red.
    # They are on by default: the stated rule is 「down arrow in downtrend」, and
    # a decline showing red levels but no arrows read as though nothing had
    # broken. --no-down-arrows restores the 升穿-only view of the reference
    # chart, which marks 升穿 alone.
    # Each marker carries the 畫圖 layer that owns it, so the menu can show the
    # entry signals without the 生死線 levels and arrows, or the reverse.
    #   "sdx"  ↑↓ breakout arrows and 清貨 — the 生死線 machinery
    #   "atk"  量增即攻 and 好友反攻 — entry triggers, read on their own
    markers = []
    for bar, direction in result.breakouts:
        up = direction is Direction.UP
        if not up and not down_arrows:
            continue
        markers.append(
            {
                "time": dates[bar],
                # ↑ above the bar, ↓ below it — matching the reference.
                "position": "aboveBar" if up else "belowBar",
                # ↓ is magenta, not red. At #dc2626 it sat in the same hue
                # family as the bearish candle body (#ef5350) and vanished into
                # the bar it was meant to mark — a 跌穿 you have to hunt for is
                # no better than one that was never drawn. ↑ keeps green: the
                # bullish body is teal (#26a69a), far enough apart already.
                #
                # Violet was the other candidate and is worse: #a855f7 sits
                # right on CLASS_COLOR[內困K] (#8b5cf6), so the arrows merge
                # into the bars the moment K線著色 goes on. Magenta's only
                # neighbour is the bearish 形態 label (#f472b6), which is a
                # captioned square on a separate toggle — no confusion in
                # practice.
                "color": "#16a34a" if up else "#ff2d95",
                "shape": "arrowUp" if up else "arrowDown",
                "text": "",
                "layer": "sdx",
            }
        )
    # 量增即攻 (R10) — the entry signal. Above the bar; Lightweight Charts stacks
    # it clear of the ↑ breakout arrows when both land on the same day.
    for i in result.buy_signals:
        markers.append(
            {
                "time": dates[i],
                "position": "aboveBar",
                "color": "#22d3ee",
                "shape": "circle",
                # Dot only — the label repeated on every signal crowds the bars.
                "text": "",
                "layer": "atk",
            }
        )
    # 好友反攻 — PLACEHOLDER, ``rally_signals`` is always empty until the
    # criteria are given. Wired up now so the toggle draws something the day the
    # detector lands, rather than needing the chart touched again.
    for i in result.rally_signals:
        markers.append(
            {
                "time": dates[i],
                # Above the bar, alongside 量增即攻 — both are entry triggers and
                # read together. Below the bar it sat against the 錘頭's long
                # lower shadow and looked like it marked the shadow rather than
                # the bar. Lightweight Charts stacks the two when they coincide.
                "position": "aboveBar",
                "color": "#f59e0b",
                "shape": "circle",
                "text": "",
                "layer": "atk",
            }
        )
    for i in result.liquidations:
        markers.append(
            {
                "time": dates[i],
                "position": "aboveBar",
                "color": "#e5e7eb",
                "shape": "circle",
                # Dot only — see the 圖例 panel for what it means.
                "text": "",
                # Its own layer rather than "sdx": 清貨 is the R9 exit, worth
                # reading on its own against the levels that produced it.
                "layer": "liq",
            }
        )
    markers.sort(key=lambda m: m["time"])

    patterns, pattern_anchor = _pattern_markers(df, dates, result.patterns)

    stop = [
        {"time": dates[i], "value": float(s)}
        for i, s in enumerate(result.current_stop)
        if s is not None
    ]

    # Each level as a horizontal segment spanning the stretch it was actually in
    # force: from the bar it became real to the bar the next level did.
    #
    # Keyed on confirmed_at, NOT on the bar the pivot sits on. Those differ, and
    # ordering by the pivot bar drew levels over stretches where they did not yet
    # exist. VOO 2025-10: the 死亡線 sits on the 10-03 頂 but confirms on 10-10,
    # the bar its 支持線 at 608.39 broke. Ordered by pivot bar it ran 10-03 to
    # 10-07 — entirely before the break — while the 支持線 ran on to 10-16,
    # straight through the 10-10 bar that took it out. So the chart showed a
    # support surviving its own break, no 死亡線 at the break, and a 死亡線 five
    # bars before anything had happened.
    #
    # A segment STARTS on its own 轉角位 and ENDS at the next level's
    # confirmation. Both halves matter and they are not the same index:
    #
    #   start = ln.bar          the 轉角位 the level IS. The line has to touch it;
    #                           starting at confirmation instead left every
    #                           segment hanging in mid-air a bar or more to the
    #                           right of the low it marks.
    #   end   = next confirmed  the level holds until another becomes active,
    #                           less one bar so the segment does not run into the
    #                           candle that ended it. Ending on the next pivot's
    #                           BAR instead cut it short of its own break — VOO's
    #                           608.39 support stopped before the 10-10 crash
    #                           that took it out.
    #
    # Activation decides WHETHER a level is drawn, not where it starts: a 轉角位
    # can never confirm on its own bar (R3 needs bar N+1, and a 支持線 needs the
    # 突破 after that), so an activation-anchored segment can never touch its
    # pivot. Every line reaching here has activated, so all are drawn, solid.
    #
    # Segments may therefore overlap: VOO's 死亡線 starts on the 10-03 頂 while
    # the 支持線 beneath it runs to its 10-10 break — an overhead level forming
    # while support still holds.
    levels = []
    ordered = sorted(result.lines, key=lambda ln: (ln.confirmed_at, ln.bar))
    for k, ln in enumerate(ordered):
        start = ln.bar
        end = (
            ordered[k + 1].confirmed_at - 1
            if k + 1 < len(ordered)
            else len(dates) - 1      # the live level runs to the right edge
        )
        # A same-bar exception (支持線/阻力線 confirmed on its own bar — see
        # sdx.engine.body_ok) can land on the very LAST bar of the whole
        # series, with no following bar to give it width. max() alone could
        # then push end past the end of `dates`; cap it back in range.
        end = min(max(end, start + 1), len(dates) - 1)
        levels.append(
            {
                "kind": ln.kind.value,
                "support": ln.is_support,
                "price": float(ln.price),
                "from": dates[start],
                "to": dates[end],
                # 復甦線/死亡線 mark 段轉向 and are drawn heavier than the
                # ordinary levels between them.
                "major": ln.kind.value in ("復甦線", "死亡線"),
            }
        )

    # The hover panel reads engine state directly rather than reverse-engineering
    # it from how that state happens to be drawn. Markers carry colour, shape and
    # layer but no text, so deriving 訊號 from them would couple the readout to
    # the drawing style; these three fields keep the two independent.
    pivot_rows = [
        {"time": dates[p.bar], "kind": p.kind.value, "price": float(p.price)}
        for p in result.pivots
    ]

    signals: dict[str, list[str]] = {}

    def _signal(bar: int, label: str) -> None:
        signals.setdefault(dates[bar], []).append(label)

    for i in result.buy_signals:
        _signal(i, "量增即攻")
    for i in result.rally_signals:
        _signal(i, "好友反攻")
    for i in result.liquidations:
        _signal(i, "清貨")
    for bar, direction in result.breakouts:
        _signal(bar, "↑ 升穿" if direction is Direction.UP else "↓ 跌穿")
    for h in result.patterns:
        _signal(h.bar, h.pattern.zh_name)

    counts: dict[str, int] = {}
    for c in classes:
        if c:
            counts[c] = counts.get(c, 0) + 1

    return {
        # All four classes, one per bar. classOverlay carries only the two that
        # get colour; the hover panel names whichever class a bar actually has.
        "classes": classes,
        "pivots": pivot_rows,
        "signals": signals,
        "classOverlay": class_overlay,
        "markers": markers,
        "patterns": patterns,
        "patternAnchor": pattern_anchor,
        "stop": stop,
        "levels": levels,
        "stats": {
            "bars": len(df),
            "classes": counts,
            "pivots": len(result.pivots),
            "legs": len(result.legs),
            "lines": len(result.lines),
            "liquidations": len(result.liquidations),
            "buys": len(result.buy_signals),
            "patterns": len(result.patterns),
            "barsWithStop": sum(1 for s in result.current_stop if s is not None),
        },
    }


def build_payload(
    df: pd.DataFrame,
    result: Optional[EngineResult],
    symbol: str,
    down_arrows: bool = True,
    params: Optional[dict] = None,
    alts: Optional[dict] = None,
    *,
    daily: bool = True,
) -> dict:
    """Chart payload for one symbol.

    ``alts`` maps a name to another run of the same bars under a different
    外擴K setting. Their ladder-dependent fields are attached under
    ``payload["alt"][name]`` so the 畫圖 toggles can swap between them without a
    round trip — the rules change what the engine COMPUTES, not merely what is
    drawn, so the browser cannot derive one ladder from another.

    ``result`` is ``None`` for bars the ladder engine never ran over (a
    non-daily Webull interval — see openspec/changes/webull-streaming-data):
    the payload then carries only ``candles``/``volume``/``indicators``, with
    every ``LADDER_KEYS`` field omitted rather than empty-populated. ``daily``
    controls the candle/volume/indicator ``time`` format — a ``YYYY-MM-DD``
    string for daily bars (Lightweight Charts' business-day format, matching
    every existing chart), a UNIX timestamp for intraday bars, which carry a
    time-of-day a date-only string would silently drop.
    """
    params = params or {}
    dates = (
        [d.strftime("%Y-%m-%d") for d in df.index]
        if daily
        else [int(d.timestamp()) for d in df.index]
    )

    candles = [
        {
            "time": dates[i],
            "open": float(df["open"].iloc[i]),
            "high": float(df["high"].iloc[i]),
            "low": float(df["low"].iloc[i]),
            "close": float(df["close"].iloc[i]),
        }
        for i in range(len(df))
    ]

    volume = [
        {
            "time": dates[i],
            "value": float(df["volume"].iloc[i]),
            "color": "#26a69a55"
            if df["close"].iloc[i] >= df["open"].iloc[i]
            else "#ef535055",
        }
        for i in range(len(df))
    ]

    # Sub-pane indicators. Lightweight Charts only draws — every value here is
    # computed in Python and shared with the engine's own maths.
    def ser(s):
        return [
            {"time": dates[i], "value": float(x)}
            for i, x in enumerate(s)
            if pd.notna(x)
        ]

    # MAVOL — defaults to RALLY_VOLUME_MA (50), the same trailing average
    # ``find_rally_attacks`` (好友反攻) checks 大量 against, drawn on the volume
    # pane so that threshold is visible rather than implicit in the signal
    # alone. User-editable via the volume pane's own gear-icon settings (see
    # IND_META.volume in the JS below) — that only ever changes what's drawn,
    # never ``RALLY_VOLUME_MA`` itself, so 好友反攻 keeps firing off the fixed
    # 50-bar average it was calibrated against regardless of what's on screen.
    volume_ma_period = params.get("volume_ma", RALLY_VOLUME_MA)
    volume_ma = ser(df["volume"].rolling(volume_ma_period).mean())

    r = rsi(df["close"], params.get("rsi", 9), params.get("rsi_signal", 6))
    m = macd(
        df["close"],
        params.get("macd_fast", 12),
        params.get("macd_slow", 26),
        params.get("macd_signal", 9),
    )
    dm = dmi(
        df["high"],
        df["low"],
        df["close"],
        params.get("di", 6),
        params.get("adx", 14),
    )

    indicators = {
        "rsi": ser(r.rsi),
        "rsiSignal": ser(r.signal),
        "dif": ser(m.dif),
        "dea": ser(m.dea),
        "hist": [
            {
                "time": d0["time"],
                "value": d0["value"],
                "color": "#26a69a" if d0["value"] >= 0 else "#ef5350",
            }
            for d0 in ser(m.hist)
        ],
        "pdi": ser(dm.pdi),
        "mdi": ser(dm.mdi),
        "adx": ser(dm.adx),
    }

    labels = {
        "rsi": f"RSI({params.get('rsi', 9)}) · SMA({params.get('rsi_signal', 6)})",
        "macd": (
            f"MACD({params.get('macd_fast', 12)},"
            f"{params.get('macd_slow', 26)},{params.get('macd_signal', 9)})"
        ),
        "dmi": f"DMI({params.get('di', 6)},{params.get('adx', 14)})",
        "volume": f"Volume · MA({volume_ma_period})",
    }

    # The numeric settings behind `labels`, for the indicator-settings popover
    # to pre-fill from — labels are prose for reading, params are numbers for
    # editing, and parsing one back out of the other would be needless and
    # fragile.
    params_out = {
        "rsi": {"period": params.get("rsi", 9), "signal": params.get("rsi_signal", 6)},
        "macd": {
            "fast": params.get("macd_fast", 12),
            "slow": params.get("macd_slow", 26),
            "signal": params.get("macd_signal", 9),
        },
        "dmi": {"di": params.get("di", 6), "adx": params.get("adx", 14)},
        "volume": {"period": volume_ma_period},
    }

    payload = {
        "symbol": symbol,
        "candles": candles,
        "volume": volume,
        "volumeMa": volume_ma,
        "indicators": indicators,
        "params": params_out,
        "labels": labels,
    }

    if result is not None:
        payload.update(_ladder_payload_fields(df, dates, result, down_arrows))
        # The Trend-mode dropdown's non-default option (5-day) at its own
        # default `PINE_TREND_BARS`, precomputed so switching TO it needs no
        # round trip — only changing "Trend in Bars" away from that default
        # does (see sdx.serve's /api/patterns endpoint).
        trend5 = patterns_for_trend_bars(df, dates, PINE_TREND_BARS)
        payload["patterns5day"] = trend5["patterns"]
        payload["patternAnchor5day"] = trend5["patternAnchor"]

        if alts:
            payload["alt"] = {
                name: {
                    k: v
                    for k, v in build_payload(df, res, symbol, down_arrows, params).items()
                    if k in LADDER_KEYS
                }
                for name, res in alts.items()
            }

    return payload


_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>生死線</title>
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0e14; color:#d7dce5;
         font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  header { padding:9px 14px; border-bottom:1px solid #1e2430;
           display:flex; gap:10px; align-items:center; position:relative; }
  h1 { font-size:14px; margin:0 6px 0 0; font-weight:600; white-space:nowrap; }
  .brandIcon { width:20px; height:20px; border-radius:4px; flex:none; }
  button { padding:4px 11px; font:inherit; font-size:12px; color:#d7dce5;
           background:#161c26; border:1px solid #2a3342; border-radius:6px;
           cursor:pointer; white-space:nowrap; }
  button:hover { background:#1e2634; border-color:#3a4658; }
  button.on { color:#0b0e14; background:#38bdf8; border-color:#38bdf8; }
  button.add { padding:4px 9px; color:#7d8797; }
  button.add:hover { color:#38bdf8; border-color:#38bdf8; }
  /* Public build: no Webull credentials are shipped, so the data-source
     toggle (yfinance/Webull picker) is hidden — not deleted, just kept out
     of reach — rather than touching the panel's markup/JS. */
  #sourceBtn { display:none !important; }
  button.iconBtn { padding:5px; display:flex; align-items:center; justify-content:center; }
  button.iconBtn svg { width:18px; height:18px; }
  button.iconBtn.busy svg { animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  button.primary { color:#0b0e14; background:#38bdf8; border-color:#38bdf8;
                   font-weight:600; }
  button.primary:hover { background:#7dd3fc; border-color:#7dd3fc; }
  button:disabled { opacity:.5; cursor:default; }
  /* Above the chart's pane separators, which Lightweight Charts puts at
     z-index 49/50 — at 20 the resize handles stayed grabbable through the
     dimmed backdrop. pointer-events on #chart is the belt-and-braces half, in
     case those z-indexes change. */
  .overlay { display:none; position:fixed; inset:0; z-index:100;
             background:rgba(4,7,12,.72); align-items:center;
             justify-content:center; }
  .overlay.open { display:flex; }
  .modal { width:min(440px, calc(100vw - 32px)); padding:20px 22px;
           background:#111721; border:1px solid #2a3342; border-radius:12px;
           box-shadow:0 20px 60px rgba(0,0,0,.6); }
  .modal h2 { margin:0; font-size:15px; font-weight:600; }
  .modal .head { display:flex; align-items:center; justify-content:space-between;
                 margin:0 0 16px; }
  .modal .xclose { padding:2px 6px; font-size:16px; line-height:1;
                    color:#7d8797; background:transparent; border:none; }
  .modal .xclose:hover { color:#e5e7eb; background:transparent; border-color:transparent; }
  .modal label { display:block; margin:12px 0 5px; font-size:11px;
                 color:#7d8797; letter-spacing:.03em; }
  .modal label:first-child { margin-top:0; }
  .modal select, .modal textarea, .modal input[type="number"], .modal input[type="text"] {
    width:100%; box-sizing:border-box; padding:8px 10px; font:inherit;
    color:#d7dce5; background:#0b0e14; border:1px solid #2a3342;
    border-radius:6px; resize:vertical; }
  .modal textarea { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .modal select:focus, .modal textarea:focus, .modal input:focus {
    outline:none; border-color:#38bdf8; }
  #indOverlay .modal { width:320px; animation:indModalIn .16s ease-out; }
  @keyframes indModalIn { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:none; } }
  #indModalBody .modalSection + .modalSection { margin-top:18px; }
  #indModalBody .modalSectionLabel { font-size:11px; color:#7d8797; letter-spacing:.03em;
    text-transform:uppercase; margin-bottom:8px; }
  #indModalBody .modalFieldGrid { display:grid; grid-template-columns:repeat(2, 1fr); gap:12px; }
  #indModalBody .modalField label { display:block; margin:0 0 5px; font-size:11px;
    color:#7d8797; letter-spacing:.03em; }
  #indModalBody input[type="number"] { width:100%; }
  /* One row: (checkbox) (name) (input number) (color) — all inline, not
     stacked. label/.modalRowToggle (the name, with or without a leading
     checkbox) is a fixed-width column so names of different lengths still
     leave inputs/colors starting at a consistent x-position across rows;
     .paramRowInputs (the number field(s) + swatch(es)) follows immediately
     after, sized to its own content. */
  #indModalBody .modalParamRow { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
  /* 13px and #d7dce5 (the app's own primary text color, e.g. .wlName),
     not the inherited 11px/#7d8797 from .modal label — that combination
     read as too faint/small next to the row's own input/color controls.
     Weight left at the inherited normal; size+color were the actual gap,
     not weight. */
  #indModalBody .modalParamRow label { flex:none; width:110px; margin:0; font-size:13px;
    color:#d7dce5; letter-spacing:.03em; white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis; }
  #indModalBody .modalRowToggle { display:flex !important; align-items:center; gap:6px; cursor:pointer;
    flex:none; width:110px; }
  /* Explicit 18px, not the native ~13px default — a bare checkbox next to
     a 28px-tall input and a 26px swatch (below) read as jarringly
     mismatched; still deliberately smaller than input/color, just not by
     as much. */
  #indModalBody .modalRowToggle input { accent-color:#38bdf8; cursor:pointer; margin:0;
    flex:none; width:18px; height:18px; }
  #indModalBody .paramRowInputs { display:flex; align-items:center; gap:6px; flex:none; }
  /* padding trimmed from .modal input[type=number]'s general 8px/10px
     (that rule stays as-is for every other modal in the app) — at the
     general padding this row's number field renders ~32px tall, well
     past the checkbox/color sizes it sits beside in one row. */
  /* 64px, not 54 — at 54 a 2-decimal value (e.g. ADX change's "1.25") ran
     right up against the native up/down spinner, the last digit sitting
     underneath it rather than beside it. */
  #indModalBody .paramRowInputs input[type="number"] { flex:none; width:64px; padding:5px 8px; }
  #indModalBody .paramRowInputs .colorSwatch { width:26px; height:26px; }
  #indModalBody .modalSwatchGrid { display:grid; grid-template-columns:repeat(2, 1fr); gap:14px 12px; }
  #indModalBody .modalSwatchField { display:flex; align-items:center; gap:10px; }
  #indModalBody .colorSwatch { flex:none; width:30px; height:30px; padding:0; box-sizing:border-box;
    border:1px solid #2a3342; border-radius:6px; cursor:pointer; }
  #indModalBody .colorSwatch:hover { border-color:#3a4658; }
  #indModalBody .modalSwatchMeta { display:flex; flex-direction:column; gap:2px; min-width:0; }
  #indModalBody .modalSwatchMeta label { font-size:11px; color:#7d8797; letter-spacing:.03em; }
  #indModalBody .swatchHex { font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:11px; color:#5d606b; text-transform:uppercase; }
  #indModalBody .modalFooter { display:flex; align-items:center; justify-content:space-between;
    margin-top:18px; }
  #indModalBody .modalFooterRight { display:flex; gap:8px; }
  /* Custom color-picker popup — a grayscale row + 7 tint/shade rows per hue,
     plus an opacity slider for swatches that carry one. Appended to <body>
     (not #indModalBody) so it isn't clipped by the modal's own box, and
     positioned via getBoundingClientRect() off whichever swatch opened it. */
  .colorPickerPopup { position:fixed; z-index:110; width:220px; padding:12px;
    background:#181c25; border:1px solid #2a3342; border-radius:10px;
    box-shadow:0 12px 30px rgba(0,0,0,.5); }
  .colorPickerPopup .swatchGrid { display:grid; grid-template-columns:repeat(10, 1fr); gap:4px; }
  .colorPickerPopup .swatchCell { width:100%; aspect-ratio:1; padding:0;
    border-radius:4px; border:2px solid transparent; cursor:pointer; }
  .colorPickerPopup .swatchCell:hover { border-color:#5d606b; }
  .colorPickerPopup .swatchCell.selected { border-color:#38bdf8; }
  .colorPickerPopup .pickerOpacity { margin-top:12px; }
  .colorPickerPopup .pickerOpacity .label { font-size:11px; color:#7d8797;
    letter-spacing:.03em; margin-bottom:6px; }
  .colorPickerPopup .pickerOpacityRow { display:flex; align-items:center; gap:8px; }
  .colorPickerPopup .pickerOpacityRow input[type="range"] { flex:1; accent-color:#38bdf8; }
  .colorPickerPopup .pickerOpacityRow input[type="number"] { width:52px; box-sizing:border-box;
    padding:5px 6px; font:inherit; font-size:12px; color:#d7dce5; background:#0b0e14;
    border:1px solid #2a3342; border-radius:6px; }
  .actions { display:flex; gap:8px; justify-content:flex-end; margin-top:18px; }
  /* Go-to-date calendar: one #gotoDateCal container re-rendered wholesale on
     every view change (day/month/year) and every nav click — simplest model
     given how small the DOM is, matches how render() rebuilds #chart. */
  .calHead { display:flex; align-items:center; justify-content:space-between;
             margin:14px 0 8px; }
  .calNav { padding:2px 8px; font-size:15px; line-height:1; color:#7d8797;
            background:transparent; border:none; }
  .calNav:hover { color:#e5e7eb; background:#1e2634; border-color:transparent; }
  .calLabel { padding:4px 10px; font-size:13px; font-weight:600; color:#d7dce5;
              background:transparent; border:none; border-radius:6px; }
  button.calLabel:hover { background:#1e2634; border-color:transparent; }
  .calWeekdays { display:grid; grid-template-columns:repeat(7, 1fr);
                 margin-bottom:2px; }
  .calWeekdays span { text-align:center; font-size:11px; color:#7d8797; padding:4px 0; }
  .calGrid { display:grid; gap:2px; }
  .calGrid.day { grid-template-columns:repeat(7, 1fr); }
  .calGrid.month { grid-template-columns:repeat(3, 1fr); }
  .calGrid.year { grid-template-columns:repeat(4, 1fr); }
  .calCell { padding:7px 0; font-size:13px; color:#d7dce5; text-align:center;
             background:transparent; border:1px solid transparent; border-radius:6px; }
  .calCell:hover { background:#1e2634; border-color:transparent; }
  .calCell.empty { visibility:hidden; }
  .calCell.today { text-decoration:underline; text-underline-offset:3px; }
  .calCell.future { color:#4b5568; }
  .calCell.selected { background:#38bdf8; color:#0b0e14; font-weight:600; }
  .calCell.selected:hover { background:#7dd3fc; }
  #gotoDateOverlay .modal { width:min(300px, calc(100vw - 32px)); }
  /* Data source flyout — same visual grammar as #drawPanel (.panel), plus a
     source toggle row and an interval <select>. */
  #sourcePanel { width:200px; }
  #sourcePanel .label { font-size:11px; color:#7d8797; letter-spacing:.03em;
                         text-transform:uppercase; margin:0 0 8px; }
  #sourcePanel .label:not(:first-child) { margin-top:12px; }
  /* #yfIntervalGroup/#webullIntervalGroup wrap their own label+row so JS can
     show/hide each as one unit (see syncSourcePanel) — that nesting makes
     the inner label a :first-child of its own wrapper, so the sibling rule
     above no longer reaches it. Put the same gap on the wrapper instead. */
  #sourcePanel #yfIntervalGroup:not(:first-child),
  #sourcePanel #webullIntervalGroup:not(:first-child) { margin-top:12px; }
  /* 畫圖 panel's three groups (生死線/訊號/其他) — same visual grammar as the
     Data Source flyout's section labels above. */
  #drawPanel .label { font-size:11px; color:#7d8797; letter-spacing:.03em;
                       text-transform:uppercase; margin:2px 6px 4px; }
  #drawPanel .hr { display:block; height:1px; margin:10px 6px 8px;
                    background:#2a3342; }
  .srcRow { display:flex; gap:6px; }
  .srcRow button { flex:1; }
  .srcRow button.on { color:#0b0e14; background:#38bdf8; border-color:#38bdf8; }
  #sourcePanel select { width:100%; box-sizing:border-box; padding:6px 8px;
                         font:inherit; color:#d7dce5; background:#0b0e14;
                         border:1px solid #2a3342; border-radius:6px; }
  #sourcePanel select:disabled { opacity:.4; }
  #sourcePanel select:focus { outline:none; border-color:#38bdf8; }
  /* Live-stream badge, appended into #symbolLegend's own innerHTML (see
     symbolLegendHtml) rather than a separate DOM node, since setSymbolLegend
     replaces that innerHTML wholesale on every hover move and render(). */
  .slLive { display:inline-flex; align-items:center; gap:5px; margin:0 8px 0 2px;
            padding:2px 7px 2px 6px; border-radius:10px; font-size:10px;
            font-weight:700; letter-spacing:.04em; vertical-align:middle; }
  .slLive.connecting { color:#f59e0b; background:#2a2210; border:1px solid #4d3d1a; }
  .slLive.live { color:#4ade80; background:#0f2e22; border:1px solid #1a4d38; }
  .slLive .dot { width:6px; height:6px; border-radius:50%; background:currentColor; }
  .slLive.live .dot { animation: livePulse 1.6s infinite; }
  @keyframes livePulse {
    0% { box-shadow:0 0 0 0 rgba(74,222,128,.55); }
    70% { box-shadow:0 0 0 6px rgba(74,222,128,0); }
    100% { box-shadow:0 0 0 0 rgba(74,222,128,0); }
  }
  /* Pinned above the jumped-to bar's high, not tied to the cursor — stays put
     until the next click anywhere clears it (see clearJumpIndicator()). */
  .jumpIndicator { position:absolute; z-index:5; transform:translate(-50%, calc(-100% - 10px));
                   padding:5px 9px; font-size:12px; color:#d7dce5; background:#1c2330;
                   border:1px solid #2a3342; border-radius:6px;
                   box-shadow:0 4px 14px rgba(0,0,0,.5); white-space:nowrap;
                   pointer-events:none; }
  .jumpIndicator::after { content:""; position:absolute; left:50%; top:100%;
                           transform:translateX(-50%); width:0; height:0;
                           border:5px solid transparent; border-top-color:#1c2330; }
  .results { margin-top:12px; font-size:12px; line-height:1.7; }
  .results div::before { margin-right:7px; }
  .results .ok { color:#7dd3fc; }
  .results .ok::before { content:"\\2713"; color:#22c55e; }
  .results .err { color:#7d8797; }
  .results .err::before { content:"\\2715"; color:#ef4444; }
  .right { margin-left:auto; display:flex; gap:8px; }
  /* Panels hang below the header rather than widening it, so the bar keeps a
     fixed height however long the stats line gets. z-index above both #rail
     (6) and .flyout (5): drawPanel drops down from the header's right edge,
     the same top-right region the sidebar occupies, and has to win that
     overlap or 畫圖 becomes unusable whenever a flyout is open. Also above
     the chart's own pane separators (z-index 49/50, see the .overlay
     comment above) — drawPanel sits directly over the chart, so without
     this the RSI/price-pane drag handle showed through and stayed
     grabbable underneath the open menu. */
  .panel { display:none; position:absolute; top:100%; right:14px; z-index:55;
           margin-top:6px; padding:10px 14px; background:#111721;
           border:1px solid #2a3342; border-radius:8px;
           box-shadow:0 8px 24px rgba(0,0,0,.5); white-space:nowrap; }
  .panel.open { display:block; }
  /* 畫圖 specifically (not every .panel — e.g. #sourcePanel is short and
     needs none of this): the per-pattern checkbox tree can add 28+ rows
     when expanded, so cap it to the viewport and scroll internally rather
     than spilling past the window. scrollbar-gutter:stable reserves the
     scrollbar's width whether or not content currently overflows — without
     it, .panel's shrink-to-fit width (white-space:nowrap, no fixed width)
     recalculates every time the scrollbar appears/disappears (e.g.
     expanding/collapsing a 單日/雙日/三日/五日 group crosses the overflow
     threshold), visibly bouncing the whole panel's width on every toggle. */
  #drawPanel { max-height:calc(100vh - 80px); overflow-y:auto; scrollbar-gutter:stable; }
  .panel b { color:#d7dce5; font-weight:600; }
  /* label.opt, not .opt: `.modal label` sets display:block at equal
     specificity and would flatten these rows. */
  label.opt { display:flex; align-items:center; gap:9px; margin:0;
              padding:5px 6px; border-radius:6px; cursor:pointer;
              font-size:13px; color:#d7dce5; letter-spacing:normal;
              white-space:nowrap; }
  label.opt:hover { background:#161c26; }
  /* Roving keyboard focus inside #drawPanel (see drawPanelCheckboxes()) —
     same highlight convention as .wlRow.on. */
  label.opt.kbdFocus { background:#132330; }
  label.opt.sub { margin-left:20px; color:#a9b2c0; font-size:12px; }
  /* 陰陽燭形態's per-pattern leaves, one level deeper than 單日/雙日/三日/五日
     (label.opt.sub) — built dynamically from PATTERNS, see buildPatternMenu(). */
  label.opt.sub2 { margin-left:36px; color:#8b93a3; font-size:11.5px; }
  label.opt input { accent-color:#38bdf8; cursor:pointer; margin:0; }
  /* 單日/雙日/三日/五日's disclosure arrow — a separate control from the
     kind checkbox beside it (label.opt.sub), so expanding/collapsing a
     kind's pattern list never itself checks/unchecks anything. On the
     right, rotating in place on open — same convention as 檢查清單's
     .clCaret, not a left-side arrow with a glyph swap. */
  .patKindRow { display:flex; align-items:center; }
  .patKindRow label.opt.sub { flex:1; min-width:0; }
  .patKindToggle { flex:none; background:none; border:none; color:#7d8797;
    font-size:9px; padding:6px 8px; cursor:pointer;
    transition:transform .15s ease, color .15s ease; }
  .patKindToggle:hover { color:#38bdf8; }
  .patKindToggle[aria-expanded="true"] { transform:rotate(-90deg); color:#38bdf8; }
  .patKindChildren { display:none; }
  /* Trend mode/bars sit under 陰陽燭形態 like the 單日/雙日/三日 checkboxes,
     but aren't a checkbox themselves — same indent, a plain div instead of
     label.opt. */
  .trendRow { display:flex; align-items:center; gap:6px; margin-left:20px;
              padding:5px 6px; color:#a9b2c0; font-size:12px; }
  .trendRow select, .trendRow input[type=number] {
    background:#0b0e14; color:#d7dce5; border:1px solid #2a3342;
    border-radius:4px; font-size:12px; padding:2px 4px; }
  .trendRow input[type=number] { width:48px; }
  .trendRow input[type=number]:disabled { opacity:.5; }
  /* Same shape as .trendRow — a plain settings div, not a checkbox — for the
     外擴K 收市比例 control, plus a small hover-tooltip info icon next to it. */
  .obFractionRow { display:flex; align-items:center; gap:6px; margin-left:20px;
              padding:5px 6px; color:#a9b2c0; font-size:12px; }
  .obFractionRow input[type=number] {
    background:#0b0e14; color:#d7dce5; border:1px solid #2a3342;
    border-radius:4px; font-size:12px; padding:2px 4px; width:52px; }
  .obFractionRow input[type=number]:disabled { opacity:.5; }
  .infoIcon { position:relative; display:inline-flex; align-items:center;
    justify-content:center; width:14px; height:14px; border-radius:50%;
    font:italic 700 10px/1 Georgia, serif; color:#7d8797; border:1px solid #3a4658;
    cursor:default; flex:none; }
  .infoIcon:hover, .infoIcon:focus { color:#38bdf8; border-color:#38bdf8; outline:none; }
  /* Right-anchored, not centered: this icon sits near the panel's own right
     edge, close to the viewport edge too, so a centered tooltip can overflow
     off-screen (measured: 242px wide box centered on the icon pushed its
     right edge 26px past a 1280px viewport). Opening it flush with the
     icon's right edge and growing leftward stays inside both. */
  .infoTip { display:none; position:absolute; bottom:calc(100% + 6px); right:-4px;
    width:220px; padding:8px 10px; border-radius:6px;
    background:#12161f; border:1px solid #2a3242; color:#d7dce5;
    font:normal 400 11px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    white-space:normal; z-index:6;
    box-shadow:0 4px 14px rgba(0,0,0,.4); cursor:auto; }
  .infoIcon:hover .infoTip, .infoIcon:focus .infoTip { display:block; }
  /* 說明 legend: 外擴K/內困K use a square swatch, everything else a circle —
     distinguishes "bar classification color" from "signal/marker color". */
  #keyPanelBody span { display:block; padding:2px 0; }
  .wk { color:#f59e0b; } .nk { color:#8b5cf6; }
  .up { color:#16a34a; } .dn { color:#ff2d95; }
  .atk { color:#22d3ee; } .rly { color:#f59e0b; }
  .liq { color:#e5e7eb; }
  /* Keyboard-shortcuts list, appended below the marker legend in the same
     #keyPanelBody. Uses <kbd>/plain text rather than <span> so it doesn't
     pick up #keyPanelBody span's display:block rule above, which is meant
     for the one-per-line legend swatches, not an inline key+label row. */
  #keyPanelBody h3 { margin:16px 0 6px; font-size:11px; font-weight:600;
                      color:#7d8797; text-transform:uppercase; letter-spacing:.04em; }
  .shortcutRow { display:flex; align-items:center; gap:10px; padding:3px 0;
                 font-size:13px; color:#b2b5be; }
  .shortcutRow kbd { display:inline-flex; align-items:center; justify-content:center;
                      min-width:20px; height:20px; padding:0 6px; border-radius:5px;
                      background:#161c26; border:1px solid #2a3342; color:#d7dce5;
                      font-family:inherit; font-size:12px; }
  /* Five panes; below ~640px the sub-panes collapse to slivers. */
  #chart { height:calc(100vh - 48px); min-height:640px; width:calc(100% - 60px); }
  body.modal-open #chart { pointer-events:none; }
  /* A flyout is 300px on top of the rail's permanent 60px — without this,
     an open flyout would sit on top of the chart as a pure overlay, hiding
     the right price scale and the rightmost candles behind it rather than
     the chart reflowing to stay fully visible next to it (real TradingView
     resizes the chart here, it doesn't cover it). resizeChartToContainer()
     below does the matching chart.resize() once this reflows. */
  #chartWrap.panelOpen #chart { width:calc(100% - 310px); }
  /* The panel is a sibling of #chart, not a child: render() calls
     chart.remove() on every symbol switch and 畫圖 toggle, which tears down the
     container's chart-owned children. */
  #chartWrap { position:relative; }

  /* --- sidebar rail + flyout panels ---------------------------------------
     The rail sits in the body area below the header, a sibling of #chart —
     absolutely positioned inside #chartWrap rather than the header, so it
     never overlaps the header row. Flyout panels are also #chartWrap
     siblings for the same reason #barPanel and .indNameHover already are:
     render() tears down everything #chart owns on every rebuild, and these
     have to survive symbol switches and 畫圖 toggles untouched. */
  #rail { position:absolute; top:0; right:0; width:60px; height:100%;
          display:flex; flex-direction:column; align-items:center;
          padding:10px 0; box-sizing:border-box; border-left:1px solid #1e2430;
          background:#0d1119; z-index:6; }
  .railSpacer { flex:1; }
  .railBtn { width:42px; height:42px; padding:0; margin:2px 0;
             display:flex; align-items:center; justify-content:center;
             background:transparent; border:1px solid transparent;
             border-radius:8px; color:#7d8797; position:relative; }
  .railBtn:hover { background:#161c26; border-color:#2a3342; color:#d7dce5; }
  .railBtn.on { color:#38bdf8; background:#132330; border-color:#1d3a4d; }
  /* Alerts count badge — hidden by default (.show toggled from JS once
     renderAlertsPanel() has a nonzero result), positioned over the icon's
     own top-right corner rather than reserving layout space for it. */
  .railBadge { display:none; position:absolute; top:2px; right:2px;
               min-width:16px; height:16px; padding:0 3px; box-sizing:border-box;
               border-radius:999px; background:#ef5350; color:#fff;
               font-size:10px; font-weight:700; line-height:16px; text-align:center; }
  .railBadge.show { display:block; }
  /* Watchlist/Alerts are TradingView's own solid glyphs (fill, no stroke,
     44x44 canvas rendered at 26px); 檢查清單/說明 are custom-drawn line icons
     (24x24 canvas rendered at 24px) matching the app's line-icon vocabulary.
     Button box is smaller than that canvas size so the icon sits close to
     the button's edge instead of floating in a wide gutter. */
  /* A thin reinforcing stroke on top of the fill, not a bigger canvas — the
     paths are TradingView's own (thin ribbon lines, thin clock hands), and
     scaling the whole glyph up would blow those out of proportion instead
     of just thickening its edges. */
  .railBtn .fillIcon { width:26px; height:26px; fill:currentColor;
                        stroke:currentColor; stroke-width:0.6; }
  /* Below the header icons' 1.6, not above: 檢查清單/說明 are sparse marks
     (small dots, short segments), and next to Watchlist/Alerts' bold
     TradingView glyphs (a solid bookmark, a reinforced ring) even the
     header's own weight read as too heavy for how little each icon draws. */
  .railBtn .strokeIcon { width:24px; height:24px; fill:none; stroke:currentColor;
                          stroke-width:1.3; stroke-linejoin:round; stroke-linecap:round; }
  /* Flyouts open immediately to the rail's left, vertically pinned to the
     header rather than to their own icon — simpler and keeps the panel from
     jumping if the rail's icon set ever changes. top+bottom (not max-height)
     so the panel always fills the whole column, not just what its content
     needs — a two-row watchlist should not leave the rest of the column
     showing chart through a gap. */
  .flyout { display:none; position:absolute; top:0; right:60px; bottom:0; z-index:5;
            width:250px; box-sizing:border-box;
            background:#111721; border-left:1px solid #2a3342;
            box-shadow:-8px 0 24px rgba(0,0,0,.35);
            flex-direction:column; overflow:hidden; }
  .flyout.open { display:flex; }
  .flyoutHead { display:flex; align-items:center; justify-content:space-between;
                padding:12px 14px; border-bottom:1px solid #1e2430; flex:none; }
  .flyoutHead h2 { margin:0; font-size:13px; font-weight:600; }
  .flyoutHeadRight { display:flex; align-items:center; gap:8px; }
  /* SVG instead of a text "+": a font glyph's ink position within its own
     line box depends on font hinting/rendering, which varies by browser/OS
     and isn't reliably fixable with a single hand-tuned nudge (confirmed —
     a translateY value tuned against one browser's rendering visibly
     mis-centered it in another). An SVG shape is centered by construction
     in its viewBox, so there's no per-environment metric to get wrong. */
  .flyoutHead button.add { display:inline-flex; align-items:center;
                            justify-content:center; padding:4px; }
  .addGlyph { width:14px; height:14px; fill:none; stroke:currentColor;
              stroke-width:2; stroke-linecap:round; }
  .flyoutEmpty { padding:24px 14px; color:#7d8797; font-size:12px; text-align:center; }
  /* Filter row: distinct tag values as toggle chips, plus an &/| switch. */
  .wlFilterRow { display:flex; align-items:flex-start; gap:8px; padding:10px 14px;
                 border-bottom:1px solid #1e2430; flex:none; }
  /* Collapsed to one row by default — a watchlist with many distinct tag
     values would otherwise push the symbol list down an unpredictable
     amount. #wlChipsToggle (shown only when there's a second row to reveal,
     see updateChipsToggle()) expands/collapses by swapping this class. */
  .wlChips { display:flex; flex-wrap:wrap; gap:6px; flex:1;
             overflow:hidden; max-height:24px; }
  .wlChips.expanded { max-height:none; }
  .wlChips:empty::after { content:"沒有標籤"; color:#4b5563; font-size:11px; }
  .chip { padding:3px 9px; font-size:11px; border-radius:999px;
          background:#161c26; border:1px solid #2a3342; color:#a9b2c0;
          cursor:pointer; }
  .chip:hover:not(.alertChip) { border-color:#3a4658; }
  .chip.on { background:#38bdf8; border-color:#38bdf8; color:#0b0e14; }
  /* Same glyph/rotate-on-state-change pattern as .clCaret (checklist rows'
     expand caret, ~line 1043): rest points left, expanded rotates -90deg to
     point down. Hidden entirely (via JS) unless the chips actually overflow
     one row, so a short tag list shows no toggle at all. */
  .wlChipsToggle { display:none; flex:none; align-self:flex-start;
                   background:none; border:none; color:#7d8797; font-size:9px;
                   padding:6px 8px; cursor:pointer;
                   transition:transform .15s ease, color .15s ease; }
  .wlChipsToggle.show { display:block; }
  .wlChipsToggle:hover { color:#38bdf8; }
  .wlChipsToggle.expanded { transform:rotate(-90deg); color:#38bdf8; }
  .wlMode, .volMode { display:flex; flex:none; border:1px solid #2a3342; border-radius:6px;
            overflow:hidden; }
  .wlMode button, .volMode button { padding:3px 9px; font-size:11px; border:none; border-radius:0; }
  /* &/| render at very different natural widths ("&" is a wide glyph, "|"
     a thin bar) — fixed width + centered text makes the two-state toggle
     read as one consistent control instead of two mismatched buttons.
     Scoped to #wlMode specifically, not the shared .wlMode class, so it
     doesn't affect the 詳細/簡潔 view toggle (#wlViewToggle), which also
     uses .wlMode but renders icons, not text, and centers those instead
     via the rule below. */
  #wlMode button { width:24px; text-align:center; }
  /* 詳細/簡潔 as icons, not text — title="" on each button carries the
     label for a11y/tooltip instead. Drawn to literally mirror what the
     toggle does rather than an abstract density metaphor: detailed shows
     two lines per row (name + the shorter tag-summary line beneath it,
     matching .wlTop/.wlTags), compact shows single lines only. */
  #wlViewToggle button { display:inline-flex; align-items:center; justify-content:center;
                          padding:4px 7px; }
  #wlViewToggle svg { width:14px; height:14px; fill:none; stroke:currentColor;
                       stroke-width:2; stroke-linecap:round; }
  .wlMode button.on { background:#38bdf8; color:#0b0e14; }
  /* Quieter than .wlMode's on-state: this toggle only reflows a local sub-list,
     it shouldn't compete with primary nav (railBtn.on) or the watchlist filter. */
  .volMode button.on { background:rgba(56,189,248,.16); color:#8fd9fb; }
  /* Smaller than .wlMode's buttons: this row shares the flyout's 250px width
     with a label, so it needs the extra margin. */
  .volMode button { padding:2px 7px; font-size:10.5px; }
  /* 檢查清單 panel — a pre-trade checklist should behave like one: tickable
     rows with a progress count, not a plain bullet dump. */
  /* The governing principle for the whole checklist, not a footnote — an
     accent rule + upright weight instead of the muted italic used for
     secondary/empty-state text (.flyoutEmpty) elsewhere in this file. */
  .checklistNote { margin:0 0 10px; padding:8px 10px; font-size:12px;
                    line-height:1.6; color:#d7dce5;
                    background:rgba(56,189,248,.08);
                    border-left:2px solid #38bdf8; border-radius:0 4px 4px 0; }
  .checklistProgress { display:flex; align-items:baseline; justify-content:flex-end;
                        margin:0 0 6px; padding:0 2px 8px; border-bottom:1px solid #1e2430;
                        font-size:11px; color:#7d8797; letter-spacing:.02em; }
  .checklistProgressCount { font-variant-numeric:tabular-nums; font-size:12px;
                             font-weight:600; color:#d7dce5; }
  .checklistProgressCount.complete { color:#22c55e; }
  .checklistRows { list-style:none; margin:0; padding:0; }
  .clRow { display:flex; flex-wrap:wrap; align-items:center; border-radius:6px; }
  .clRow:hover { background:#161c26; }
  .clCheckArea { display:flex; flex:1; min-width:0; align-items:center; gap:9px;
                 padding:6px 4px; font:inherit; text-align:left; color:inherit;
                 background:none; border:none; cursor:pointer; }
  .clCheck { flex:none; width:15px; height:15px; border:1.5px solid #3a4658;
             border-radius:4px; position:relative; }
  .clCheckArea:hover .clCheck { border-color:#38bdf8; }
  .clRow.checked .clCheck { background:#22c55e; border-color:#22c55e; }
  .clRow.checked .clCheck::after { content:""; position:absolute; left:4px; top:1px;
                                    width:4px; height:8px; transform:rotate(45deg);
                                    border:solid #0b0e14; border-width:0 2px 2px 0; }
  .clLabel { font-size:12.5px; color:#d7dce5; }
  .clRow.checked .clLabel { color:#5b6472; text-decoration:line-through;
                             text-decoration-color:#3a4658; }
  .clCaret { flex:none; background:none; border:none; color:#7d8797; font-size:9px;
             padding:6px 8px; cursor:pointer; transition:transform .15s ease, color .15s ease; }
  .clCaret:hover { color:#38bdf8; }
  /* Glyph points left at rest; -90deg (counter-clockwise) turns it to point
     down on open, same "opens downward" result as a right-pointing glyph
     rotated +90deg would give, just starting from the other side. */
  .clRow[data-open="true"] .clCaret { transform:rotate(-90deg); color:#38bdf8; }
  /* box-sizing:border-box here is load-bearing: flex-basis:100% sizes this to
     the row's full content width, so the indent MUST come from padding (inside
     that width) rather than margin (added on top of it) — margin here overflows
     the flyout's own boundary and gets clipped by its overflow:hidden. */
  .clBody { flex:0 0 100%; box-sizing:border-box; display:none;
            margin:0 0 8px; padding-left:24px; }
  .clRow[data-open="true"] .clBody { display:block; }
  .volQuoteBlock { margin-top:2px; }
  .volQuoteHead { display:flex; align-items:center; justify-content:space-between;
                  margin-bottom:4px; }
  .volQuoteHead span { font-size:11px; color:#8b93a3; }
  .volQuoteList { margin:0; padding:0 0 0 16px; font-size:12.5px; line-height:1.6; color:#a9b2c0; }
  .volQuoteList li { padding:2px 4px 2px 0; }
  .volQuoteList[data-state="collapsed"] { display:none; }
  .volQuoteList[data-state="key"] li:not(.keyLine) { display:none; }
  .volQuoteList .volBold { font-weight:700; }
  .volQuoteList .volMark { background:rgba(56,189,248,.15); border-radius:3px; padding:2px 4px; }
  .indSubList { margin:0; padding:0 0 0 2px; font-size:12.5px; line-height:1.6; color:#a9b2c0; }
  .indSubList ul { margin:2px 0 4px; padding:0 0 0 16px; }
  .indSubList li { padding:2px 0; }
  /* 趨勢 row: a plain check can't capture which way — selecting a state
     both records it and counts the row as reviewed. Colors mirror the
     chart's own candle colors (upColor/downColor, see createChart), so 升/跌
     mean the same thing here as they do on the candles. */
  .clRow.clTrend { padding:6px 4px 6px 28px; justify-content:space-between; }
  .trendMode { display:flex; flex:none; border:1px solid #2a3342; border-radius:6px;
               overflow:hidden; }
  .trendMode button { padding:2px 9px; font-size:11px; font:inherit; border:none;
                       border-radius:0; background:none; color:#7d8797; cursor:pointer; }
  .trendMode button:hover { color:#d7dce5; }
  .trendMode button[data-v="up"].on { background:rgba(38,166,154,.22); color:#26a69a; }
  .trendMode button[data-v="down"].on { background:rgba(239,83,80,.22); color:#ef5350; }
  .trendMode button[data-v="flat"].on { background:rgba(125,135,151,.22); color:#a9b2c0; }
  .clRow.clTrend.checked .clLabel { color:#5b6472; }
  /* Symbol rows */
  .wlList { flex:1; overflow:auto; }
  /* Thin, theme-matched scrollbar — shared by every scrollable list in the
     watchlist UI (the sidebar itself and the keyboard quick-switch results)
     instead of leaving the bulky OS-default scrollbar on each. */
  .wlList, #symbolSwitchResults {
    scrollbar-width:thin; scrollbar-color:#2a3342 transparent; }
  .wlList::-webkit-scrollbar, #symbolSwitchResults::-webkit-scrollbar { width:8px; }
  .wlList::-webkit-scrollbar-track, #symbolSwitchResults::-webkit-scrollbar-track { background:transparent; }
  .wlList::-webkit-scrollbar-thumb, #symbolSwitchResults::-webkit-scrollbar-thumb {
    background:#2a3342; border-radius:4px; }
  .wlList::-webkit-scrollbar-thumb:hover, #symbolSwitchResults::-webkit-scrollbar-thumb:hover {
    background:#3a4557; }
  .wlRow { display:flex; align-items:center; gap:8px; padding:8px 14px;
           border-bottom:1px solid rgba(30,36,48,.6); cursor:pointer;
           position:relative; }
  .wlRow:hover { background:#161c26; }
  .wlRow.on { background:#132330; }
  .wlRow.dragging, .wlSection.dragging { opacity:.4; }
  /* Insertion-point indicator while dragging a row/section to reorder —
     see wireDrag(). */
  .wlDropLine { height:2px; margin:-1px 12px; background:#38bdf8; border-radius:1px; }
  /* 特別關注 ribbon: pinned to the row's left edge, overlapping its own
     padding gutter rather than consuming row width. Hidden at rest when
     inactive — appears on row hover as a click hint, brighter on direct
     hover, fully solid once active regardless of hover (the :not(.on)
     scoping on the hover rules is load-bearing: without it,
     `.wlRow:hover .wlFlag` (3 selectors) would outrank `.wlFlag.on` (2
     selectors) and an active flag would visibly dim while its row is
     hovered). */
  .wlFlag { position:absolute; left:0; top:50%; transform:translateY(-50%);
            width:13px; height:18px; background:#e5b04b;
            clip-path:polygon(0 0, 65% 0, 100% 50%, 65% 100%, 0 100%);
            opacity:0; cursor:pointer; transition:opacity .12s ease; z-index:1; }
  .wlRow:hover .wlFlag:not(.on) { opacity:.35; }
  /* Same specificity as the row-hover rule above (both 4 class-level
     selectors — `:not(.on)` counts its argument's class specificity) so
     source order decides the tie in this rule's favor; a bare
     `.wlFlag:not(.on):hover` (3 class-level selectors) would lose to the
     row-hover rule above whenever the flag itself is hovered, since
     hovering the flag always also satisfies `.wlRow:hover` — confirmed
     live, this was silently stuck at .35 instead of reaching .65. */
  .wlRow .wlFlag:not(.on):hover { opacity:.65; }
  .wlFlag.on { opacity:1; }
  /* margin-left, not the row's own gap (that only applies to .wlRow's flex
     children — .wlFlag is taken out of flow by position:absolute so it
     never participates): without it, .wlFlag's right edge (13px from the
     row's left) sits almost flush against .wlMain's content, which starts
     at the row's own 14px left padding — about 1px of visual gap. */
  .wlMain { flex:1; min-width:0; margin-left:3px; }
  /* center, not baseline: .wlHeld has no text of its own to align a
     baseline against, so baseline alignment synthesizes one from its
     bottom edge and the badge reads as sitting lower than .wlName/.wlChg
     next to it. */
  .wlTop { display:flex; align-items:center; gap:6px; }
  .wlName { font-size:13px; font-weight:600; color:#d7dce5; }
  /* 持有 badge — NCS blue, deliberately distinct from the app's existing
     accent blue (#38bdf8, used for active filter chips/mode toggles) so
     the two don't read as the same status. The "H" is drawn as SVG, not
     text: a font glyph's ink position within its own line box depends on
     font hinting/rendering, which varies by browser/OS — a translateY nudge
     tuned against one browser's rendering (confirmed via zoomed
     screenshots) still visibly mis-centered it in another. An SVG shape is
     centered by construction in its viewBox, so there's no per-environment
     metric to get wrong. */
  .wlHeld { flex:none; box-sizing:border-box; display:inline-flex; align-items:center;
            justify-content:center; width:15px; height:15px;
            border-radius:4px; background:#0087bd; }
  .wlHeldGlyph { width:9px; height:9px; fill:#eaf6fb; }
  .wlChg { flex:none; padding:1px 6px; font-size:11px; font-weight:600; border-radius:999px; }
  .wlChg.pos { color:#26a69a; background:rgba(38,166,154,.14); }
  .wlChg.neg { color:#ef5350; background:rgba(239,83,80,.14); }
  .wlTags { display:block; margin-top:2px; font-size:11px; color:#7d8797;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .wl-compact .wlTags { display:none; }
  /* Alerts panel rows — same base layout as .wlRow (held badge, symbol
     name) but read-only: no drag handles, no edit/delete icons, and
     .alertChips replaces .wlTags as the row's second line. */
  .alertRow { cursor:pointer; }
  /* Ticked off = "I've seen this" — dims the row (chips stay legible enough
     to still glance at) and drops it out of the rail badge's count, without
     removing it from the list entirely. Session-only, not persisted. */
  .alertRow.acked { opacity:.5; }
  .alertAck { flex:none; width:15px; height:15px; margin:0; cursor:pointer;
              accent-color:#38bdf8; }
  .alertChips { display:flex; flex-wrap:wrap; gap:4px; margin-top:7px; }
  /* Status chips, not toggles — same pill base as .chip but no hover/.on
     interaction state. Colors match the marker legend the chart already
     uses for these three signals (atk/rly/liq), and the app's existing
     up/down candle colors for divergence direction. cursor:default alone
     doesn't cancel .chip:hover's border-color change above (see the
     :not(.alertChip) added there) — .alertChip carries the shared .chip
     class for its pill styling but never gets a click handler or any
     other reason to react to hover. */
  .alertChip { cursor:default; }
  .alertChip-atk { color:#22d3ee; border-color:#22d3ee; }
  .alertChip-rly { color:#f59e0b; border-color:#f59e0b; }
  .alertChip-liq { color:#e5e7eb; border-color:#e5e7eb; }
  .alertChip-bull { color:#0b0e14; background:#26a69a; border-color:#26a69a; }
  .alertChip-bear { color:#0b0e14; background:#ef5350; border-color:#ef5350; }
  /* 初步 (provisional) — same hue as its confirmed counterpart but hollow
     and muted, so a tentative signal never reads with the same confidence
     as a confirmed one even before the "·初步" label suffix is read. */
  .alertChip-bull-prov { color:#26a69a; background:transparent;
                          border:1px dashed #26a69a; opacity:.75; }
  .alertChip-bear-prov { color:#ef5350; background:transparent;
                          border:1px dashed #ef5350; opacity:.75; }
  .wlIcons { display:flex; flex:none; gap:2px; }
  /* Hidden until the row/section is hovered — .forceOn overrides this for
     the confirm/cancel pair, which must stay visible even if the pointer
     drifts off the row while a delete is pending (see armPendingDelete()). */
  .wlIcon { flex:none; width:26px; height:26px; padding:0; display:flex;
            align-items:center; justify-content:center; background:transparent;
            border:none; color:#7d8797; border-radius:6px;
            opacity:0; transition:opacity .12s ease; }
  .wlRow:hover .wlIcon, .wlSection:hover .wlIcon, .wlIcon.forceOn { opacity:1; }
  .wlIcon:hover { background:#1e2634; color:#e5e7eb; border-color:transparent; }
  .wlIcon.wlDel:hover { color:#ef4444; }
  .wlIcon.wlConfirm:hover { color:#26a69a; background:#122824; }
  .wlIcon.wlCancel:hover { color:#ef5350; background:#2a1516; }
  .wlIcon svg { width:15px; height:15px; fill:none; stroke:currentColor;
                stroke-width:1.6; stroke-linejoin:round; stroke-linecap:round; }
  /* Section header — a plain grouping line, not a wrapper: tickers above
     or below it are unaffected by its presence except via the collapsed
     flag (see renderWatchlistRows()'s hideUntilNextSection tracking).
     Deliberately thinner than a .wlRow (4px vs 8px vertical padding, and
     its own smaller .wlIcon override below) — it's a divider label, not a
     row of ticker data, so it shouldn't claim the same visual weight. */
  .wlSection { display:flex; align-items:center; gap:6px; padding:4px 14px;
               background:#0d1119; border-top:1px solid #1e2430;
               border-bottom:1px solid #1e2430; cursor:grab; }
  .wlSection + .wlSection, .wlList > .wlSection:first-child { border-top:none; }
  .wlSection .wlIcon { width:20px; height:20px; }
  .wlSection .wlIcon svg { width:12px; height:12px; }
  .wlChevron { flex:none; width:16px; height:16px; padding:0; background:transparent;
               border:none; color:#7d8797; cursor:pointer; display:flex;
               align-items:center; justify-content:center; }
  .wlChevron svg { width:10px; height:10px; fill:none; stroke:currentColor;
                    stroke-width:2; stroke-linecap:round; stroke-linejoin:round;
                    transition:transform .12s ease; }
  .wlSection.collapsed .wlChevron svg { transform:rotate(-90deg); }
  .wlSecName { flex:1; min-width:0; font-size:11px; font-weight:700; letter-spacing:.03em;
               color:#a9b2c0; text-transform:uppercase; overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; }
  .wlSecName input { width:100%; box-sizing:border-box; font:inherit; font-size:11px;
                      font-weight:700; letter-spacing:.03em; text-transform:uppercase;
                      color:#d7dce5; background:#0b0e14; border:1px solid #38bdf8;
                      border-radius:4px; padding:2px 5px; }
  /* Right-click context menu for a ticker/section — same fixed-position,
     cursor-anchored, viewport-clamped popup pattern as .colorPickerPopup
     (see openColorPicker()), just with a menu-item list instead of a
     swatch grid. */
  .ctxMenu { position:fixed; z-index:110; min-width:168px; background:#161c26;
             border:1px solid #2a3342; border-radius:8px; padding:5px;
             box-shadow:0 12px 28px rgba(0,0,0,.5); }
  .ctxItem { display:flex; align-items:center; gap:9px; padding:7px 9px;
             border-radius:6px; font-size:12px; color:#d7dce5; cursor:pointer; }
  .ctxItem:hover { background:#1e2634; }
  .ctxItem.danger:hover { background:#2a1516; color:#ef5350; }
  /* fill:none/stroke:currentColor here (not fill:currentColor) so the
     reused stroke-style EDIT_ICON/TRASH_ICON render correctly; the filled
     SECTION_ICON/TAG_ICON set fill="currentColor" on their own <path>s,
     which — as a declared value on the element itself — wins over this
     inherited fill:none regardless of the low specificity either side of
     that has. */
  .ctxItem svg { width:15px; height:15px; flex:none; fill:none; stroke:currentColor;
                 stroke-width:1.6; stroke-linejoin:round; stroke-linecap:round; }
  /* Keyboard quick-switch (see #symbolSwitchOverlay): reuses .wlRow/.wlMain/
     .wlName/.wlTags styling for result rows, just scoped/scrollable here
     since it isn't inside .wlList's flex parent. Fixed height, not
     max-height: #overlay centers .modal with flex align-items:center, so a
     height that shrinks as filtering narrows the match list re-centers the
     whole modal on every keystroke, reading as the box visibly jumping up
     and down. A constant height keeps the modal's vertical position fixed
     regardless of how many rows currently match. */
  #symbolSwitchResults { height:320px; overflow:auto; margin-top:10px; }
  #symbolSwitchResults .wlRow:last-child { border-bottom:none; }
  /* Edit-tags modal: pill-style chip inputs with per-category autocomplete. */
  .chipField { position:relative; }
  .pills { display:flex; flex-wrap:wrap; gap:6px; padding:7px 8px;
           background:#0b0e14; border:1px solid #2a3342; border-radius:6px; }
  .pills:focus-within { border-color:#38bdf8; }
  .pill { display:flex; align-items:center; gap:5px; padding:2px 4px 2px 9px;
          font-size:12px; background:#161c26; border:1px solid #2a3342;
          border-radius:999px; color:#d7dce5; }
  .pill button { padding:0 4px; font-size:13px; line-height:1; color:#7d8797;
                 background:transparent; border:none; }
  .pill button:hover { color:#ef4444; background:transparent; border-color:transparent; }
  .pillInput { flex:1; min-width:80px; border:none; background:transparent;
               color:#d7dce5; font:inherit; outline:none; padding:2px 2px; }
  .acList { position:absolute; top:100%; left:0; right:0; z-index:2; margin-top:4px;
            max-height:180px; overflow:auto; background:#161c26;
            border:1px solid #2a3342; border-radius:6px; box-shadow:0 8px 24px rgba(0,0,0,.5); }
  .acItem { display:flex; justify-content:space-between; gap:10px; padding:6px 10px;
            font-size:12px; cursor:pointer; }
  .acItem.hi, .acItem:hover { background:#1e2634; }
  .acItem .acCount { color:#7d8797; }
  /* left:14px matches header's own left padding, so the symbol name lines up
     with 生死線 above it. Never needs repositioning on flyout toggle, unlike
     #barPanel — it's anchored from the left, and only #chart's right edge
     moves when a flyout opens. */
  #symbolLegend { position:absolute; top:10px; left:14px; z-index:3;
                  display:flex; align-items:baseline; gap:10px;
                  pointer-events:none; font-variant-numeric:tabular-nums;
                  font-family:-apple-system,"system-ui","Trebuchet MS",Roboto,Ubuntu,sans-serif; }
  #symbolLegend .slName { font-size:15px; font-weight:400; color:rgb(178,181,190);
                           white-space:nowrap; }
  #symbolLegend .slOhlc { font-size:13px; color:#7d8797; white-space:nowrap; }
  /* Midpoint between .slName's rgb(178,181,190) and .slOhlc's own #7d8797 —
     the ticker sits between the company name and the interval it now
     labels, so its color does too, rather than blending into either. */
  #symbolLegend .slOhlc .slSym { color:#989eab; }
  #symbolLegend .slOhlc .v { color:#d7dce5; }
  #symbolLegend .slOhlc .v.pos, #symbolLegend .slOhlc .chg.pos { color:#26a69a; }
  #symbolLegend .slOhlc .v.neg, #symbolLegend .slOhlc .chg.neg { color:#ef5350; }
  /* Same visual grammar as the header dropdowns, pinned inside the chart and
     driven by the crosshair. pointer-events:none so it never swallows a hover
     — the panel sits over the plot area it is reporting on. */
  /* right:76px, not 16px — clears the price axis, which is ~60px wide and was
     hidden behind the panel at the top of the scale. */
  #barPanel { display:none; position:absolute; top:10px; right:76px; z-index:4;
              padding:9px 12px; min-width:200px; font-size:12px;
              background:rgba(17,23,33,.92); border:1px solid #2a3342;
              border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,.5);
              pointer-events:none; }
  #barPanel.on { display:block; }
  /* Range measure tool: a #chartWrap sibling, not #chart's own child, so it
     survives render()'s chart.remove() on every symbol switch/畫圖 toggle —
     same reasoning as #barPanel/.indNameHover above. Positioned in JS via
     pane-rect math (positionMeasurements()), not native layout. inset:0
     with pointer-events:none so the (mostly empty) layer never blocks the
     rail or the chart itself; only the individual line/box/delete-button/
     dot elements inside it opt back into pointer-events. z-index:52, not the
     5 other #chartWrap overlays use — an endpoint whose price sits outside
     pane 0's own visible range (dragged down past the price axis, or just a
     measurement anchored somewhere no longer in view) renders past pane 0's
     bottom edge into a sub-pane's (volume/RSI/MACD/DMI) own screen area, and
     those sub-panes' canvases sit at the pane-separator z-index (49/50, see
     the .overlay comment above) — at 5 those dots/lines were being visually
     AND interactively covered there, silently swallowing their clicks.
     Still below .panel's 55 so an open 畫圖/資料來源 flyout stays on top. */
  #measureLayer { position:absolute; inset:0; z-index:52; pointer-events:none; }
  /* .6 alpha, not .95 — deliberately see-through so the box reads as an
     annotation floating over the candles, not an opaque panel hiding them. */
  /* Default sits above its anchor (an "up"/gaining measurement's box goes
     above the line's upper point); .below flips it under the anchor
     instead, for a "down"/losing measurement's box under the line's lower
     point — set in JS per-measurement in positionMeasurements(). */
  .measureBox { position:absolute; transform:translate(-50%, calc(-100% - 10px));
                pointer-events:auto; cursor:pointer; white-space:nowrap;
                padding:6px 10px; font-size:11px; line-height:1.6;
                background:rgba(17,23,33,.6); border:1px solid rgba(42,51,66,.7);
                border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,.35); }
  .measureBox.below { transform:translate(-50%, 10px); }
  .measureBox .mPct { font-weight:600; }
  .measureBox .mPct.pos { color:#26a69a; }
  .measureBox .mPct.neg { color:#ef5350; }
  .measureBox .mSub { color:#7d8797; }
  /* Icon-only delete button, shown only for the currently-selected (clicked)
     measurement — positioned in JS at that measurement's own box's top-right
     corner, via the box's actual rendered rect rather than a guessed offset. */
  .measureDel { position:absolute; width:24px; height:24px; padding:0;
                display:flex; align-items:center; justify-content:center;
                pointer-events:auto; color:#ef5350; }
  .measureDel:hover { background:#2a1516; border-color:#ef5350; }
  .measureDel svg { width:14px; height:14px; fill:none; stroke:currentColor;
                     stroke-width:1.6; stroke-linejoin:round; stroke-linecap:round; }
  /* Edit-mode endpoint handles — same #measureLayer children as the box/del
     button above, shown only for the selected measurement, draggable to
     move that one endpoint (see attachMeasureDot()). */
  .measureDot { position:absolute; width:10px; height:10px;
                transform:translate(-50%, -50%); border-radius:50%;
                background:#38bdf8; border:2px solid #0b0e14;
                box-shadow:0 0 0 1px rgba(56,189,248,.6);
                pointer-events:auto; cursor:grab; }
  .measureDot:active { cursor:grabbing; }
  #barPanel div { display:flex; justify-content:space-between; gap:22px;
                  padding:1px 0; }
  #barPanel .k { color:#7d8797; }
  /* Tabular figures so the right-aligned column does not jitter bar to bar. */
  #barPanel .v { color:#d7dce5; font-variant-numeric:tabular-nums; }
  #barPanel .hr { display:block; height:1px; padding:0; margin:6px 0;
                  background:#2a3342; }
  /* Same sibling-of-#chart pattern as #barPanel, for the same reason:
     render() tears down everything #chart owns on every rebuild, and a
     param edit deliberately never calls render(). Positioned in JS from
     each pane's own getHTMLElement() rect; sized by ordinary flex layout
     around the live .indName text, not measured/tracked pixel math.

     .indName itself is a REAL DOM element, always visible (not a canvas
     watermark) — required, not stylistic: a Lightweight Charts watermark
     renders on a layer beneath series/marker content regardless of
     creation order or a markers plugin's own zOrder option (confirmed
     live — 頂背馳/牛差離/熊差離 labels visibly covered the RSI/MACD watermark
     text when scrolled near the pane's top-left corner, and setting
     zOrder:'bottom' on the markers plugin made no difference). A plain
     DOM element has no such problem — it always paints above the canvas.
     setLegends() below writes the same base+live-value text here that a
     canvas watermark would have shown, for any pane with this box (Volume/
     RSI/MACD/DMI — addLegend()'s hasDomLabel check picks this path
     automatically for any pane whose indHover box exists).

     Only the GEAR ICON stays hover-only — the box itself is still the
     hover target for revealing it, sized via ordinary flex layout around
     the now-always-visible name text plus the icon's reserved space
     (visibility:hidden preserves layout, so revealing it doesn't reflow
     anything). Plain CSS :hover rather than JS mouseenter/mouseleave
     bookkeeping, so moving from the name onto the icon inside the same
     box can never flicker it closed. */
  .indNameHover { display:flex; position:absolute; z-index:4;
                   align-items:center; gap:4px; padding:2px 4px 2px 6px;
                   border-radius:6px; border:1px solid transparent; }
  .indNameHover .indName { font-size:11px; color:rgba(125,135,151,0.9);
                            white-space:nowrap; }
  .indNameHover .gear { display:flex; align-items:center; justify-content:center;
                         width:22px; height:20px; padding:0; border-radius:4px;
                         background:transparent; border:none; color:#8b95a5;
                         visibility:hidden; }
  .indNameHover:hover { background:#12161f; border-color:#2a3242;
                         box-shadow:0 4px 14px rgba(0,0,0,.45); }
  .indNameHover:hover .gear { visibility:visible; }
  .indNameHover .gear:hover { background:#1e2634; color:#e5e7eb; }
  .indNameHover .gear svg { width:14px; height:14px; fill:none; stroke:currentColor;
                             stroke-width:1.6; stroke-linejoin:round; stroke-linecap:round; }
</style>
<header>
  <img class="brandIcon" alt="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAH/ElEQVR4AXRWS3McVxk9373dM2NLshRbI78NfjuOI0sJeEFBFSlWWUMVLGAVflaWWWSTbbxhYVJQBQUkjgJUJD+lRJrRa6SRrNE8ero555sZWQHS1efe736Pc757u6elgGPXjRs3ytVq9YPp6elPiVWiXa1O5zMz1eLc2Wpx4fzZ4uKFc45LF88VVy6dP4L8ip8/N1OcZa5qWJ8TbUJcn1bJLY1jkjhqgMG53d3dhaIoPmTC+8QFQ142gwUDQgiEIeGiFA3lJKIUB0hjoM0Y/WaG4ADnglZRBnDBzN7P8/zDRqOxIC36/A4a5aDwI9q3CL8NBQUDSQws5gxERhKKpWlEJTVUSoTmNKBEnxpJ2UQMNqhh02YsIhf5Zch/q9/vP5qqTs3JEXQkDH7MxSQxvAsmgjCHCBOSJUngToET7KSSBoxVEoyfSHGyHEETZYqnSfRTigFea2bePI5dZjYZ8vixtAOP/XeM3SaGt8TtdTGJEu4oklxHX0oMJ8oppsZKmDlVxtmpMia9iQSKlSxHQvVowYUphsGVD6bhaMBtPo7fkh6/5AkM3aAwjgrNwGM3BBImzEzZRKWUYLwSUZ2sYO7mNN69VcXZ02U/jUopRcpE3qxhHQl4k9NgAMw04ugKIfwqUHz2yHPMUG7gOgTzhpIQfGclspdLEZNjKR48mMV7v/gpzkxUUE4jxQ0xBrBPqDayNpLIzFxc75XZ0OYM4H7gfIYG2AingoAnAwVn2oEFMEQyijzhOhJZVuDx51/hT5/9Ge1OD3RhFJMdSMwy+llv4tIMXsVQiyZwmieAsps+FD5qMGOBGLgIw2YSNRHMmynyPvp8rJOnz6PIC6RRfvMmAk8rhsLF8T2XNkyUSDnIoJ4bchzZFOPtgmbmhIGONAQkFJyeOYfJ8RLjFI5ApI8hCDQ935gPXiznqPv1Jrli+HUEZgYO0PFHGBh1EnGIUKIJF4PdAWW+kMaT8GfL/MiY3hXNRhYuYTSCBp4iXf9za8OvncWgO9Z4oWoCcm9Czz8woN2JT9C6YHWgEegQ1LQ3EAyDxoBAi2kwM5gMEfsM8B0QhVYFzIwAOMKMI1+eAECk5EMIAZGOYAYh57PvZDlzi8Gbz/w0KBb4iAJzCsJYR7BGJwu/2Npws6RzD4yTmbEAhMFIFC1AzzUG+gV2Hszw7jtv4fKVS8j4Fm7vvMLY2Bhu3b6KM29MwEy5cA7VRSqQhg0GhGAeNzZqZtDFMOgEgwGRCUmM0Cc3jYG/a9AXfDda6zHo9x6ZNzU1gXYGPHu5iouXL6N6+hRuXr+CmTOn/NMsjhgCUoqlnGMSkGjmyxRDYIOgriG8ffMCZm9dxP3bl3H/zhXH/N0fYv7uDzB39xrm7l3Dfc737lzF2/du4Ec/nsM/vnqGV90CzVddrDc72Ng9wN8XnqGb9XHt6iX87Cfv4MH8Hczfu47Zt6471/yb5HvzCuakcfsSZqkn7fD1yzoWlzew5NjE4so20cDS6h6e1l5hae0AS/VDLK138M/lffzx8SrqrTI+W6jhb0ubWN3N8fAvz/HlygH+8Pkq/rrYwKMvvsW/lpt4vtHGi81DPK238HS1iSff7OCJ+Jc38fWLOv79vI7Q3DvEbrOF7eYBtnf3sbW9i42tHWxs7qC+0cBafRsvV2o86jU8eVnD4os1bCp/r4v9LtBs9dDtB0y8UeXzquD5N+tYWdvE8uomVlY3UGN9fX3LObcaTWzt7KHBE9uh3t5+a/AryPMcOV+ofr8AX2yiQJZl/MS20el03O5lFOr20Ov1sb9/gPr6JrrdPnr8FUycmsS3qzXUauv09dDtZczj3O16vWrFl/Gb4RrUAy9+CRE489dRQAshZwdCr9+ncAH2hIxDp5O5WJ+Ndru02UhGW+s+a7odirFBmsj4LnSYI794cv7kcv6CqH+kIy1pewNKkCNnZwWhOaeo+yiS9XrosyGhx5Ppcp1xnTN2cmwcje2G7zhjTFBen3af8SIHcv7h+m9x6YjfG1AnWrA99NWACtU1bREWSiCcmMIqVpN04cSJCg4ODrxOHMpRTDWyPZdcst1PW3mKS+87DShBTp2IbMGTWfQdm2vlqVntchSTb5SvWf6MDUt8tJZPtsAPAd8BWSLUdgiJq2t38wQ06xRyEklgRKD58LCFUppAtmrUkGwJCsonpcflF5f7hnoGFAGGLoafxdGxqAkJqmg0i3y0G5HL3mnsYG2tBtmK67l7DRvX7L5h4xJ3qCPCjH8PgE6gvUUc3WrCE9mlN8LZ10NSEasp7WTi1DjGx0/KhHwS1GmpQeXpVLyWHJ40modqPIFG4K9jwdfHgiqSuGaHxBkXqSChnK934Dc9JpHcub+Eih0XVS2DvAuH64wG8nGzX/KvAj4Z+UazGXvTQkkUF9EIEhEy/ta3trb4COrI+JOTTzk5a9xmndtcU4j75AMXJzFaU+WT0G61PmJoiX6/RcJ2waCvNXiBiEQq8LnqmHv82HT58ZGtOgkLo1MQj9ezVnxmHAkzA+/Fdrv1kd6BThLDrwFrSgj/5xL5cWhnWkvYBSlwXFQ8igtqwoacWgs8jiY1f0O3v4T6kDzmvwHvMfHoJNQiE/ymH4IWA4LBMx0JSUT+41Cug1vFEGYGA5aSxH7Oj9djxXUCmtFqtb7odA5nk2i/h9lD/khqFGjThgPgZA5afGqMgpd2P4QaoQfmA0cz0Nembo2uhzHgA2mMxOnDfwAAAP//0EWiuQAAAAZJREFUAwB5kCjEnBKUiwAAAABJRU5ErkJggg=="/>
  <h1>生死線</h1>
  <div class="right">
    <button id="drawBtn" class="iconBtn" title="畫圖">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5"/>
      </svg>
    </button>
    <button id="measureBtn" class="iconBtn" title="量度 (M)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="8" width="18" height="8" rx="1.5" transform="rotate(-35 12 12)"/>
        <g transform="rotate(-35 12 12)">
          <path d="M7 9.3v1.6"/><path d="M9.4 9.3v2.6"/><path d="M11.8 9.3v1.6"/>
          <path d="M14.2 9.3v2.6"/><path d="M16.6 9.3v1.6"/>
        </g>
      </svg>
    </button>
    <button id="toggle" class="iconBtn" title="只看K線">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
        <line x1="8" y1="3" x2="8" y2="21"/><rect x="6" y="8" width="4" height="7" rx="1" fill="currentColor" stroke="none"/>
        <line x1="16" y1="5" x2="16" y2="19"/><rect x="14" y="10" width="4" height="6" rx="1" fill="currentColor" stroke="none"/>
      </svg>
    </button>
    <button id="gotoDateBtn" class="iconBtn" title="Go to date">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28"><path fill="currentColor" stroke="currentColor" stroke-width="0.6" stroke-linejoin="round" fill-rule="evenodd" d="M11 4h-1v2H7.5A2.5 2.5 0 0 0 5 8.5V13h1v-2h16v8.5c0 .83-.67 1.5-1.5 1.5H14v1h6.5a2.5 2.5 0 0 0 2.5-2.5v-11A2.5 2.5 0 0 0 20.5 6H18V4h-1v2h-6V4Zm6 4V7h-6v1h-1V7H7.5C6.67 7 6 7.67 6 8.5V10h16V8.5c0-.83-.67-1.5-1.5-1.5H18v1h-1Zm-5.15 10.15-3.5-3.5-.7.7L10.29 18H4v1h6.3l-2.65 2.65.7.7 3.5-3.5.36-.35-.36-.35Z"></path></svg>
    </button>
    <button id="sourceBtn" class="iconBtn" title="Data source: yfinance">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="6" width="18" height="12" rx="2"/><path d="M3 14h18"/><circle cx="17" cy="16" r="0.8" fill="currentColor" stroke="none"/>
      </svg>
    </button>
    <button id="refreshBtn" class="iconBtn" title="重新整理（點擊＝目前股票，長按＝整個股票列表）">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 12a9 9 0 0 1 15-6.7L21 8"/>
        <path d="M21 3v5h-5"/>
        <path d="M21 12a9 9 0 0 1-15 6.7L3 16"/>
        <path d="M3 21v-5h5"/>
      </svg>
    </button>
  </div>
  <div class="panel" id="drawPanel">
    <div class="label">生死線</div>
    <label class="opt"><input type="checkbox" id="optSdx" checked> 生死線</label>
    <label class="opt sub"><input type="checkbox" id="optArrows"> 突破箭咀</label>
    <label class="opt"><input type="checkbox" id="optOb"> 外擴K 轉角位</label>
    <label class="opt sub"><input type="checkbox" id="optObBearish" checked> 陰燭 外擴K</label>
    <label class="opt sub"><input type="checkbox" id="optObBullish"> 陽燭 外擴K</label>
    <div class="opt sub obFractionRow">
      <span>收市比例</span>
      <input type="number" id="obFraction" min="0" max="1" step="0.05" value="0.6" title="外擴K 收市比例">
      <span class="infoIcon" tabindex="0">i<span class="infoTip">外擴K同日確認阻力線/支持線嘅收市強度門檻：收市價須達前一日全日波幅嘅此比例（陽燭由前一日低位起計，陰燭由前一日高位起計）。數值愈細愈寬鬆，愈大愈嚴格。預設 0.6。</span></span>
    </div>

    <div class="hr"></div>
    <div class="label">訊號</div>
    <label class="opt"><input type="checkbox" id="optAtk" checked> 量增即攻+好友反攻</label>
    <label class="opt"><input type="checkbox" id="optLiq" checked> 清貨</label>
    <label class="opt"><input type="checkbox" id="optPat"> 陰陽燭形態</label>
    <div id="patMenu"></div>
    <!-- 單日/雙日/三日/五日 kind toggles, and one leaf checkbox per pattern
         under each, are built from PATTERNS by buildPatternMenu() — see
         below. Not hand-written: this tree stays in sync automatically if
         patterns are ever added/removed on the Python side. -->
    <div class="opt sub trendRow">
      <span>Trend</span>
      <select id="trendMode">
        <option value="regime">Regime</option>
        <option value="5day" selected>Days</option>
      </select>
      <input type="number" id="trendBars" min="1" step="1" value="5" title="Trend in Bars" disabled>
    </div>

    <div class="hr"></div>
    <div class="label">其他</div>
    <label class="opt"><input type="checkbox" id="optClass"> K線著色</label>
    <label class="opt"><input type="checkbox" id="optGmma"> GMMA</label>
  </div>
  <div class="panel" id="sourcePanel">
    <div class="label">Data Source</div>
    <div class="srcRow">
      <button type="button" id="srcYf" class="on">yfinance</button>
      <button type="button" id="srcWb">Webull</button>
    </div>
    <div class="label">Price
      <span class="infoIcon" tabindex="0">i<span class="infoTip">Adjusted：還原股價，計入股息同拆股（同 Futu／Webull 一致）。Raw：只還原拆股、唔還原股息（同 TradingView 預設一致）。淨係影響 yfinance（D/M/Y）。</span></span>
    </div>
    <div class="srcRow">
      <button type="button" id="priceRaw" class="on">Raw</button>
      <button type="button" id="priceAdj">Adjusted</button>
    </div>
    <div id="yfIntervalGroup">
      <div class="label">Interval (yfinance)
        <span class="infoIcon" tabindex="0">i<span class="infoTip">獨立於 Webull 嘅 Interval —— M/Y 由 yfinance 日線自己 resample，唔係用 Webull 嘅 M/Y（Webull 嘅 M/Y 淨係有還原股息版本，冇 Raw 選項）。</span></span>
      </div>
      <div class="srcRow">
        <button type="button" id="yfD" class="on">D</button>
        <button type="button" id="yfM">M</button>
        <button type="button" id="yfY">Y</button>
      </div>
    </div>
    <div id="webullIntervalGroup">
      <div class="label">Interval (Webull)</div>
      <select id="srcInterval">
        <option value="5m">5m</option>
        <option value="15m">15m</option>
        <option value="30m">30m</option>
        <option value="1h">1h</option>
        <option value="4h">4h</option>
        <option value="D" selected>D</option>
        <option value="M">M</option>
        <option value="Y">Y</option>
      </select>
    </div>
  </div>
</header>
<div id="chartWrap">
  <div id="chart"></div>
  <div id="symbolLegend"></div>
  <div id="barPanel"></div>
  <div id="measureLayer"></div>
  <div class="indNameHover" id="indHover-volume" data-ind="volume">
    <span class="indName"></span>
    <button class="gear" title="設定"><svg viewBox="0 0 24 24"><path d="M19.79 7.5 12 3 4.21 7.5v9l7.79 4.5 7.79-4.5z"/><circle cx="12" cy="12" r="3"/></svg></button>
  </div>
  <div class="indNameHover" id="indHover-rsi" data-ind="rsi">
    <span class="indName"></span>
    <button class="gear" title="設定"><svg viewBox="0 0 24 24"><path d="M19.79 7.5 12 3 4.21 7.5v9l7.79 4.5 7.79-4.5z"/><circle cx="12" cy="12" r="3"/></svg></button>
  </div>
  <div class="indNameHover" id="indHover-macd" data-ind="macd">
    <span class="indName"></span>
    <button class="gear" title="設定"><svg viewBox="0 0 24 24"><path d="M19.79 7.5 12 3 4.21 7.5v9l7.79 4.5 7.79-4.5z"/><circle cx="12" cy="12" r="3"/></svg></button>
  </div>
  <div class="indNameHover" id="indHover-dmi" data-ind="dmi">
    <span class="indName"></span>
    <button class="gear" title="設定"><svg viewBox="0 0 24 24"><path d="M19.79 7.5 12 3 4.21 7.5v9l7.79 4.5 7.79-4.5z"/><circle cx="12" cy="12" r="3"/></svg></button>
  </div>

  <div id="rail">
    <button class="railBtn" id="railWatchlist" title="Watchlist">
      <svg class="fillIcon" viewBox="0 0 44 44"><path d="M28 16H16v1h12v-1ZM28 20H16v1h12v-1ZM16 24h12v1H16v-1Z"></path><path fill-rule="evenodd" d="m22 30-10 4V12a1 1 0 0 1 1-1h18a1 1 0 0 1 1 1v22l-10-4Zm-9 2.52V12h18v20.52l-9-3.6-9 3.6Z"></path></svg>
    </button>
    <button class="railBtn" id="railAlerts" title="Alerts">
      <svg class="fillIcon" viewBox="0 0 44 44"><path d="M35 14.66 29.16 9l-.7.72 5.84 5.66.7-.72ZM9 14.66 14.84 9l.7.72-5.84 5.66-.7-.72ZM22 16v7h-5v1h6v-8h-1Z"></path><path fill-rule="evenodd" d="M22 33a11 11 0 1 0 0-22 11 11 0 0 0 0 22Zm0-1a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z"></path></svg>
      <span class="railBadge" id="railAlertsBadge"></span>
    </button>
    <button class="railBtn" id="railChecklist" title="檢查清單">
      <svg class="strokeIcon" viewBox="0 0 24 24">
        <circle cx="9" cy="7" r="1"/><path d="M13 7h6"/>
        <circle cx="9" cy="12" r="1"/><path d="M13 12h6"/>
        <path d="M7.5 16.5l1 1 2-2.2"/><path d="M13 17h6"/>
      </svg>
    </button>
    <div class="railSpacer"></div>
    <button class="railBtn" id="railKey" title="說明">
      <svg class="strokeIcon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><line x1="12" y1="10.5" x2="12" y2="16"/><circle cx="12" cy="7.5" r="0.9" fill="currentColor" stroke="none"/></svg>
    </button>
  </div>

  <div class="flyout" id="watchlistFlyout">
    <div class="flyoutHead">
      <h2>股票列表</h2>
      <div class="flyoutHeadRight">
        <div class="wlMode" id="wlViewToggle">
          <button data-mode="detailed" class="on" title="詳細">
            <svg viewBox="0 0 24 24"><line x1="4" y1="6" x2="18" y2="6"/><line x1="4" y1="10" x2="12" y2="10"/><line x1="4" y1="16" x2="18" y2="16"/><line x1="4" y1="20" x2="12" y2="20"/></svg>
          </button>
          <button data-mode="compact" title="簡潔">
            <svg viewBox="0 0 24 24"><line x1="4" y1="6" x2="18" y2="6"/><line x1="4" y1="12" x2="18" y2="12"/><line x1="4" y1="18" x2="18" y2="18"/></svg>
          </button>
        </div>
        <button class="add" id="wlAddBtn" title="加入股票">
          <svg class="addGlyph" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </button>
      </div>
    </div>
    <div class="wlFilterRow">
      <div class="wlChips" id="wlChips"></div>
      <button class="wlChipsToggle" id="wlChipsToggle" aria-label="展開全部標籤">&#9664;</button>
      <div class="wlMode" id="wlMode">
        <button data-mode="&amp;" class="on">&amp;</button><button data-mode="|">|</button>
      </div>
    </div>
    <div class="wlList" id="wlList"></div>
  </div>

  <div class="flyout" id="alertsFlyout">
    <div class="flyoutHead"><h2>Alerts</h2></div>
    <div class="wlList" id="alertsBody"></div>
  </div>

  <div class="flyout" id="checklistFlyout">
    <div class="flyoutHead"><h2>檢查清單</h2></div>
    <div style="padding:10px 14px 14px">
      <p class="checklistNote">「指標係助證，用陰陽燭成交量做決策」</p>
      <div class="checklistProgress">
        <span class="checklistProgressCount" id="checklistProgressCount">0 / 6</span>
      </div>
      <ul class="checklistRows" id="checklistRows">
        <li class="clRow clTrend">
          <span class="clLabel">趨勢</span>
          <div class="trendMode" id="trendMode">
            <button data-v="up">升</button>
            <button data-v="down">跌</button>
            <button data-v="flat">橫</button>
          </div>
        </li>
        <li class="clRow">
          <button class="clCheckArea"><span class="clCheck"></span><span class="clLabel">阻力／支持</span></button>
        </li>
        <li class="clRow">
          <button class="clCheckArea"><span class="clCheck"></span><span class="clLabel">K 線形態</span></button>
        </li>
        <li class="clRow clExpandable" data-open="false">
          <button class="clCheckArea"><span class="clCheck"></span><span class="clLabel">成交量的匹配</span></button>
          <button class="clCaret" aria-label="展開">&#9664;</button>
          <div class="clBody">
            <div class="volQuoteBlock">
              <div class="volQuoteHead">
                <span>量訣18句</span>
                <div class="volMode" id="volMode">
                  <button data-state="collapsed">收起</button>
                  <button data-state="key" class="on">精簡</button>
                  <button data-state="all">全部</button>
                </div>
              </div>
              <ol class="volQuoteList" id="volQuoteList" data-state="key">
                <li class="volBold volMark keyLine">量增即攻是買點</li>
                <li>量增不跌低點現</li>
                <li>量縮勢歇宜離市</li>
                <li class="volMark keyLine">量增反跌是跌市</li>
                <li class="volBold">量增不攻高點現</li>
                <li>量價匹配勢延續</li>
                <li>量能一枝草，勢不穩</li>
                <li>量能突變，勢蹊蹺</li>
                <li class="volBold volMark keyLine">量增狀弱高點現</li>
                <li>量增無低空自減</li>
                <li>量增減速視量縮</li>
                <li>量能平衡勢不變</li>
                <li>量增價行，趨向明</li>
                <li>量小狀強勢必堅</li>
                <li>量縮價強勢堅定</li>
                <li>量縮價弱叛前兆</li>
                <li>量增無高好轉淡</li>
                <li>量增無低空自減</li>
              </ol>
            </div>
            <div class="volQuoteBlock">
              <div class="volQuoteHead">
                <span>價量關係</span>
              </div>
              <ol class="volQuoteList" data-state="all">
                <li><span class="volMark">同一個段之內</span>：睇成交量喺呢一段升／跌入面點變化（例如逐日遞增定遞減）</li>
                <li>同<span class="volMark">前一個段</span>比較：唔理類型，就係「上一段」，可以係相反方向</li>
                <li>同前一個<span class="volMark">同類型</span>嘅段比較：升 vs 升、跌 vs 跌</li>
              </ol>
            </div>
          </div>
        </li>
        <li class="clRow">
          <button class="clCheckArea"><span class="clCheck"></span><span class="clLabel">形態</span></button>
        </li>
        <li class="clRow clExpandable" data-open="false">
          <button class="clCheckArea"><span class="clCheck"></span><span class="clLabel">技術指標</span></button>
          <button class="clCaret" aria-label="展開">&#9664;</button>
          <div class="clBody">
            <ul class="indSubList">
              <li>RSI
                <ul>
                  <li>超賣/超買 (25/75)</li>
                  <li>背馳</li>
                </ul>
              </li>
              <li>MACD
                <ul>
                  <li>&gt; / &lt; 0</li>
                  <li>背離</li>
                  <li>差離</li>
                </ul>
              </li>
              <li>DJI
                <ul>
                  <li>+DI vs -DI</li>
                  <li>ADX 方向</li>
                  <li>ADX 數值 (20/40)</li>
                </ul>
              </li>
            </ul>
          </div>
        </li>
      </ul>
    </div>
  </div>

  <div class="flyout" id="keyFlyout">
    <div class="flyoutHead"><h2>說明</h2></div>
    <div id="keyPanelBody" style="padding:10px 14px 14px;">
      <span class="wk">■ 外擴K</span><span class="nk">■ 內困K</span>
      <span class="up">● ↑ 升穿</span><span class="dn">● ↓ 跌穿</span>
      <span class="atk">● 量增即攻</span><span class="rly">● 好友反攻</span><span class="liq">● 清貨</span>
      <h3>鍵盤快捷鍵</h3>
      <div class="shortcutRow"><kbd>/</kbd>切換股票</div>
      <div class="shortcutRow"><kbd>W</kbd>股票列表</div>
      <div class="shortcutRow"><kbd>A</kbd>Alerts</div>
      <div class="shortcutRow"><kbd>C</kbd>檢查清單</div>
      <div class="shortcutRow"><kbd>D</kbd>畫圖選單</div>
      <div class="shortcutRow"><kbd>M</kbd>量度</div>
      <div class="shortcutRow"><kbd>F</kbd>只看K線</div>
    </div>
  </div>
</div>
<div class="overlay" id="indOverlay">
  <div class="modal">
    <div class="head">
      <h2 id="indModalTitle"></h2>
      <button class="xclose" id="indModalXClose" title="關閉">✕</button>
    </div>
    <div id="indModalBody"></div>
  </div>
</div>
<div class="overlay" id="tagOverlay">
  <div class="modal">
    <div class="head">
      <h2 id="tagModalTitle"></h2>
      <button class="xclose" id="tagModalXClose" title="關閉">✕</button>
    </div>
    <label class="opt" style="margin:0 0 14px;"><input type="checkbox" id="tagHeld"> 持有</label>
    <label>策略</label>
    <div class="chipField">
      <div class="pills" id="pills-strategies"><input class="pillInput" id="input-strategies" placeholder="新增策略…" autocomplete="off"></div>
      <div class="acList" id="ac-strategies" style="display:none"></div>
    </div>
    <label>階段</label>
    <div class="chipField">
      <div class="pills" id="pills-stages"><input class="pillInput" id="input-stages" placeholder="新增階段…" autocomplete="off"></div>
      <div class="acList" id="ac-stages" style="display:none"></div>
    </div>
    <label>形態</label>
    <div class="chipField">
      <div class="pills" id="pills-patterns"><input class="pillInput" id="input-patterns" placeholder="新增形態…" autocomplete="off"></div>
      <div class="acList" id="ac-patterns" style="display:none"></div>
    </div>
    <div class="actions">
      <button id="tagCancel">取消</button>
      <button id="tagSave" class="primary">儲存</button>
    </div>
  </div>
</div>
<div class="overlay" id="gotoDateOverlay">
  <div class="modal">
    <div class="head">
      <h2>Go to</h2>
      <button class="xclose" id="gotoDateXClose" title="Close">✕</button>
    </div>
    <input type="text" id="gotoDateInput" placeholder="YYYY-MM-DD" autocomplete="off">
    <div id="gotoDateCal"></div>
    <div class="actions">
      <button id="gotoDateCancel">Cancel</button>
      <button id="gotoDateSubmit" class="primary">Go to</button>
    </div>
  </div>
</div>
<div class="overlay" id="symbolSwitchOverlay">
  <div class="modal">
    <div class="head">
      <h2>切換股票</h2>
      <button class="xclose" id="symbolSwitchXClose" title="關閉">✕</button>
    </div>
    <input type="text" id="symbolSwitchInput" placeholder="輸入代號或公司名稱…" autocomplete="off">
    <div id="symbolSwitchResults"></div>
  </div>
</div>
<div class="overlay" id="overlay">
  <div class="modal">
    <h2>加入股票</h2>
    <label for="addSyms">代號 — 可用逗號、空格或換行分隔多個</label>
    <textarea id="addSyms" rows="4" placeholder="0700.HK, NVDA&#10;MSFT"></textarea>
    <div class="results" id="addResults"></div>
    <div class="actions">
      <button id="addCancel">取消</button>
      <button id="addGo" class="primary">加入</button>
    </div>
  </div>
</div>
<script>__BUNDLE__</script>
<script>
const ALL = __DATA__;
// Every Pattern's {value, zh, kind} — see PATTERN_CATALOG in sdx/viz.py.
// Builds the 陰陽燭形態 menu's per-pattern checkbox tree; stays in sync
// automatically if patterns are ever added/removed on the Python side.
const PATTERNS = __PATTERNS__;
// `live` is set only by sdx.serve. A file:// page can neither fetch bars nor
// write watchlists.json, so the add/remove affordances stay hidden there.
const LIVE = !!ALL.live;
let subPanesOn = true;
let classColorOn = false;
let drawingsOn = true;
// ↑↓ breakout arrows, a sub-toggle under 生死線: the levels are the point,
// the arrows mark where one activated — off by default like the rest of
// 陰陽燭形態/量增即攻, separable from 生死線 when wanted.
let arrowsOn = false;
let atkOn = true;   // 量增即攻 + 好友反攻
let liqOn = true;    // 清貨 — the R9 exit, on its own layer
let rsiDivOn = true;   // RSI背馳 — overbought/oversold shading is unconditional
let macdDivOn = true;   // MACD背馳 — MACD-line (底/頂背馳) only, own color pair
let macdChaOn = true;   // MACD差離 — histogram (牛/熊差離) only, own color pair
let dmiBgOn = true;   // DMI背景 — the day-over-day ADX-change band (rising+falling)
// Per-line visibility for every other row in each indicator's own settings
// modal (RSI/MACD/DMI's own main line + the "DMI"/"ADX" rows are exempt —
// see IND_META below) — one checkbox per row, each gating exactly the
// chart element(s) that row's own color swatch(es) already control.
let rsiMaOn = true;              // RSI MA (signal) line
let rsiOverboughtOn = true;      // Overbought reference line
let rsiOversoldOn = true;        // Oversold reference line
let macdSignalOn = true;         // DEA line
let macdHistOn = true;           // Histogram bars
let dmiLowerLevelOn = true;      // 有力 reference line
let dmiUpperLevelOn = true;      // 超買 reference line
let volumeMaOn = false;          // Volume MA line
// A lone 外擴K seating a level when no 轉角位 has confirmed — gated by the
// engine's own body test (OUTSIDE_BAR_CLOSE_FRACTION) whenever the relevant
// sub-toggle is on, no separate toggle for that test itself. 陰燭 外擴K
// (阻力線 side) defaults on; 陽燭 外擴K (支持線 side) defaults off — applying
// it symmetrically produced a false 死亡線 on XLF 2026-06-17/25 (an ordinary
// uptrend pullback bar wrongly seating a premature 支持線). This changes
// what the ENGINE computes, so the payload ships all four ladders and this
// picks one — see ladderOf().
let obOn = false;
let obBearishOn = true;
let obBullishOn = false;
// GMMA (Guppy Multiple Moving Average) — 12 EMAs on close, short-term group
// (3/5/8/10/12/15) plus long-term group (30/35/40/45/50/60). Off by default
// like the rest of 其他's opt-in overlays (陰陽燭形態, K線著色) — 12 lines
// compete hard with everything else on the price pane.
let gmmaOn = false;
// The body test's own strictness (sdx.engine.OUTSIDE_BAR_CLOSE_FRACTION),
// user-adjustable via 收市比例. The four precomputed ladders above only
// cover the default fraction, so a non-default value needs its own round
// trip — see fetchCustomLadder()/refreshObCustomIfNeeded(), the same shape
// as the Trend-in-Bars control's trendCustom.
let obCloseFraction = 0.6;
let obCustom = null;   // {symbol, bearish, bullish, fraction, ...ladderFields}
// 陰陽燭形態 defaults OFF: at any real zoom the labels blanket the price
// pane and bury the levels and arrows underneath them. One entry per
// Pattern (keyed by its English `.value`, the stable id PATTERNS/each
// marker carry — see PATTERN_CATALOG in sdx/viz.py) rather than one
// per-kind boolean, so 單日/雙日/三日/五日 and 陰陽燭形態 itself are all
// just aggregates over this — see syncPatParents() below.
const patOn = Object.fromEntries(PATTERNS.map(p => [p.value, false]));
// 5-day (the source Pine script's own literal open-vs-open-N-bars-back
// proxy) is the default; Regime (this app's own swing-structure trend) is
// the alternative — Regime's trend gate reads the *same* bar it's
// classifying, so a bar whose own move is what flips the regime (e.g. a
// reversal bar breaking the last support) can fail its own trend gate and
// suppress an otherwise-valid pattern; 5-day's older-bar comparison doesn't
// have that same-bar self-exclusion.
// D.patterns/patternAnchor is always Regime mode; D.patterns5day/
// patternAnchor5day is 5-day mode at the default PINE_TREND_BARS (5) — both
// ship in every payload, so switching between them needs no round trip.
// Only a non-default trendBars value fetches fresh via /api/patterns.
let trendMode = '5day';
let trendBars = 5;
let trendCustom = null;   // {patterns, patternAnchor} for a non-default trendBars, per symbol
let chart = null;
let candleSeries = null;
let subSeries = {};   // { rsi:{main,signal}, macd:{hist,dif,dea}, dmi:{pdi,mdi,adx} }
let savedRange = null;   // visible logical range carried across render() rebuilds
//: Visible price range, only when the user has manually zoomed the right
//: price scale (autoScale off). While on autoScale, each rebuild refits to
//: whatever is now shown, which is what you want when a toggle changes the
//: displayed levels (e.g. 外擴K 轉角位 swaps in a different ladder) — pinning
//: a stale manual range across that would clip the new content instead.
let savedPriceRange = null;

// --- indicator math (JS port of sdx/indicators.py) --------------------------
// Faithful port, not a reimplementation from formulas — must match the
// server's numbers exactly so re-entering the same period never visibly
// moves a line. `null` stands in for pandas' NaN throughout.

// Wilder/EMA recursive smoothing: alpha·x[i] + (1-alpha)·y[i-1], adjust=False
// semantics. A leading run of nulls (e.g. from diffJS on the first bar) stays
// null until the first real value seeds the recursion; any null AFTER that
// carries the last smoothed value forward unchanged — verified directly
// against pandas' ewm(adjust=False), which behaves the same way.
function ewmJS(values, alpha) {
  const out = new Array(values.length).fill(null);
  let y = null;
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    if (x === null || Number.isNaN(x)) { out[i] = y; continue; }
    y = (y === null) ? x : alpha * x + (1 - alpha) * y;
    out[i] = y;
  }
  return out;
}
const wilderJS = (values, period) => ewmJS(values, 1 / period);
const emaSpanJS = (values, span) => ewmJS(values, 2 / (span + 1));

function diffJS(values) {
  const out = new Array(values.length).fill(null);
  for (let i = 1; i < values.length; i++) out[i] = values[i] - values[i - 1];
  return out;
}

// pandas .rolling(window).mean() with its default min_periods == window: a
// position needs `window` consecutive non-null values behind it, or the
// result is null — one null anywhere in the window nulls the whole mean.
function rollingMeanJS(values, window) {
  const out = new Array(values.length).fill(null);
  for (let i = window - 1; i < values.length; i++) {
    let sum = 0, ok = true;
    for (let k = i - window + 1; k <= i; k++) {
      const v = values[k];
      if (v === null || Number.isNaN(v)) { ok = false; break; }
      sum += v;
    }
    out[i] = ok ? sum / window : null;
  }
  return out;
}

function computeRSI(closes, period, signalPeriod) {
  const delta = diffJS(closes);
  const gain = delta.map(d => d === null ? null : Math.max(d, 0));
  const loss = delta.map(d => d === null ? null : Math.max(-d, 0));
  const avgGain = wilderJS(gain, period);
  const avgLoss = wilderJS(loss, period);
  const rsi = avgGain.map((ag, i) => {
    const al = avgLoss[i];
    if (ag === null || al === null) return null;
    if (al === 0) return 100.0;      // all-gain stretch — RSI 100 by definition
    if (ag === 0) return 0.0;        // all-loss stretch
    return 100 - 100 / (1 + ag / al);
  });
  return { rsi, signal: rollingMeanJS(rsi, signalPeriod) };
}

function computeMACD(closes, fast, slow, signal) {
  const emaFast = emaSpanJS(closes, fast);
  const emaSlow = emaSpanJS(closes, slow);
  const dif = emaFast.map((f, i) => (f === null || emaSlow[i] === null) ? null : f - emaSlow[i]);
  const dea = emaSpanJS(dif, signal);
  // Doubled to match 通達信's convention, not TradingView's raw difference.
  const hist = dif.map((d, i) => (d === null || dea[i] === null) ? null : (d - dea[i]) * 2);
  return { dif, dea, hist };
}

// RSI背馳 — ported from the reference Pine Script's divergence block
// (lookbackLeft/Right = 5, rangeLower/Upper = 5/60, hardcoded there too, not
// user inputs). A one-shot full-array pass, not true bar-by-bar streaming —
// the chart re-renders from a static bar array, so there is no "live bar"
// to stream against; a single forward scan tracking the most recent
// confirmed pivot low/high reproduces the same result.
function computeRsiDivergence(candles, rsiAligned) {
  const LB = 5, RANGE_MIN = 5, RANGE_MAX = 60;
  const n = candles.length;
  const bull = [], bear = [];
  let prevLow = null, prevHigh = null;

  for (let i = LB; i < n - LB; i++) {
    const v = rsiAligned[i];
    if (v === null) continue;
    let isLow = true, isHigh = true;
    for (let k = i - LB; k <= i + LB && (isLow || isHigh); k++) {
      if (k === i) continue;
      const vk = rsiAligned[k];
      if (vk === null) { isLow = false; isHigh = false; break; }
      if (vk < v) isLow = false;
      if (vk > v) isHigh = false;
    }
    if (isLow) {
      const priceLow = candles[i].low;
      if (prevLow) {
        const dist = i - prevLow.idx;
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v > prevLow.rsiVal && priceLow < prevLow.priceLow) {
          bull.push({ p1: { idx: prevLow.idx, value: prevLow.rsiVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevLow = { idx: i, rsiVal: v, priceLow };
    }
    if (isHigh) {
      const priceHigh = candles[i].high;
      if (prevHigh) {
        const dist = i - prevHigh.idx;
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v < prevHigh.rsiVal && priceHigh > prevHigh.priceHigh) {
          bear.push({ p1: { idx: prevHigh.idx, value: prevHigh.rsiVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevHigh = { idx: i, rsiVal: v, priceHigh };
    }
  }
  return { bull, bear };
}

// MACD背馳/差離 — same pivot/range-filter mechanism as computeRsiDivergence
// above, kept as its own function rather than generalizing that one: the
// two real rule differences (MACD-line divergence needs a zero-line
// filter RSI has no equivalent of; MACD-histogram divergence explicitly
// omits one even though it's structurally closest to MACD-line) would
// cost more in indirection to force into one parameterized helper than
// the ~15 duplicated lines save (see this change's design.md Non-Goals).
//
// zeroLineFilter, when true, requires the oscillator to have stayed on
// one side of zero across a FIXED trailing window of `LB+LB+5` (15) bars
// ending at the confirming pivot bar — matching the reference script's
// `highest(osc, lbL+lbR+5)`/`lowest(...)`, a fixed-length rolling window
// evaluated at the current bar, not the (variable) span back to the
// previous pivot.
function computeMacdDivergence(candles, oscAligned, zeroLineFilter) {
  const LB = 5, RANGE_MIN = 5, RANGE_MAX = 60, ZERO_WINDOW = LB + LB + 5;
  const n = candles.length;
  const bull = [], bear = [];
  let prevLow = null, prevHigh = null;

  const staysBelowZero = i => {
    for (let k = Math.max(0, i - ZERO_WINDOW + 1); k <= i; k++) {
      const vk = oscAligned[k];
      if (vk !== null && vk >= 0) return false;
    }
    return true;
  };
  const staysAboveZero = i => {
    for (let k = Math.max(0, i - ZERO_WINDOW + 1); k <= i; k++) {
      const vk = oscAligned[k];
      if (vk !== null && vk <= 0) return false;
    }
    return true;
  };

  for (let i = LB; i < n - LB; i++) {
    const v = oscAligned[i];
    if (v === null) continue;
    let isLow = true, isHigh = true;
    for (let k = i - LB; k <= i + LB && (isLow || isHigh); k++) {
      if (k === i) continue;
      const vk = oscAligned[k];
      if (vk === null) { isLow = false; isHigh = false; break; }
      if (vk < v) isLow = false;
      if (vk > v) isHigh = false;
    }
    if (isLow) {
      const priceLow = candles[i].low;
      if (prevLow) {
        const dist = i - prevLow.idx;
        // Same histogram color (both bars >= 0, or both < 0) — the two
        // pivots being compared don't need the oscillator to have stayed
        // on one side for the whole window between them (it may cross
        // zero and come back), only the two COMPARED bars themselves need
        // to match. Redundant with the zero-line filter for MACD-line
        // divergence (that filter already implies same-sign endpoints),
        // but is the actual fix for histogram divergence (差離), which has
        // no zero-line filter and, without this, was comparing pivots
        // straddling zero — e.g. a red (below-zero) low against a green
        // (above-zero) low, which don't represent the same condition.
        const sameSide = (prevLow.oscVal >= 0) === (v >= 0);
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v > prevLow.oscVal && priceLow < prevLow.priceLow
            && sameSide && (!zeroLineFilter || staysBelowZero(i))) {
          bull.push({ p1: { idx: prevLow.idx, value: prevLow.oscVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevLow = { idx: i, oscVal: v, priceLow };
    }
    if (isHigh) {
      const priceHigh = candles[i].high;
      if (prevHigh) {
        const dist = i - prevHigh.idx;
        const sameSide = (prevHigh.oscVal >= 0) === (v >= 0);
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v < prevHigh.oscVal && priceHigh > prevHigh.priceHigh
            && sameSide && (!zeroLineFilter || staysAboveZero(i))) {
          bear.push({ p1: { idx: prevHigh.idx, value: prevHigh.oscVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevHigh = { idx: i, oscVal: v, priceHigh };
    }
  }
  return { bull, bear };
}

// Alerts-only early-warning variants of the two divergence detectors above.
// Identical pivot/pairing rules — the only change is the lookforward side of
// the pivot window, relaxed from "must have the full LB=5 bars after" down
// to "must have at least K" (0 <= K <= LB), so a fresh pivot can register
// before the confirmed functions above would ever see it. The lookback side
// is untouched: a candidate always still needs the full LB bars before it,
// exactly like the confirmed algorithm. K = LB reproduces the confirmed
// function's output exactly. See docs/superpowers/specs/
// 2026-08-13-watchlist-alerts-design.md for why K=0 (zero lookforward) was
// tried and rejected, and how the per-indicator K values were chosen.
function computeRsiDivergenceProvisional(candles, rsiAligned, K) {
  const LB = 5, RANGE_MIN = 5, RANGE_MAX = 60;
  const n = candles.length;
  const bull = [], bear = [];
  let prevLow = null, prevHigh = null;

  for (let i = LB; i < n - K; i++) {
    const v = rsiAligned[i];
    if (v === null) continue;
    let isLow = true, isHigh = true;
    const kEnd = Math.min(n - 1, i + LB);
    for (let k = i - LB; k <= kEnd && (isLow || isHigh); k++) {
      if (k === i) continue;
      const vk = rsiAligned[k];
      if (vk === null) { isLow = false; isHigh = false; break; }
      if (vk < v) isLow = false;
      if (vk > v) isHigh = false;
    }
    if (isLow) {
      const priceLow = candles[i].low;
      if (prevLow) {
        const dist = i - prevLow.idx;
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v > prevLow.rsiVal && priceLow < prevLow.priceLow) {
          bull.push({ p1: { idx: prevLow.idx, value: prevLow.rsiVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevLow = { idx: i, rsiVal: v, priceLow };
    }
    if (isHigh) {
      const priceHigh = candles[i].high;
      if (prevHigh) {
        const dist = i - prevHigh.idx;
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v < prevHigh.rsiVal && priceHigh > prevHigh.priceHigh) {
          bear.push({ p1: { idx: prevHigh.idx, value: prevHigh.rsiVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevHigh = { idx: i, rsiVal: v, priceHigh };
    }
  }
  return { bull, bear };
}

function computeMacdDivergenceProvisional(candles, oscAligned, K, zeroLineFilter) {
  const LB = 5, RANGE_MIN = 5, RANGE_MAX = 60, ZERO_WINDOW = LB + LB + 5;
  const n = candles.length;
  const bull = [], bear = [];
  let prevLow = null, prevHigh = null;

  const staysBelowZero = i => {
    for (let k = Math.max(0, i - ZERO_WINDOW + 1); k <= i; k++) {
      const vk = oscAligned[k];
      if (vk !== null && vk >= 0) return false;
    }
    return true;
  };
  const staysAboveZero = i => {
    for (let k = Math.max(0, i - ZERO_WINDOW + 1); k <= i; k++) {
      const vk = oscAligned[k];
      if (vk !== null && vk <= 0) return false;
    }
    return true;
  };

  for (let i = LB; i < n - K; i++) {
    const v = oscAligned[i];
    if (v === null) continue;
    let isLow = true, isHigh = true;
    const kEnd = Math.min(n - 1, i + LB);
    for (let k = i - LB; k <= kEnd && (isLow || isHigh); k++) {
      if (k === i) continue;
      const vk = oscAligned[k];
      if (vk === null) { isLow = false; isHigh = false; break; }
      if (vk < v) isLow = false;
      if (vk > v) isHigh = false;
    }
    if (isLow) {
      const priceLow = candles[i].low;
      if (prevLow) {
        const dist = i - prevLow.idx;
        const sameSide = (prevLow.oscVal >= 0) === (v >= 0);
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v > prevLow.oscVal && priceLow < prevLow.priceLow
            && sameSide && (!zeroLineFilter || staysBelowZero(i))) {
          bull.push({ p1: { idx: prevLow.idx, value: prevLow.oscVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevLow = { idx: i, oscVal: v, priceLow };
    }
    if (isHigh) {
      const priceHigh = candles[i].high;
      if (prevHigh) {
        const dist = i - prevHigh.idx;
        const sameSide = (prevHigh.oscVal >= 0) === (v >= 0);
        if (dist >= RANGE_MIN && dist <= RANGE_MAX
            && v < prevHigh.oscVal && priceHigh > prevHigh.priceHigh
            && sameSide && (!zeroLineFilter || staysAboveZero(i))) {
          bear.push({ p1: { idx: prevHigh.idx, value: prevHigh.oscVal },
                      p2: { idx: i, value: v } });
        }
      }
      prevHigh = { idx: i, oscVal: v, priceHigh };
    }
  }
  return { bull, bear };
}

// Label anchor: horizontal midpoint between the two pivots, value linearly
// interpolated by bar-index fraction so the anchor sits on the line.
function divergenceMidpoint(pair, candles) {
  const i1 = pair.p1.idx, i2 = pair.p2.idx;
  const midIdx = Math.round((i1 + i2) / 2);
  const frac = (midIdx - i1) / (i2 - i1);
  const value = pair.p1.value + (pair.p2.value - pair.p1.value) * frac;
  return { time: candles[midIdx].time, value };
}

function hexToRgba(hex, opacityPct) {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + (opacityPct / 100) + ')';
}

function trueRangeJS(highs, lows, closes) {
  const out = new Array(highs.length);
  for (let i = 0; i < highs.length; i++) {
    const hl = highs[i] - lows[i];
    if (i === 0) { out[i] = hl; continue; }
    const pc = closes[i - 1];
    out[i] = Math.max(hl, Math.abs(highs[i] - pc), Math.abs(lows[i] - pc));
  }
  return out;
}

function computeDMI(highs, lows, closes, diPeriod, adxPeriod) {
  const n = highs.length;
  const plusDM = new Array(n).fill(0);
  const minusDM = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const up = highs[i] - highs[i - 1];
    const down = lows[i - 1] - lows[i];
    if (up > down && up > 0) plusDM[i] = up;
    if (down > up && down > 0) minusDM[i] = down;
  }
  const atr = wilderJS(trueRangeJS(highs, lows, closes), diPeriod)
    .map(v => (v === null || v === 0) ? null : v);
  const plusWilder = wilderJS(plusDM, diPeriod);
  const minusWilder = wilderJS(minusDM, diPeriod);
  const pdi = plusWilder.map((v, i) => (v === null || atr[i] === null) ? null : 100 * v / atr[i]);
  const mdi = minusWilder.map((v, i) => (v === null || atr[i] === null) ? null : 100 * v / atr[i]);
  const dx = pdi.map((p, i) => {
    const m = mdi[i];
    if (p === null || m === null) return 0;      // dx.fillna(0)
    const total = p + m;
    return total === 0 ? 0 : 100 * Math.abs(p - m) / total;
  });
  return { pdi, mdi, adx: wilderJS(dx, adxPeriod) };
}

// Mirrors build_payload()'s ser(): drop points with no value yet, same as
// dropping pandas NaN before JSON-encoding.
function toPointsJS(dates, values) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i] !== null && !Number.isNaN(values[i])) out.push({ time: dates[i], value: values[i] });
  }
  return out;
}

function labelsFor(params) {
  return {
    rsi: `RSI(${params.rsi.period}) · SMA(${params.rsi.signal})`,
    macd: `MACD(${params.macd.fast},${params.macd.slow},${params.macd.signal})`,
    dmi: `DMI(${params.dmi.di},${params.dmi.adx})`,
    volume: `Volume · MA(${params.volume.period})`,
  };
}

// Same pandas .rolling(window).mean() semantics as rollingMeanJS (used by
// RSI's own signal line) — over raw volume rather than an oscillator.
function computeVolumeMA(volume, period) {
  return toPointsJS(volume.map(v => v.time), rollingMeanJS(volume.map(v => v.value), period));
}

// The one function both the page-load bootstrap and a newly-added symbol
// (see the "+" handler below) call to recompute a payload's indicators from
// a params object — so neither path can drift from the other.
function recomputeIndicatorsFor(D, params) {
  const dates = D.candles.map(c => c.time);
  const closes = D.candles.map(c => c.close);
  const highs = D.candles.map(c => c.high);
  const lows = D.candles.map(c => c.low);

  const r = computeRSI(closes, params.rsi.period, params.rsi.signal);
  const m = computeMACD(closes, params.macd.fast, params.macd.slow, params.macd.signal);
  const dm = computeDMI(highs, lows, closes, params.dmi.di, params.dmi.adx);

  D.indicators = {
    rsi: toPointsJS(dates, r.rsi),
    rsiSignal: toPointsJS(dates, r.signal),
    dif: toPointsJS(dates, m.dif),
    dea: toPointsJS(dates, m.dea),
    hist: toPointsJS(dates, m.hist).map(p => ({
      ...p, color: p.value >= 0 ? '#26a69a' : '#ef5350',
    })),
    pdi: toPointsJS(dates, dm.pdi),
    mdi: toPointsJS(dates, dm.mdi),
    adx: toPointsJS(dates, dm.adx),
  };
  D.volumeMa = computeVolumeMA(D.volume, params.volume.period);
  D.labels = { ...D.labels, ...labelsFor(params) };
  D.params = params;
}

// --- collapsible panels (畫圖, 資料來源) --------------------------------------
// panelButtons tracks every wirePanel() trigger so opening one panel — or
// clicking outside all of them — clears `.on` from every trigger button, not
// just the first one. Only mattered once a second wirePanel() user (the data
// source flyout) showed up; drawBtn alone never exposed the gap.
const panelButtons = [];
function closeAllPanels() {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('open'));
  panelButtons.forEach(b => b.classList.remove('on'));
}
function wirePanel(btnId, panelId) {
  const btn = document.getElementById(btnId);
  const panel = document.getElementById(panelId);
  panelButtons.push(btn);
  btn.addEventListener('click', e => {
    e.stopPropagation();
    document.querySelectorAll('.panel').forEach(p => {
      if (p !== panel) { p.classList.remove('open'); }
    });
    panelButtons.forEach(b => { if (b !== btn) b.classList.remove('on'); });
    panel.classList.toggle('open');
    btn.classList.toggle('on', panel.classList.contains('open'));
  });
}
wirePanel('drawBtn', 'drawPanel');
addEventListener('click', closeAllPanels);
document.querySelectorAll('.panel').forEach(p =>
  p.addEventListener('click', e => e.stopPropagation()));

// --- 畫圖 panel keyboard nav: Up/Down move, Enter toggles, Left/Right
// expand/collapse a 陰陽燭形態 kind group ------------------------------------
// The navigable list is recomputed on every keypress, not cached: a kind
// group's children appear/disappear (display:none) as it's expanded or
// collapsed, and optOb's sub-checkboxes flip disabled — both change which
// checkboxes are actually reachable while the panel stays open.
const drawPanelEl = document.getElementById('drawPanel');
let drawFocusIdx = -1;
document.getElementById('drawBtn').addEventListener('click', () => {
  drawFocusIdx = -1;
  drawPanelEl.querySelectorAll('.opt.kbdFocus').forEach(el => el.classList.remove('kbdFocus'));
});
function drawPanelCheckboxes() {
  return [...drawPanelEl.querySelectorAll('input[type=checkbox]')]
    .filter(cb => !cb.disabled && cb.offsetParent !== null);
}
function setDrawFocus(idx) {
  const boxes = drawPanelCheckboxes();
  drawPanelEl.querySelectorAll('.opt.kbdFocus').forEach(el => el.classList.remove('kbdFocus'));
  if (!boxes.length) { drawFocusIdx = -1; return; }
  drawFocusIdx = Math.max(0, Math.min(boxes.length - 1, idx));
  const cb = boxes[drawFocusIdx];
  cb.closest('.opt').classList.add('kbdFocus');
  cb.focus();
  cb.scrollIntoView({ block: 'nearest' });
}
addEventListener('keydown', e => {
  if (!drawPanelEl.classList.contains('open')) return;
  // obFraction/trendBars (number inputs) and trendMode (select) keep their
  // own native arrow-key/Enter behavior instead of being hijacked here.
  const active = document.activeElement;
  const inOtherField = active &&
    ((active.tagName === 'INPUT' && active.type !== 'checkbox') || active.tagName === 'SELECT');
  if (inOtherField) return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    setDrawFocus(drawFocusIdx === -1 ? 0 : drawFocusIdx + 1);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    setDrawFocus(drawFocusIdx === -1 ? drawPanelCheckboxes().length - 1 : drawFocusIdx - 1);
  } else if (e.key === 'Enter') {
    const boxes = drawPanelCheckboxes();
    if (drawFocusIdx >= 0 && boxes[drawFocusIdx]) { e.preventDefault(); boxes[drawFocusIdx].click(); }
  } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    // Only the four kind-row checkboxes (單日/雙日/三日/五日, data-kind) carry
    // a disclosure arrow beside them — every other checkbox ignores Left/Right.
    const cb = drawPanelCheckboxes()[drawFocusIdx];
    if (!cb || !cb.dataset.kind) return;
    const toggle = document.querySelector(`#patMenu button[data-kind-toggle="${cb.dataset.kind}"]`);
    if (!toggle) return;
    const expanded = toggle.getAttribute('aria-expanded') === 'true';
    if (e.key === 'ArrowRight' && !expanded) { e.preventDefault(); toggle.click(); }
    if (e.key === 'ArrowLeft' && expanded) { e.preventDefault(); toggle.click(); }
  }
});

// --- sidebar rail + flyout panels --------------------------------------------
// autoSize is off (see render()'s createChart call), so the chart's canvas
// only ever matches #chart's box because something explicitly resizes it —
// on window resize (below) and here, whenever a flyout's open/closed state
// changes #chart's own CSS width (see #chartWrap.panelOpen). Without this,
// #chart would reflow narrower/wider but the canvas itself would not, so the
// chart would either overflow its box or leave a stale gap — either way it
// would look like the plotted candles had shifted relative to the axes.
function resizeChartToContainer() {
  if (!chart) return;
  const el = document.getElementById('chart');
  chart.resize(el.clientWidth, el.clientHeight);
  positionIndHovers();
  positionBarPanel();
  positionMeasurements();
}

// Clicking a rail icon opens its flyout immediately to the icon's left,
// closing any other open flyout — at most one is open at a time. Unlike the
// 畫圖/統計-style .panel dropdowns, a flyout does NOT close on an outside
// click: it stays open while the user works the chart (hovering bars,
// toggling 畫圖, switching symbols), and closes only when its own rail icon
// is clicked again.
const chartWrapEl = document.getElementById('chartWrap');

function wireRail(btnId, flyoutId, onOpen) {
  const btn = document.getElementById(btnId);
  const flyout = document.getElementById(flyoutId);
  btn.addEventListener('click', e => {
    e.stopPropagation();
    document.querySelectorAll('.flyout').forEach(f => {
      if (f !== flyout) f.classList.remove('open');
    });
    document.querySelectorAll('.railBtn').forEach(b => {
      if (b !== btn) b.classList.remove('on');
    });
    flyout.classList.toggle('open');
    btn.classList.toggle('on', flyout.classList.contains('open'));
    chartWrapEl.classList.toggle('panelOpen', flyout.classList.contains('open'));
    // Next frame: #chart's own width has to have actually reflowed to the
    // new CSS value before reading clientWidth off it.
    requestAnimationFrame(resizeChartToContainer);
    if (onOpen && flyout.classList.contains('open')) onOpen();
  });
}
// updateChipsToggle() re-measures #wlChips's scrollHeight against its
// collapsed height to decide whether the expand chevron is needed — that
// measurement is meaningless while .watchlistFlyout is display:none (both
// read as 0), so the initial page-load render (before the panel is ever
// opened) always finds "no overflow" and the chevron never gets a second
// chance to appear. Re-running it right as the panel opens catches the
// real, now-visible layout instead of leaving that stale false reading.
wireRail('railWatchlist', 'watchlistFlyout', updateChipsToggle);
wireRail('railAlerts', 'alertsFlyout');
wireRail('railChecklist', 'checklistFlyout');
wireRail('railKey', 'keyFlyout');

// --- watchlist panel: list, filter, switch -----------------------------------
// `watchlist` mirrors the server's flat {symbol: {held, strategies, stages,
// patterns}} map; every mutation (add/tag/remove) updates it in place and
// re-renders, so the panel never has to re-fetch to stay in sync with itself.
let watchlist = ALL.watchlist;
// `layout` is the sidebar's display order — an array mixing
// {type:'ticker', symbol} and {type:'section', id, name, collapsed}
// entries, reconciled server-side against `watchlist` on every load (see
// sdx/watchlist_layout.py) so it always covers exactly the current
// ticker set without needing a migration step.
let layout = ALL.layout;
// The default symbol at page load is the first *ticker* in the sidebar's
// visual order (layout), not Object.keys(watchlist)[0] — that's raw
// watchlists.json key order, which reordering the sidebar never changes
// (only watchlist_layout.json does), so it silently stuck at whichever
// symbol was added first regardless of how the list was reordered.
// Falls back to dict order only if layout somehow has no ticker entries
// (defensive — reconciliation guarantees one per watchlist symbol).
const firstLayoutTicker = layout.find(item => item.type === 'ticker');
let current = firstLayoutTicker ? firstLayoutTicker.symbol : Object.keys(watchlist)[0];
let filterSelected = new Set();
let filterMode = '&';
let filterChipsExpanded = false;

function symbolTagValues(sym) {
  const t = watchlist[sym];
  const vals = [...t.strategies, ...t.stages, ...t.patterns];
  if (t.held) vals.push('持有');
  return vals;
}

function allTagValues() {
  const vals = new Set();
  Object.keys(watchlist).forEach(sym => symbolTagValues(sym).forEach(v => vals.add(v)));
  return [...vals];
}

function matchesFilter(sym) {
  if (!filterSelected.size) return true;
  const vals = symbolTagValues(sym);
  const has = v => vals.includes(v);
  return filterMode === '&'
    ? [...filterSelected].every(has)
    : [...filterSelected].some(has);
}

// 持有 is deliberately excluded here — it now has its own badge (.wlHeld,
// see renderWatchlistRows) instead of appearing as text in this summary.
// symbolTagValues() (above) still includes it, so it stays filterable via
// .wlChips even though it no longer shows up in this line.
function tagSummary(tags) {
  const parts = [...tags.strategies, ...tags.stages, ...tags.patterns];
  return parts.join(', ');
}

function renderFilterChips() {
  const chipsEl = document.getElementById('wlChips');
  chipsEl.innerHTML = '';
  allTagValues().forEach(val => {
    const chip = document.createElement('button');
    chip.className = 'chip' + (filterSelected.has(val) ? ' on' : '');
    chip.textContent = val;
    chip.addEventListener('click', () => {
      if (filterSelected.has(val)) filterSelected.delete(val); else filterSelected.add(val);
      renderWatchlistPanel();
      renderAlertsPanel();
    });
    chipsEl.appendChild(chip);
  });
  chipsEl.classList.toggle('expanded', filterChipsExpanded);
  updateChipsToggle();
}

// Shows the toggle only when the chips actually wrap past one row — scrollHeight
// (the content's real height) exceeds clientHeight (the collapsed max-height cap)
// only in that case, regardless of the chip count, so a short tag list never
// shows a toggle with nothing extra to reveal.
function updateChipsToggle() {
  const chipsEl = document.getElementById('wlChips');
  const toggle = document.getElementById('wlChipsToggle');
  // >4, not >0: sub-pixel layout rounding alone (e.g. 25 vs 24) can make
  // scrollHeight exceed clientHeight by a stray 1px with zero actual second
  // row present — confirmed live with a single chip. A real overflowing row
  // adds a full chip's height (~20px+), well clear of that noise floor.
  const overflows = filterChipsExpanded || chipsEl.scrollHeight > chipsEl.clientHeight + 4;
  toggle.classList.toggle('show', overflows);
  toggle.classList.toggle('expanded', filterChipsExpanded);
  toggle.setAttribute('aria-label', filterChipsExpanded ? '收起標籤' : '展開全部標籤');
}

document.getElementById('wlChipsToggle').addEventListener('click', () => {
  filterChipsExpanded = !filterChipsExpanded;
  document.getElementById('wlChips').classList.toggle('expanded', filterChipsExpanded);
  updateChipsToggle();
});

document.querySelectorAll('#wlMode button').forEach(b => {
  b.addEventListener('click', () => {
    filterMode = b.dataset.mode === '|' ? '|' : '&';
    document.querySelectorAll('#wlMode button').forEach(x => x.classList.toggle('on', x === b));
    renderWatchlistPanel();
    renderAlertsPanel();
  });
});

document.querySelectorAll('#volMode button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#volMode button').forEach(x => x.classList.toggle('on', x === b));
    document.getElementById('volQuoteList').dataset.state = b.dataset.state;
  });
});

// 檢查清單 rows: click the check area to tick off a step (session-only, no
// persistence — resets on reload same as volMode, it's a focus aid for the
// current symbol's review, not a saved record). The caret is a separate
// control from the checkmark so expanding a row's detail never toggles it.
(() => {
  const rows = document.querySelectorAll('#checklistRows .clRow');
  const countEl = document.getElementById('checklistProgressCount');
  function updateProgress() {
    const done = document.querySelectorAll('#checklistRows .clRow.checked').length;
    countEl.textContent = done + ' / ' + rows.length;
    countEl.classList.toggle('complete', done === rows.length);
  }
  rows.forEach(row => {
    const checkArea = row.querySelector('.clCheckArea');
    if (checkArea) checkArea.addEventListener('click', () => {
      row.classList.toggle('checked');
      updateProgress();
    });
    const caret = row.querySelector('.clCaret');
    if (caret) caret.addEventListener('click', () => {
      row.dataset.open = row.dataset.open === 'true' ? 'false' : 'true';
    });
  });
  // 趨勢: picking 升/跌/橫 both records the read and counts the row as
  // reviewed — there's no separate checkbox, the choice IS the check.
  const trendRow = document.querySelector('.clRow.clTrend');
  trendRow.querySelectorAll('.trendMode button').forEach(b => {
    b.addEventListener('click', () => {
      trendRow.querySelectorAll('.trendMode button').forEach(x => x.classList.toggle('on', x === b));
      trendRow.classList.add('checked');
      updateProgress();
    });
  });
})();

const EDIT_ICON = '<svg viewBox="0 0 24 24"><path d="M4 20L5 16L16 5L19 8L8 19Z"/><path d="M14 7L17 10"/></svg>';
const TRASH_ICON = '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/><path d="M10 11v6M14 11v6"/></svg>';
const CHECK_ICON = '<svg viewBox="0 0 24 24"><path d="M4 13l5 5L20 6"/></svg>';
const CANCEL_ICON = '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>';
const CHEVRON_ICON = '<svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>';
// Context-menu icons are filled (fill="currentColor"), not stroke-style
// like .wlIcon's — .wlIcon svg forces fill:none, which would render a
// filled icon invisible, so these are sized/colored via .ctxItem svg
// instead (see the .ctxMenu CSS block).
const SECTION_ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28"><path fill="currentColor" d="M4 10h5V9H4v1zM11 10h6V9h-6v1zM24 10h-5V9h5v1zM6.5 14c-.83 0-1.5.67-1.5 1.5v3c0 .83.67 1.5 1.5 1.5h15c.83 0 1.5-.67 1.5-1.5v-3c0-.83-.67-1.5-1.5-1.5h-15zM6 15.5c0-.28.22-.5.5-.5h15c.28 0 .5.22.5.5v3a.5.5 0 0 1-.5.5h-15a.5.5 0 0 1-.5-.5v-3z"/></svg>';
const TAG_ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M12.41 2.5H4A1.5 1.5 0 0 0 2.5 4v8.41c0 .4.16.78.44 1.06l9.5 9.5a1.5 1.5 0 0 0 2.12 0l8.41-8.41a1.5 1.5 0 0 0 0-2.12l-9.5-9.5a1.5 1.5 0 0 0-1.06-.44ZM7 8.5A1.5 1.5 0 1 1 7 5.5a1.5 1.5 0 0 1 0 3Z"/></svg>';

// #sourceBtn swaps between these two so the header icon itself shows which
// data source is active without opening #sourcePanel. yfinance gets a disk
// icon — sdx.data.load caches every symbol to a local CSV and serves from
// disk with no network call whenever the request is already covered (see
// that module's docstring); Webull gets a broadcast icon (a dot radiating
// paired arcs left/right) — it's the live, still-arriving network stream.
const SOURCE_ICON_YF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M3 14h18"/><circle cx="17" cy="16" r="0.8" fill="currentColor" stroke="none"/></svg>';
const SOURCE_ICON_WB = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1.8" fill="currentColor" stroke="none"/><path d="M9 9a4 4 0 0 0 0 6"/><path d="M15 9a4 4 0 0 1 0 6"/><path d="M6 6a9 9 0 0 0 0 12"/><path d="M18 6a9 9 0 0 1 0 12"/></svg>';

// Latest-day % change for a symbol, from data already loaded client-side
// (no fetch triggered here) — same close-vs-prev-close math as the top-left
// OHLC legend (symbolLegendHtml), just applied per watchlist row instead of
// only to the current symbol. Returns '' (no chip) when the symbol's data
// isn't loaded yet rather than fabricating a placeholder/zero.
function chgChipHtml(sym) {
  const data = ALL.symbols[sym];
  const candles = data && data.candles;
  if (!candles || candles.length < 2) return '';
  const last = candles[candles.length - 1];
  const prev = candles[candles.length - 2];
  if (!prev.close) return '';
  const pct = (last.close - prev.close) / prev.close * 100;
  const cls = pct >= 0 ? 'pos' : 'neg';
  const sign = pct >= 0 ? '+' : '';
  return '<span class="wlChg ' + cls + '">' + sign + num(pct, 2) + '%</span>';
}

// idx of the layout entry (ticker or section) currently showing the
// inline confirm/cancel icon pair in place of edit/delete, or null.
let confirmDeleteIdx = null;
// id of the section currently in inline rename mode, or null.
let renamingSectionId = null;

// Builds the hover-revealed icon pair for a ticker row or section header:
// edit/delete normally, or — while `confirmDeleteIdx === idx` — a
// confirm/cancel pair in the same slot (see armPendingDelete()). Kept in
// one place since ticker rows and section headers share this exactly.
function buildIconPair(idx, editTitle, delTitle, onEdit, onConfirmDelete) {
  const wrap = document.createElement('div');
  wrap.className = 'wlIcons';
  wrap.dataset.idx = idx;
  if (confirmDeleteIdx === idx) {
    const confirm = document.createElement('button');
    confirm.className = 'wlIcon wlConfirm forceOn';
    confirm.title = '確認移除';
    confirm.innerHTML = CHECK_ICON;
    confirm.addEventListener('click', e => { e.stopPropagation(); onConfirmDelete(); });
    const cancel = document.createElement('button');
    cancel.className = 'wlIcon wlCancel forceOn';
    cancel.title = '取消';
    cancel.innerHTML = CANCEL_ICON;
    cancel.addEventListener('click', e => { e.stopPropagation(); cancelPendingDelete(); });
    wrap.appendChild(confirm);
    wrap.appendChild(cancel);
  } else {
    const edit = document.createElement('button');
    edit.className = 'wlIcon wlEdit';
    edit.title = editTitle;
    edit.innerHTML = EDIT_ICON;
    edit.addEventListener('click', e => { e.stopPropagation(); onEdit(); });
    const del = document.createElement('button');
    del.className = 'wlIcon wlDel';
    del.title = delTitle;
    del.innerHTML = TRASH_ICON;
    del.addEventListener('click', e => { e.stopPropagation(); armPendingDelete(idx); });
    wrap.appendChild(edit);
    wrap.appendChild(del);
  }
  return wrap;
}

function armPendingDelete(idx) {
  confirmDeleteIdx = idx;
  renderWatchlistPanel();
  renderAlertsPanel();
  // Deferred so the same click that armed this doesn't immediately
  // register as an "outside" click and cancel it right back — same guard
  // openColorPicker() uses.
  setTimeout(() => {
    document.addEventListener('mousedown', outsideConfirmClick);
    document.addEventListener('keydown', confirmDeleteEscape);
  }, 0);
}
function cancelPendingDelete() {
  if (confirmDeleteIdx === null) return;
  confirmDeleteIdx = null;
  document.removeEventListener('mousedown', outsideConfirmClick);
  document.removeEventListener('keydown', confirmDeleteEscape);
  renderWatchlistPanel();
  renderAlertsPanel();
}
function outsideConfirmClick(e) {
  if (confirmDeleteIdx === null) return;
  const iconsEl = e.target.closest('.wlIcons');
  if (iconsEl && Number(iconsEl.dataset.idx) === confirmDeleteIdx) return;
  cancelPendingDelete();
}
function confirmDeleteEscape(e) {
  if (e.key === 'Escape') cancelPendingDelete();
}

// For each section index, whether at least one ticker between it and the
// next section boundary passes the current filter — a section with none
// simply isn't rendered (no empty header left behind by filtering), unlike
// a collapsed section, which still shows its header. Computed as one pass
// so renderWatchlistRows() doesn't need to look ahead while iterating.
function sectionFilterVisibility() {
  const visible = new Map();
  let curIdx = null, any = false;
  layout.forEach((item, idx) => {
    if (item.type === 'section') {
      if (curIdx !== null) visible.set(curIdx, any);
      curIdx = idx;
      any = false;
    } else if (curIdx !== null && matchesFilter(item.symbol)) {
      any = true;
    }
  });
  if (curIdx !== null) visible.set(curIdx, any);
  return visible;
}

function renderWatchlistRows() {
  const listEl = document.getElementById('wlList');
  listEl.innerHTML = '';
  const sectionVisible = sectionFilterVisibility();
  let hideUntilNextSection = false;

  layout.forEach((item, idx) => {
    if (item.type === 'section') {
      hideUntilNextSection = item.collapsed;
      if (sectionVisible.get(idx) === false) return;   // every ticker under it is filtered out

      const sec = document.createElement('div');
      sec.className = 'wlSection' + (item.collapsed ? ' collapsed' : '');
      sec.dataset.idx = idx;
      if (LIVE) { sec.draggable = true; wireDrag(sec, idx); }

      const chevron = document.createElement('button');
      chevron.className = 'wlChevron';
      chevron.innerHTML = CHEVRON_ICON;
      chevron.addEventListener('click', e => { e.stopPropagation(); toggleSectionCollapsed(idx); });
      sec.appendChild(chevron);

      const name = document.createElement('div');
      name.className = 'wlSecName';
      if (renamingSectionId === item.id) {
        name.innerHTML = '<input type="text">';
        const input = name.querySelector('input');
        input.value = item.name;
        setTimeout(() => { input.focus(); input.select(); }, 0);
        input.addEventListener('click', e => e.stopPropagation());
        input.addEventListener('keydown', e => {
          if (e.key === 'Enter') commitRenameSection(idx, input.value);
          if (e.key === 'Escape') cancelRenameSection();
        });
        input.addEventListener('blur', () => commitRenameSection(idx, input.value));
      } else {
        name.textContent = item.name;
      }
      sec.appendChild(name);

      if (LIVE) {
        sec.appendChild(buildIconPair(idx, '重新命名', '移除分類',
          () => startRenameSection(item.id),
          () => removeSectionEntry(idx)));
        sec.addEventListener('contextmenu', e => { e.preventDefault(); openSectionCtxMenu(e, idx, item); });
      }

      listEl.appendChild(sec);
      return;
    }

    // ticker entry
    if (hideUntilNextSection) return;
    const sym = item.symbol;
    if (!matchesFilter(sym)) return;

    const row = document.createElement('div');
    row.className = 'wlRow' + (sym === current ? ' on' : '');
    row.dataset.sym = sym;
    row.dataset.idx = idx;
    if (LIVE) { row.draggable = true; wireDrag(row, idx); }
    // On the row itself, not just .wlMain — .wlRow has its own padding
    // (padding:8px 14px) .wlMain doesn't cover, and cursor:pointer already
    // implies the whole row is clickable. .wlFlag and the edit/delete
    // icons each stopPropagation() their own clicks, so this doesn't
    // double-fire for those.
    row.addEventListener('click', () => select(sym));

    const flag = document.createElement('div');
    flag.className = 'wlFlag' + (watchlist[sym].special ? ' on' : '');
    flag.title = '特別關注';
    if (LIVE) flag.addEventListener('click', e => { e.stopPropagation(); toggleSpecial(sym); });
    row.appendChild(flag);

    const heldHtml = watchlist[sym].held
      ? '<span class="wlHeld"><svg class="wlHeldGlyph" viewBox="0 0 24 24">' +
        '<rect x="5" y="4" width="4" height="16"/><rect x="15" y="4" width="4" height="16"/>' +
        '<rect x="5" y="10" width="14" height="4"/></svg></span>'
      : '';
    const main = document.createElement('div');
    main.className = 'wlMain';
    main.innerHTML = '<span class="wlTop">' + heldHtml +
      `<span class="wlName">${sym}</span>${chgChipHtml(sym)}</span>` +
      `<span class="wlTags">${tagSummary(watchlist[sym]) || '—'}</span>`;
    row.appendChild(main);

    if (LIVE) {
      row.appendChild(buildIconPair(idx, '編輯標籤', '移除',
        () => openTagModal(sym),
        () => removeSymbol(sym)));
      row.addEventListener('contextmenu', e => { e.preventDefault(); openTickerCtxMenu(e, idx, sym); });
    }
    listEl.appendChild(row);
  });
}

function renderWatchlistPanel() {
  renderFilterChips();
  renderWatchlistRows();
}

// --- alerts panel: which 持倉/特別關注 symbols currently have a signal -------
// LB duplicates computeRsiDivergence/computeMacdDivergence's own hardcoded
// LB=5 rather than having those functions export it — same small-duplication-
// over-coupling precedent those two functions already set between themselves.
const ALERT_LB = 5;
// Per-indicator confirming-bar requirement for the 初步 (provisional) tier,
// chosen from an empirical walk-forward backtest (real daily bars,
// VOO/AAPL/TSLA) comparing each K's false-flag rate against what the
// confirmed algorithm eventually reports for the same pivot — see
// docs/superpowers/specs/2026-08-13-watchlist-alerts-design.md.
const PROVISIONAL_K = { rsi: 3, macd: 2 };

// Each chip carries its own log key (symbol is added by the caller,
// renderAlertsPanel, since symbolAlerts() doesn't thread it through here) —
// `condition` is always the bare label with no "·初步" suffix, `tier` is
// separate, so a pivot's provisional and (later) confirmed sightings are
// two distinct, individually-acknowledgeable log rows rather than one
// mutating row that would silently "forget" it was ever provisional.
function divTag(dirLabel, cls, provisional, date) {
  return provisional
    ? { label: dirLabel + '·初步', cls: cls + '-prov', condition: dirLabel, tier: 'provisional', date }
    : { label: dirLabel, cls, condition: dirLabel, tier: 'confirmed', date };
}

function symbolAlerts(sym) {
  const D = ALL.symbols[sym];
  const tags = watchlist[sym];
  if (!D || !tags || !(tags.held || tags.special)) return null;

  const candles = D.candles;
  const n = candles.length;
  if (!n) return null;
  const lastDate = candles[n - 1].time;
  // n-10: within 5 days of its OWN confirmation day (a pivot first becomes
  // confirmable exactly LB=5 bars after it forms — this window is a lower
  // bound only; computeRsiDivergence/computeMacdDivergence's own loop bound
  // already guarantees the upper bound (p2.idx <= n-1-LB), so this must
  // NOT also re-derive that upper bound (`idx >= n-1-LB` alone would only
  // ever match the single day a pivot first becomes confirmable, then drop
  // it forever the next day).
  const confirmedRecent = idx => idx >= n - 1 - ALERT_LB - ALERT_LB;
  // n-5: no K here — existence in provPairs already guarantees
  // idx <= n-1-K (the provisional detector's own loop bound), so this only
  // needs to trim off ancient history, not re-derive the upper bound either
  // (the same one-day-then-gone mistake applies here too: `idx >= n-1-K`
  // combined with that existence bound leaves exactly one valid day).
  const provisionalRecent = idx => idx >= n - ALERT_LB;

  const chips = [];
  const today = D.signals[lastDate] || [];
  if (today.includes('量增即攻')) chips.push({ label: '量增即攻', cls: 'atk', condition: '量增即攻', tier: 'confirmed', date: lastDate });
  if (today.includes('好友反攻')) chips.push({ label: '好友反攻', cls: 'rly', condition: '好友反攻', tier: 'confirmed', date: lastDate });
  if (tags.held && today.includes('清貨')) chips.push({ label: '清貨訊號', cls: 'liq', condition: '清貨訊號', tier: 'confirmed', date: lastDate });

  // confPairs/provPairs passed SEPARATELY, never merged — confirmedRecent
  // must only ever be checked against pairs the confirmed detector itself
  // produced, or a provisional-only pair could pass confirmedRecent purely
  // by date and get mislabeled as 確認 before it was ever actually confirmed.
  // find(), not some(): the log key needs the actual matching pair's own
  // p2 bar date, not just "a pair exists somewhere in the window".
  const addDiv = (confPairs, provPairs, dirLabel, cls) => {
    const confHit = confPairs.find(p => confirmedRecent(p.p2.idx));
    if (confHit) {
      chips.push(divTag(dirLabel, cls, false, candles[confHit.p2.idx].time));
      return;
    }
    const provHit = provPairs.find(p => provisionalRecent(p.p2.idx));
    if (provHit) chips.push(divTag(dirLabel, cls, true, candles[provHit.p2.idx].time));
  };

  const rsiByTime = new Map(D.indicators.rsi.map(p => [p.time, p.value]));
  const rsiAligned = candles.map(c => { const v = rsiByTime.get(c.time); return v === undefined ? null : v; });
  const rsiConf = computeRsiDivergence(candles, rsiAligned);
  const rsiProv = computeRsiDivergenceProvisional(candles, rsiAligned, PROVISIONAL_K.rsi);
  addDiv(rsiConf.bull, rsiProv.bull, 'RSI背馳(底)', 'bull');
  addDiv(rsiConf.bear, rsiProv.bear, 'RSI背馳(頂)', 'bear');

  const alignTo = key => {
    const byTime = new Map(D.indicators[key].map(p => [p.time, p.value]));
    return candles.map(c => { const v = byTime.get(c.time); return v === undefined ? null : v; });
  };
  const difAligned = alignTo('dif'), histAligned = alignTo('hist');

  const difConf = computeMacdDivergence(candles, difAligned, true);
  const difProv = computeMacdDivergenceProvisional(candles, difAligned, PROVISIONAL_K.macd, true);
  addDiv(difConf.bull, difProv.bull, 'MACD背馳(底)', 'bull');
  addDiv(difConf.bear, difProv.bear, 'MACD背馳(頂)', 'bear');

  const histConf = computeMacdDivergence(candles, histAligned, false);
  const histProv = computeMacdDivergenceProvisional(candles, histAligned, PROVISIONAL_K.macd, false);
  addDiv(histConf.bull, histProv.bull, 'MACD差離(牛)', 'bull');
  addDiv(histConf.bear, histProv.bear, 'MACD差離(熊)', 'bear');

  return chips.length ? { sym, chips } : null;
}

// Persisted alert-occurrence log (sdx/alerts_log.py, data/alerts_log.json)
// — each row is a distinct (symbol, date, condition, tier) occurrence, so
// acknowledgment survives reloads without going stale: a signal on a NEW
// date is always a new, unacknowledged row, nothing to invalidate. Fetched
// once at boot (see the bottom of this script); ALERT_LOG_MAP is rebuilt
// from it after every fetch/mutation for O(1) per-chip lookups.
let ALERT_LOG = [];
let ALERT_LOG_MAP = new Map();
function alertKeyStr(symbol, date, condition, tier) {
  return symbol + '|' + date + '|' + condition + '|' + tier;
}
function rebuildAlertLogMap() {
  ALERT_LOG_MAP = new Map(ALERT_LOG.map(e => [alertKeyStr(e.symbol, e.date, e.condition, e.tier), e.acked]));
}

function renderAlertsPanel() {
  const results = layout
    .filter(item => item.type === 'ticker')
    .map(item => symbolAlerts(item.symbol))
    .filter(Boolean);

  // Any chip whose key isn't in the log yet is a genuinely new occurrence
  // — persist it (fire-and-forget; the backend is the authoritative dedup,
  // this client-side check just avoids resending what we already know is
  // there). Optimistically added to ALERT_LOG_MAP as acked:false right
  // away so this same render pass (and any re-render before the request
  // resolves) doesn't collect and resend it again.
  const newKeys = [];
  results.forEach(({ sym, chips }) => chips.forEach(c => {
    const k = alertKeyStr(sym, c.date, c.condition, c.tier);
    if (!ALERT_LOG_MAP.has(k)) {
      ALERT_LOG_MAP.set(k, false);
      newKeys.push({ symbol: sym, date: c.date, condition: c.condition, tier: c.tier });
    }
  }));
  if (newKeys.length) {
    fetch('/api/alerts/log', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newKeys),
    }).then(r => r.json()).then(log => { ALERT_LOG = log; rebuildAlertLogMap(); })
      .catch(err => console.error('alerts log append failed:', err.message));
  }

  const unacked = results.filter(({ sym, chips }) =>
    chips.some(c => !ALERT_LOG_MAP.get(alertKeyStr(sym, c.date, c.condition, c.tier)))).length;

  const badge = document.getElementById('railAlertsBadge');
  badge.textContent = String(unacked);
  badge.classList.toggle('show', unacked > 0);

  const body = document.getElementById('alertsBody');
  if (!results.length) {
    body.innerHTML = '<div class="flyoutEmpty">沒有符合條件的股票</div>';
    return;
  }
  body.innerHTML = '';
  results.forEach(({ sym, chips }) => {
    const rowKeys = chips.map(c => ({ symbol: sym, date: c.date, condition: c.condition, tier: c.tier }));
    const acked = rowKeys.every(k => ALERT_LOG_MAP.get(alertKeyStr(k.symbol, k.date, k.condition, k.tier)));
    const row = document.createElement('div');
    row.className = 'wlRow alertRow' + (acked ? ' acked' : '');
    row.dataset.sym = sym;
    const heldHtml = watchlist[sym].held
      ? '<span class="wlHeld"><svg class="wlHeldGlyph" viewBox="0 0 24 24">' +
        '<rect x="5" y="4" width="4" height="16"/><rect x="15" y="4" width="4" height="16"/>' +
        '<rect x="5" y="10" width="14" height="4"/></svg></span>'
      : '';
    row.innerHTML = `<input type="checkbox" class="alertAck" ${acked ? 'checked' : ''} title="標記已讀">` +
      `<div class="wlMain">` +
      `<span class="wlTop">${heldHtml}<span class="wlName">${sym}</span></span>` +
      `<div class="alertChips">${chips.map(c =>
        `<span class="chip alertChip alertChip-${c.cls}">${c.label}</span>`).join('')}</div>` +
      `</div>`;
    row.querySelector('.alertAck').addEventListener('click', e => {
      e.stopPropagation();
      const wantAcked = e.target.checked;
      rowKeys.forEach(k => ALERT_LOG_MAP.set(alertKeyStr(k.symbol, k.date, k.condition, k.tier), wantAcked));
      renderAlertsPanel();
      fetch('/api/alerts/log/ack', {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keys: rowKeys, acked: wantAcked }),
      }).then(r => r.json()).then(log => { ALERT_LOG = log; rebuildAlertLogMap(); renderAlertsPanel(); })
        .catch(err => console.error('alerts log ack failed:', err.message));
    });
    row.addEventListener('click', () => select(sym));
    body.appendChild(row);
  });
}

// Toggles special (特別關注) directly from the ribbon flag — no modal, unlike
// held/strategies/stages/patterns which go through openTagModal. The PATCH
// endpoint replaces the whole entry, so the symbol's current tag set is
// resent with only `special` flipped (same full-entry-replace pattern the
// tag-save handler below uses).
async function toggleSpecial(sym) {
  const tags = watchlist[sym];
  const body = {
    held: tags.held,
    special: !tags.special,
    strategies: tags.strategies,
    stages: tags.stages,
    patterns: tags.patterns,
  };
  try {
    const res = await fetch('/api/watchlist/' + encodeURIComponent(sym), {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const out = await res.json();
    watchlist[sym] = out.tags;
    renderWatchlistPanel();
    renderAlertsPanel();
  } catch (err) {
    console.error('toggle special failed:', err.message);
  }
}

async function removeSymbol(sym) {
  try {
    const res = await fetch('/api/watchlist/' + encodeURIComponent(sym), { method: 'DELETE' });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    delete watchlist[sym];
    delete ALL.symbols[sym];
    // Drop the matching ticker entry locally too — the stored
    // watchlist_layout.json self-heals against watchlists.json on its next
    // read (see sdx/watchlist_layout.py), but that reconciliation is
    // read-time only, so the in-memory `layout` this render loop uses
    // needs the same fix-up immediately, not just on the next page load.
    const layoutIdx = layout.findIndex(e => e.type === 'ticker' && e.symbol === sym);
    if (layoutIdx !== -1) layout.splice(layoutIdx, 1);
    confirmDeleteIdx = null;
    filterSelected.forEach(v => { if (!allTagValuesIncludes(v)) filterSelected.delete(v); });
    if (current === sym) {
      const remaining = Object.keys(watchlist);
      if (remaining.length) select(remaining[0]);
    }
    renderWatchlistPanel();
    renderAlertsPanel();
  } catch (err) {
    console.error('remove failed:', err.message);
  }
}

// --- watchlist layout: sections, rename, context menus, drag-and-drop ------

async function saveLayout() {
  try {
    const res = await fetch('/api/watchlist/layout', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(layout),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    layout = await res.json();
  } catch (err) {
    console.error('save layout failed:', err.message);
  }
  renderWatchlistPanel();
  renderAlertsPanel();
}

function toggleSectionCollapsed(idx) {
  layout[idx].collapsed = !layout[idx].collapsed;
  saveLayout();
}

function startRenameSection(id) {
  renamingSectionId = id;
  renderWatchlistPanel();
  renderAlertsPanel();
}
function cancelRenameSection() {
  renamingSectionId = null;
  renderWatchlistPanel();
  renderAlertsPanel();
}
function commitRenameSection(idx, value) {
  if (renamingSectionId === null) return;   // already committed/cancelled (blur firing after Enter)
  const trimmed = value.trim();
  renamingSectionId = null;
  if (trimmed) {
    layout[idx].name = trimmed;
    saveLayout();
  } else {
    renderWatchlistPanel();
    renderAlertsPanel();
  }
}

function removeSectionEntry(idx) {
  layout.splice(idx, 1);
  confirmDeleteIdx = null;
  saveLayout();
}

// No client-generated id here — the server mints one (wll.new_section_id())
// for any section entry that omits it. Once saveLayout() swaps `layout`
// for the server's response, `layout[tickerIdx]` carries the real id, so
// renaming can target it.
async function addSectionAboveTicker(tickerIdx) {
  layout.splice(tickerIdx, 0, { type: 'section', name: '新分類', collapsed: false });
  await saveLayout();
  if (layout[tickerIdx] && layout[tickerIdx].type === 'section') {
    renamingSectionId = layout[tickerIdx].id;
    renderWatchlistPanel();
    renderAlertsPanel();
  }
}

// Drops a filter chip whose only symbol was just removed, so a stale chip
// with no matching row cannot linger in the filter row.
function allTagValuesIncludes(val) {
  return allTagValues().includes(val);
}

// --- watchlist: right-click context menus -------------------------------------
// Same fixed-position, cursor-anchored, viewport-clamped popup pattern as
// openColorPicker()'s .colorPickerPopup, just listing menu items instead
// of a swatch grid.

let ctxMenuEl = null;

function closeCtxMenu() {
  if (!ctxMenuEl) return;
  ctxMenuEl.remove();
  ctxMenuEl = null;
  document.removeEventListener('mousedown', outsideCtxMenuClick);
  document.removeEventListener('keydown', ctxMenuEscape);
}
function outsideCtxMenuClick(e) {
  if (ctxMenuEl && !ctxMenuEl.contains(e.target)) closeCtxMenu();
}
function ctxMenuEscape(e) {
  if (e.key === 'Escape') closeCtxMenu();
}

function showCtxMenu(x, y, items) {
  closeCtxMenu();
  const menu = document.createElement('div');
  menu.className = 'ctxMenu';
  menu.innerHTML = items.map((it, i) =>
    `<div class="ctxItem${it.danger ? ' danger' : ''}" data-i="${i}">${it.icon}<span>${it.label}</span></div>`
  ).join('');
  document.body.appendChild(menu);

  const rect = menu.getBoundingClientRect();
  let left = x, top = y;
  if (left + rect.width > innerWidth - 8) left = innerWidth - rect.width - 8;
  if (top + rect.height > innerHeight - 8) top = innerHeight - rect.height - 8;
  menu.style.left = left + 'px';
  menu.style.top = top + 'px';

  menu.querySelectorAll('.ctxItem').forEach(el => {
    el.addEventListener('click', () => { closeCtxMenu(); items[Number(el.dataset.i)].run(); });
  });

  ctxMenuEl = menu;
  // Deferred so the same right-click that opened this menu doesn't
  // immediately register as an "outside" click — same guard
  // openColorPicker() uses.
  setTimeout(() => {
    document.addEventListener('mousedown', outsideCtxMenuClick);
    document.addEventListener('keydown', ctxMenuEscape);
  }, 0);
}

function openTickerCtxMenu(e, idx, sym) {
  showCtxMenu(e.clientX, e.clientY, [
    { label: '新增分類', icon: SECTION_ICON, run: () => addSectionAboveTicker(idx) },
    { label: '加標籤', icon: TAG_ICON, run: () => openTagModal(sym) },
  ]);
}

function openSectionCtxMenu(e, idx, item) {
  showCtxMenu(e.clientX, e.clientY, [
    { label: '重新命名', icon: EDIT_ICON, run: () => startRenameSection(item.id) },
    // Arms the same inline confirm/cancel step as clicking the section's
    // own trash-bin icon (armPendingDelete) — must NOT remove immediately.
    { label: '移除分類', icon: TRASH_ICON, run: () => armPendingDelete(idx), danger: true },
  ]);
}

// --- watchlist: drag-and-drop reordering ---------------------------------------
// Both .wlRow and .wlSection share this: desktop uses native HTML5 drag
// (dragstart fires immediately, no arm delay); touch has no native
// drag-and-drop, so a ~350ms hold-without-scrolling arms it first (see
// wireTouchDrag()). Both converge on reorderLayout() so the reorder math
// itself is exercised the same way regardless of input device.

let dragSrcIdx = null;

function clearDropLine() {
  document.querySelectorAll('.wlDropLine').forEach(n => n.remove());
}

// Moves the entry at `fromIdx` to sit just before `toIdx` (as measured in
// the array *before* removal) and persists the result. Dragging a section
// moves only that entry — its tickers are never touched here, so after the
// splice/insert they simply belong to whichever section (if any) is now
// above them.
function reorderLayout(fromIdx, toIdx) {
  if (fromIdx === toIdx || fromIdx === toIdx - 1) return;
  const [item] = layout.splice(fromIdx, 1);
  const insertAt = fromIdx < toIdx ? toIdx - 1 : toIdx;
  layout.splice(insertAt, 0, item);
  saveLayout();
}

// Blends every text-bearing descendant's own `color` toward the flyout
// panel's background (#111721) — a literal repaint, not a compositing
// effect. Needed because the drag-ghost clone's setDragImage snapshot
// doesn't apply opacity/filter to its own text at all (confirmed live:
// with filter:opacity() on the ghost, its background/border fades but the
// glyphs stay full-brightness) — muting the color value directly is the
// only thing that reliably shows up in that snapshot.
function muteGhostText(root, alpha) {
  const BG = [17, 23, 33];   // #111721
  const blend = colorStr => {
    const m = colorStr.match(/\\d+(\\.\\d+)?/g);
    if (!m || m.length < 3) return null;
    const [r, g, b] = m.map(Number);
    const mix = (a, bg) => Math.round(a * alpha + bg * (1 - alpha));
    return `rgb(${mix(r, BG[0])}, ${mix(g, BG[1])}, ${mix(b, BG[2])})`;
  };
  root.querySelectorAll('.wlName, .wlTags, .wlChg, .wlSecName').forEach(node => {
    const blended = blend(getComputedStyle(node).color);
    if (blended) node.style.color = blended;
  });
}

function wireDrag(el, idx) {
  el.addEventListener('dragstart', e => {
    dragSrcIdx = idx;
    el.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    // Chrome's default drag image is a near-opaque snapshot of the
    // element, which sits right under the cursor and hides the drop-line
    // indicator/target it's supposed to be revealing — swap in a
    // translucent clone instead so the row being dragged over stays
    // legible underneath. Two separate mechanisms, since they don't
    // behave consistently in this snapshot: a moderate filter:opacity()
    // fades the container/background/border (kept well short of 50% —
    // that made the background nearly disappear), while the text itself
    // is muted directly via muteGhostText() since filter/opacity don't
    // reach it at all.
    const ghost = el.cloneNode(true);
    ghost.style.position = 'absolute';
    ghost.style.top = '-9999px';
    ghost.style.left = '-9999px';
    ghost.style.width = el.offsetWidth + 'px';
    ghost.style.filter = 'opacity(80%)';
    document.body.appendChild(ghost);
    muteGhostText(ghost, 0.55);
    e.dataTransfer.setDragImage(ghost, e.offsetX, e.offsetY);
    setTimeout(() => ghost.remove(), 0);
  });
  el.addEventListener('dragend', () => {
    el.classList.remove('dragging');
    clearDropLine();
    dragSrcIdx = null;
  });
  el.addEventListener('dragover', e => {
    if (dragSrcIdx === null) return;
    e.preventDefault();
    clearDropLine();
    // A collapsed section's tickers aren't rendered, so they can't be
    // compared against by position — dropping anywhere on the header
    // targets "first entry under this section" instead of an
    // insertion-line position.
    if (el.classList.contains('wlSection') && el.classList.contains('collapsed')) return;
    const rect = el.getBoundingClientRect();
    const before = (e.clientY - rect.top) < rect.height / 2;
    const line = document.createElement('div');
    line.className = 'wlDropLine';
    if (before) el.before(line); else el.after(line);
  });
  el.addEventListener('drop', e => {
    if (dragSrcIdx === null) return;
    e.preventDefault();
    clearDropLine();
    const srcIdx = dragSrcIdx;
    dragSrcIdx = null;
    if (srcIdx === idx) return;

    if (el.classList.contains('wlSection') && el.classList.contains('collapsed')) {
      reorderLayout(srcIdx, idx + 1);
      return;
    }
    const rect = el.getBoundingClientRect();
    const before = (e.clientY - rect.top) < rect.height / 2;
    reorderLayout(srcIdx, before ? idx : idx + 1);
  });

  wireTouchDrag(el, idx);
}

// Touch has no native drag-and-drop. A ~350ms hold arms the drag; moving
// past a small threshold before that fires cancels it (treated as a
// scroll, not a drag) rather than hijacking every touchmove.
const TOUCH_HOLD_MS = 350;
const TOUCH_MOVE_CANCEL_PX = 10;

function wireTouchDrag(el, idx) {
  let holdTimer = null;
  let armed = false;
  let startX = 0, startY = 0;

  el.addEventListener('touchstart', e => {
    if (dragSrcIdx !== null) return;   // another element is already mid-drag
    const t = e.touches[0];
    startX = t.clientX; startY = t.clientY;
    armed = false;
    holdTimer = setTimeout(() => {
      armed = true;
      dragSrcIdx = idx;
      el.classList.add('dragging');
    }, TOUCH_HOLD_MS);
  }, { passive: true });

  el.addEventListener('touchmove', e => {
    if (armed) {
      e.preventDefault();
      const t = e.touches[0];
      const target = document.elementFromPoint(t.clientX, t.clientY);
      const hovered = target && target.closest('.wlRow, .wlSection');
      clearDropLine();
      if (hovered && hovered !== el) {
        if (hovered.classList.contains('wlSection') && hovered.classList.contains('collapsed')) return;
        const rect = hovered.getBoundingClientRect();
        const before = (t.clientY - rect.top) < rect.height / 2;
        const line = document.createElement('div');
        line.className = 'wlDropLine';
        if (before) hovered.before(line); else hovered.after(line);
      }
      return;
    }
    const t = e.touches[0];
    if (Math.hypot(t.clientX - startX, t.clientY - startY) > TOUCH_MOVE_CANCEL_PX) {
      clearTimeout(holdTimer);
    }
  }, { passive: false });

  el.addEventListener('touchend', e => {
    clearTimeout(holdTimer);
    if (!armed) return;
    armed = false;
    el.classList.remove('dragging');
    clearDropLine();
    const srcIdx = dragSrcIdx;
    dragSrcIdx = null;
    const t = e.changedTouches[0];
    const target = document.elementFromPoint(t.clientX, t.clientY);
    const hovered = target && target.closest('.wlRow, .wlSection');
    if (!hovered) return;
    const toIdx = Number(hovered.dataset.idx);
    if (Number.isNaN(toIdx) || srcIdx === toIdx) return;

    if (hovered.classList.contains('wlSection') && hovered.classList.contains('collapsed')) {
      reorderLayout(srcIdx, toIdx + 1);
      return;
    }
    const rect = hovered.getBoundingClientRect();
    const before = (t.clientY - rect.top) < rect.height / 2;
    reorderLayout(srcIdx, before ? toIdx : toIdx + 1);
  });

  el.addEventListener('touchcancel', () => {
    clearTimeout(holdTimer);
    armed = false;
    dragSrcIdx = null;
    el.classList.remove('dragging');
    clearDropLine();
  });
}

// --- watchlist: compact/detailed view toggle ------------------------------------

(function initViewToggle() {
  const toggle = document.getElementById('wlViewToggle');
  const saved = localStorage.getItem('wlViewMode');
  if (saved === 'compact') {
    document.getElementById('watchlistFlyout').classList.add('wl-compact');
    toggle.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.mode === 'compact'));
  }
  toggle.addEventListener('click', e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    toggle.querySelectorAll('button').forEach(b => b.classList.toggle('on', b === btn));
    const compact = btn.dataset.mode === 'compact';
    document.getElementById('watchlistFlyout').classList.toggle('wl-compact', compact);
    localStorage.setItem('wlViewMode', compact ? 'compact' : 'detailed');
  });
})();

// --- add-symbol modal ---------------------------------------------------------
const overlay   = document.getElementById('overlay');
const symsInput = document.getElementById('addSyms');
const results   = document.getElementById('addResults');
const goBtn     = document.getElementById('addGo');

function openAddModal() {
  symsInput.value = '';
  results.innerHTML = '';
  overlay.classList.add('open');
  document.body.classList.add('modal-open');
  symsInput.focus();
}

// Uppercase as-you-type — submission already uppercases each symbol (see
// the .toUpperCase() in the split/map below), this just makes the textarea
// itself read that way while typing instead of only at submit time.
symsInput.addEventListener('input', () => {
  const start = symsInput.selectionStart;
  const end = symsInput.selectionEnd;
  symsInput.value = symsInput.value.toUpperCase();
  symsInput.setSelectionRange(start, end);
});

function closeModal() {
  overlay.classList.remove('open');
  syncModalOpen();
}

// One class drives #chart's pointer-events, so it must reflect every overlay.
function syncModalOpen() {
  const anyOpen = [...document.querySelectorAll('.overlay')]
    .some(o => o.classList.contains('open'));
  document.body.classList.toggle('modal-open', anyOpen);
}

if (LIVE) {
  document.getElementById('wlAddBtn').addEventListener('click', e => {
    e.stopPropagation(); openAddModal();
  });
} else {
  document.getElementById('wlAddBtn').style.display = 'none';
}

// --- keyboard quick-switch: press "/" to jump to a watchlist symbol by ticker or name ---
// Independent of LIVE — like clicking a .wlRow, switching symbols this way
// needs no server write, just the already-mirrored `watchlist` object.
const symbolSwitchOverlay = document.getElementById('symbolSwitchOverlay');
const symbolSwitchInput = document.getElementById('symbolSwitchInput');
const symbolSwitchResults = document.getElementById('symbolSwitchResults');
let symbolSwitchMatches = [];
let symbolSwitchHighlight = -1;

// Special (特別關注) first, then held (持倉), then alphabetically by ticker
// within each tier — not insertion order, so the picker stays predictable
// as the watchlist grows regardless of when each symbol was added.
function symbolSwitchTier(sym) {
  const tags = watchlist[sym];
  if (tags.special) return 0;
  if (tags.held) return 1;
  return 2;
}
function symbolSwitchSort(syms) {
  return syms.slice().sort((a, b) =>
    symbolSwitchTier(a) - symbolSwitchTier(b) || a.localeCompare(b));
}

function symbolSwitchMatchList(query) {
  const q = query.trim().toLowerCase();
  if (!q) return symbolSwitchSort(Object.keys(watchlist));
  return symbolSwitchSort(Object.keys(watchlist).filter(sym => {
    const name = (watchlist[sym].name || '').toLowerCase();
    return sym.toLowerCase().includes(q) || name.includes(q);
  }));
}

function paintSymbolSwitchResults() {
  if (!symbolSwitchMatches.length) {
    symbolSwitchResults.innerHTML = '<div class="flyoutEmpty">找不到符合的股票</div>';
    return;
  }
  symbolSwitchResults.innerHTML = symbolSwitchMatches.map((sym, i) => {
    const name = watchlist[sym].name;
    return `<div class="wlRow${i === symbolSwitchHighlight ? ' on' : ''}" data-sym="${sym}">` +
      `<div class="wlMain"><span class="wlName">${sym}</span>` +
      (name ? `<span class="wlTags">${name}</span>` : '') +
      `</div></div>`;
  }).join('');
  symbolSwitchResults.querySelectorAll('.wlRow').forEach(row => {
    // mousedown, not click: the input's blur (from clicking outside it)
    // would otherwise close/clear the overlay before a click ever lands —
    // same reasoning as the tag-autocomplete dropdown's own mousedown use.
    row.addEventListener('mousedown', e => {
      e.preventDefault();
      confirmSymbolSwitch(row.dataset.sym);
    });
  });
}

function renderSymbolSwitchResults() {
  symbolSwitchMatches = symbolSwitchMatchList(symbolSwitchInput.value);
  symbolSwitchHighlight = symbolSwitchMatches.length ? 0 : -1;
  paintSymbolSwitchResults();
}

function moveSymbolSwitchHighlight(delta) {
  if (!symbolSwitchMatches.length) return;
  symbolSwitchHighlight = Math.max(0, Math.min(symbolSwitchMatches.length - 1, symbolSwitchHighlight + delta));
  paintSymbolSwitchResults();
}

function confirmSymbolSwitch(sym) {
  closeSymbolSwitch();
  select(sym);
}

function openSymbolSwitch() {
  symbolSwitchOverlay.classList.add('open');
  syncModalOpen();
  renderSymbolSwitchResults();
  symbolSwitchInput.focus();
}

function closeSymbolSwitch() {
  symbolSwitchOverlay.classList.remove('open');
  syncModalOpen();
  symbolSwitchInput.value = '';
  symbolSwitchMatches = [];
  symbolSwitchHighlight = -1;
}

symbolSwitchInput.addEventListener('input', () => {
  // Uppercase as-you-type. Matching already lowercases both sides (see
  // symbolSwitchMatchList), so this is purely cosmetic — it doesn't affect
  // which symbols match, just how the query reads back to the user.
  const pos = symbolSwitchInput.selectionStart;
  symbolSwitchInput.value = symbolSwitchInput.value.toUpperCase();
  symbolSwitchInput.setSelectionRange(pos, pos);
  renderSymbolSwitchResults();
});
symbolSwitchInput.addEventListener('keydown', e => {
  if (e.key === 'ArrowDown') { e.preventDefault(); moveSymbolSwitchHighlight(1); return; }
  if (e.key === 'ArrowUp') { e.preventDefault(); moveSymbolSwitchHighlight(-1); return; }
  if (e.key === 'Enter') {
    e.preventDefault();
    if (symbolSwitchHighlight >= 0 && symbolSwitchMatches[symbolSwitchHighlight]) {
      confirmSymbolSwitch(symbolSwitchMatches[symbolSwitchHighlight]);
    }
  }
});
document.getElementById('symbolSwitchXClose').addEventListener('click', closeSymbolSwitch);
symbolSwitchOverlay.addEventListener('click', e => {
  if (e.target === symbolSwitchOverlay) closeSymbolSwitch();
});
addEventListener('keydown', e => {
  if (e.key === 'Escape' && symbolSwitchOverlay.classList.contains('open')) closeSymbolSwitch();
});

// Global trigger: guarded so "/" typed into any real text field (add-symbol
// textarea, tag inputs, goto-date input, this overlay's own input, etc.) or
// while any other overlay is open never gets hijacked into opening this one.
addEventListener('keydown', e => {
  if (e.key !== '/') return;
  const active = document.activeElement;
  const inTextField = active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA');
  const anyOverlayOpen = [...document.querySelectorAll('.overlay')]
    .some(o => o.classList.contains('open'));
  if (inTextField || anyOverlayOpen) return;
  e.preventDefault();
  openSymbolSwitch();
});

// w/a/c/d/f each just click a real header/rail button rather than
// duplicating its logic: w/a/c open the Watchlist/Alerts/Checklist rail
// panels (reuses wireRail's toggle-open/close-others/resize logic), d opens
// the 畫圖 panel (wirePanel's toggle-open/close logic), f flips 只看K線
// (toggleSubPanes). Same overlay guard as "/" above, plus a modifier check:
// Ctrl/Cmd/Alt held means the browser owns this combo (e.g. Cmd+W closes the
// tab), so it's left alone rather than hijacked into clicking a button
// instead. The text-field guard only excludes real typing surfaces — NOT
// type=checkbox — so d/f keep working to close/toggle while a checkbox
// inside #drawPanel has keyboard focus (see the Up/Down/Enter navigation
// below); a plain `tagName === 'INPUT'` check would treat that focused
// checkbox as a text field and silently swallow every one of these keys.
const CLICK_SHORTCUTS = {
  w: 'railWatchlist', a: 'railAlerts', c: 'railChecklist', d: 'drawBtn', f: 'toggle',
  m: 'measureBtn',
};
addEventListener('keydown', e => {
  const btnId = CLICK_SHORTCUTS[e.key.toLowerCase()];
  if (!btnId || e.ctrlKey || e.metaKey || e.altKey) return;
  const active = document.activeElement;
  const inTextField = active &&
    ((active.tagName === 'INPUT' && active.type !== 'checkbox') || active.tagName === 'TEXTAREA');
  const anyOverlayOpen = [...document.querySelectorAll('.overlay')]
    .some(o => o.classList.contains('open'));
  if (inTextField || anyOverlayOpen) return;
  e.preventDefault();
  document.getElementById(btnId).click();
});

// --- header refresh button: short click = current symbol, long-press = whole watchlist ---
const REFRESH_LONGPRESS_MS = 500;
const refreshBtn = document.getElementById('refreshBtn');

if (LIVE) {
  let pressTimer = null;
  let longPressed = false;

  function setRefreshBusy(busy) {
    refreshBtn.classList.toggle('busy', busy);
    refreshBtn.disabled = busy;
  }

  async function refreshCurrent() {
    setRefreshBusy(true);
    try {
      const res = await fetch('/api/refresh/' + encodeURIComponent(current), { method: 'POST' });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const { payload } = await res.json();
      // /api/refresh is yfinance-only — keep YF_PAYLOADS in sync too, or a
      // later Webull->yfinance switch would restore the pre-refresh data.
      YF_PAYLOADS[current] = payload;
      ALL.symbols[current] = payload;
      render();
      renderAlertsPanel();
    } catch (err) {
      console.error('refresh failed:', err.message);
    } finally {
      setRefreshBusy(false);
    }
  }

  async function refreshAll() {
    setRefreshBusy(true);
    try {
      const res = await fetch('/api/refresh', { method: 'POST' });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const { refreshed } = await res.json();
      refreshed.forEach(({ symbol, payload }) => {
        YF_PAYLOADS[symbol] = payload;
        ALL.symbols[symbol] = payload;
      });
      render();
      renderAlertsPanel();
    } catch (err) {
      console.error('refresh all failed:', err.message);
    } finally {
      setRefreshBusy(false);
    }
  }

  refreshBtn.addEventListener('pointerdown', () => {
    longPressed = false;
    pressTimer = setTimeout(() => { longPressed = true; refreshAll(); }, REFRESH_LONGPRESS_MS);
  });
  refreshBtn.addEventListener('pointerup', () => {
    clearTimeout(pressTimer);
    if (!longPressed) refreshCurrent();
  });
  refreshBtn.addEventListener('pointerleave', () => clearTimeout(pressTimer));
} else {
  refreshBtn.style.display = 'none';
}

// --- data source (yfinance / Webull) + live streaming ------------------
// Spec: openspec/changes/webull-streaming-data. yfinance stays the
// zero-fetch default — ALL.symbols already holds every symbol's payload
// from the server, exactly as before this feature existed. Webull is
// fetched on demand into the same ALL.symbols slot, never touching
// YF_PAYLOADS, so switching back to yfinance for a symbol needs no
// re-fetch either.
//
// `activateSymbol` and `liveState` are declared here, at top level, so
// select() (defined below) and symbolLegendHtml (which reads liveState to
// render the badge) can reach them regardless of load order — everything
// else stays private inside the `if (LIVE)` block below.
let activateSymbol = sym => render();   // no-op-ish default for a static (non-LIVE) page
let liveState = 'off';                  // 'off' | 'connecting' | 'live'

const SOURCE_KEY = 'sdx_data_source';
const YF_PAYLOADS = Object.assign({}, ALL.symbols);

// Mirrors sdx.providers.webull's LADDER_INTERVALS/NO_LIVE_INTERVALS — kept
// as an explicit list here rather than inferred from the payload, since
// "does this payload have ladder data" isn't reliably derivable after the
// fact (a real ladder run can legitimately have zero pivots/lines).
const LADDER_INTERVALS = ['D', '4h', 'M', 'Y'];
// Live streaming isn't offered for Month/Year — see the matching comment on
// sdx.providers.webull.NO_LIVE_INTERVALS for why (no trading calendar to
// bucket a "currently forming" bar against Webull's last-trading-day
// anchor).
const NO_LIVE_INTERVALS = ['M', 'Y'];

// A payload the ladder engine never ran over (a non-ladder-eligible Webull
// interval) is missing every LADDER_KEYS-ish field — see sdx/viz.py
// build_payload's result=None case. Every existing
// render()/buildLookups()/ladderOf() call site assumes those fields exist,
// even empty (ladderOf() reads D.alt.noPivot by DEFAULT, since obOn starts
// false). Backfilling once here keeps the rest of the chart code oblivious
// to which provider a payload came from, rather than guarding a dozen call
// sites individually.
const EMPTY_LADDER = {
  classes: [], pivots: [], signals: {}, classOverlay: [], markers: [],
  patterns: [], patternAnchor: [], stop: [], levels: [],
  stats: { bars: 0, classes: {}, pivots: 0, legs: 0, lines: 0,
           liquidations: 0, buys: 0, patterns: 0, barsWithStop: 0 },
};
function fillLadderless(payload) {
  const filled = Object.assign({}, EMPTY_LADDER, payload);
  filled.stats = Object.assign({}, EMPTY_LADDER.stats, { bars: payload.candles.length });
  filled.alt = { bothOn: filled, bullishOnly: filled, noPivot: filled };
  return filled;
}

// One global {source, interval} — deliberately NOT per-symbol: switching
// stock keeps whatever data source/interval was active, rather than each
// symbol remembering its own independently. `sym` stays as a parameter at
// every call site for a smaller diff, but is no longer part of the storage
// key.
function sourceFor(sym) {
  try {
    const raw = JSON.parse(localStorage.getItem(SOURCE_KEY));
    if (raw && raw.source) return raw;
  } catch {}
  return { source: 'yfinance', interval: 'D' };
}
function setSourceFor(sym, source, interval) {
  try { localStorage.setItem(SOURCE_KEY, JSON.stringify({ source, interval })); } catch {}
}

// Same "one global value, not per-symbol" reasoning as sourceFor/setSourceFor
// above. Only meaningful for the yfinance source (D, or resampled M/Y — see
// sdx.serve.bars_payload_for) — split- and dividend-adjusted (matches
// Futu/Webull) vs. split-adjusted-only (matches TradingView's own default),
// see sdx.data's module docstring. Raw is the default, matching
// sdx.serve.payload_for's default — see yfKey.
const PRICE_ADJ_KEY = 'sdx_price_adjusted';
function adjustedFor() {
  try {
    const raw = localStorage.getItem(PRICE_ADJ_KEY);
    if (raw !== null) return raw === 'true';
  } catch {}
  return false;
}
function setAdjustedFor(adjusted) {
  try { localStorage.setItem(PRICE_ADJ_KEY, String(adjusted)); } catch {}
}
// yfinance's own interval choice — independent of sourceFor's `interval`,
// which is Webull's (D/5m/.../M/Y, see srcIntervalSel). Deliberately kept
// separate rather than shared: Webull's interval list includes values
// (5m, 1h, ...) yfinance has no cache pipeline for at all (sdx.data is
// daily-only — M/Y are resampled from it, see bars_payload_for), so a
// single shared field would leave yfinance's own interval buttons with
// nothing selected whenever the stored value came from a Webull pick.
const YF_INTERVAL_KEY = 'sdx_yf_interval';
function yfIntervalFor() {
  try {
    const raw = localStorage.getItem(YF_INTERVAL_KEY);
    if (raw === 'D' || raw === 'M' || raw === 'Y') return raw;
  } catch {}
  return 'D';
}
function setYfIntervalFor(interval) {
  try { localStorage.setItem(YF_INTERVAL_KEY, interval); } catch {}
}
// YF_PAYLOADS holds at most one cached payload per (symbol, interval,
// adjusted) triple — the key is unsuffixed only for the (D, raw) pair,
// matching every payload the server ever seeds unsuffixed: initial held-
// symbol load, refresh, add-symbol — all always D/raw by default, see
// payload_for. Any other combination (a non-D interval, or adjusted=true)
// is fetched lazily only once a user switches to it.
function yfKey(sym, adjusted, interval) {
  const suffix = (interval && interval !== 'D' ? '|' + interval : '') + (adjusted ? '|adj' : '');
  return sym + suffix;
}

if (LIVE) {
  const sourceBtn = document.getElementById('sourceBtn');
  const srcYfBtn = document.getElementById('srcYf');
  const srcWbBtn = document.getElementById('srcWb');
  const srcIntervalSel = document.getElementById('srcInterval');
  const priceAdjBtn = document.getElementById('priceAdj');
  const priceRawBtn = document.getElementById('priceRaw');
  const yfDBtn = document.getElementById('yfD');
  const yfMBtn = document.getElementById('yfM');
  const yfYBtn = document.getElementById('yfY');
  const yfIntervalGroup = document.getElementById('yfIntervalGroup');
  const webullIntervalGroup = document.getElementById('webullIntervalGroup');

  function syncSourcePanel() {
    const { source, interval } = sourceFor(current);
    const adjusted = adjustedFor();
    const yfInterval = yfIntervalFor();
    srcYfBtn.classList.toggle('on', source === 'yfinance');
    srcWbBtn.classList.toggle('on', source === 'webull');
    srcIntervalSel.value = interval;
    sourceBtn.innerHTML = source === 'yfinance' ? SOURCE_ICON_YF : SOURCE_ICON_WB;
    sourceBtn.title = 'Data source: ' + (source === 'yfinance' ? 'yfinance' : 'Webull');
    // Each source gets only its own Interval group — a Webull-only value
    // (5m, 1h, ...) has no meaning for yfinance and vice versa, so showing
    // both side by side (one merely disabled) was more confusing than
    // useful. Whichever isn't the active source is hidden outright.
    yfIntervalGroup.style.display = source === 'yfinance' ? '' : 'none';
    webullIntervalGroup.style.display = source === 'webull' ? '' : 'none';
    yfDBtn.classList.toggle('on', yfInterval === 'D');
    yfMBtn.classList.toggle('on', yfInterval === 'M');
    yfYBtn.classList.toggle('on', yfInterval === 'Y');
    // Adjustable whenever bars come from yfinance: the D/webull-seeded case
    // (see bars_payload_for), or an explicit yfinance-source pick at any of
    // its own D/M/Y intervals. A non-D Webull interval has no such toggle.
    const priceApplicable = source === 'yfinance' || interval === 'D';
    priceAdjBtn.disabled = !priceApplicable;
    priceRawBtn.disabled = !priceApplicable;
    priceAdjBtn.classList.toggle('on', adjusted);
    priceRawBtn.classList.toggle('on', !adjusted);
  }

  let liveSocket = null;
  let liveReconnectTimer = null;
  const LIVE_RECONNECT_MS = 3000;

  function setLiveState(state) {
    liveState = state;
    setSymbolLegend(lastHoverTime);   // symbolLegendHtml reads liveState directly
  }

  function closeLiveSocket() {
    if (liveReconnectTimer) { clearTimeout(liveReconnectTimer); liveReconnectTimer = null; }
    if (liveSocket) { liveSocket.onclose = null; liveSocket.close(); liveSocket = null; }
    setLiveState('off');
  }

  function applyLiveBar(bar) {
    if (!candleSeries) return;
    candleSeries.update(bar);

    // Keep the data model in sync with the chart, not just the visual —
    // otherwise a later render() (a 畫圖 toggle, another symbol switch and
    // back) would redraw from a candle list one bar stale.
    const candles = ALL.symbols[current].candles;
    const last = candles[candles.length - 1];
    if (last && last.time === bar.time) candles[candles.length - 1] = bar;
    else candles.push(bar);

    if (BAR) {
      const i = BAR.idx.has(bar.time) ? BAR.idx.get(bar.time) : BAR.candles.length;
      BAR.candles[i] = bar;
      BAR.idx.set(bar.time, i);
    }
    if (!lastHoverTime) setSymbolLegend(null);   // refresh the OHLC readout off-hover
  }

  function openLiveSocket(sym, interval) {
    closeLiveSocket();
    setLiveState('connecting');
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(proto + '//' + location.host + '/ws/bars/' +
                              encodeURIComponent(sym) + '?interval=' + interval);
    liveSocket = ws;
    ws.onmessage = e => {
      if (liveState !== 'live') setLiveState('live');
      applyLiveBar(JSON.parse(e.data));
    };
    ws.onclose = () => {
      if (liveSocket !== ws) return;   // superseded by a newer socket already
      liveSocket = null;
      setLiveState('connecting');
      liveReconnectTimer = setTimeout(() => openLiveSocket(sym, interval), LIVE_RECONNECT_MS);
    };
    ws.onerror = () => ws.close();
  }

  activateSymbol = async function activateSymbol(sym) {
    // Same reasoning as select()'s reset, and the same trap: a saved
    // logical range is bar indices, and a source/interval change is exactly
    // as much a "different bar count and spacing" event as switching
    // symbols is — carrying a 5m series' range over onto a daily series (or
    // the reverse) lands somewhere arbitrary. The chart has to go FIRST,
    // same as select() — render() re-captures savedRange from whatever
    // chart still exists on its way out, so nulling it while the old chart
    // is still alive just gets overwritten right back.
    if (chart) { chart.remove(); chart = null; }
    savedRange = null;
    savedPriceRange = null;
    const { source, interval } = sourceFor(sym);
    const adjusted = adjustedFor();
    const yfInterval = yfIntervalFor();
    syncSourcePanel();

    closeLiveSocket();

    if (source === 'yfinance') {
      const key = yfKey(sym, adjusted, yfInterval);
      if (YF_PAYLOADS[key]) {
        ALL.symbols[sym] = YF_PAYLOADS[key];
        render();
        return;
      }
      // Not preloaded — only 持有 (held) symbols are fetched eagerly at page
      // load (see sdx.serve._watchlist_and_payloads); every other watchlist
      // symbol is fetched lazily, the first time it's selected, then cached
      // here same as a held one. Only the (D, raw) variant of a held symbol
      // IS preloaded (server always seeds YF_PAYLOADS D/raw by default, see
      // yfKey), so this fetch runs for any other symbol/interval/adjusted
      // combination not yet seen.
      try {
        const res = await fetch('/api/bars/' + encodeURIComponent(sym) +
                                 '?source=yfinance&interval=' + yfInterval + '&adjusted=' + adjusted);
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        const { payload } = await res.json();
        const now = sourceFor(current);
        if (sym !== current || now.source !== 'yfinance' ||
            adjustedFor() !== adjusted || yfIntervalFor() !== yfInterval) return;
        YF_PAYLOADS[key] = payload;
        ALL.symbols[sym] = payload;
        render();
      } catch (err) {
        console.error('yfinance bars failed:', err.message);
      }
      return;
    }

    try {
      const res = await fetch('/api/bars/' + encodeURIComponent(sym) +
                               '?source=webull&interval=' + interval + '&adjusted=' + adjusted);
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const { payload } = await res.json();
      // The user may have switched symbol/source/interval/adjusted again
      // while this request was in flight — a stale response must not
      // clobber it.
      const now = sourceFor(current);
      if (sym !== current || now.source !== 'webull' || now.interval !== interval ||
          adjustedFor() !== adjusted) return;
      ALL.symbols[sym] = LADDER_INTERVALS.includes(interval) ? payload : fillLadderless(payload);
      render();
      if (!NO_LIVE_INTERVALS.includes(interval)) openLiveSocket(sym, interval);
    } catch (err) {
      console.error('webull bars failed:', err.message);
    }
  };

  srcYfBtn.addEventListener('click', () => {
    setSourceFor(current, 'yfinance', sourceFor(current).interval);
    activateSymbol(current);
  });
  srcWbBtn.addEventListener('click', () => {
    setSourceFor(current, 'webull', sourceFor(current).interval);
    activateSymbol(current);
  });
  srcIntervalSel.addEventListener('change', () => {
    setSourceFor(current, 'webull', srcIntervalSel.value);
    activateSymbol(current);
    closeAllPanels();   // interval is the last step of the flow — dismiss once chosen
  });
  priceAdjBtn.addEventListener('click', () => {
    setAdjustedFor(true);
    activateSymbol(current);
  });
  priceRawBtn.addEventListener('click', () => {
    setAdjustedFor(false);
    activateSymbol(current);
  });
  yfDBtn.addEventListener('click', () => {
    setYfIntervalFor('D');
    activateSymbol(current);
  });
  yfMBtn.addEventListener('click', () => {
    setYfIntervalFor('M');
    activateSymbol(current);
  });
  yfYBtn.addEventListener('click', () => {
    setYfIntervalFor('Y');
    activateSymbol(current);
  });

  wirePanel('sourceBtn', 'sourcePanel');
} else {
  document.getElementById('sourceBtn').style.display = 'none';
}

document.getElementById('addCancel').addEventListener('click', closeModal);
overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });
addEventListener('keydown', e => {
  if (!overlay.classList.contains('open')) return;
  if (e.key === 'Escape') closeModal();
  // Cmd/Ctrl+Enter submits — plain Enter has to stay free for multi-line input.
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) goBtn.click();
});

goBtn.addEventListener('click', async () => {
  // Tickers are conventionally upper-case (0388.HK, AAPL) — uppercase
  // whatever case the user pasted or typed so "aapl"/"Aapl" all resolve to
  // the same symbol instead of silently failing to match.
  const symbols = symsInput.value.split(/[\\s,;]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
  if (!symbols.length) return;

  goBtn.disabled = true; goBtn.textContent = '加入中…';
  results.innerHTML = '';
  try {
    const res = await fetch('/api/watchlist', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({symbols}),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const out = await res.json();

    out.added.forEach(a => {
      YF_PAYLOADS[a.symbol] = a.payload;   // /api/watchlist adds via yfinance
      ALL.symbols[a.symbol] = a.payload;
      // A newly-added symbol arrives with the server's startup params, not
      // whatever's currently in effect (§6.3 / design.md decision 5) — apply
      // the same recompute the bootstrap path uses so it doesn't silently
      // disagree with every other loaded symbol until the next reload.
      if (currentParams) recomputeIndicatorsFor(a.payload, currentParams);
      watchlist[a.symbol] = { held: false, strategies: [], stages: [], patterns: [] };
      // watchlist_layout.json self-heals against watchlists.json on its next
      // read (see sdx/watchlist_layout.py), but that reconciliation is
      // read-time only — same mirror-image fix-up removeSymbol() does on
      // delete, needed here too or the new symbol has no layout row and
      // renderWatchlistRows() (which walks `layout`, not `watchlist`) never
      // shows it until the next page load.
      layout.push({ type: 'ticker', symbol: a.symbol });
    });

    results.innerHTML =
      out.added.map(a => `<div class="ok">${a.symbol} — ${a.payload.stats.bars} bars</div>`).join('') +
      // The API names the symbol in its error too ("BOGUS.XX: no bars"), which
      // would read twice once the row is already labelled with it.
      out.failed.map(f => {
        const msg = f.error.startsWith(f.symbol + ':')
          ? f.error.slice(f.symbol.length + 1).trim() : f.error;
        return `<div class="err">${f.symbol} — ${msg}</div>`;
      }).join('');

    if (out.added.length) {
      renderWatchlistPanel();
      renderAlertsPanel();
      select(out.added[out.added.length - 1].symbol);
      symsInput.value = '';
      if (!out.failed.length) setTimeout(closeModal, 700);
    }
  } catch (err) {
    results.innerHTML = `<div class="err">${err.message}</div>`;
  } finally {
    goBtn.disabled = false; goBtn.textContent = '加入';
  }
});

// --- edit-tags modal: pill chip inputs + per-category autocomplete -----------
const TAG_CATEGORIES = ['strategies', 'stages', 'patterns'];
const tagOverlay = document.getElementById('tagOverlay');
const tagHeldBox = document.getElementById('tagHeld');
let editingSymbol = null;
let editingTags = null;
let acItems = [];
let acHighlight = -1;

function renderPills(category) {
  const pillsEl = document.getElementById('pills-' + category);
  const input = document.getElementById('input-' + category);
  pillsEl.querySelectorAll('.pill').forEach(p => p.remove());
  editingTags[category].forEach(val => {
    const pill = document.createElement('span');
    pill.className = 'pill';
    const label = document.createElement('span');
    label.textContent = val;
    const x = document.createElement('button');
    x.type = 'button';
    x.textContent = '×';
    x.addEventListener('click', () => {
      editingTags[category] = editingTags[category].filter(v => v !== val);
      renderPills(category);
    });
    pill.appendChild(label);
    pill.appendChild(x);
    pillsEl.insertBefore(pill, input);
  });
}

function commitPill(category, raw) {
  const val = raw.trim();
  if (!val || editingTags[category].includes(val)) return;
  editingTags[category].push(val);
  renderPills(category);
}

// Suggestions are scoped to this one category, annotated with how many other
// symbols carry that exact value, and exclude values already on the symbol
// being edited — see design.md decision 5.
function tagUsageCounts(category, excludeSymbol) {
  const counts = new Map();
  for (const sym of Object.keys(watchlist)) {
    if (sym === excludeSymbol) continue;
    for (const v of watchlist[sym][category] || []) counts.set(v, (counts.get(v) || 0) + 1);
  }
  return counts;
}

function updateAutocomplete(category) {
  const input = document.getElementById('input-' + category);
  const acEl = document.getElementById('ac-' + category);
  const q = input.value.trim().toLowerCase();
  const counts = tagUsageCounts(category, editingSymbol);
  acItems = [...counts.entries()]
    .filter(([val]) => !editingTags[category].includes(val))
    .filter(([val]) => !q || val.toLowerCase().includes(q))
    .sort((a, b) => b[1] - a[1]);
  acHighlight = -1;
  if (!acItems.length) { acEl.style.display = 'none'; acEl.innerHTML = ''; return; }
  acEl.innerHTML = acItems
    .map(([val, n], i) => `<div class="acItem" data-i="${i}"><span>${val}</span><span class="acCount">${n}隻股票</span></div>`)
    .join('');
  acEl.style.display = 'block';
  acEl.querySelectorAll('.acItem').forEach(el => {
    // mousedown, not click: click fires after the input's blur already hid
    // the dropdown, so a click on it would never land.
    el.addEventListener('mousedown', e => {
      e.preventDefault();
      commitPill(category, acItems[Number(el.dataset.i)][0]);
      input.value = '';
      updateAutocomplete(category);
      input.focus();
    });
  });
}

function highlightAc(category, delta) {
  if (!acItems.length) return;
  acHighlight = (acHighlight + delta + acItems.length) % acItems.length;
  document.querySelectorAll('#ac-' + category + ' .acItem').forEach((el, i) =>
    el.classList.toggle('hi', i === acHighlight));
}

TAG_CATEGORIES.forEach(category => {
  const input = document.getElementById('input-' + category);
  input.addEventListener('input', () => updateAutocomplete(category));
  input.addEventListener('focus', () => updateAutocomplete(category));
  input.addEventListener('blur', () => setTimeout(() => {
    document.getElementById('ac-' + category).style.display = 'none';
  }, 150));
  input.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); highlightAc(category, 1); return; }
    if (e.key === 'ArrowUp') { e.preventDefault(); highlightAc(category, -1); return; }
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      if (acHighlight >= 0 && acItems[acHighlight]) commitPill(category, acItems[acHighlight][0]);
      else commitPill(category, input.value);
      input.value = '';
      updateAutocomplete(category);
      return;
    }
    if (e.key === 'Backspace' && !input.value && editingTags[category].length) {
      editingTags[category].pop();
      renderPills(category);
    }
  });
});

function openTagModal(sym) {
  editingSymbol = sym;
  const tags = watchlist[sym];
  editingTags = {
    held: tags.held,
    special: tags.special,   // no modal control for this — carried through unchanged
    strategies: [...tags.strategies],
    stages: [...tags.stages],
    patterns: [...tags.patterns],
  };
  document.getElementById('tagModalTitle').textContent = sym;
  tagHeldBox.checked = editingTags.held;
  TAG_CATEGORIES.forEach(c => {
    renderPills(c);
    document.getElementById('input-' + c).value = '';
    document.getElementById('ac-' + c).style.display = 'none';
  });
  tagOverlay.classList.add('open');
  syncModalOpen();
}

function closeTagModal() {
  tagOverlay.classList.remove('open');
  syncModalOpen();
}

tagHeldBox.addEventListener('change', () => { editingTags.held = tagHeldBox.checked; });
document.getElementById('tagModalXClose').addEventListener('click', closeTagModal);
document.getElementById('tagCancel').addEventListener('click', closeTagModal);
tagOverlay.addEventListener('click', e => { if (e.target === tagOverlay) closeTagModal(); });
addEventListener('keydown', e => {
  if (e.key === 'Escape' && tagOverlay.classList.contains('open')) closeTagModal();
});

document.getElementById('tagSave').addEventListener('click', async () => {
  const saveBtn = document.getElementById('tagSave');
  saveBtn.disabled = true;
  try {
    const res = await fetch('/api/watchlist/' + encodeURIComponent(editingSymbol), {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(editingTags),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const out = await res.json();
    watchlist[editingSymbol] = out.tags;
    renderWatchlistPanel();
    renderAlertsPanel();
    closeTagModal();
  } catch (err) {
    console.error('save failed:', err.message);
  } finally {
    saveBtn.disabled = false;
  }
});

function select(sym) {
  // Already viewing this symbol AND a chart already exists — a full
  // chart.remove()/rebuild here would be pure waste and visibly flashes
  // every pane (including DMI's colored background bands redrawing from
  // scratch) for zero actual change. The `chart` check is load-bearing:
  // at boot, select(current) is called specifically to trigger the FIRST
  // render — sym === current is trivially true there (nothing has run
  // yet to make them differ), so guarding on sym === current alone skips
  // activateSymbol() forever and the chart never gets built at all.
  if (sym === current && chart) return;
  // Drop the saved zoom on a symbol switch: logical ranges are bar indices, so
  // carrying 388.HK's over to a series with a different bar count lands
  // somewhere arbitrary. Each symbol opens on its own default window.
  //
  // The chart has to go FIRST. render() re-captures the outgoing chart's range
  // on its way out, so clearing savedRange while that chart still exists just
  // gets the old value written straight back — switching 388.HK→AAPL then
  // opened AAPL on 60 bars instead of its own 9-month window.
  if (chart) { chart.remove(); chart = null; }
  savedRange = null;
  savedPriceRange = null;   // a different symbol trades on a different price scale
  current = sym;
  document.querySelectorAll('.wlRow').forEach(r =>
    r.classList.toggle('on', r.dataset.sym === sym));
  document.title = '生死線 — ' + sym;
  trendCustom = null;   // scoped to the outgoing symbol — drop it, refetch below
  refreshTrendCustomIfNeeded();
  obCustom = null;      // same — scoped to the outgoing symbol
  refreshObCustomIfNeeded();
  activateSymbol(sym);   // yfinance renders immediately; Webull fetches first
}

// Which of the two shipped ladders the 外擴K 轉角位 toggle selects.
function ladderOf(D) {
  if (!obOn) return Object.assign({}, D, D.alt.noPivot);
  // A non-default 收市比例 fetched and matching the CURRENT symbol/toggle
  // state wins over every precomputed ladder — those only ever cover the
  // default fraction. Stale (wrong symbol/toggle/fraction) falls through to
  // the precomputed ladders below, same graceful-degradation as trendSource().
  if (obCustom && obCustom.symbol === current && obCustom.bearish === obBearishOn
      && obCustom.bullish === obBullishOn && obCustom.fraction === obCloseFraction) {
    return Object.assign({}, D, obCustom);
  }
  if (obBearishOn && obBullishOn) return Object.assign({}, D, D.alt.bothOn);
  if (!obBearishOn && obBullishOn) return Object.assign({}, D, D.alt.bullishOnly);
  if (!obBearishOn && !obBullishOn) return Object.assign({}, D, D.alt.noPivot);
  return D;   // 陰燭 on, 陽燭 off — the default combination, ships as D itself
}

// 陰陽燭形態's Trend control. Regime and 5-day-at-the-default-trendBars both
// ship in every payload (see LADDER_KEYS), so those two switch instantly;
// a non-default trendBars uses whatever fetchTrendPatterns() last resolved
// for the CURRENT symbol, falling back to the 5-day default while a fetch
// is in flight or hasn't been triggered yet.
function trendSource(D) {
  if (trendMode === 'regime') {
    return { patterns: D.patterns, patternAnchor: D.patternAnchor };
  }
  if (trendBars !== 5 && trendCustom && trendCustom.symbol === current
      && trendCustom.trendBars === trendBars) {
    return trendCustom;
  }
  return { patterns: D.patterns5day, patternAnchor: D.patternAnchor5day };
}

// --- hover readout ----------------------------------------------------------
// One crosshair subscription drives both the bar panel and the pane legends.
// It is keyed on param.time rather than on a pane or a series, which is what
// makes the sub-panes work for free: hovering RSI, MACD, DMI or volume reports
// the same time as hovering the candles, so the readout follows whichever pane
// the cursor happens to be over.
const WEEKDAY = ['SUN','MON','TUE','WED','THU','FRI','SAT'];
const LEGEND_STYLE = { color:'rgba(125,135,151,0.9)', fontSize:11 };
const barPanelEl = document.getElementById('barPanel');

let BAR = null;        // per-symbol lookup tables, rebuilt with the chart
let legends = [];      // [{wm, base, parts:[{name, map, digits}]}]

// A fixed table rather than toLocaleDateString, so the label cannot change with
// the viewer's locale; getUTCDay rather than getDay, because the series keys are
// plain YYYY-MM-DD dates and a negative UTC offset rolls them back a day.
// Intraday Webull bars key on a UNIX timestamp (seconds) instead — see
// build_payload's daily=False case — handled as the second branch.
function weekday(time) {
  const d = typeof time === 'string'
    ? new Date(time + 'T00:00:00Z')
    : new Date(time * 1000);
  return WEEKDAY[d.getUTCDay()];
}

// The hover panel's first row shows this as the bar's own label — a bare
// numeric timestamp would be unreadable, so intraday bars get formatted as
// a UTC date+time instead of the daily YYYY-MM-DD string used as-is.
function formatBarTime(time) {
  if (typeof time === 'string') return time;
  return new Date(time * 1000).toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
}

function num(v, digits) {
  if (v === undefined || v === null || Number.isNaN(v)) return '\\u2014';
  return v.toLocaleString('en-US', { minimumFractionDigits: digits,
                                     maximumFractionDigits: digits });
}

const mapOf = rows => new Map(rows.map(r => [r.time, r.value]));

function buildLookups(D) {
  const idx = new Map();
  D.candles.forEach((c, i) => idx.set(c.time, i));

  const group = (rows, key) => {
    const m = new Map();
    for (const r of rows || []) {
      if (!m.has(r[key])) m.set(r[key], []);
      m.get(r[key]).push(r);
    }
    return m;
  };

  // Lookup by time, built once per render — a linear scan per mouse move would
  // walk ~1600 bars on every pixel of travel.
  BAR = {
    idx,
    candles: D.candles,
    classes: D.classes || [],
    vol: mapOf(D.volume),
    stop: mapOf(D.stop),
    pivots: group(D.pivots, 'time'),
    lines: group(D.levels, 'from'),
    signals: D.signals || {},
  };
}

function row(k, v, cls) {
  return '<div><span class="k">' + k + '</span><span class="v' +
         (cls ? ' ' + cls : '') + '">' + v + '</span></div>';
}

function showBar(time) {
  const i = BAR.idx.get(time);
  if (i === undefined) { hideBar(); return; }

  // O/H/L/C, Change, Change % and Volume live in the top-left symbol legend
  // now (setSymbolLegend), not here — this panel is for what's left: class,
  // pivots, lines, stop, signal.
  const out = [row(formatBarTime(time), weekday(time))];
  out.push('<div class="hr"></div>');

  // Labels are English; values keep the 生死線 vocabulary. 上移K, 頂/底 and
  // 量增即攻 are the words R1, R3 and R10 are written in, and translating them
  // would put a second vocabulary between the chart and the spec.
  const klass = BAR.classes[i];
  if (klass) out.push(row('Class', klass + 'K'));
  for (const p of BAR.pivots.get(time) || []) {
    out.push(row('Pivot', p.kind + ' ' + num(p.price, 2)));
  }
  for (const lv of BAR.lines.get(time) || []) {
    out.push(row('Line', lv.kind + ' ' + num(lv.price, 2)));
  }

  // Always drawn, '—' when flat. R9 makes this the only liquidation trigger, so
  // "no stop right now" is the answer rather than an absence — and a row that
  // vanished would read as a rendering bug.
  out.push(row('Stop', num(BAR.stop.get(time), 2)));

  const sig = BAR.signals[time];
  if (sig && sig.length) out.push(row('Signal', sig.join('  ')));

  barPanelEl.innerHTML = out.join('');
  barPanelEl.classList.add('on');
}

function hideBar() { barPanelEl.classList.remove('on'); }

// Values are named and pipe-separated — a single text run (watermark line
// or DOM text) carries one colour, so naming is what tells DIF from DEA;
// tinting each value to match its series, the way TradingView does, is
// not available through either mechanism.
function setLegends(time) {
  for (const L of legends) {
    let text = L.base;
    if (time !== null) {
      for (const p of L.parts) {
        const v = num(p.map.get(time), p.digits);
        text += ' | ' + (p.name ? p.name + ' ' + v : v);
      }
    }
    const domName = document.querySelector('#indHover-' + L.key + ' .indName');
    if (domName) domName.textContent = text;
    else L.wm.applyOptions({ lines: [Object.assign({ text }, LEGEND_STYLE)] });
  }
}

// --- top-left symbol/OHLC legend ---------------------------------------------
// A plain HTML overlay, not a chart watermark: the desired look mixes three
// styles on one line (bright symbol name, dim O/H/L/C labels, bright values,
// coloured change) — a watermark line only ever has one colour for its whole
// text. Sibling of #chart for the usual reason (render() tears down
// everything #chart owns; a crosshair move deliberately never calls
// render()), positioned with plain CSS since, unlike #barPanel, it's
// left-anchored and so never needs to react to the sidebar flyout narrowing
// #chart from the right.
const symbolLegendEl = document.getElementById('symbolLegend');

function symbolLegendHtml(i) {
  const c = BAR.candles[i];
  // Coloured by the change vs the previous close, not this bar's own O-vs-C —
  // those can disagree (a bar can close above its own open yet still be down
  // on the day), and a negative change shown in green would read as a bug.
  // Bar 0 has no previous close, so O/H/L/C stay neutral and there's no
  // change figure, rather than a fabricated zero/direction.
  let cls = '', chgHtml = '';
  if (i > 0) {
    const prev = BAR.candles[i - 1].close;
    const chg = c.close - prev;
    const sign = chg >= 0 ? '+' : '';
    cls = chg >= 0 ? 'pos' : 'neg';
    chgHtml = ' <span class="chg ' + cls + '">' + sign + num(chg, 2) +
              ' (' + sign + num(prev ? chg / prev * 100 : 0, 2) + '%)</span>';
  }
  const v = val => '<span class="v ' + cls + '">' + num(val, 2) + '</span>';
  // yfinance is always daily regardless of what interval value happens to
  // be remembered from a prior Webull selection (setSourceFor keeps it
  // around for next time Webull is picked) — show the interval actually in
  // effect, not the stored one. Plain text alongside the O/H/L/C letters
  // inside .slOhlc, so it picks up that same neutral colour for free.
  const { source: legendSource, interval: legendInterval } = sourceFor(current);
  const intervalLabel = legendSource === 'yfinance' ? 'D' : legendInterval;
  // Ticker moves down here, ahead of the interval it now labels, since the
  // name slot above shows the company name instead — the ticker still needs
  // to be visible somewhere. Plain text alongside intervalLabel inside
  // .slOhlc, so it picks up the same dim colour/format for free, same as
  // intervalLabel itself does.
  const ohlc = '<span class="slSym">' + current + '</span> · ' + intervalLabel + ' · O ' + v(c.open) + ' H ' + v(c.high) +
               ' L ' + v(c.low) + ' C ' + v(c.close) + chgHtml;
  // Three states: absent (not streaming), a dim static dot (connecting /
  // reconnecting), a pulsing dot + "LIVE" (receiving bar updates).
  const badge = liveState === 'off' ? '' :
    ' <span class="slLive ' + liveState + '"><span class="dot"></span>' +
    (liveState === 'live' ? 'LIVE' : '') + '</span>';
  // Falls back to the ticker when no company name was ever fetched
  // (fetch_company_name is best-effort in serve.py and can come back empty)
  // — an empty name slot would read as broken, unlike a redundant ticker.
  const displayName = (watchlist[current] && watchlist[current].name) || current;
  return '<span class="slName">' + displayName + '</span>' + badge +
         '<span class="slOhlc">' + ohlc + '</span>';
}

// Falls back to the latest bar off-hover rather than going blank — a symbol
// quote reads as broken with nothing displayed, unlike the indicator legends
// above, which have no meaningful "current" value to fall back to.
function setSymbolLegend(time) {
  if (!BAR || !BAR.candles.length) return;
  const i = (time !== null ? BAR.idx.get(time) : undefined) ??
            BAR.candles.length - 1;
  symbolLegendEl.innerHTML = symbolLegendHtml(i);
}

// --- chart ------------------------------------------------------------------
function render() {
  // Rebuild wholesale on every switch. Swapping data in place would mean
  // reconciling a variable number of level series between symbols; recreating
  // is a few ms and cannot leave a stale series behind.
  if (chart) {
    savedRange = chart.timeScale().getVisibleLogicalRange();
    // Unconditional, same as savedRange above — not gated on whether
    // autoScale was on. A toggle like 陰陽燭形態 adds text-labelled markers
    // whose on-pane primitive reserves its own vertical margin to fit the
    // labels (separate from the anchor series' own values, which
    // autoscaleInfoProvider below already excludes) — that margin alone
    // was enough to visibly grow the autoscaled range on every toggle, with
    // no zoom involved. Capturing/restoring the exact prior range this way
    // makes any 畫圖 toggle's rebuild inert to the price scale, not just a
    // manually-zoomed one.
    savedPriceRange = chart.priceScale('right').getVisibleRange();
    chart.remove();
    chart = null;
  }
  const D = ladderOf(ALL.symbols[current]);

  chart = LightweightCharts.createChart(document.getElementById('chart'), {
    // fontSize is the one global text-size knob Lightweight Charts exposes —
    // it drives every axis/tick label AND series-marker text (底背馳/頂背馳/
    // 牛差離/熊差離 have no font-size option of their own; createSeriesMarkers
    // always renders at this chart-wide size, confirmed against the bundled
    // library source). Set to 11 (default is 12) specifically to shrink
    // those marker labels; the one-point-smaller axis text is an accepted
    // side effect, not itself the goal.
    layout: { background:{color:'#0b0e14'}, textColor:'#7d8797', fontSize:11,
              panes:{separatorColor:'#1e2430'} },
    grid: { vertLines:{color:'#141922'}, horzLines:{color:'#141922'} },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor:'#1e2430' },
    timeScale: { borderColor:'#1e2430', rightOffset:8 },
  });

  const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {
    upColor:'#26a69a', downColor:'#ef5350',
    borderUpColor:'#26a69a', borderDownColor:'#ef5350',
    wickUpColor:'#26a69a', wickDownColor:'#ef5350',
    priceLineVisible:false, lastValueVisible:false,
  }, 0);
  candleSeries = candles;
  applyCandleColors();

  // GMMA — 12 EMAs of close, computed client-side with the same emaSpanJS
  // helper MACD uses. Short-term group (orange) reads compression/expansion
  // of near-term momentum; long-term group (cyan) reads the underlying
  // trend — one flat color per group, not a gradient, so the two groups
  // read as two ribbons rather than 12 individually-distinguishable lines.
  // Added before the 生死線 level lines below so levels draw on top of them.
  if (gmmaOn) {
    const gmmaCloses = D.candles.map(c => c.close);
    const gmmaTimes = D.candles.map(c => c.time);
    const GMMA_SHORT = [3, 5, 8, 10, 12, 15].map(p => [p, '#FF8D1E']);
    const GMMA_LONG = [30, 35, 40, 45, 50, 60].map(p => [p, '#13FFFF']);
    for (const [period, color] of [...GMMA_SHORT, ...GMMA_LONG]) {
      const s = chart.addSeries(LightweightCharts.LineSeries, {
        color, lineWidth: 1, lineStyle: 0,
        priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false,
      }, 0);
      const ema = emaSpanJS(gmmaCloses, period);
      s.setData(gmmaTimes.map((time, i) => ({ time, value: ema[i] })));
    }
  }

  if (drawingsOn) for (const lv of D.levels) {
    const s = chart.addSeries(LightweightCharts.LineSeries, {
      color: lv.support ? '#2e7d63' : '#a13d3d',
      lineWidth: lv.major ? 3 : 1.5,
      // Always solid: a level is drawn only once active, so there is nothing
      // inert left to distinguish.
      lineStyle: 0,
      priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false,
    }, 0);
    s.setData([{time:lv.from, value:lv.price}, {time:lv.to, value:lv.price}]);
  }

  // One markers plugin per series, so every layer goes up in a single call —
  // a second createSeriesMarkers on `candles` would replace the first. 清貨
  // is the one exception, split onto its own carrier series below: the
  // plugin's zOrder is a per-series setting, so giving 清貨 its own "top"
  // zOrder without also lifting 量增即攻/好友反攻/突破箭咀 needs a series of
  // its own to attach to.
  //
  // {autoScale: false} — the plugin's OWN default (confirmed straight from
  // the bundled library: `{autoScale:true, zOrder:"normal"}`), independent
  // of `candles`' own price data. Left at its default, the plugin reserves
  // its own vertical margin to fit whichever glyphs/labels are attached, so
  // toggling 突破箭咀/量增即攻/清貨 changed the visible price range with no
  // zoom involved — this cannot be fixed by anything on the `candles`
  // series itself (its real OHLC data must stay autoscale-eligible), only
  // by telling the MARKERS PLUGIN not to contribute its own margin.
  const LAYER_ON = { atk: () => atkOn, sdx: () => drawingsOn && arrowsOn };
  const marks = D.markers.filter(m => m.layer !== 'liq' && LAYER_ON[m.layer]());
  if (marks.length) {
    LightweightCharts.createSeriesMarkers(candles, marks, { autoScale: false });
  }

  // 清貨's own carrier: a second, fully-transparent CandlestickSeries fed the
  // exact same OHLC as `candles`, so "aboveBar" anchors to the identical position
  // — only the marker plugin's zOrder differs. zOrder:'top' is what this
  // exists for: at typical zoom a 生死線 level line (added below, its own
  // LineSeries per active level) sits at nearly the same price a 清貨 signal
  // fired against, and at the default zOrder the line painted over the
  // marker, visibly slicing the dot in half. Left off `liqOn` to skip the
  // extra series entirely when the layer isn't shown.
  if (liqOn) {
    const liqMarks = D.markers.filter(m => m.layer === 'liq');
    if (liqMarks.length) {
      // Transparent, not visible:false — a hidden series' own attached
      // markers plugin never paints either (confirmed live: visible:false
      // here silently dropped every 清貨 dot, worse than the overlap bug
      // this carrier exists to fix). Fully transparent colors keep the
      // series (and its markers primitive) actually rendering.
      const liqCarrier = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: 'rgba(0,0,0,0)', downColor: 'rgba(0,0,0,0)',
        borderUpColor: 'rgba(0,0,0,0)', borderDownColor: 'rgba(0,0,0,0)',
        wickUpColor: 'rgba(0,0,0,0)', wickDownColor: 'rgba(0,0,0,0)',
        priceLineVisible: false, lastValueVisible: false,
      }, 0);
      liqCarrier.setData(D.candles);
      LightweightCharts.createSeriesMarkers(liqCarrier, liqMarks, { autoScale: false, zOrder: 'top' });
    }
  }

  // Pattern labels hang on their own invisible series, offset from the high or
  // low, so they clear the candle. A second markers plugin on `candles` would
  // replace the first, hence the separate carrier.
  //
  // TWO anchor series, not one: a LineSeries takes one value per time, but an
  // aboveBar pattern (anchored off the high) and a belowBar one (anchored off
  // the low) can land on the SAME bar — the Pine port has no "continue" after
  // Doji the way the old detector did, so e.g. Doji + Hammer routinely
  // coincide. One shared anchor series would let whichever pattern's anchor
  // was set last for that bar win, and the OTHER pattern silently render at
  // that same (wrong-direction) height. Splitting by position keeps each
  // series to one direction, where multiple same-direction hits on one bar
  // already stack correctly (same trick 量增即攻/好友反攻 use).
  const patSource = trendSource(D);
  const patsAbove = [], patsBelow = [];
  const anchorAbove = new Map(), anchorBelow = new Map();
  patSource.patterns.forEach((p, i) => {
    if (!patOn[p.pattern]) return;
    const a = patSource.patternAnchor[i];
    if (p.position === 'aboveBar') { patsAbove.push(p); anchorAbove.set(a.time, a); }
    else { patsBelow.push(p); anchorBelow.set(a.time, a); }
  });
  const addPatternLayer = (pats, anchorByTime) => {
    if (!pats.length) return;
    const anchor = chart.addSeries(LightweightCharts.LineSeries, {
      color:'rgba(0,0,0,0)', lineWidth:1, pointMarkersVisible:false,
      priceLineVisible:false, lastValueVisible:false,
      crosshairMarkerVisible:false,
      // Excluded from the right price scale's autoscale — this series only
      // carries synthetic values (high/low offset by PATTERN_OFFSET) to
      // anchor marker positions, not a real price the axis should fit to.
      // Without this, toggling 陰陽燭形態 widened the visible range even on
      // autoScale with no manual zoom involved, the same class of jump
      // savedRange/savedPriceRange (above) fixes for an actually-zoomed view.
      autoscaleInfoProvider: () => null,
    }, 0);
    anchor.setData([...anchorByTime.values()]);
    // Belt and suspenders with the series' own autoscaleInfoProvider above:
    // the markers plugin's margin contribution (see the {autoScale:false}
    // on the atk/liq/sdx markers below) is independent of the underlying
    // series' data-based autoscale info, so both need excluding separately.
    LightweightCharts.createSeriesMarkers(anchor, pats, { autoScale: false });
  };
  addPatternLayer(patsAbove, anchorAbove);
  addPatternLayer(patsBelow, anchorBelow);

  buildSubPanes(D);                      // volume always; the rest per 只看K線

  // Off-chart is hidden, not last-bar: the panel disappears and every legend
  // reverts to its static name and parameters.
  buildLookups(D);
  hideBar();
  clearJumpIndicator();
  setLegends(null);
  setSymbolLegend(null);                 // shows the latest bar until hovered
  positionBarPanel();                    // correct even if a flyout is already open

  // The bar panel (class/pivots/lines/stop/signal) has nothing to show for
  // an interval the ladder engine never ran over — every field would just
  // read as a placeholder dash. Computed once per render() from the
  // source/interval that produced D, not from D's shape, since a real
  // ladder run can legitimately have zero pivots/lines on a short series.
  const { source: hoverSource, interval: hoverInterval } = sourceFor(current);
  const hasLadder = hoverSource === 'yfinance' || LADDER_INTERVALS.includes(hoverInterval);
  // sourceFor's `interval` is Webull's own field (see yfIntervalFor's own
  // comment on why they're kept separate) — for yfinance it's stale/always
  // 'D', so applyDefaultRange below needs the real displayed interval or a
  // yfinance M/Y view wrongly gets D's 9-month window instead of fitContent.
  const effectiveInterval = hoverSource === 'yfinance' ? yfIntervalFor() : hoverInterval;

  chart.subscribeCrosshairMove(param => {
    lastHoverTime = param.time || null;
    if (!param.time) { hideBar(); setLegends(null); setSymbolLegend(null); return; }
    if (hasLadder) showBar(param.time); else hideBar();
    setLegends(param.time);
    setSymbolLegend(param.time);
  });

  // TradingView-style double-click-to-maximize: only the main price pane
  // (paneIndex 0), not volume/RSI/MACD/DMI, toggles 只看K線 — same action as
  // clicking #toggle, so a second double-click restores the sub-panes.
  chart.subscribeDblClick(param => {
    if (param.paneIndex === 0) toggleSubPanes();
  });

  // Keeps every persisted measurement's line/box glued to its actual
  // time/price anchor as the visible range pans/zooms — measurements store
  // data values, not pixels, so without this they'd stay frozen at wherever
  // they were drawn instead of tracking the chart underneath them.
  chart.timeScale().subscribeVisibleTimeRangeChange(positionMeasurements);

  // Restore the zoom the rebuild just threw away, else fall back to the
  // default window. Every 畫圖 toggle calls render(), which recreates the
  // chart — without this, flipping 形態 off to see the levels underneath also
  // snapped the view back and you had to find your place again.
  if (savedRange) chart.timeScale().setVisibleLogicalRange(savedRange);
  else applyDefaultRange(D, effectiveInterval);

  // Same for the price scale — null only on the very first chart (nothing to
  // carry over yet). Deliberately NOT re-enabling autoScale afterward, even
  // though setVisibleRange() just turned it off: two 畫圖 toggles clicked in
  // quick succession (well within realistic testing/clicking speed, not
  // just a synthetic same-tick double-click) raced with the library's own
  // deferred autoscale recompute on the second chart, silently overriding
  // this restore a frame or two later — reliably reproduced by firing two
  // render()s with no gap, and confirmed fixed by never turning autoScale
  // back on here. A symbol switch already bypasses this entirely (`select`
  // nulls both saved* variables before tearing the chart down, so a fresh
  // chart is exactly as auto-scaled as chart creation always defaults to);
  // only a toggle-driven rebuild pins the price scale the way a manual zoom
  // already did before this was made unconditional.
  if (savedPriceRange) chart.priceScale('right').setVisibleRange(savedPriceRange);

  positionMeasurements();
  // Every render() rebuild creates a fresh `chart` with handleScroll/
  // handleScale back at their defaults (true) — resync to whatever measure
  // mode was already in effect, same reasoning as setMeasureMode() above.
  chart.applyOptions({ handleScroll: !measureMode, handleScale: !measureMode });
}

// The opening view. fitContent() over the full history squeezes ~1600 bars
// into the pane and the candles collapse into unreadable slivers, so the
// default is the recent window; the whole series is still one scroll away.
const DEFAULT_MONTHS = 9;

function applyDefaultRange(D, interval) {
  const n = D.candles.length;
  if (!n) return;
  const ts = chart.timeScale();

  // Month/Year bars are inherently long-term — a stock's whole history is
  // rarely more than a few hundred bars even at monthly granularity, and a
  // "last 9 months" cutoff is meaningless at yearly granularity (it leaves
  // 1-2 giant bars visible, not a chart). Show the full history instead,
  // same as the existing "shorter history than the window" fallback below.
  if (interval === 'M' || interval === 'Y') { ts.fitContent(); return; }

  // Intraday Webull bars carry a UNIX timestamp (seconds), not a
  // YYYY-MM-DD string (see build_payload's daily=False case) — `new
  // Date(numberOfSeconds)` would misread it as milliseconds (landing in
  // 1970), and comparing a number to the `iso` string below would silently
  // string-coerce into a meaningless lexicographic comparison. Branch on the
  // time type rather than threading a `daily` flag through from render().
  const lastTime = D.candles[n - 1].time;
  const daily = typeof lastTime === 'string';
  const cutoff = new Date(daily ? lastTime : lastTime * 1000);
  cutoff.setMonth(cutoff.getMonth() - DEFAULT_MONTHS);

  // Counted off the real dates rather than assuming bars-per-month: HK and US
  // calendars differ, and 停牌 gaps make any fixed multiple wrong.
  let first = 0;
  if (daily) {
    const iso = cutoff.toISOString().slice(0, 10);
    while (first < n && D.candles[first].time < iso) first++;
  } else {
    const cutoffSec = Math.floor(cutoff.getTime() / 1000);
    while (first < n && D.candles[first].time < cutoffSec) first++;
  }

  // Shorter history than the window — nothing to trim. Intraday bars are
  // capped by Webull's per-call bar count (a few sessions' worth at 5m
  // granularity), so this is the common case there, not just an edge case.
  if (first === 0) { ts.fitContent(); return; }

  // +3 leaves the newest bar off the right edge instead of flush against it.
  ts.setVisibleLogicalRange({ from: first, to: n - 1 + 3 });
}

// Recolour 外擴K / 內困K in place so classification is auditable by eye. Toggled
// off, the candles revert to plain up/down — the levels and arrows are easier
// to read without the extra hues competing.
function applyCandleColors() {
  if (!candleSeries) return;
  const D = ALL.symbols[current];
  if (!classColorOn) { candleSeries.setData(D.candles); return; }
  const byTime = new Map(D.classOverlay.map(o => [o.time, o.color]));
  candleSeries.setData(D.candles.map(c => {
    const col = byTime.get(c.time);
    return col ? {...c, color:col, borderColor:col, wickColor:col} : c;
  }));
}

// 畫圖 menu. Four independent layers over the price pane:
//   生死線             levels + ↑↓ arrows (arrows separable via optArrows)
//   量增即攻+好友反攻    the entry triggers, split out so they can be read alone
//   陰陽燭形態          single-bar reversal candles
//   K線著色             外擴K/內困K tinting — off by default, it competes with the rest
// Only K線著色 avoids a rebuild, since nothing but the candle bodies change.
document.getElementById('optSdx').addEventListener('change', e => {
  drawingsOn = e.target.checked;
  // The arrows sub-toggle only means anything while 生死線 itself is on.
  document.getElementById('optArrows').disabled = !drawingsOn;
  render();
});
document.getElementById('optArrows').addEventListener('change', e => {
  arrowsOn = e.target.checked; render();
});
document.getElementById('optLiq').addEventListener('change', e => {
  liqOn = e.target.checked; render();
});
document.getElementById('optOb').addEventListener('change', e => {
  obOn = e.target.checked;
  // The two sub-toggles and the fraction input only mean anything while the
  // master is on.
  document.getElementById('optObBearish').disabled = !obOn;
  document.getElementById('optObBullish').disabled = !obOn;
  document.getElementById('obFraction').disabled = !obOn;
  refreshObCustomIfNeeded();
  render();
});
document.getElementById('optObBearish').addEventListener('change', e => {
  obBearishOn = e.target.checked;
  refreshObCustomIfNeeded();
  render();
});
document.getElementById('optObBullish').addEventListener('change', e => {
  obBullishOn = e.target.checked;
  refreshObCustomIfNeeded();
  render();
});
document.getElementById('obFraction').addEventListener('change', e => {
  const v = parseFloat(e.target.value);
  if (!isFinite(v) || v < 0 || v > 1) { e.target.value = obCloseFraction; return; }
  obCloseFraction = v;
  saveObFraction();
  refreshObCustomIfNeeded();
  render();
});
document.getElementById('optAtk').addEventListener('change', e => {
  atkOn = e.target.checked; render();
});
// 陰陽燭形態 is a parent over 單日/雙日/三日/五日 (kind), and each kind is in
// turn a parent over its own patterns — a three-level indeterminate-parent
// tree (leaf checkboxes built from PATTERNS, one per Pattern, see
// buildPatternMenu()), one level deeper than the old 陰陽燭形態→單日/雙日/
// 三日/五日 tree this replaces. patOn (declared above) is the one source
// of truth; every checkbox at every level is just a view onto it.
const optPat = document.getElementById('optPat');
const KINDS = ['單日', '雙日', '三日', '五日'];
const patternsByKind = Object.fromEntries(
  KINDS.map(k => [k, PATTERNS.filter(p => p.kind === k)])
);

// Leaf checkboxes are grouped by (kind, label) rather than one-per-Pattern:
// most patterns already have a unique `p.zh` (PATTERN_CATALOG appends
// （看好）/（看淡） in Python wherever a shared name has a direction to
// disambiguate with — see PATTERN_CATALOG's own comment), but a
// direction-neutral pair like 陀螺 (Spinning Top White/Black) has no
// direction to append and would otherwise show as two identical,
// unexplained "陀螺" rows. Grouping by label — not a hardcoded pattern
// list — collapses any such case into one checkbox that toggles every
// member together, so this stays correct if a future pattern is ever added
// under a name that collides with an existing neutral one.
const patternGroupsByKind = Object.fromEntries(
  KINDS.map(k => {
    const groups = [];
    const byZh = new Map();
    patternsByKind[k].forEach(p => {
      if (!byZh.has(p.zh)) { const g = { zh: p.zh, members: [] }; byZh.set(p.zh, g); groups.push(g); }
      byZh.get(p.zh).members.push(p);
    });
    return [k, groups];
  })
);

// Each kind's pattern list starts collapsed — 28 leaf checkboxes fully
// expanded would dominate 畫圖 on open. The disclosure arrow is a separate
// control from the kind checkbox: clicking the arrow only shows/hides that
// kind's patterns, it must never also flip which patterns are on, and
// clicking the checkbox must never also expand the list just to check it.
function buildPatternMenu() {
  const menu = document.getElementById('patMenu');
  menu.innerHTML = KINDS.map(k => `
    <div class="patKindRow">
      <label class="opt sub"><input type="checkbox" data-kind="${k}"> ${k}</label>
      <button type="button" class="patKindToggle" data-kind-toggle="${k}" aria-expanded="false" aria-label="展開">&#9664;</button>
    </div>
    <div class="patKindChildren" data-kind-children="${k}" style="display:none">
      ${patternGroupsByKind[k].map(g => `
        <label class="opt sub2"><input type="checkbox" data-pattern-group="${g.members.map(p => p.value).join(',')}"> ${g.zh}</label>
      `).join('')}
    </div>
  `).join('');

  menu.querySelectorAll('input[data-pattern-group]').forEach(cb => {
    cb.addEventListener('change', e => {
      e.target.dataset.patternGroup.split(',').forEach(v => { patOn[v] = e.target.checked; });
      syncPatParents();
      render();
    });
  });
  menu.querySelectorAll('input[data-kind]').forEach(cb => {
    cb.addEventListener('change', e => {
      const k = e.target.dataset.kind;
      patternsByKind[k].forEach(p => { patOn[p.value] = e.target.checked; });
      syncPatParents();
      render();
    });
  });
  menu.querySelectorAll('button[data-kind-toggle]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const k = btn.dataset.kindToggle;
      const children = menu.querySelector(`[data-kind-children="${k}"]`);
      const expanded = children.style.display !== 'none';
      children.style.display = expanded ? 'none' : 'block';
      btn.setAttribute('aria-expanded', String(!expanded));
      btn.setAttribute('aria-label', expanded ? '展開' : '收起');
    });
  });
}

// Recomputes every checkbox's checked/indeterminate from patOn — called
// after ANY pattern/kind/top-level change, not just the one that fired,
// since a leaf change can flip its kind parent's and 陰陽燭形態's state too.
function syncPatParents() {
  const menu = document.getElementById('patMenu');
  let anyOn = false, allOn = true;
  KINDS.forEach(k => {
    const vals = patternsByKind[k].map(p => patOn[p.value]);
    const kindAllOn = vals.every(Boolean);
    const kindAnyOn = vals.some(Boolean);
    if (kindAnyOn) anyOn = true;
    if (!kindAllOn) allOn = false;
    const kindBox = menu.querySelector(`input[data-kind="${k}"]`);
    kindBox.checked = kindAllOn;
    kindBox.indeterminate = kindAnyOn && !kindAllOn;
    patternGroupsByKind[k].forEach(g => {
      const groupVals = g.members.map(p => patOn[p.value]);
      const box = menu.querySelector(`input[data-pattern-group="${g.members.map(p => p.value).join(',')}"]`);
      box.checked = groupVals.every(Boolean);
      box.indeterminate = groupVals.some(Boolean) && !groupVals.every(Boolean);
    });
  });
  optPat.checked = allOn;
  optPat.indeterminate = anyOn && !allOn;
}

optPat.addEventListener('change', e => {
  PATTERNS.forEach(p => { patOn[p.value] = e.target.checked; });
  syncPatParents();
  render();
});

buildPatternMenu();
syncPatParents();

// Trend mode/bars — see trendSource() for how these pick a pattern set.
// Regime and 5-day-at-the-default both ship in every payload; only a
// non-default "Trend in Bars" needs a round trip, so this is the one
// 畫圖-menu control (besides Data Source) that talks to the server.
const trendModeSel = document.getElementById('trendMode');
const trendBarsInput = document.getElementById('trendBars');
const TREND_KEY = 'sdx.trend';

function loadSavedTrend() {
  try {
    const raw = localStorage.getItem(TREND_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}
function saveTrend() {
  try { localStorage.setItem(TREND_KEY, JSON.stringify({ trendMode, trendBars })); }
  catch (e) { /* falls back to session-only, silently */ }
}

// A static (non-LIVE) page has no server to answer /api/patterns — same
// graceful-degradation as every other served-only affordance here, falling
// back to the 5-day default view (see trendSource()) rather than erroring.
async function fetchTrendPatterns(sym, bars) {
  if (!LIVE) return;
  try {
    const res = await fetch('/api/patterns/' + encodeURIComponent(sym) + '?trend_bars=' + bars);
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    // Stale if the symbol or the inputs moved on while this was in flight.
    if (sym !== current || trendMode !== '5day' || trendBars !== bars) return;
    trendCustom = { symbol: sym, trendBars: bars, patterns: data.patterns, patternAnchor: data.patternAnchor };
    // A newly-selected symbol's own bars may still be in flight (lazy fetch
    // — see activateSymbol()) if this resolves first; that fetch's own
    // render() call will pick up trendCustom, already set above, once it
    // lands. Rendering now would crash on ALL.symbols[sym] being absent.
    if (ALL.symbols[sym]) render();
  } catch (err) {
    console.error('trend patterns fetch failed:', err.message);
  }
}

function refreshTrendCustomIfNeeded() {
  if (trendMode === '5day' && trendBars !== 5) fetchTrendPatterns(current, trendBars);
}

const OB_FRACTION_KEY = 'sdx.obFraction';

function loadSavedObFraction() {
  try {
    const raw = localStorage.getItem(OB_FRACTION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}
function saveObFraction() {
  try { localStorage.setItem(OB_FRACTION_KEY, JSON.stringify({ obCloseFraction })); }
  catch (e) { /* falls back to session-only, silently */ }
}

// A static (non-LIVE) page has no server to answer /api/ladder — same
// graceful-degradation as fetchTrendPatterns(), falling back to the default
// fraction's precomputed ladder (see ladderOf()) rather than erroring.
async function fetchCustomLadder(sym, bearish, bullish, fraction) {
  if (!LIVE) return;
  try {
    const url = '/api/ladder/' + encodeURIComponent(sym)
      + '?bearish=' + bearish + '&bullish=' + bullish + '&close_fraction=' + fraction;
    const res = await fetch(url);
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    // Stale if the symbol or the inputs moved on while this was in flight.
    if (sym !== current || obBearishOn !== bearish || obBullishOn !== bullish
        || obCloseFraction !== fraction) return;
    obCustom = Object.assign({ symbol: sym, bearish, bullish, fraction }, data);
    if (ALL.symbols[sym]) render();
  } catch (err) {
    console.error('custom ladder fetch failed:', err.message);
  }
}

function refreshObCustomIfNeeded() {
  if (obOn && obCloseFraction !== 0.6) {
    fetchCustomLadder(current, obBearishOn, obBullishOn, obCloseFraction);
  } else {
    obCustom = null;
  }
}

trendModeSel.addEventListener('change', e => {
  trendMode = e.target.value;
  trendBarsInput.disabled = trendMode !== '5day';
  saveTrend();
  refreshTrendCustomIfNeeded();
  render();
});
trendBarsInput.addEventListener('change', e => {
  const n = parseInt(e.target.value, 10);
  if (!Number.isInteger(n) || n < 1) { e.target.value = trendBars; return; }
  trendBars = n;
  saveTrend();
  refreshTrendCustomIfNeeded();
  render();
});
document.getElementById('optClass').addEventListener('change', e => {
  classColorOn = e.target.checked; applyCandleColors();
});
document.getElementById('optGmma').addEventListener('change', e => {
  gmmaOn = e.target.checked; render();
});
// RSI背馳/MACD背馳/DMI背景 toggles moved into each indicator's own settings
// modal (IND_META rows' `toggle` key, wired in renderIndModalBody below) —
// 畫圖 is for toggles that change the main bar chart, not sub-panes.

// 只看K線 keeps price AND volume — volume is part of reading the bars (量增即攻,
// 量價背馳), not an indicator overlay. Only RSI/MACD/DMI are dropped.
function buildSubPanes(D) {
  const I = D.indicators, LBL = D.labels;
  subSeries = {};   // reset — the chart (and every series on it) is fresh

  // Volume/RSI/MACD/DMI all render their legend into the real .indName DOM
  // element (see the CSS comment above .indNameHover for why — a canvas
  // watermark can end up visually covered by divergence labels / background
  // bands). Any future pane with no such box falls back to the
  // canvas-watermark path below; the watermark handle is retained there so
  // the label can be rewritten per crosshair move via applyOptions() rather
  // than torn down and recreated.
  legends = [];
  const addLegend = (paneIdx, key, base, parts) => {
    const hasDomLabel = !!document.querySelector('#indHover-' + key + ' .indName');
    const wm = hasDomLabel ? null : LightweightCharts.createTextWatermark(chart.panes()[paneIdx], {
      horzAlign:'left', vertAlign:'top',
      lines:[Object.assign({ text: base }, LEGEND_STYLE)],
    });
    legends.push({ key, wm, base, parts });
  };

  const line = (pane, data, color, width) => {
    const s = chart.addSeries(LightweightCharts.LineSeries,
      { color, lineWidth: width || 1.5, priceLineVisible:false,
        lastValueVisible:false, crosshairMarkerVisible:false }, pane);
    s.setData(data);
    return s;
  };

  chart.addPane();
  const vol = chart.addSeries(LightweightCharts.HistogramSeries,
    { priceFormat:{type:'volume'}, priceLineVisible:false }, 1);
  vol.setData(D.volume);
  // MAVOL — defaults to RALLY_VOLUME_MA (50), the same trailing average
  // 好友反攻 checks 大量 against (see sdx/candles.py), drawn on the shared
  // volume price scale so that threshold reads directly off the chart
  // instead of being implicit in the signal alone. Period/color are
  // user-editable via this pane's own gear-icon settings (IND_META.volume
  // below) the same way RSI/MACD/DMI are — that only ever changes what's
  // drawn, never RALLY_VOLUME_MA itself, so 好友反攻 keeps firing off the
  // fixed 50-bar average it was calibrated against.
  const volMa = line(1, D.volumeMa, currentParams.volume.colors.ma, 1.5);
  subSeries.volume = { ma: volMa };
  updateVolumeOverlay(D);
  addLegend(1, 'volume', LBL.volume, [
    { name:'', map: mapOf(D.volume), digits: 0 },
    { name:'MA', map: mapOf(D.volumeMa), digits: 0 },
  ]);
  // Volume's own hover box is independent of subPanesOn — pane 1 exists in
  // both view modes, unlike RSI/MACD/DMI's panes 2-4 — so it is shown and
  // labeled here unconditionally rather than folding into the two
  // RSI/MACD/DMI-only loops below.
  const volHover = document.getElementById('indHover-volume');
  volHover.style.display = 'flex';
  volHover.querySelector('.indName').textContent = LBL.volume;

  if (!subPanesOn) {                     // volume only
    chart.panes()[0].setStretchFactor(6.5);
    chart.panes()[1].setStretchFactor(1.4);
    closeIndModal();
    ['rsi', 'macd', 'dmi'].forEach(ind =>
      document.getElementById('indHover-' + ind).style.display = 'none');
    requestAnimationFrame(positionIndHovers);
    return;
  }

  chart.addPane();                       // 2 — RSI
  const rsiColors = currentParams.rsi.colors;

  // Overbought/oversold background: two Area series (Lightweight Charts has
  // no built-in pane background fill), each covering the full pane height
  // for a bar where that condition holds, added BEFORE the RSI/signal lines
  // so those lines render on top of the tint. Area, not Histogram: a
  // Histogram draws discrete gapped bars (reads as blocky columns), where
  // Area draws one continuous filled polygon between consecutive points —
  // no gap artifacts. `lineType: WithSteps` (not the default straight-line
  // interpolation) is required, not cosmetic: with plain linear
  // interpolation, a single isolated triggered bar ramps from its 0-value
  // neighbors up to 100 and back down, rendering as a pointy triangle that
  // also smears color into the untriggered bars on either side of it
  // (visible live, flagged by the user). Steps hold each bar's own value
  // flat until the next bar's own boundary, so a triggered run — even a
  // run of one — renders as a flat-topped rectangle confined to exactly
  // its own bar(s), no bleed into neighbors. No `baseValue` option needed:
  // Area fills down to the bottom of the visible price range by default,
  // and that range is pinned to a fixed [0,100] below, so "bottom" is
  // already exactly 0 — required both for a stable coordinate space and so
  // the shading never itself drives, or is driven by, the pane's
  // autoscale (see this project's CLAUDE.md on the autoScale-drift class
  // of bug).
  const rsiOverSeries = chart.addSeries(LightweightCharts.AreaSeries,
    { lineVisible:false, priceLineVisible:false, lastValueVisible:false,
      crosshairMarkerVisible:false, pointMarkersVisible:false,
      lineType: LightweightCharts.LineType.WithSteps }, 2);
  const rsiUnderSeries = chart.addSeries(LightweightCharts.AreaSeries,
    { lineVisible:false, priceLineVisible:false, lastValueVisible:false,
      crosshairMarkerVisible:false, pointMarkersVisible:false,
      lineType: LightweightCharts.LineType.WithSteps }, 2);
  // scaleMargins zeroed too — Lightweight Charts pads a fixed range with
  // its own default top/bottom margin regardless of autoScale, so without
  // this the fill (and the 0/100 ends of the pinned range generally) stop
  // short of the pane's actual top/bottom edge instead of reaching it.
  rsiOverSeries.priceScale().applyOptions({ autoScale: false, scaleMargins: { top: 0, bottom: 0 } });
  rsiOverSeries.priceScale().setVisibleRange({ from: 0, to: 100 });

  const rsiSeries = line(2, I.rsi, rsiColors.line, 1.5);
  const rsiSignalSeries = line(2, I.rsiSignal, rsiColors.signalLine, 1);

  // RSI背馳 — connecting line + midpoint label per confirmed divergence pair.
  // The connecting lines are NOT one shared series: Lightweight Charts has
  // no "break the line here" primitive (a `{time}`-only whitespace entry
  // does not create a visual gap in a LineSeries — it is simply skipped,
  // and the line still connects the nearest real points on either side).
  // The correct precedent already in this file is the ladder support/
  // resistance levels below (`for (const lv of D.levels) { ... }` in
  // render()) — one dedicated 2-point series per segment. updateRsiOverlay
  // creates/destroys these dynamically into `divLines`; only the anchor
  // series for labels stays fixed here, since markers don't have this
  // connecting-line problem (see the 陰陽燭形態 anchor pattern below).
  const rsiDivAnchorBull = chart.addSeries(LightweightCharts.LineSeries, {
    color:'rgba(0,0,0,0)', lineWidth:1, pointMarkersVisible:false,
    priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false,
    autoscaleInfoProvider: () => null,
  }, 2);
  const rsiDivAnchorBear = chart.addSeries(LightweightCharts.LineSeries, {
    color:'rgba(0,0,0,0)', lineWidth:1, pointMarkersVisible:false,
    priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false,
    autoscaleInfoProvider: () => null,
  }, 2);
  const rsiDivMarkersBull = LightweightCharts.createSeriesMarkers(rsiDivAnchorBull, [], { autoScale: false });
  const rsiDivMarkersBear = LightweightCharts.createSeriesMarkers(rsiDivAnchorBear, [], { autoScale: false });

  // 超買 / 超賣 reference levels — read from the same configurable thresholds
  // driving the shading above, not a hardcoded 25/75. Handles kept so a
  // later threshold change (gear modal) can reposition them in place.
  const rsiOversoldLine = rsiSeries.createPriceLine({
    price: currentParams.rsi.oversold,
    color: hexToRgba(rsiColors.oversoldLineColor, rsiColors.oversoldLineOpacity), lineWidth:1,
    lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible:true,
  });
  const rsiOverboughtLine = rsiSeries.createPriceLine({
    price: currentParams.rsi.overbought,
    color: hexToRgba(rsiColors.overboughtLineColor, rsiColors.overboughtLineOpacity), lineWidth:1,
    lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible:true,
  });

  subSeries.rsi = {
    main: rsiSeries, signal: rsiSignalSeries,
    over: rsiOverSeries, under: rsiUnderSeries,
    oversoldLine: rsiOversoldLine, overboughtLine: rsiOverboughtLine,
    divLines: [],   // one 2-point LineSeries per confirmed divergence pair
    divAnchorBull: rsiDivAnchorBull, divAnchorBear: rsiDivAnchorBear,
    divMarkersBull: rsiDivMarkersBull, divMarkersBear: rsiDivMarkersBear,
  };
  updateRsiOverlay(D);

  chart.addPane();                       // 3 — MACD
  // A newly-added pane's default 'right' price scale inherits whatever
  // options were most recently applied via applyOptions() to an
  // earlier-created pane's 'right' scale, rather than the library's own
  // built-in default — confirmed live (chart.priceScale('right', 3)
  // .options() reads back RSI's exact {autoScale:false, scaleMargins:
  // {top:0,bottom:0}} pinning, applied two panes earlier, on a pane whose
  // own code never touches autoScale at all). Left alone, MACD's price
  // scale is stuck at RSI's pinned range until the user double-clicks it
  // (Lightweight Charts' own autoScale-reset gesture) — explicitly restore
  // the library's normal autoscaling defaults here rather than relying on
  // whatever the previous pane happened to leave behind.
  chart.priceScale('right', 3).applyOptions({ autoScale: true, scaleMargins: { top: 0.2, bottom: 0.1 } });
  const hist = chart.addSeries(LightweightCharts.HistogramSeries,
    { priceLineVisible:false, lastValueVisible:false }, 3);
  const difSeries = line(3, I.dif, currentParams.macd.colors.dif, 1.5);
  const deaSeries = line(3, I.dea, currentParams.macd.colors.dea, 1);

  // MACD背馳/差離 — connecting line + midpoint label per confirmed divergence
  // pair, both the MACD-line (底背馳/頂背馳) and histogram (牛差離/熊差離)
  // variants sharing one pair of anchor/marker series since they share one
  // toggle and one Bullish/Bearish color pair, distinguished only by label
  // text. Same one-series-per-pair rendering as the RSI pane (see this
  // file's `divLines` comment above for why a shared whitespace-gapped
  // series doesn't work).
  const macdDivAnchorBull = chart.addSeries(LightweightCharts.LineSeries, {
    color:'rgba(0,0,0,0)', lineWidth:1, pointMarkersVisible:false,
    priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false,
    autoscaleInfoProvider: () => null,
  }, 3);
  const macdDivAnchorBear = chart.addSeries(LightweightCharts.LineSeries, {
    color:'rgba(0,0,0,0)', lineWidth:1, pointMarkersVisible:false,
    priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false,
    autoscaleInfoProvider: () => null,
  }, 3);
  const macdDivMarkersBull = LightweightCharts.createSeriesMarkers(macdDivAnchorBull, [], { autoScale: false });
  const macdDivMarkersBear = LightweightCharts.createSeriesMarkers(macdDivAnchorBear, [], { autoScale: false });

  subSeries.macd = {
    hist, dif: difSeries, dea: deaSeries,
    divLines: [],   // one 2-point LineSeries per confirmed divergence pair
    divAnchorBull: macdDivAnchorBull, divAnchorBear: macdDivAnchorBear,
    divMarkersBull: macdDivMarkersBull, divMarkersBear: macdDivMarkersBear,
  };
  updateMacdOverlay(D);

  chart.addPane();                       // 4 — DMI
  // Same inherited-defaults fix as MACD's pane above, same reason: without
  // this, DMI's price scale is also stuck at RSI's pinned range on load.
  chart.priceScale('right', 4).applyOptions({ autoScale: true, scaleMargins: { top: 0.2, bottom: 0.1 } });
  const dmiColors = currentParams.dmi.colors;

  // DMI背景 — day-over-day ADX-change band, full pane height. Added BEFORE
  // the +DI/-DI/ADX lines below so those lines render on top of the tint,
  // not under it (series added later draw over series added earlier —
  // the same ordering RSI's shading uses relative to its RSI/signal
  // lines). This pane's price scale is NOT pinned — it keeps its natural
  // line-driven autoscale (a deliberate choice from the original design;
  // see design.md's Non-Goals) — but the bands themselves are still filled
  // to fixed sentinel bounds (DMI_BAND_BOTTOM/TOP, defined above
  // computeDmiBandArrays()) rather than the pane's actual current top/bottom
  // — see redrawDmiBands()'s own comment for why chasing the real bounds
  // caused a flicker on every 畫圖 toggle.
  //
  // Two Area series — bottom-of-range default baseline fits a full-height
  // band exactly. autoscaleInfoProvider null on both so they never feed
  // back into the very autoscale they read from — that feedback loop is
  // the one thing that would make this pane's scale unstable. (An earlier
  // version also had a red top-1/5 "ADX extreme" band via a Baseline
  // series; removed per user feedback — hard to read, and the 40
  // reference line already conveys the same information.)
  const dmiRisingSeries = chart.addSeries(LightweightCharts.AreaSeries, {
    lineVisible:false, priceLineVisible:false, lastValueVisible:false,
    crosshairMarkerVisible:false, pointMarkersVisible:false,
    lineType: LightweightCharts.LineType.WithSteps,
    autoscaleInfoProvider: () => null,
  }, 4);
  const dmiFallingSeries = chart.addSeries(LightweightCharts.AreaSeries, {
    lineVisible:false, priceLineVisible:false, lastValueVisible:false,
    crosshairMarkerVisible:false, pointMarkersVisible:false,
    lineType: LightweightCharts.LineType.WithSteps,
    autoscaleInfoProvider: () => null,
  }, 4);

  const pdiSeries = line(4, I.pdi, dmiColors.pdi, 1);
  const mdiSeries = line(4, I.mdi, dmiColors.mdi, 1);
  const adxSeries = line(4, I.adx, dmiColors.adx, 1.5);

  // Reference lines, value AND color independently configurable per line
  // (unlike RSI's 25/75 lines, which share one fixed gray) — same Dashed
  // style as RSI's lines. Handles kept so updateDmiOverlay() can
  // reposition/recolor them in place on a gear-modal edit. Kept even
  // though the red "extreme" band was removed — the upper line is still a
  // standard ADX reading threshold worth marking on its own.
  const dmiLowerLine = adxSeries.createPriceLine({
    price: currentParams.dmi.lowerLevel,
    color: hexToRgba(dmiColors.lowerLevelColor, dmiColors.lowerLevelOpacity),
    lineWidth:1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible:true,
  });
  const dmiUpperLine = adxSeries.createPriceLine({
    price: currentParams.dmi.upperLevel,
    color: hexToRgba(dmiColors.upperLevelColor, dmiColors.upperLevelOpacity),
    lineWidth:1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible:true,
  });

  subSeries.dmi = {
    pdi: pdiSeries, mdi: mdiSeries, adx: adxSeries,
    rising: dmiRisingSeries, falling: dmiFallingSeries,
    lowerLine: dmiLowerLine, upperLine: dmiUpperLine,
  };
  updateDmiOverlay(D);

  // Stretch factors, not setHeight: setHeight is absolute, so consecutive calls
  // fight over the same total and the sub-panes end up far taller than asked.
  chart.panes()[0].setStretchFactor(6.5);
  chart.panes()[1].setStretchFactor(1.4);
  [2, 3, 4].forEach(i => chart.panes()[i].setStretchFactor(1.8));

  addLegend(2, 'rsi', LBL.rsi, [
    { name:'RSI', map: mapOf(I.rsi), digits: 2 },
    { name:'SMA', map: mapOf(I.rsiSignal), digits: 2 },
  ]);
  addLegend(3, 'macd', LBL.macd, [
    { name:'DIF', map: mapOf(I.dif), digits: 2 },
    { name:'DEA', map: mapOf(I.dea), digits: 2 },
    { name:'HIST', map: mapOf(I.hist), digits: 2 },
  ]);
  addLegend(4, 'dmi', LBL.dmi, [
    { name:'PDI', map: mapOf(I.pdi), digits: 2 },
    { name:'MDI', map: mapOf(I.mdi), digits: 2 },
    { name:'ADX', map: mapOf(I.adx), digits: 2 },
  ]);

  ['rsi', 'macd', 'dmi'].forEach(ind => {
    const hover = document.getElementById('indHover-' + ind);
    hover.style.display = 'flex';
    hover.querySelector('.indName').textContent = LBL[ind];
  });

  // Deferred one frame: getHTMLElement() on a pane added earlier in this same
  // synchronous call returns null until the browser has laid out its row —
  // confirmed live, calling this inline (no rAF) silently no-ops every time.
  requestAnimationFrame(positionIndHovers);
}

// Same role as updateRsiOverlay/updateMacdOverlay/updateDmiOverlay below,
// for the volume pane's own MA line — color, opacity and on/off all live in
// currentParams.volume, this just re-applies them in place. Unlike those
// three this pane always exists regardless of subPanesOn, but subSeries.volume
// is only ever populated inside buildSubPanes, so the guard is still needed
// during the brief window before the first buildSubPanes() call completes.
function updateVolumeOverlay(D) {
  const S = subSeries.volume;
  if (!S) return;
  const colors = currentParams.volume.colors;
  S.ma.applyOptions({ color: hexToRgba(colors.ma, colors.maOpacity), visible: volumeMaOn });
}

// Recomputes and re-applies everything the RSI pane's gear-modal controls
// affect — thresholds, colors, and the overlays derived from them — from
// `currentParams` and the given payload's current indicator data. Shared by
// buildSubPanes() (initial paint) and pushIndicatorToChart() (in-place edit,
// no chart rebuild), so both stay identical by construction rather than by
// convention. A no-op if the RSI pane doesn't exist (只看K線 mode).
function updateRsiOverlay(D) {
  const S = subSeries.rsi;
  if (!S) return;
  const colors = currentParams.rsi.colors;
  const overbought = currentParams.rsi.overbought;
  const oversold = currentParams.rsi.oversold;

  S.main.applyOptions({ color: hexToRgba(colors.line, colors.lineOpacity) });
  S.signal.applyOptions({ color: hexToRgba(colors.signalLine, colors.signalLineOpacity), visible: rsiMaOn });
  S.oversoldLine.applyOptions({
    price: oversold, color: hexToRgba(colors.oversoldLineColor, colors.oversoldLineOpacity),
    lineVisible: rsiOversoldOn, axisLabelVisible: rsiOversoldOn,
  });
  S.overboughtLine.applyOptions({
    price: overbought, color: hexToRgba(colors.overboughtLineColor, colors.overboughtLineOpacity),
    lineVisible: rsiOverboughtOn, axisLabelVisible: rsiOverboughtOn,
  });

  // Flat fill (top/bottom colors identical — no gradient) at the swatch's
  // own color+opacity, full pane height (0 or 100) for a bar where that
  // threshold is crossed, 0 otherwise. Every bar gets an explicit value —
  // not just the bars that qualify — so the Area series has one contiguous
  // array to draw a continuous polygon from; a sparse array here would
  // bridge unrelated occurrences together the same way it did for the
  // divergence lines (see that decision above) before being fixed to use
  // per-pair series instead.
  const bearFill = hexToRgba(colors.bearish, colors.bearishOpacity);
  const bullFill = hexToRgba(colors.bullish, colors.bullishOpacity);
  S.over.applyOptions({ topColor: bearFill, bottomColor: bearFill });
  S.under.applyOptions({ topColor: bullFill, bottomColor: bullFill });
  const overData = [], underData = [];
  D.indicators.rsi.forEach(p => {
    overData.push({ time: p.time, value: p.value >= overbought ? 100 : 0 });
    underData.push({ time: p.time, value: p.value <= oversold ? 100 : 0 });
  });
  S.over.setData(overData);
  S.under.setData(underData);

  // Divergence connecting lines are rebuilt wholesale on every call, same
  // "tear down and recreate rather than reconcile" philosophy render()
  // itself uses — simpler than diffing which pairs changed, and cheap
  // enough at chart-data scale (see the comment on `divLines` above).
  S.divLines.forEach(s => chart.removeSeries(s));
  S.divLines = [];
  S.divAnchorBull.setData([]);
  S.divAnchorBear.setData([]);
  S.divMarkersBull.setMarkers([]);
  S.divMarkersBear.setMarkers([]);

  if (!rsiDivOn) return;

  const rsiByTime = new Map(D.indicators.rsi.map(p => [p.time, p.value]));
  const candles = D.candles;
  const rsiAligned = candles.map(c => {
    const v = rsiByTime.get(c.time);
    return v === undefined ? null : v;
  });
  const { bull, bear } = computeRsiDivergence(candles, rsiAligned);

  const addDivLine = (pair, color) => {
    const s = chart.addSeries(LightweightCharts.LineSeries, {
      color, lineWidth:2, priceLineVisible:false, lastValueVisible:false,
      crosshairMarkerVisible:false, pointMarkersVisible:false,
    }, 2);
    s.setData([
      { time: candles[pair.p1.idx].time, value: pair.p1.value },
      { time: candles[pair.p2.idx].time, value: pair.p2.value },
    ]);
    S.divLines.push(s);
  };
  bull.forEach(p => addDivLine(p, colors.bullish));
  bear.forEach(p => addDivLine(p, colors.bearish));

  const bullAnchors = bull.map(p => divergenceMidpoint(p, candles));
  const bearAnchors = bear.map(p => divergenceMidpoint(p, candles));
  S.divAnchorBull.setData(bullAnchors);
  S.divAnchorBear.setData(bearAnchors);
  S.divMarkersBull.setMarkers(bullAnchors.map(a => ({
    time: a.time, position:'belowBar', color: colors.bullish, shape:'arrowUp', size:0, text:'底背馳',
  })));
  S.divMarkersBear.setMarkers(bearAnchors.map(a => ({
    time: a.time, position:'aboveBar', color: colors.bearish, shape:'arrowDown', size:0, text:'頂背馳',
  })));
}

// Same role as updateRsiOverlay() above, for the MACD pane: colors, the
// histogram's per-bar sign-based recolor, and both divergence overlays
// (MACD-line + histogram) sharing one Bullish/Bearish anchor/marker pair.
// A no-op if the MACD pane doesn't exist (只看K線 mode).
function updateMacdOverlay(D) {
  const S = subSeries.macd;
  if (!S) return;
  const colors = currentParams.macd.colors;

  S.dif.applyOptions({ color: hexToRgba(colors.dif, colors.difOpacity) });
  S.dea.applyOptions({ color: hexToRgba(colors.dea, colors.deaOpacity), visible: macdSignalOn });

  // Histogram bar colors are assigned server-side by sign in build_payload()
  // and mirrored the same way in recomputeIndicatorsFor() — recolor the
  // already-loaded data in place by that same sign rather than round-
  // tripping to the server for a pure color change.
  S.hist.applyOptions({ visible: macdHistOn });
  const histUpFill = hexToRgba(colors.histUp, colors.histUpOpacity);
  const histDownFill = hexToRgba(colors.histDown, colors.histDownOpacity);
  S.hist.setData(D.indicators.hist.map(p => ({
    time: p.time, value: p.value, color: p.value >= 0 ? histUpFill : histDownFill,
  })));

  // Divergence connecting lines rebuilt wholesale on every call, same as
  // the RSI pane's equivalent.
  S.divLines.forEach(s => chart.removeSeries(s));
  S.divLines = [];
  S.divAnchorBull.setData([]);
  S.divAnchorBear.setData([]);
  S.divMarkersBull.setMarkers([]);
  S.divMarkersBear.setMarkers([]);

  if (!macdDivOn && !macdChaOn) return;

  const candles = D.candles;
  const alignTo = key => {
    const byTime = new Map(D.indicators[key].map(p => [p.time, p.value]));
    return candles.map(c => { const v = byTime.get(c.time); return v === undefined ? null : v; });
  };

  const addDivLine = (pair, color) => {
    const s = chart.addSeries(LightweightCharts.LineSeries, {
      color, lineWidth:2, priceLineVisible:false, lastValueVisible:false,
      crosshairMarkerVisible:false, pointMarkersVisible:false,
    }, 3);
    s.setData([
      { time: candles[pair.p1.idx].time, value: pair.p1.value },
      { time: candles[pair.p2.idx].time, value: pair.p2.value },
    ]);
    S.divLines.push(s);
  };

  // 背馳 (MACD-line divergence) and 差離 (histogram divergence) each gated
  // and colored independently now — collected into the same bull/bear
  // anchor arrays (each entry carrying its OWN color, not a single shared
  // bullish/bearish) so both can still land on one shared marker series
  // per direction, same as before the split.
  const bullAnchors = [], bearAnchors = [];
  if (macdDivOn) {
    const difDiv = computeMacdDivergence(candles, alignTo('dif'), true);
    difDiv.bull.forEach(p => addDivLine(p, colors.bullish));
    difDiv.bear.forEach(p => addDivLine(p, colors.bearish));
    bullAnchors.push(...difDiv.bull.map(p => ({ ...divergenceMidpoint(p, candles), text:'底背馳', color: colors.bullish })));
    bearAnchors.push(...difDiv.bear.map(p => ({ ...divergenceMidpoint(p, candles), text:'頂背馳', color: colors.bearish })));
  }
  if (macdChaOn) {
    const histDiv = computeMacdDivergence(candles, alignTo('hist'), false);
    histDiv.bull.forEach(p => addDivLine(p, colors.histBullish));
    histDiv.bear.forEach(p => addDivLine(p, colors.histBearish));
    bullAnchors.push(...histDiv.bull.map(p => ({ ...divergenceMidpoint(p, candles), text:'牛差離', color: colors.histBullish })));
    bearAnchors.push(...histDiv.bear.map(p => ({ ...divergenceMidpoint(p, candles), text:'熊差離', color: colors.histBearish })));
  }

  // Sorted by time — the two sources (dif/hist) are each individually
  // in ascending order, but merging them isn't; a LineSeries requires
  // its data in ascending time order.
  const byTime = (a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0);
  bullAnchors.sort(byTime);
  bearAnchors.sort(byTime);
  S.divAnchorBull.setData(bullAnchors.map(a => ({ time: a.time, value: a.value })));
  S.divAnchorBear.setData(bearAnchors.map(a => ({ time: a.time, value: a.value })));
  S.divMarkersBull.setMarkers(bullAnchors.map(a => ({
    time: a.time, position:'belowBar', color: a.color, shape:'arrowUp', size:0, text: a.text,
  })));
  S.divMarkersBear.setMarkers(bearAnchors.map(a => ({
    time: a.time, position:'aboveBar', color: a.color, shape:'arrowDown', size:0, text: a.text,
  })));
}

// Sentinel top/bottom for the DMI bands below — see redrawDmiBands()'s
// comment for why these are fixed constants rather than the pane's actual
// current price-scale bounds. Real PDI/MDI/ADX values live in ~0-100, so
// ±1000 clears the visible pane on either side by roughly 10x with room to
// spare, however the pane's autoscale (fed only by the PDI/MDI/ADX lines —
// see autoscaleInfoProvider below) happens to be zoomed/margined at the
// moment.
const DMI_BAND_TOP = 1000;
const DMI_BAND_BOTTOM = -1000;

// Per-bar band geometry for the DMI background — full pane height per
// triggered bar, from DMI_BAND_BOTTOM to DMI_BAND_TOP. An earlier version
// split this into a bottom-4/5 band plus a separate top-1/5 "ADX extreme"
// band; the extreme band was removed per user feedback (hard to read),
// simplifying this back to a single full-height fill per band. Every bar
// gets an explicit value (not just triggered ones), same reasoning as RSI:
// a sparse array would let the Area series' interpolation connect across
// untriggered bars.
function computeDmiBandArrays(candles, adxAligned, from, to, changeThreshold) {
  const rising = [], falling = [];
  for (let i = 0; i < candles.length; i++) {
    const v = adxAligned[i];
    const prev = i > 0 ? adxAligned[i - 1] : null;
    const delta = (v !== null && prev !== null) ? v - prev : null;
    const t = candles[i].time;
    rising.push({ time: t, value: (delta !== null && delta >= changeThreshold) ? to : from });
    falling.push({ time: t, value: (delta !== null && delta <= -changeThreshold) ? to : from });
  }
  return { rising, falling };
}

// Recomputes and re-applies the DMI band colors + geometry from
// `currentParams`. Split out from updateDmiOverlay() below purely for the
// same colors/geometry separation as updateRsiOverlay()/updateMacdOverlay()
// — nothing here is geometry that needs re-deriving on pan/zoom (see below).
//
// `from`/`to` used to be read via S.adx.coordinateToPrice(0)/(paneHeight) —
// the pane's actual pixel-mapped top/bottom, needed because a plain
// priceScale.getVisibleRange() (the TIGHT PDI/MDI/ADX data range) left the
// margin strips uncolored and let the Area series' fill-to-bottom default
// bleed a sliver of the wrong color into the bottom margin. But
// coordinateToPrice() only reflects reality once the chart has actually
// painted a frame at the current visible range — and buildSubPanes() below
// calls this (via updateDmiOverlay()) synchronously, before render()
// restores the saved/default view. Reading it that early returned
// transiently-wrong bounds (confirmed live: instrumenting this function and
// toggling any 畫圖 checkbox showed the very first paint covering ~1/4 of
// the pane's actual colored area, then snapping to the correct ~full
// coverage a frame later) — a real, visible flicker on every single toggle,
// worse than the RSI pane's shading because RSI's domain is the fixed
// [0,100] the design doc calls out, needing no scale-dependent geometry at
// all.
//
// Since these two series are the band fill ONLY — never a real price a
// user reads off the axis — they don't need the pane's true current bounds
// at all, just something that safely covers whatever those bounds happen to
// be. DMI_BAND_BOTTOM/TOP (defined above computeDmiBandArrays()) does that
// with fixed constants no chart paint is needed to compute, closing the
// race at its root rather than timing around it. Confirmed live (see this
// function's own history) that this is visually identical to the old
// coordinateToPrice-based fill, both immediately on the very first paint
// and under an 8-toggle rapid-fire stress sequence, with per-frame canvas
// pixel sampling showing the same ~1-pixel antialiasing wobble RSI's own
// (never-raced) shading already has — i.e. parity with RSI, not just "close
// enough". autoscaleInfoProvider: () => null on both series (set where
// they're created, in buildSubPanes) is what makes these sentinel values
// safe: they never feed back into the pane's own PDI/MDI/ADX-driven
// autoscale.
function redrawDmiBands(D) {
  const S = subSeries.dmi;
  if (!S) return;
  const colors = currentParams.dmi.colors;
  const risingFill = hexToRgba(colors.rising, colors.risingOpacity);
  const fallingFill = hexToRgba(colors.falling, colors.fallingOpacity);
  S.rising.applyOptions({ topColor: risingFill, bottomColor: risingFill });
  S.falling.applyOptions({ topColor: fallingFill, bottomColor: fallingFill });

  if (!dmiBgOn) {
    S.rising.setData([]);
    S.falling.setData([]);
    return;
  }

  const candles = D.candles;
  const adxByTime = new Map(D.indicators.adx.map(p => [p.time, p.value]));
  const adxAligned = candles.map(c => {
    const v = adxByTime.get(c.time);
    return v === undefined ? null : v;
  });
  const { rising, falling } = computeDmiBandArrays(
    candles, adxAligned, DMI_BAND_BOTTOM, DMI_BAND_TOP, currentParams.dmi.adxChangeThreshold);
  S.rising.setData(rising);
  S.falling.setData(falling);
}

// Same role as updateRsiOverlay()/updateMacdOverlay() above, for the DMI
// pane: +DI/-DI/ADX line colors, the two reference lines' value+color,
// then delegates band geometry+color to redrawDmiBands(). A no-op if the
// DMI pane doesn't exist (只看K線 mode). Safe to call synchronously right
// after the DMI series are created (buildSubPanes below) or in place
// (pushIndicatorToChart) alike — redrawDmiBands() no longer depends on the
// chart having painted a frame, see its own comment.
function updateDmiOverlay(D) {
  const S = subSeries.dmi;
  if (!S) return;
  const colors = currentParams.dmi.colors;
  S.pdi.applyOptions({ color: hexToRgba(colors.pdi, colors.pdiOpacity) });
  S.mdi.applyOptions({ color: hexToRgba(colors.mdi, colors.mdiOpacity) });
  S.adx.applyOptions({ color: hexToRgba(colors.adx, colors.adxOpacity) });
  S.lowerLine.applyOptions({
    price: currentParams.dmi.lowerLevel,
    color: hexToRgba(colors.lowerLevelColor, colors.lowerLevelOpacity),
    lineVisible: dmiLowerLevelOn, axisLabelVisible: dmiLowerLevelOn,
  });
  S.upperLine.applyOptions({
    price: currentParams.dmi.upperLevel,
    color: hexToRgba(colors.upperLevelColor, colors.upperLevelOpacity),
    lineVisible: dmiUpperLevelOn, axisLabelVisible: dmiUpperLevelOn,
  });
  redrawDmiBands(D);
}

// --- indicator settings: hover-to-reveal name box + centered modal ---------
// The .indNameHover elements are siblings of #chart, not children of a pane
// (see #barPanel's own comment above): render() tears down everything
// #chart owns on every rebuild, and a param edit deliberately never calls
// render() — these have to survive edits the chart itself doesn't rebuild
// for. Show/hide is plain CSS :hover (see the stylesheet) — the box is
// sized to the name text plus the icon by ordinary flex layout, so hovering
// the live crosshair values appended further right, outside the box, does
// nothing; only the name (and, once revealed, the icon beside it) does.
// Defaults for fields that live only client-side (no server-computed
// value to seed from, unlike period/signal) — RSI's thresholds and colors.
// Also the fallback for a `colors`/threshold key missing from a saved
// localStorage blob written before this feature existed (see
// withIndicatorDefaults below).
const IND_DEFAULTS = {
  rsi: {
    overbought: 75, oversold: 25,
    colors: {
      line:'#f59e0b', lineOpacity:100,
      signalLine:'#38bdf8', signalLineOpacity:100,
      overboughtLineColor:'#4b5563', overboughtLineOpacity:100,
      oversoldLineColor:'#4b5563', oversoldLineOpacity:100,
      bullish:'#22c55e', bullishOpacity:15,
      bearish:'#ef4444', bearishOpacity:15,
    },
  },
  macd: {
    colors: {
      dif:'#f59e0b', difOpacity:100,
      dea:'#38bdf8', deaOpacity:100,
      histUp:'#26a69a', histUpOpacity:100,
      histDown:'#ef5350', histDownOpacity:100,
      // Unlike RSI's Bullish/Bearish (which drive both a fill AND an
      // always-solid divergence line, so opacity is reserved for the fill
      // alone), MACD's Bullish/Bearish drive only the divergence line —
      // opacity applies directly, so it defaults to fully solid like the
      // other four MACD swatches rather than RSI's low fill-opacity default.
      // 背馳 (MACD-line divergence: 底/頂背馳) and 差離 (histogram
      // divergence: 牛/熊差離) each own a separate Bullish/Bearish pair —
      // same starting colors so the split is invisible until customized,
      // but independently editable and independently toggleable from here on.
      bullish:'#22c55e', bullishOpacity:100,
      bearish:'#ef4444', bearishOpacity:100,
      histBullish:'#22c55e', histBullishOpacity:100,
      histBearish:'#ef4444', histBearishOpacity:100,
    },
  },
  dmi: {
    // How many ADX points of day-over-day change count as "rising"/
    // "falling" for the background band — user-configurable (originally
    // hardcoded at 4, the reference brainstorm's value, but that fires
    // only 13 times across a 25-year daily history on a Wilder-smoothed
    // ADX(14) — nearly invisible in normal use, so it's a tunable
    // threshold rather than a fixed constant, defaulting to 1).
    adxChangeThreshold: 1,
    // The two reference-line values — originally hardcoded 20/40 (a
    // standard ADX reading convention), now user-adjustable.
    lowerLevel: 20, upperLevel: 40,
    colors: {
      pdi:'#26a69a', pdiOpacity:100,
      mdi:'#ef5350', mdiOpacity:100,
      adx:'#eab308', adxOpacity:100,
      // The two bands, like RSI's fill colors, default to a low
      // opacity — they're a background tint, not a line.
      rising:'#84cc16', risingOpacity:15,
      falling:'#f97316', fallingOpacity:15,
      // #f59e0b — the same orange as RSI's line/MACD's DIF, this app's
      // default-palette orange, not the earlier gray.
      lowerLevelColor:'#f59e0b', lowerLevelOpacity:100,
      upperLevelColor:'#a855f7', upperLevelOpacity:100,
    },
  },
  // `period` normally comes from the server payload (build_payload()'s
  // volume_ma_period, defaulting to RALLY_VOLUME_MA), same as RSI's
  // period/signal — but unlike those, which have carried a period in every
  // saved localStorage blob since they were first added, "volume" is a
  // brand-new params key: a blob saved before this feature existed has no
  // `volume` key at all, so withIndicatorDefaults() would otherwise leave
  // period undefined (an empty, invalid number input) until the next full
  // page reload re-seeds it from the server. 50 matches RALLY_VOLUME_MA so
  // this fallback is invisible in the common case.
  volume: {
    period: 50,
    colors: { ma:'#38bdf8', maOpacity:50 },
  },
};

const IND_META = {
  rsi:  { pane: 2, fields: [['period', 'Period'], ['signal', 'SMA period'],
                             ['overbought', '超買'], ['oversold', '超賣']],
          colorFields: [
            { key:'line', label:'RSI line', opacityKey:'lineOpacity' },
            { key:'signalLine', label:'Signal line', opacityKey:'signalLineOpacity' },
            { key:'overboughtLineColor', label:'超買 line', opacityKey:'overboughtLineOpacity' },
            { key:'oversoldLineColor', label:'超賣 line', opacityKey:'oversoldLineOpacity' },
            { key:'bullish', label:'Bullish', opacityKey:'bullishOpacity' },
            { key:'bearish', label:'Bearish', opacityKey:'bearishOpacity' },
          ],
          // Paired-row layout (see renderIndModalBody), matching the
          // MACD/DMI convention: each period field sits beside the color it
          // drives (RSI line, its own MA line, each reference line's own
          // color), with 背馳 as a colors-only row since divergence has no
          // period of its own. Every row but "RSI" itself (the main line —
          // always on, no reason to hide the indicator's own namesake line)
          // carries a `toggle`, each gating exactly the chart element(s)
          // that row's own color(s) already control.
          rows: [
            { label:'RSI', fields:['period'], colors:['line'] },
            { label:'RSI MA', fields:['signal'], colors:['signalLine'], toggle:'rsiMaOn' },
            { fields:['overbought'], colors:['overboughtLineColor'], toggle:'rsiOverboughtOn' },
            { fields:['oversold'], colors:['oversoldLineColor'], toggle:'rsiOversoldOn' },
            { label:'背馳', colors:['bullish', 'bearish'], toggle:'rsiDivOn' },
          ] },
  macd: { pane: 3, fields: [['fast', 'Fast'], ['slow', 'Slow'], ['signal', 'Signal']],
          colorFields: [
            { key:'dif', label:'DIF line', opacityKey:'difOpacity' },
            { key:'dea', label:'DEA line', opacityKey:'deaOpacity' },
            { key:'histUp', label:'Histogram up', opacityKey:'histUpOpacity' },
            { key:'histDown', label:'Histogram down', opacityKey:'histDownOpacity' },
            { key:'bullish', label:'背馳 Bullish', opacityKey:'bullishOpacity' },
            { key:'bearish', label:'背馳 Bearish', opacityKey:'bearishOpacity' },
            { key:'histBullish', label:'差離 Bullish', opacityKey:'histBullishOpacity' },
            { key:'histBearish', label:'差離 Bearish', opacityKey:'histBearishOpacity' },
          ],
          // Paired-row layout (see renderIndModalBody). Fast+Slow together
          // are what "MACD" (the DIF line) means, so they share a row with
          // the DIF color; Histogram, 背馳, and 差離 have no period field of
          // their own (histogram is derived, divergence has no period), so
          // those rows are colors-only with an explicit label. Every row
          // but "MACD" itself (the main DIF line) carries a `toggle`. 背馳
          // (MACD-line divergence) and 差離 (histogram divergence) are two
          // separate rows, not one combined row — each with its own
          // Bullish/Bearish pair and its own checkbox, independently
          // configurable and independently toggleable.
          rows: [
            { label:'MACD', fields:['fast', 'slow'], colors:['dif'] },
            { fields:['signal'], colors:['dea'], toggle:'macdSignalOn' },
            { label:'柱狀體', colors:['histUp', 'histDown'], toggle:'macdHistOn' },
            { label:'背馳', colors:['bullish', 'bearish'], toggle:'macdDivOn' },
            { label:'差離', colors:['histBullish', 'histBearish'], toggle:'macdChaOn' },
          ] },
  dmi:  { pane: 4, fields: [['di', 'DI'], ['adx', 'ADX'], ['adxChangeThreshold', 'ADX change'],
                             ['lowerLevel', '有力'], ['upperLevel', '超買']],
          colorFields: [
            { key:'pdi', label:'+DI line', opacityKey:'pdiOpacity' },
            { key:'mdi', label:'-DI line', opacityKey:'mdiOpacity' },
            { key:'adx', label:'ADX line', opacityKey:'adxOpacity' },
            { key:'rising', label:'ADX rising', opacityKey:'risingOpacity' },
            { key:'falling', label:'ADX falling', opacityKey:'fallingOpacity' },
            { key:'lowerLevelColor', label:'Lower line', opacityKey:'lowerLevelOpacity' },
            { key:'upperLevelColor', label:'Upper line', opacityKey:'upperLevelOpacity' },
          ],
          // Paired-row layout (see renderIndModalBody) — each field sits
          // beside the color(s) it drives, rather than in a separate
          // Colors section: DI's own two line colors (+DI/-DI), ADX's own
          // line color, the change-threshold's two band colors, and each
          // reference line's own value beside its own color. "DMI" (+DI/-DI)
          // and "ADX" are DMI's two main lines and stay exempt from a
          // toggle, same reasoning as RSI's "RSI" row and MACD's "MACD"
          // row; every other row carries one. 背景 (day-over-day ADX-change
          // band) rides on the "ADX change" row rather than a separate
          // lone row, since that row already owns the rising/falling
          // colors the band itself is filled with.
          rows: [
            { label:'DMI', fields:['di'], colors:['pdi', 'mdi'] },
            { label:'ADX', fields:['adx'], colors:['adx'] },
            { fields:['adxChangeThreshold'], colors:['rising', 'falling'], toggle:'dmiBgOn' },
            { fields:['lowerLevel'], colors:['lowerLevelColor'], toggle:'dmiLowerLevelOn' },
            { fields:['upperLevel'], colors:['upperLevelColor'], toggle:'dmiUpperLevelOn' },
          ] },
  // The volume histogram itself has no settings here (no period, no color —
  // it's the raw bars, colored by up/down like every candle chart's volume
  // pane); this pane's only configurable thing is its MA line, so unlike
  // RSI/MACD/DMI there is no untoggleable "main line" row — the one row here
  // carries a toggle, same reasoning as RSI's own "RSI MA" row.
  volume: { pane: 1, fields: [['period', 'Period']],
            colorFields: [
              { key:'ma', label:'MA line', opacityKey:'maOpacity' },
            ],
            rows: [
              { label:'Volume MA', fields:['period'], colors:['ma'], toggle:'volumeMaOn' },
            ] },
};

// get/set for the sub-pane visibility toggles a modal row can carry (`row.
// toggle`, see IND_META above) — these are plain `let` bindings, not
// window properties, so renderIndModalBody needs an explicit lookup rather
// than `window[key]`.
const TOGGLE_VARS = {
  rsiDivOn: { get: () => rsiDivOn, set: v => { rsiDivOn = v; } },
  macdDivOn: { get: () => macdDivOn, set: v => { macdDivOn = v; } },
  macdChaOn: { get: () => macdChaOn, set: v => { macdChaOn = v; } },
  dmiBgOn: { get: () => dmiBgOn, set: v => { dmiBgOn = v; } },
  rsiMaOn: { get: () => rsiMaOn, set: v => { rsiMaOn = v; } },
  rsiOverboughtOn: { get: () => rsiOverboughtOn, set: v => { rsiOverboughtOn = v; } },
  rsiOversoldOn: { get: () => rsiOversoldOn, set: v => { rsiOversoldOn = v; } },
  macdSignalOn: { get: () => macdSignalOn, set: v => { macdSignalOn = v; } },
  macdHistOn: { get: () => macdHistOn, set: v => { macdHistOn = v; } },
  dmiLowerLevelOn: { get: () => dmiLowerLevelOn, set: v => { dmiLowerLevelOn = v; } },
  dmiUpperLevelOn: { get: () => dmiUpperLevelOn, set: v => { dmiUpperLevelOn = v; } },
  volumeMaOn: { get: () => volumeMaOn, set: v => { volumeMaOn = v; } },
};

// Fills in any IND_DEFAULTS field/sub-field a params object doesn't carry —
// the server payload (no overbought/oversold/colors at all) and a saved
// localStorage blob from before this feature existed both hit this, and
// both must keep working without a migration step.
function withIndicatorDefaults(params) {
  const out = { ...params };
  for (const ind of Object.keys(IND_DEFAULTS)) {
    const defaults = IND_DEFAULTS[ind];
    const cur = out[ind] || {};
    out[ind] = { ...defaults, ...cur, colors: { ...defaults.colors, ...(cur.colors || {}) } };
  }
  return out;
}
let lastHoverTime = null;
let currentParams = null;   // set at bootstrap, below
let indModalSnapshot = null;   // deep copy of currentParams[ind] taken when its modal opened, for Cancel

const INDICATOR_PARAMS_KEY = 'sdx.indicatorParams';

function loadSavedParams() {
  try {
    const raw = localStorage.getItem(INDICATOR_PARAMS_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }   // private browsing, etc. — session-only
}

function saveParams(params) {
  try { localStorage.setItem(INDICATOR_PARAMS_KEY, JSON.stringify(params)); }
  catch (e) { /* falls back to session-only, silently */ }
}

// Fields that may carry up to 2 decimal places instead of the default
// positive-integer-only rule below — currently only DMI's ADX change
// threshold (day-over-day ADX points counting as "rising"/"falling" is
// meaningfully sub-integer, e.g. 0.5, unlike a period which Wilder's
// alpha = 1/period has no sane fractional reading for).
const DECIMAL_FIELDS = { dmi: new Set(['adxChangeThreshold']) };

function isDecimalField(ind, field) {
  return !!(DECIMAL_FIELDS[ind] && DECIMAL_FIELDS[ind].has(field));
}

// Positive integers only by default — anything else is left uncommitted
// rather than crashing Wilder's alpha = 1/period on a zero or negative
// period. DECIMAL_FIELDS above opts specific fields into positive numbers
// with up to 2 decimal places instead.
function isValidFieldValue(ind, field, raw) {
  const s = String(raw).trim();
  const re = isDecimalField(ind, field) ? /^[0-9]+([.][0-9]{1,2})?$/ : /^[1-9][0-9]*$/;
  return re.test(s) && parseFloat(s) > 0;
}

// Repositions the three name boxes from each pane's own getHTMLElement()
// rect, rather than tracked pixel math — stays correct across resize and
// stretch-factor changes for free, the same reason #barPanel avoided that
// drift. Width/height are NOT set here: the box's own flex layout around
// the live .indName text sizes it, so a param edit that changes the label's
// width (e.g. "RSI(9)" -> "RSI(20)") is handled by the browser, not by us.
// #barPanel's `right` CSS was a fixed 76px clearing the price axis, which
// only held while #chart's own right edge coincided with #chartWrap's — true
// until a flyout opens and #chart narrows (#chartWrap.panelOpen). Re-measured
// off #chart's actual rect on every resize/toggle rather than hand-tracking
// the flyout's width, so it stays correct regardless of what else changes it.
function positionBarPanel() {
  if (!chart) return;
  const wrapRect = document.getElementById('chartWrap').getBoundingClientRect();
  const chartRect = document.getElementById('chart').getBoundingClientRect();
  barPanelEl.style.right = (wrapRect.right - chartRect.right + 76) + 'px';
}

function positionIndHovers() {
  if (!chart) return;
  const wrapRect = document.getElementById('chartWrap').getBoundingClientRect();
  const panes = chart.panes();
  for (const ind of Object.keys(IND_META)) {
    const hover = document.getElementById('indHover-' + ind);
    const paneIdx = IND_META[ind].pane;
    const el = paneIdx < panes.length ? panes[paneIdx].getHTMLElement() : null;
    if (!el) continue;
    const r = el.getBoundingClientRect();
    hover.style.top = (r.top - wrapRect.top + 1) + 'px';
    hover.style.left = (r.left - wrapRect.left + 6) + 'px';
  }
}

// Rebuilds one pane's legend base text and hover-readout maps from freshly
// recomputed data — setLegends() reads both, so a param edit that skipped
// this would redraw the chart line but leave the legend (and crosshair
// hover values) showing the old period. Also updates the DOM .indName span
// that sits over the canvas text on hover, so the two never disagree.
function updateLegendFor(ind, D) {
  const L = legends.find(l => l.key === ind);
  if (!L) return;
  L.base = D.labels[ind];
  document.querySelector('#indHover-' + ind + ' .indName').textContent = D.labels[ind];
  const I = D.indicators;
  if (ind === 'rsi') {
    L.parts[0].map = mapOf(I.rsi);
    L.parts[1].map = mapOf(I.rsiSignal);
  } else if (ind === 'macd') {
    L.parts[0].map = mapOf(I.dif);
    L.parts[1].map = mapOf(I.dea);
    L.parts[2].map = mapOf(I.hist);
  } else if (ind === 'dmi') {
    L.parts[0].map = mapOf(I.pdi);
    L.parts[1].map = mapOf(I.mdi);
    L.parts[2].map = mapOf(I.adx);
  } else if (ind === 'volume') {
    L.parts[1].map = mapOf(D.volumeMa);
  }
}

// In-place .setData() on the series buildSubPanes() stashed in subSeries —
// never render(), so the visible time/price range is structurally untouched.
function pushIndicatorToChart(ind, D) {
  if (ind === 'rsi' && subSeries.rsi) {
    subSeries.rsi.main.setData(D.indicators.rsi);
    subSeries.rsi.signal.setData(D.indicators.rsiSignal);
    updateRsiOverlay(D);
  } else if (ind === 'macd' && subSeries.macd) {
    subSeries.macd.dif.setData(D.indicators.dif);
    subSeries.macd.dea.setData(D.indicators.dea);
    updateMacdOverlay(D);
  } else if (ind === 'dmi' && subSeries.dmi) {
    subSeries.dmi.pdi.setData(D.indicators.pdi);
    subSeries.dmi.mdi.setData(D.indicators.mdi);
    subSeries.dmi.adx.setData(D.indicators.adx);
    updateDmiOverlay(D);
  } else if (ind === 'volume' && subSeries.volume) {
    subSeries.volume.ma.setData(D.volumeMa);
    updateVolumeOverlay(D);
  }
  updateLegendFor(ind, D);
}

// One edit → every loaded symbol recomputed (so switching tickers is never a
// moment where one still shows the old period), the visible chart updated in
// place for whichever symbol is on screen, and the new values persisted.
function applyIndicatorParamChange(ind, field, rawValue) {
  if (!isValidFieldValue(ind, field, rawValue)) return;
  const value = isDecimalField(ind, field) ? parseFloat(rawValue) : parseInt(rawValue, 10);
  currentParams = {
    ...currentParams,
    [ind]: { ...currentParams[ind], [field]: value },
  };
  for (const sym of Object.keys(ALL.symbols)) {
    recomputeIndicatorsFor(ALL.symbols[sym], currentParams);
  }
  pushIndicatorToChart(ind, ALL.symbols[current]);
  setLegends(lastHoverTime);
  saveParams(currentParams);
}

const indOverlay = document.getElementById('indOverlay');

// Colors (and, for RSI, thresholds) are display-only — no per-symbol data to
// recompute, unlike period/signal — so this updates currentParams directly
// and repaints only the currently visible symbol's chart, rather than
// looping every loaded symbol the way applyIndicatorParamChange does.
function applyIndicatorColorChange(ind, key, rawValue, isOpacity) {
  const value = isOpacity
    ? Math.max(0, Math.min(100, parseInt(rawValue, 10) || 0))
    : rawValue;
  currentParams = {
    ...currentParams,
    [ind]: { ...currentParams[ind], colors: { ...currentParams[ind].colors, [key]: value } },
  };
  pushIndicatorToChart(ind, ALL.symbols[current]);
  saveParams(currentParams);
}

function resetIndicatorColors(ind) {
  currentParams = {
    ...currentParams,
    [ind]: { ...currentParams[ind], colors: { ...IND_DEFAULTS[ind].colors } },
  };
  pushIndicatorToChart(ind, ALL.symbols[current]);
  saveParams(currentParams);
  renderIndModalBody(ind);   // re-render only — indModalSnapshot stays the pre-open state, so Cancel still reverts to before the modal was ever opened, not just to before Reset
}

// --- custom swatch-grid color picker ---------------------------------------
// A grayscale row plus 7 tint/shade rows per hue (10 hues x 8 rows incl.
// grayscale), matching the reference picker's grid layout. HSL-generated
// rather than a literal copy of any specific app's palette — the ask was to
// match the picker's structure/behaviour, not reproduce proprietary hex
// values pulled from a screenshot.
const SWATCH_HUES = [4, 30, 48, 122, 168, 190, 221, 262, 291, 340];
const SWATCH_GRAYS = ['#ffffff','#d1d4dc','#b2b5be','#9598a1','#787b86','#5d606b','#434651','#2a2e39','#131722','#000000'];
const SWATCH_LIGHTNESS_ROWS = [50, 85, 74, 63, 45, 32, 20];   // row2 (vivid) through row8 (darkest)

function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const toHex = x => Math.round(255 * x).toString(16).padStart(2, '0');
  return '#' + toHex(f(0)) + toHex(f(8)) + toHex(f(4));
}

function buildSwatchPalette() {
  const rows = [SWATCH_GRAYS];
  SWATCH_LIGHTNESS_ROWS.forEach(L => {
    rows.push(SWATCH_HUES.map(h => hslToHex(h, 70, L)));
  });
  return rows;
}

let openPicker = null;   // { el } for the currently open popup, if any

function closeColorPicker() {
  if (!openPicker) return;
  openPicker.el.remove();
  openPicker = null;
  document.removeEventListener('mousedown', outsidePickerClick);
  document.removeEventListener('keydown', pickerEscape);
}
function outsidePickerClick(e) {
  if (openPicker && !openPicker.el.contains(e.target)) closeColorPicker();
}
function pickerEscape(e) {
  if (e.key === 'Escape') closeColorPicker();
}

function openColorPicker(triggerEl, ind, field, opacityKey) {
  closeColorPicker();
  const colors = currentParams[ind].colors;
  const currentHex = colors[field].toLowerCase();

  const popup = document.createElement('div');
  popup.className = 'colorPickerPopup';
  const gridHtml = buildSwatchPalette().map(row => row.map(hex => `
    <button type="button" class="swatchCell${hex === currentHex ? ' selected' : ''}"
            style="background:${hex}" data-hex="${hex}"></button>
  `).join('')).join('');
  const opacityHtml = opacityKey ? `
    <div class="pickerOpacity">
      <div class="label">Opacity</div>
      <div class="pickerOpacityRow">
        <input type="range" min="0" max="100" step="1" value="${colors[opacityKey]}">
        <input type="number" min="0" max="100" step="1" value="${colors[opacityKey]}">
      </div>
    </div>` : '';
  popup.innerHTML = `<div class="swatchGrid">${gridHtml}</div>${opacityHtml}`;
  document.body.appendChild(popup);

  // Positioned off the trigger swatch, clamped so it never runs off-screen.
  const rect = triggerEl.getBoundingClientRect();
  const popupRect = popup.getBoundingClientRect();
  let top = rect.bottom + 6, left = rect.left;
  if (left + popupRect.width > innerWidth - 8) left = innerWidth - popupRect.width - 8;
  if (top + popupRect.height > innerHeight - 8) top = rect.top - popupRect.height - 6;
  popup.style.top = top + 'px';
  popup.style.left = left + 'px';

  popup.querySelectorAll('.swatchCell').forEach(cell => {
    cell.addEventListener('click', () => {
      applyIndicatorColorChange(ind, field, cell.dataset.hex, false);
      triggerEl.style.background = cell.dataset.hex;
      const caption = document.querySelector('[data-hex-for="' + field + '"]');
      if (caption) caption.textContent = cell.dataset.hex;
      popup.querySelectorAll('.swatchCell').forEach(c => c.classList.remove('selected'));
      cell.classList.add('selected');
    });
  });
  if (opacityKey) {
    const range = popup.querySelector('input[type="range"]');
    const number = popup.querySelector('input[type="number"]');
    const sync = v => {
      range.value = v; number.value = v;
      applyIndicatorColorChange(ind, opacityKey, v, true);
    };
    range.addEventListener('input', e => sync(e.target.value));
    number.addEventListener('input', e => sync(e.target.value));
  }

  openPicker = { el: popup };
  // Deferred so the same click that opened the popup doesn't immediately
  // register as an "outside" click and close it right back.
  setTimeout(() => {
    document.addEventListener('mousedown', outsidePickerClick);
    document.addEventListener('keydown', pickerEscape);
  }, 0);
}

function openIndSettingsModal(ind) {
  indModalSnapshot = JSON.parse(JSON.stringify(currentParams[ind]));
  renderIndModalBody(ind);

  indOverlay.classList.add('open');
  syncModalOpen();
}

// Rebuilds #indModalBody's fields/swatches/footer from the current live
// state. Split out from openIndSettingsModal() so resetIndicatorColors()
// can re-render after a reset WITHOUT recapturing indModalSnapshot — Cancel
// must still undo everything back to how the indicator looked before the
// modal was opened at all, not just back to before the last Reset click.
function renderIndModalBody(ind) {
  closeColorPicker();
  const meta = IND_META[ind];
  document.getElementById('indModalTitle').textContent = ind.toUpperCase();
  const body = document.getElementById('indModalBody');

  let fieldsHtml, colorHtml = '';
  if (meta.rows) {
    // Paired-row layout: each row's numeric field(s) sit directly beside
    // the color swatch(es) they control (DI period next to its DI colors,
    // Fast+Slow next to the DIF/"MACD" line color, Histogram's up/down
    // colors with no field at all, etc.) — the generic Parameters-grid-
    // then-Colors-grid split below reads as two unrelated lists once a
    // concept has more than one closely-related color, which several of
    // MACD's and DMI's do. `row.fields`/`row.colors` are each 0+ keys;
    // `row.label` overrides the row heading for rows with zero or more
    // than one field (where there's no single field name to reuse, e.g.
    // "Histogram" or "背馳 & 差離"). Opt-in via `rows` so RSI, which
    // doesn't need this, keeps the simpler generic layout untouched.
    const fieldsByKey = Object.fromEntries(meta.fields);
    const colorsByKey = Object.fromEntries((meta.colorFields || []).map(cf => [cf.key, cf]));
    const colors = currentParams[ind].colors;
    fieldsHtml = `
      <div class="modalSection">
        <div class="modalSectionLabel">Parameters</div>
        ${meta.rows.map(row => {
          const fields = row.fields || [];
          const rowLabel = row.label || fieldsByKey[fields[0]];
          const inputsHtml = fields.map(f => {
            const decimals = isDecimalField(ind, f);
            return `
            <input type="number" min="${decimals ? '0.01' : '1'}" step="${decimals ? '0.01' : '1'}"
                   data-field="${f}" title="${fieldsByKey[f]}" value="${currentParams[ind][f]}">
          `;
          }).join('');
          const colorsHtml = (row.colors || []).map(ck => {
            const cf = colorsByKey[ck];
            return `<button type="button" class="colorSwatch" data-color="${cf.key}"
                      data-opacity-key="${cf.opacityKey || ''}" title="${cf.label}"
                      style="background:${colors[cf.key]}"></button>`;
          }).join('');
          // row.toggle (RSI背馳/MACD背馳&差離/DMI背景, see IND_META and
          // TOGGLE_VARS) — a sub-pane visibility switch riding along in the
          // row that already groups that concept's colors/fields, rather
          // than a whole separate section for one checkbox.
          const labelHtml = row.toggle
            ? `<label class="modalRowToggle"><input type="checkbox" data-toggle="${row.toggle}"
                 ${TOGGLE_VARS[row.toggle].get() ? 'checked' : ''}> ${rowLabel}</label>`
            : `<label>${rowLabel}</label>`;
          return `
            <div class="modalParamRow">
              ${labelHtml}
              <div class="paramRowInputs">${inputsHtml}${colorsHtml}</div>
            </div>`;
        }).join('')}
      </div>`;
  } else {
    // Paired 2-up grids, not a stacked list — Period pairs with SMA period
    // as "how RSI is calculated," Bullish pairs with Bearish as "which
    // direction this color means." The grid itself carries that grouping.
    fieldsHtml = `
      <div class="modalSection">
        <div class="modalSectionLabel">Parameters</div>
        <div class="modalFieldGrid">
          ${meta.fields.map(([field, label]) => {
            const decimals = isDecimalField(ind, field);
            return `
            <div class="modalField">
              <label>${label}</label>
              <input type="number" min="${decimals ? '0.01' : '1'}" step="${decimals ? '0.01' : '1'}"
                     data-field="${field}" value="${currentParams[ind][field]}">
            </div>
          `;
          }).join('')}
        </div>
      </div>`;

    if (meta.colorFields) {
      const colors = currentParams[ind].colors;
      colorHtml = `
        <div class="modalSection">
          <div class="modalSectionLabel">Colors</div>
          <div class="modalSwatchGrid">
            ${meta.colorFields.map(cf => `
              <div class="modalSwatchField">
                <button type="button" class="colorSwatch" data-color="${cf.key}"
                        data-opacity-key="${cf.opacityKey || ''}" style="background:${colors[cf.key]}"></button>
                <div class="modalSwatchMeta">
                  <label>${cf.label}</label>
                  <span class="swatchHex" data-hex-for="${cf.key}">${colors[cf.key]}</span>
                </div>
              </div>
            `).join('')}
          </div>
        </div>`;
    }
  }

  const footerHtml = `
    <div class="modalFooter">
      ${meta.colorFields ? '<button type="button" id="indResetColors">Reset to default</button>' : '<span></span>'}
      <div class="modalFooterRight">
        <button type="button" id="indCancelBtn">Cancel</button>
        <button type="button" id="indOkBtn" class="primary">OK</button>
      </div>
    </div>`;

  body.innerHTML = fieldsHtml + colorHtml + footerHtml;

  body.querySelectorAll('input[data-field]').forEach(input => {
    input.addEventListener('input', e => {
      applyIndicatorParamChange(ind, e.target.dataset.field, e.target.value);
    });
  });
  // Applies immediately and is NOT part of Cancel's snapshot/restore below
  // (that snapshot only covers currentParams[ind] — params/colors), and is
  // never persisted — same "takes effect on change, no separate commit"
  // behavior these toggles already had as 畫圖-menu checkboxes before this
  // session moved them here. pushIndicatorToChart, not render(): every
  // other field in this modal (period/color) already updates in place
  // without rebuilding the whole chart (see the comment above IND_DEFAULTS
  // on why param edits deliberately never call render()) — these toggles
  // now live in the same modal as those fields, so they follow the same
  // lightweight path instead of the 畫圖-menu-checkbox convention they used
  // when they lived there.
  body.querySelectorAll('input[data-toggle]').forEach(cb => {
    cb.addEventListener('change', e => {
      TOGGLE_VARS[e.target.dataset.toggle].set(e.target.checked);
      pushIndicatorToChart(ind, ALL.symbols[current]);
    });
  });
  body.querySelectorAll('.colorSwatch').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      openColorPicker(btn, ind, btn.dataset.color, btn.dataset.opacityKey || null);
    });
  });
  const resetBtn = document.getElementById('indResetColors');
  if (resetBtn) resetBtn.addEventListener('click', () => resetIndicatorColors(ind));
  document.getElementById('indCancelBtn').addEventListener('click', () => cancelIndModal(ind));
  document.getElementById('indOkBtn').addEventListener('click', closeIndModal);
}

// Every edit inside this modal (period/threshold fields, colors) applies
// live as you make it — there's no separate "commit" step — so Cancel has
// to actively restore the pre-open snapshot rather than simply discarding
// unsaved input, matching how OK is just a close (nothing left to apply).
function cancelIndModal(ind) {
  if (indModalSnapshot) {
    currentParams = { ...currentParams, [ind]: indModalSnapshot };
    for (const sym of Object.keys(ALL.symbols)) {
      recomputeIndicatorsFor(ALL.symbols[sym], currentParams);
    }
    pushIndicatorToChart(ind, ALL.symbols[current]);
    setLegends(lastHoverTime);
    saveParams(currentParams);
  }
  closeIndModal();
}

function closeIndModal() {
  closeColorPicker();
  indModalSnapshot = null;
  indOverlay.classList.remove('open');
  syncModalOpen();
}

document.querySelectorAll('.indNameHover .gear').forEach(btn => {
  btn.addEventListener('click', e => {
    e.stopPropagation();
    openIndSettingsModal(btn.closest('.indNameHover').dataset.ind);
  });
});
document.getElementById('indModalXClose').addEventListener('click', closeIndModal);
indOverlay.addEventListener('click', e => { if (e.target === indOverlay) closeIndModal(); });
addEventListener('keydown', e => {
  if (e.key === 'Escape' && indOverlay.classList.contains('open')) closeIndModal();
});

// --- Go to date modal ---------------------------------------------------
// A TradingView-style day/month/year drill-down. calCursor is the browsed
// position (shared across all three views, not just day view) so entering
// month/year view and coming back down lands on what was being browsed,
// independent of calSelected (the actual chosen date, synced with the text
// input). The whole #gotoDateCal container is re-rendered wholesale on every
// view change or nav click — same "rebuild rather than patch" model #chart
// itself uses.
const gotoDateOverlay = document.getElementById('gotoDateOverlay');
const gotoDateInput = document.getElementById('gotoDateInput');
const gotoDateCal = document.getElementById('gotoDateCal');
const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

let calView = 'day';                   // 'day' | 'month' | 'year'
let calCursor = { year: 0, month: 0 }; // browsed position; month is 0-based
let calSelected = null;                // 'YYYY-MM-DD' or null

function pad2(n) { return String(n).padStart(2, '0'); }

function calDateStr(y, m, d) { return `${y}-${pad2(m + 1)}-${pad2(d)}`; }

// Rejects both malformed strings and calendar-invalid ones (e.g. 2026-02-30)
// by round-tripping through Date.UTC and checking the fields didn't roll over.
function parseDateStr(s) {
  const m = /^(\\d{4})-(\\d{2})-(\\d{2})$/.exec(s || '');
  if (!m) return null;
  const y = +m[1], mo = +m[2], d = +m[3];
  const dt = new Date(Date.UTC(y, mo - 1, d));
  if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d) return null;
  return { year: y, month: mo - 1, day: d };
}

function shiftCalCursor(dir) {
  if (calView === 'day') {
    let { year, month } = calCursor;
    month += dir;
    if (month < 0) { month = 11; year -= 1; }
    else if (month > 11) { month = 0; year += 1; }
    calCursor = { year, month };
  } else if (calView === 'month') {
    calCursor = Object.assign({}, calCursor, { year: calCursor.year + dir });
  } else {
    calCursor = Object.assign({}, calCursor, { year: calCursor.year + dir * 20 });
  }
  renderCalendar();
}

function renderCalendar() {
  if (calView === 'day') renderDayView();
  else if (calView === 'month') renderMonthView();
  else renderYearView();
}

function renderDayView() {
  const { year, month } = calCursor;
  const startWeekday = new Date(Date.UTC(year, month, 1)).getUTCDay(); // 0=Sun
  const daysInMonth = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  const todayStr = new Date().toISOString().slice(0, 10);

  let cells = '';
  for (let i = 0; i < startWeekday; i++) cells += '<button class="calCell empty" tabindex="-1"></button>';
  for (let d = 1; d <= daysInMonth; d++) {
    const ds = calDateStr(year, month, d);
    const cls = ['calCell'];
    if (ds === todayStr) cls.push('today');
    if (ds > todayStr) cls.push('future');
    if (ds === calSelected) cls.push('selected');
    cells += `<button class="${cls.join(' ')}" data-date="${ds}">${d}</button>`;
  }

  gotoDateCal.innerHTML = `
    <div class="calHead">
      <button class="calNav" id="calPrev">‹</button>
      <button class="calLabel" id="calLabel">${MONTH_NAMES[month]} ${year}</button>
      <button class="calNav" id="calNext">›</button>
    </div>
    <div class="calWeekdays"><span>Su</span><span>Mo</span><span>Tu</span><span>We</span><span>Th</span><span>Fr</span><span>Sa</span></div>
    <div class="calGrid day">${cells}</div>
  `;

  gotoDateCal.querySelectorAll('.calCell[data-date]').forEach(btn => {
    btn.addEventListener('click', () => {
      calSelected = btn.dataset.date;
      gotoDateInput.value = calSelected;
      calView = 'day';
      renderCalendar();
    });
  });
  document.getElementById('calLabel').addEventListener('click', () => { calView = 'month'; renderCalendar(); });
  document.getElementById('calPrev').addEventListener('click', () => shiftCalCursor(-1));
  document.getElementById('calNext').addEventListener('click', () => shiftCalCursor(1));
}

function renderMonthView() {
  const { year } = calCursor;
  let cells = '';
  for (let m = 0; m < 12; m++) {
    const cls = ['calCell'];
    if (m === calCursor.month) cls.push('selected');
    cells += `<button class="${cls.join(' ')}" data-month="${m}">${MONTH_NAMES[m]}</button>`;
  }

  gotoDateCal.innerHTML = `
    <div class="calHead">
      <button class="calNav" id="calPrev">‹</button>
      <button class="calLabel" id="calLabel">${year}</button>
      <button class="calNav" id="calNext">›</button>
    </div>
    <div class="calGrid month">${cells}</div>
  `;

  gotoDateCal.querySelectorAll('.calCell[data-month]').forEach(btn => {
    btn.addEventListener('click', () => {
      calCursor = Object.assign({}, calCursor, { month: +btn.dataset.month });
      calView = 'day';
      renderCalendar();
    });
  });
  document.getElementById('calLabel').addEventListener('click', () => { calView = 'year'; renderCalendar(); });
  document.getElementById('calPrev').addEventListener('click', () => shiftCalCursor(-1));
  document.getElementById('calNext').addEventListener('click', () => shiftCalCursor(1));
}

function renderYearView() {
  const blockStart = Math.floor(calCursor.year / 20) * 20;
  let cells = '';
  for (let y = blockStart; y < blockStart + 20; y++) {
    const cls = ['calCell'];
    if (y === calCursor.year) cls.push('selected');
    cells += `<button class="${cls.join(' ')}" data-year="${y}">${y}</button>`;
  }

  gotoDateCal.innerHTML = `
    <div class="calHead">
      <button class="calNav" id="calPrev">‹</button>
      <span class="calLabel">${blockStart} - ${blockStart + 19}</span>
      <button class="calNav" id="calNext">›</button>
    </div>
    <div class="calGrid year">${cells}</div>
  `;

  gotoDateCal.querySelectorAll('.calCell[data-year]').forEach(btn => {
    btn.addEventListener('click', () => {
      calCursor = Object.assign({}, calCursor, { year: +btn.dataset.year });
      calView = 'month';
      renderCalendar();
    });
  });
  document.getElementById('calPrev').addEventListener('click', () => shiftCalCursor(-1));
  document.getElementById('calNext').addEventListener('click', () => shiftCalCursor(1));
}

// Typing follows the same calSelected/calCursor state the calendar clicks
// use, so a valid typed date repaints the day grid on the right month — it
// does not force calView back to 'day' by itself (only clicking a day does).
gotoDateInput.addEventListener('input', () => {
  const p = parseDateStr(gotoDateInput.value);
  if (!p) return;
  calSelected = gotoDateInput.value;
  calCursor = { year: p.year, month: p.month };
  if (calView === 'day') renderCalendar();
});

function openGotoDateModal() {
  const range = chart.timeScale().getVisibleLogicalRange();
  const lastIdx = BAR.candles.length - 1;
  const idx = range ? Math.max(0, Math.min(lastIdx, Math.round(range.to))) : lastIdx;
  const ds = BAR.candles[idx].time;
  const p = parseDateStr(ds);

  calSelected = ds;
  gotoDateInput.value = ds;
  calCursor = { year: p.year, month: p.month };
  calView = 'day';
  renderCalendar();

  gotoDateOverlay.classList.add('open');
  syncModalOpen();
}

function closeGotoDateModal() {
  gotoDateOverlay.classList.remove('open');
  syncModalOpen();
}

// Sorted YYYY-MM-DD strings sort lexically the same as chronologically —
// same property applyDefaultRange() already relies on — so a plain string
// binary search finds the bracketing bars without parsing every date.
function nearestBarIndex(ds) {
  const arr = BAR.candles;
  if (BAR.idx.has(ds)) return BAR.idx.get(ds);

  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].time < ds) lo = mid + 1; else hi = mid;
  }
  if (lo === 0) return 0;
  if (lo === arr.length) return arr.length - 1;

  const target = new Date(ds + 'T00:00:00Z').getTime();
  const before = new Date(arr[lo - 1].time + 'T00:00:00Z').getTime();
  const after = new Date(arr[lo].time + 'T00:00:00Z').getTime();
  // Strict less-than so an exact tie (equidistant before/after) keeps lo-1 —
  // the earlier bar, per the "ties break to the earlier bar" decision.
  return (after - target) < (target - before) ? lo : lo - 1;
}

// Post-jump indicator: a plain positioned DOM label rather than a chart-native
// series marker, since a series marker would persist through further pans/
// zooms and need its own cleanup path. Positioned from timeToCoordinate/
// priceToCoordinate offset by the candle pane's own getHTMLElement() rect
// relative to #chartWrap — the same rect-based-not-hand-tracked pattern
// positionIndHovers()/positionBarPanel() already use.
let jumpIndicatorEl = null;
let jumpIndicatorDismiss = null;

function formatJumpLabel(ds) {
  const [y, m, d] = ds.split('-').map(Number);
  const wd = weekday(ds);
  const wdShort = wd[0] + wd.slice(1).toLowerCase();
  return `${wdShort} ${d} ${MONTH_NAMES[m - 1]} '${String(y).slice(-2)}`;
}

function clearJumpIndicator() {
  if (jumpIndicatorEl) { jumpIndicatorEl.remove(); jumpIndicatorEl = null; }
  if (jumpIndicatorDismiss) { document.removeEventListener('click', jumpIndicatorDismiss); jumpIndicatorDismiss = null; }
}

function showJumpIndicator(time) {
  clearJumpIndicator();
  requestAnimationFrame(() => {
    const idx = BAR.idx.get(time);
    if (idx === undefined) return;
    const x = chart.timeScale().timeToCoordinate(time);
    const y = candleSeries.priceToCoordinate(BAR.candles[idx].high);
    if (x === null || y === null) return;

    const pane0 = chart.panes()[0].getHTMLElement();
    const paneRect = pane0.getBoundingClientRect();
    const wrapRect = document.getElementById('chartWrap').getBoundingClientRect();

    const el = document.createElement('div');
    el.className = 'jumpIndicator';
    el.textContent = formatJumpLabel(time);
    el.style.left = (x + paneRect.left - wrapRect.left) + 'px';
    el.style.top = (y + paneRect.top - wrapRect.top) + 'px';
    document.getElementById('chartWrap').appendChild(el);
    jumpIndicatorEl = el;

    // Deferred a tick so the same click that triggered the jump (the Go to
    // button itself) doesn't immediately dismiss the indicator it just
    // created — a listener added to document mid-bubble still fires for the
    // event currently in flight.
    setTimeout(() => {
      jumpIndicatorDismiss = clearJumpIndicator;
      document.addEventListener('click', jumpIndicatorDismiss, { once: true });
    }, 0);
  });
}

function submitGotoDate() {
  if (!BAR || !BAR.candles.length) return;
  const p = parseDateStr(gotoDateInput.value);
  if (!p) return;

  const idx = nearestBarIndex(gotoDateInput.value);
  const range = chart.timeScale().getVisibleLogicalRange();
  const span = range ? (range.to - range.from) : (BAR.candles.length - 1);
  chart.timeScale().setVisibleLogicalRange({ from: idx - span / 2, to: idx + span / 2 });
  closeGotoDateModal();
  showJumpIndicator(BAR.candles[idx].time);
}

document.getElementById('gotoDateBtn').addEventListener('click', e => {
  e.stopPropagation();
  openGotoDateModal();
});
document.getElementById('gotoDateXClose').addEventListener('click', closeGotoDateModal);
document.getElementById('gotoDateCancel').addEventListener('click', closeGotoDateModal);
document.getElementById('gotoDateSubmit').addEventListener('click', submitGotoDate);
gotoDateInput.addEventListener('keydown', e => { if (e.key === 'Enter') submitGotoDate(); });
gotoDateOverlay.addEventListener('click', e => { if (e.target === gotoDateOverlay) closeGotoDateModal(); });
addEventListener('keydown', e => {
  if (e.key === 'Escape' && gotoDateOverlay.classList.contains('open')) closeGotoDateModal();
});

// --- range measure tool ----------------------------------------------------
// Persisted as {t1,p1,t2,p2} time/price values, never pixels, so every entry
// in `measurements` survives both a render() rebuild (chart.remove() on any
// 畫圖 toggle — same reason savedRange/savedPriceRange exist) AND a symbol
// switch, unlike savedRange/savedPriceRange which are explicitly nulled
// there: a measurement is meant to outlive the symbol it was drawn on.
// positionMeasurements() (defined below) redraws the whole set on demand;
// it silently skips (never deletes) any entry whose time/price doesn't
// currently resolve to a coordinate — panned out of view, or a symbol whose
// axis doesn't cover that point — so it reappears once back in range.
let measureMode = false;
let measurements = [];        // persisted: [{t1,p1,t2,p2}]
let measureDragging = false;
let measureDragStart = null;  // {t,p,x,y}, pane-relative; set on mousedown
let measureDragEnd = null;
let selectedMeasure = -1;     // index into `measurements` showing its delete button

function setMeasureMode(on) {
  measureMode = on;
  document.getElementById('measureBtn').classList.toggle('on', on);
  document.getElementById('chart').style.cursor = on ? 'crosshair' : '';
  // Without this, the chart's own drag-to-pan/scroll-to-zoom handling (on
  // the canvas itself) still fires alongside our mousedown/mousemove
  // listeners below — the two aren't mutually exclusive at the DOM-event
  // level, so a measure drag was also panning the chart underneath it.
  // Re-applied in render() too, since 畫圖 toggles rebuild `chart` from
  // scratch (fresh options, always handleScroll/handleScale defaulted back
  // to true) without going through setMeasureMode again.
  if (chart) chart.applyOptions({ handleScroll: !on, handleScale: !on });
  if (!on) {
    measureDragging = false;
    measureDragStart = null;
    measureDragEnd = null;
    positionMeasurements();
  }
}
document.getElementById('measureBtn').addEventListener('click', e => {
  e.stopPropagation();
  setMeasureMode(!measureMode);
});
// Capture phase, not bubble: showCtxMenu()'s own Escape handler
// (ctxMenuEscape) is registered on `document` in the bubble phase, which —
// since bare addEventListener here means `window`, and bubbling reaches
// document before window — would otherwise run FIRST and null out ctxMenuEl
// before this code ever sees it, so pressing Escape to dismiss only this
// measurement's right-click delete menu silently exited measure mode too.
// Confirmed live (dispatching a synthetic Escape and inspecting ctxMenuEl
// immediately after showed it already null by the time this ran) — capture
// phase runs window before document, fixing the ordering.
addEventListener('keydown', e => {
  if (e.key === 'Escape' && measureMode && !ctxMenuEl) setMeasureMode(false);
}, true);

// Pane-relative client coords -> {t,p,x,y}, gated to inside pane 0 (the main
// price pane) — same paneIndex===0 restriction subscribeDblClick uses above,
// so a drag started over a volume/RSI/MACD/DMI sub-pane is simply ignored
// rather than measuring against the wrong series.
// Pane-relative pixel -> {t,p}, no bounds gate (unlike measurePoint below) —
// used for whole-line dragging, where an endpoint can briefly track a
// cursor position slightly outside the pane during a fast drag.
function measurePointFromCoord(x, y) {
  if (!chart || !candleSeries) return null;
  const t = chart.timeScale().coordinateToTime(x);
  const p = candleSeries.coordinateToPrice(y);
  if (t === null || p === null) return null;
  return { t, p };
}

function measurePoint(clientX, clientY) {
  if (!chart || !candleSeries) return null;
  const panes = chart.panes();
  if (!panes.length) return null;
  const rect = panes[0].getHTMLElement().getBoundingClientRect();
  const x = clientX - rect.left, y = clientY - rect.top;
  if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;
  const tp = measurePointFromCoord(x, y);
  if (!tp) return null;
  return { t: tp.t, p: tp.p, x, y };
}

// {t,p} -> pane-relative pixel, the inverse of measurePointFromCoord — used
// to capture a line's current on-screen position before a whole-line drag.
function measureCoord(t, p) {
  if (!chart || !candleSeries) return null;
  const x = chart.timeScale().timeToCoordinate(t);
  const y = candleSeries.priceToCoordinate(p);
  if (x === null || y === null) return null;
  return { x, y };
}

// These three listeners are wired once at load, not re-wired per render() —
// chart/candleSeries are module-level vars every render() reassigns, so
// looking them up fresh (via measurePoint(), itself called fresh) on each
// event keeps this correct across chart rebuilds without ever needing to
// re-attach.
document.addEventListener('mousedown', e => {
  if (!measureMode || e.button !== 0) return;
  const pt = measurePoint(e.clientX, e.clientY);
  if (!pt) return;
  measureDragging = true;
  measureDragStart = pt;
  measureDragEnd = pt;
  positionMeasurements();
});
// Throttles the "something might have moved, reposition" path (used below
// for drags that aren't our own measure-drag — e.g. a price-axis rescale)
// to once per animation frame, deferred rather than synchronous. This is
// load-bearing, not an optimization: rebuilding #measureLayer synchronously
// inside a mouseup listener destroys and recreates the .measureBox/hit-line
// elements before the browser's own mousedown+mouseup -> click synthesis
// resolves — confirmed live, a plain click on a measurement's box stopped
// registering as a click at all once this ran synchronously on every
// mouseup, because the element it was about to fire 'click' at no longer
// existed. Our OWN measure-drag repositioning (below) stays synchronous
// since a real drag never gets a synthesized 'click' to break.
let measureRafPending = false;
function scheduleMeasureReposition() {
  if (measureRafPending) return;
  measureRafPending = true;
  requestAnimationFrame(() => { measureRafPending = false; positionMeasurements(); });
}

document.addEventListener('mousemove', e => {
  if (measureDragging) {
    const pt = measurePoint(e.clientX, e.clientY);
    if (pt) measureDragEnd = pt;
    positionMeasurements();
    return;
  }
  // Not gated on measureMode: Lightweight Charts doesn't expose a "visible
  // price range changed" subscription the way it does
  // subscribeVisibleTimeRangeChange for the time axis, so a price-axis drag
  // (rescaling the right price scale) never repositioned persisted
  // measurements on its own — every line/box just sat frozen at its old
  // pixel position while the candles rescaled underneath it. Reposition on
  // any left-button drag, not just our own.
  if (e.buttons & 1) scheduleMeasureReposition();
});
document.addEventListener('mouseup', () => {
  if (!measureDragging) { scheduleMeasureReposition(); return; }
  measureDragging = false;
  // A plain click (no real drag) creates nothing — only an actual drag past
  // a small pixel threshold counts, so arming measure mode and clicking
  // around doesn't litter the chart with zero-length measurements.
  const moved = measureDragStart && measureDragEnd &&
    (Math.abs(measureDragEnd.x - measureDragStart.x) > 4 ||
     Math.abs(measureDragEnd.y - measureDragStart.y) > 4);
  if (moved) {
    measurements.push({
      t1: measureDragStart.t, p1: measureDragStart.p,
      t2: measureDragEnd.t, p2: measureDragEnd.p,
    });
    // One measurement and done — matches how 畫圖's own one-shot actions
    // behave, and avoids the tool staying armed (crosshair cursor, chart
    // panning disabled) after the user is done with it.
    setMeasureMode(false);
  }
  measureDragStart = null;
  measureDragEnd = null;
  positionMeasurements();
});

function measureBarCount(t1, t2) {
  if (!BAR || !BAR.idx) return null;
  const i1 = BAR.idx.get(t1), i2 = BAR.idx.get(t2);
  if (i1 === undefined || i2 === undefined) return null;
  return Math.abs(i2 - i1);
}

// Daily bars carry a 'YYYY-MM-DD' string, intraday a UNIX-seconds number —
// same type branch applyDefaultRange() uses above for the same reason.
function measureDayCount(t1, t2) {
  const ms1 = typeof t1 === 'string' ? new Date(t1 + 'T00:00:00Z').getTime() : t1 * 1000;
  const ms2 = typeof t2 === 'string' ? new Date(t2 + 'T00:00:00Z').getTime() : t2 * 1000;
  return Math.round(Math.abs(ms2 - ms1) / 86400000);
}

function measureBoxHTML(p1, p2, bars, days) {
  const pct = (p2 - p1) / p1 * 100;
  const cls = pct >= 0 ? 'pos' : 'neg';
  const pctSign = pct >= 0 ? '+' : '';
  const chgSign = (p2 - p1) >= 0 ? '+' : '';
  const parts = [];
  if (bars !== null) parts.push(`${bars} 支K棒`);
  parts.push(`${days} 天`);
  return `<div class="mPct ${cls}">${pctSign}${pct.toFixed(2)}%</div>` +
         `<div class="mSub">${chgSign}${(p2 - p1).toFixed(2)} · ${parts.join(' · ')}</div>`;
}

// A normal click on a measurement's line/box enters edit mode for it (shows
// its delete button and two draggable endpoint dots, see positionMeasurements()
// below) — stopPropagation so the same click doesn't immediately re-trigger
// the outside-click exit listener further down. suppressNextDeselect (set by
// the drag handlers below, whenever a drag actually moved something) skips
// the toggle for the click that follows a real drag — without it, dragging
// the box or a dot would immediately deselect right after, since that
// trailing click still lands on this same listener.
let suppressNextDeselect = false;
function attachMeasureSelect(el, i) {
  el.addEventListener('click', e => {
    e.stopPropagation();
    if (suppressNextDeselect) { suppressNextDeselect = false; return; }
    selectedMeasure = (selectedMeasure === i) ? -1 : i;
    positionMeasurements();
  });
}

// Edit-mode endpoint dragging: which:1 drags {t1,p1}, which:2 drags {t2,p2}
// of measurements[i], mutated directly (not replaced) so positionMeasurements()
// picks up the live change on every move.
let editDragging = null;      // {i, which} while dragging a dot
function attachMeasureDot(el, i, which) {
  el.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    editDragging = { i, which };
  });
  // Not a drag — e.g. a stray click on a dot without moving it — should
  // stay in edit mode too, same reasoning as attachMeasureSelect() above.
  el.addEventListener('click', e => e.stopPropagation());
}

// Whole-line dragging via the box, only once the measurement is already
// selected (selectedMeasure === i — checked on mousedown, dynamically, not
// just when the listener was attached, since the box is re-rendered on
// every positionMeasurements() call regardless of selection state). Tracks
// the drag in pane-relative PIXEL space (both endpoints' on-screen position
// at drag start, plus the client-coordinate delta since) rather than doing
// arithmetic on `t`/`p` directly — `t` is either a 'YYYY-MM-DD' string or a
// UNIX-seconds number depending on interval (see measureDayCount() above),
// and the chart's time axis is per-bar/categorical, not continuous, so pixel
// deltas re-resolved through the chart's own coordinate mapping are the only
// robust way to move both ends together by "the same amount".
let boxDragging = null;  // {i, x1,y1,x2,y2 (pane px at drag start), startX,startY (client px), moved}
function attachMeasureBoxDrag(el, i) {
  el.addEventListener('mousedown', e => {
    if (e.button !== 0 || selectedMeasure !== i) return;
    const m = measurements[i];
    const c1 = measureCoord(m.t1, m.p1), c2 = measureCoord(m.t2, m.p2);
    if (!c1 || !c2) return;
    e.stopPropagation();
    e.preventDefault();
    boxDragging = { i, x1: c1.x, y1: c1.y, x2: c2.x, y2: c2.y,
                     startX: e.clientX, startY: e.clientY, moved: false };
  });
}

document.addEventListener('mousemove', e => {
  if (editDragging) {
    const pt = measurePoint(e.clientX, e.clientY);
    if (!pt) return;
    const m = measurements[editDragging.i];
    if (!m) { editDragging = null; return; }
    if (editDragging.which === 1) { m.t1 = pt.t; m.p1 = pt.p; }
    else { m.t2 = pt.t; m.p2 = pt.p; }
    positionMeasurements();
  } else if (boxDragging) {
    const dx = e.clientX - boxDragging.startX, dy = e.clientY - boxDragging.startY;
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) boxDragging.moved = true;
    const m = measurements[boxDragging.i];
    if (!m) { boxDragging = null; return; }
    const p1 = measurePointFromCoord(boxDragging.x1 + dx, boxDragging.y1 + dy);
    const p2 = measurePointFromCoord(boxDragging.x2 + dx, boxDragging.y2 + dy);
    // Skip rather than partially apply — e.g. dragged far enough that one
    // end no longer resolves to a coordinate — so one endpoint doesn't snap
    // somewhere unrelated while the other keeps tracking the cursor.
    if (!p1 || !p2) return;
    m.t1 = p1.t; m.p1 = p1.p;
    m.t2 = p2.t; m.p2 = p2.p;
    positionMeasurements();
  }
});
document.addEventListener('mouseup', () => {
  if (editDragging) {
    editDragging = null;
  } else if (boxDragging) {
    if (!boxDragging.moved) { boxDragging = null; return; }
    boxDragging = null;
  } else {
    return;
  }
  // The dragged dot/box tracks the cursor and gets rebuilt under it on every
  // frame, so the native 'click' event synthesized right after this mouseup
  // can land back on it (harmless, it stops its own propagation) or on
  // whatever else is now under the cursor, which would otherwise bubble to
  // the outside-click exit listener below and end edit mode the instant a
  // drag finishes. Suppress exactly one click, self-clearing on the next
  // tick so it can never swallow a later, real outside click on the
  // occasions no click event fires here at all.
  suppressNextDeselect = true;
  setTimeout(() => { suppressNextDeselect = false; }, 0);
  positionMeasurements();
});

addEventListener('click', () => {
  if (suppressNextDeselect) { suppressNextDeselect = false; return; }
  if (selectedMeasure === -1) return;
  selectedMeasure = -1;
  positionMeasurements();
});

// Renders every persisted measurement plus the in-progress drag (if any) as
// a dashed connector line (SVG, sized/positioned to just the two points'
// bounding box rather than spanning the whole chart) and a floating stat
// box — same rect-based-not-hand-tracked positioning as showJumpIndicator()
// below uses (pane rect relative to #chartWrap rect). Rebuilds #measureLayer
// from scratch on every call rather than diffing, since the set is small.
function positionMeasurements() {
  const layer = document.getElementById('measureLayer');
  layer.innerHTML = '';
  if (!chart || !candleSeries) return;
  const panes = chart.panes();
  if (!panes.length) return;
  const pane0 = panes[0].getHTMLElement();
  const paneRect = pane0.getBoundingClientRect();
  const wrapRect = document.getElementById('chartWrap').getBoundingClientRect();
  const ts = chart.timeScale();

  const items = measurements.map((m, i) => ({ m, i, live: false }));
  if (measureDragging && measureDragStart && measureDragEnd) {
    items.push({
      m: { t1: measureDragStart.t, p1: measureDragStart.p,
           t2: measureDragEnd.t, p2: measureDragEnd.p },
      i: -1, live: true,
    });
  }

  items.forEach(({ m, i, live }) => {
    const x1 = ts.timeToCoordinate(m.t1), x2 = ts.timeToCoordinate(m.t2);
    const y1 = candleSeries.priceToCoordinate(m.p1), y2 = candleSeries.priceToCoordinate(m.p2);
    if (x1 === null || x2 === null || y1 === null || y2 === null) return;

    const wx1 = x1 + paneRect.left - wrapRect.left, wy1 = y1 + paneRect.top - wrapRect.top;
    const wx2 = x2 + paneRect.left - wrapRect.left, wy2 = y2 + paneRect.top - wrapRect.top;

    const svgNS = 'http://www.w3.org/2000/svg';
    const left = Math.min(wx1, wx2), top = Math.min(wy1, wy2);
    const w = Math.max(Math.abs(wx2 - wx1), 1), h = Math.max(Math.abs(wy2 - wy1), 1);
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);
    svg.style.position = 'absolute';
    svg.style.left = left + 'px';
    svg.style.top = top + 'px';
    svg.style.overflow = 'visible';
    svg.style.pointerEvents = 'none';
    const lx1 = wx1 - left, ly1 = wy1 - top, lx2 = wx2 - left, ly2 = wy2 - top;

    // A fat transparent hit-stroke under the thin visible dashed line — a
    // literal 1.5px line is nearly impossible to land a click on.
    if (!live) {
      const hit = document.createElementNS(svgNS, 'line');
      hit.setAttribute('x1', lx1); hit.setAttribute('y1', ly1);
      hit.setAttribute('x2', lx2); hit.setAttribute('y2', ly2);
      hit.setAttribute('stroke', 'transparent');
      hit.setAttribute('stroke-width', '10');
      hit.style.pointerEvents = 'auto';
      hit.style.cursor = 'pointer';
      attachMeasureSelect(hit, i);
      svg.appendChild(hit);
    }

    const vis = document.createElementNS(svgNS, 'line');
    vis.setAttribute('x1', lx1); vis.setAttribute('y1', ly1);
    vis.setAttribute('x2', lx2); vis.setAttribute('y2', ly2);
    vis.setAttribute('stroke', '#38bdf8');
    vis.setAttribute('stroke-width', '1.5');
    vis.setAttribute('stroke-dasharray', '4 3');
    svg.appendChild(vis);
    layer.appendChild(svg);

    // Centered on the date range (x midpoint of the two anchors), not
    // pinned to the drag's end point — reads as labelling the whole range
    // rather than just where the mouse happened to let go. A losing (price
    // down) measurement sits below the line's lower point instead of above
    // its upper one — above would either overlap the line itself or, for a
    // steep drop, land awkwardly off to one side of it.
    const isDown = m.p2 < m.p1;
    const box = document.createElement('div');
    box.className = 'measureBox' + (isDown ? ' below' : '');
    box.style.left = ((wx1 + wx2) / 2) + 'px';
    box.style.top = (isDown ? Math.max(wy1, wy2) : Math.min(wy1, wy2)) + 'px';
    box.innerHTML = measureBoxHTML(m.p1, m.p2, measureBarCount(m.t1, m.t2), measureDayCount(m.t1, m.t2));
    if (live) {
      box.style.pointerEvents = 'none';
    } else {
      attachMeasureSelect(box, i);
      // A no-op until the measurement is already selected (see
      // attachMeasureBoxDrag()'s own selectedMeasure check) — safe to wire
      // up unconditionally on every render the same way attachMeasureSelect
      // is above, rather than only once selected.
      attachMeasureBoxDrag(box, i);
      box.style.cursor = (i === selectedMeasure) ? 'move' : 'pointer';
    }
    layer.appendChild(box);

    // Edit mode for the selected measurement: draggable endpoint dots plus
    // the delete button, positioned off the box's own actual rendered rect
    // (not a guessed offset) so the button lands correctly regardless of how
    // long the box's day/bar-count text is.
    if (!live && i === selectedMeasure) {
      const dot1 = document.createElement('div');
      dot1.className = 'measureDot';
      dot1.style.left = wx1 + 'px';
      dot1.style.top = wy1 + 'px';
      attachMeasureDot(dot1, i, 1);
      layer.appendChild(dot1);

      const dot2 = document.createElement('div');
      dot2.className = 'measureDot';
      dot2.style.left = wx2 + 'px';
      dot2.style.top = wy2 + 'px';
      attachMeasureDot(dot2, i, 2);
      layer.appendChild(dot2);

      const boxRect = box.getBoundingClientRect();
      const del = document.createElement('button');
      del.className = 'measureDel';
      del.title = '刪除量度';
      del.innerHTML = TRASH_ICON;
      del.style.left = (boxRect.right - wrapRect.left + 6) + 'px';
      del.style.top = (boxRect.top - wrapRect.top) + 'px';
      del.addEventListener('click', e => {
        e.stopPropagation();
        measurements.splice(i, 1);
        selectedMeasure = -1;
        positionMeasurements();
      });
      layer.appendChild(del);
    }
  });
}

const toggleBtn = document.getElementById('toggle');
function toggleSubPanes() {
  subPanesOn = !subPanesOn;
  toggleBtn.title = subPanesOn ? '只看K線' : '顯示指標';
  toggleBtn.classList.toggle('on', !subPanesOn);
  render();
}
toggleBtn.addEventListener('click', e => {
  e.stopPropagation();
  toggleSubPanes();
});

// Seed currentParams from whatever the server actually computed (every
// symbol's payload carries the same shared settings), then, if a saved
// preference exists, override it and recompute every symbol before the
// first render() — so the very first paint already reflects it instead of
// showing the server defaults and then visibly jumping.
currentParams = withIndicatorDefaults(ALL.symbols[current].params);
const savedParams = loadSavedParams();
if (savedParams) {
  currentParams = withIndicatorDefaults(savedParams);
  for (const sym of Object.keys(ALL.symbols)) {
    recomputeIndicatorsFor(ALL.symbols[sym], currentParams);
  }
}

document.getElementById('optObBearish').disabled = !obOn;
document.getElementById('optObBullish').disabled = !obOn;
document.getElementById('obFraction').disabled = !obOn;

// Read before the first select()/render() so the very first paint already
// reflects a saved Trend preference, same as indicator params above.
const savedTrend = loadSavedTrend();
if (savedTrend && (savedTrend.trendMode === 'regime' || savedTrend.trendMode === '5day')) {
  trendMode = savedTrend.trendMode;
  if (Number.isInteger(savedTrend.trendBars) && savedTrend.trendBars >= 1) {
    trendBars = savedTrend.trendBars;
  }
}
trendModeSel.value = trendMode;
trendBarsInput.value = trendBars;
trendBarsInput.disabled = trendMode !== '5day';

// Same idea for 收市比例 — read before select() so a saved non-default
// value is already in effect for the very first render/fetch.
const savedObFraction = loadSavedObFraction();
if (savedObFraction && typeof savedObFraction.obCloseFraction === 'number'
    && savedObFraction.obCloseFraction >= 0 && savedObFraction.obCloseFraction <= 1) {
  obCloseFraction = savedObFraction.obCloseFraction;
}
document.getElementById('obFraction').value = obCloseFraction;

renderWatchlistPanel();
renderAlertsPanel();
// The alerts log isn't in the server-rendered __DATA__ payload (unlike
// watchlist/layout) — fetched once here, then the panel re-renders with
// real acked state once it lands. Renders once already above so the panel
// isn't empty/stuck while this is in flight; that first pass just treats
// everything as unacked, corrected the moment this resolves.
fetch('/api/alerts/log').then(r => r.json()).then(log => {
  ALERT_LOG = log;
  rebuildAlertLogMap();
  renderAlertsPanel();
}).catch(err => console.error('alerts log fetch failed:', err.message));
select(current);
addEventListener('resize', resizeChartToContainer);
</script>
"""


def resolve_watchlist(args) -> dict:
    """Work out which symbols to render: positional symbols override the config file.

    Positional symbols render with every tag empty (there is no source to
    resolve tags from on the command line); otherwise every symbol in
    ``watchlists.json`` is rendered with its stored tags.
    """
    if args.symbols:
        return {
            sym: {"held": False, "strategies": [], "stages": [], "patterns": []}
            for sym in args.symbols
        }

    from .watchlist import load as load_watchlist

    if args.watchlists and args.watchlists.exists():
        return load_watchlist(args.watchlists)
    return {}


def render(
    payloads: dict,
    watchlist: dict,
    out: Optional[Path] = None,
) -> Path:
    """Write one standalone HTML holding every symbol.

    A single page rather than a file per ticker: the chart is rebuilt in the
    browser on switch, so comparing symbols is a click instead of opening
    another tab, and the 196KB charting bundle is inlined once instead of per
    symbol.
    """
    if not BUNDLE.exists():
        raise FileNotFoundError(
            f"Missing charting bundle at {BUNDLE}. "
            "Copy lightweight-charts.standalone.production.js into vendor/."
        )

    out = out or ROOT / "out" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    html = (
        _TEMPLATE.replace("__BUNDLE__", BUNDLE.read_text(encoding="utf-8"))
        .replace("__DATA__", json.dumps({"symbols": payloads, "watchlist": watchlist}))
        .replace("__PATTERNS__", json.dumps(PATTERN_CATALOG))
    )
    out.write_text(html, encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .data import load

    ap = argparse.ArgumentParser(description="Render a 生死線 chart.")
    ap.add_argument(
        "symbols",
        nargs="*",
        help="Symbols to render. Overrides the tagged watchlist.",
    )
    ap.add_argument(
        "--watchlists",
        type=Path,
        default=ROOT / "watchlists.json",
        help="Tagged watchlist file (default watchlists.json)",
    )
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--open", action="store_true", help="open in the browser")
    ap.add_argument(
        "--no-down-arrows",
        action="store_false",
        dest="down_arrows",
        help="mark 升穿 only, matching the reference chart",
    )
    ap.add_argument("--rsi", type=int, default=9, help="RSI period")
    ap.add_argument("--rsi-signal", type=int, default=6, help="SMA of RSI")
    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)
    ap.add_argument("--di", type=int, default=6, help="DMI +DI/-DI period")
    ap.add_argument("--adx", type=int, default=14, help="ADX smoothing period")
    args = ap.parse_args(argv)

    params = {
        "rsi": args.rsi,
        "rsi_signal": args.rsi_signal,
        "macd_fast": args.macd_fast,
        "macd_slow": args.macd_slow,
        "macd_signal": args.macd_signal,
        "di": args.di,
        "adx": args.adx,
    }

    watchlist = resolve_watchlist(args)

    payloads = {}
    for symbol in watchlist:
        df = load(symbol, args.start, args.end)
        result, alts = _ladders(df)
        payloads[symbol] = build_payload(
            df, result, symbol, args.down_arrows, params, alts
        )
        print(f"  {symbol}: {len(df)} bars")

    if not payloads:
        print("No symbols to render — check watchlists.json")
        return 1

    path = render(payloads, watchlist)
    print(path)
    if args.open:
        webbrowser.open(path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""陰陽燭形態 — candlestick pattern detection.

Ported from a Pine Script v6 indicator (`Candlestick Patterns Identified,
update 1-17-26`, repo32) — originally 15 patterns across single-, two-, and
three-candle shapes, plus Dark Cloud Cover (`sdx`-original — the source
script never had it; added as Piercing Line's literal bearish mirror so the
雙日 set isn't missing its one-sided pair) and minus Bullish Belt (dropped
post-port). `Pattern.value` keeps Pine's own `title=` strings for source
parity (see `openspec/specs/candlestick-patterns/spec.md`); user-facing
display goes through the separate `zh_name`/`marker_text` Chinese
translation layer near the bottom of this module instead.

Two things the source script leaves as an indicator input are exposed here
as function parameters rather than hardcoded: the "trend" bar-count and its
alternative, this app's own swing-structure `regime` (see
:func:`_trend_flags`). Everything else — doji size, all shape thresholds —
is a faithful, literal port of the script's math.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from .types import Direction


class Pattern(str, Enum):
    # 單日 — one candle
    DOJI = "Doji"
    HAMMER = "Hammer"
    INVERTED_HAMMER = "Inverted Hammer"
    HANGING_MAN = "Hanging Man"
    SHOOTING_STAR = "Shooting Star"
    SPINNING_TOP_WHITE = "Spinning Top White"
    SPINNING_TOP_BLACK = "Spinning Top Black"

    # 雙日 — two candles
    BEARISH_HARAMI = "Bearish Harami"
    BULLISH_HARAMI = "Bullish Harami"
    BEARISH_ENGULFING = "Bearish Engulfing"
    BULLISH_ENGULFING = "Bullish Engulfing"
    PIERCING_LINE = "Piercing Line"
    DARK_CLOUD_COVER = "Dark Cloud Cover"
    HARAMI_CROSS_BULLISH = "Harami Cross Bullish"
    HARAMI_CROSS_BEARISH = "Harami Cross Bearish"
    TWEEZER_BOTTOM = "Tweezer Bottom"
    TWEEZER_TOP = "Tweezer Top"

    # 三日 — three candles
    EVENING_STAR = "Evening Star"
    MORNING_STAR = "Morning Star"
    ABANDONED_BABY_BULLISH = "Abandoned Baby Bullish"
    ABANDONED_BABY_BEARISH = "Abandoned Baby Bearish"
    EVENING_DOJI_STAR = "Evening Doji Star"
    MORNING_DOJI_STAR = "Morning Doji Star"
    THREE_BLACK_CROWS = "Three Black Crows"
    THREE_WHITE_SOLDIERS = "Three White Soldiers"

    # 五日 — five candles
    RISING_THREE_METHODS = "Rising Three Methods"
    FALLING_THREE_METHODS = "Falling Three Methods"

    @property
    def kind(self) -> str:
        """單日/雙日/三日/五日, for filtering on the chart."""
        if self in _FIVE_BAR:
            return "五日"
        if self in _THREE_BAR:
            return "三日"
        if self in _TWO_BAR:
            return "雙日"
        return "單日"

    @property
    def color(self) -> str:
        """Pine's own default palette — bullColor/bearColor/dojiColor."""
        return _COLOR[self]

    @property
    def above_bar(self) -> bool:
        """Marker position, mirroring the source script's `location=`
        (unset in Pine defaults to above-bar)."""
        return self in _ABOVE_BAR

    @property
    def shape(self) -> str:
        """Marker shape. Pine's own `style=` is `shape.arrowup` for every
        bullColor pattern and `shape.arrowdown` for every bearColor one —
        mapped losslessly here. Doji (`shape.cross`) and Hammer/Inverted
        Hammer (`shape.diamond`) have no equivalent in this app's charting
        library (Lightweight Charts supports only circle/square/arrowUp/
        arrowDown), so those three fall back to `circle`."""
        if self in _NEUTRAL_SHAPE:
            return "circle"
        return "arrowUp" if self.color == _BULL else "arrowDown"

    @property
    def zh_name(self) -> str:
        """Chinese display name — see module-level `_ZH`. Kept separate
        from `.value` (Pine's own English `title=`, preserved for source
        parity) so this is purely a display-layer translation."""
        return _ZH[self]

    @property
    def marker_text(self) -> str:
        """On-chart label, in Chinese. Shortened for Hammer/Inverted Hammer
        (mirroring Pine's own shorter `text=` vs `title=` for those two:
        "H"/"IH") — every other pattern's marker text is its full
        `zh_name`."""
        return _ZH_MARKER_TEXT.get(self, self.zh_name)

    @property
    def direction(self) -> str | None:
        """看好 (bullish) / 看淡 (bearish) / None (direction-neutral — Doji,
        Hammer/Inverted Hammer, Spinning Top). Not read anywhere on-chart —
        markers/hover text already carry direction via `.color`, and a text
        suffix there would be redundant (see `zh_name`'s own docstring on
        why Bearish/Bullish Harami etc. deliberately share one name). Exists
        for the 畫圖 menu (sdx/viz.py's PATTERN_CATALOG), which has no color
        coding of its own: patterns whose `zh_name` is ambiguous across
        directions (身懷六甲/穿頭破腳/十字胎) get this appended there to tell
        their two checkboxes apart."""
        if self.color == _BULL:
            return "看好"
        if self.color == _BEAR:
            return "看淡"
        return None


_TWO_BAR = frozenset(
    {
        Pattern.BEARISH_HARAMI,
        Pattern.BULLISH_HARAMI,
        Pattern.BEARISH_ENGULFING,
        Pattern.BULLISH_ENGULFING,
        Pattern.PIERCING_LINE,
        Pattern.DARK_CLOUD_COVER,
        Pattern.HARAMI_CROSS_BULLISH,
        Pattern.HARAMI_CROSS_BEARISH,
        Pattern.TWEEZER_BOTTOM,
        Pattern.TWEEZER_TOP,
    }
)

_THREE_BAR = frozenset(
    {
        Pattern.EVENING_STAR,
        Pattern.MORNING_STAR,
        Pattern.ABANDONED_BABY_BULLISH,
        Pattern.ABANDONED_BABY_BEARISH,
        Pattern.EVENING_DOJI_STAR,
        Pattern.MORNING_DOJI_STAR,
        Pattern.THREE_BLACK_CROWS,
        Pattern.THREE_WHITE_SOLDIERS,
    }
)

_FIVE_BAR = frozenset({Pattern.RISING_THREE_METHODS, Pattern.FALLING_THREE_METHODS})

#: Pine's `bearColor`/`bullColor`/`dojiColor` defaults, with `bullColor`
#: toned down from Pine's pure `#00ff00` — too sharp against this app's dark
#: theme — to a softer green. `bearColor`/`dojiColor` stay Pine's exact
#: `#ff0000`/`#ffffff`.
_BULL = "#4ade80"
_BEAR = "#ff0000"
_NEUTRAL = "#ffffff"

#: Not derivable from shape direction alone — Hammer/Inverted Hammer read
#: as directional shapes but are plotted in `dojiColor`, same as Doji.
_COLOR: dict[Pattern, str] = {
    Pattern.DOJI: _NEUTRAL,
    Pattern.HAMMER: _NEUTRAL,
    Pattern.INVERTED_HAMMER: _NEUTRAL,
    Pattern.SPINNING_TOP_WHITE: _NEUTRAL,
    Pattern.SPINNING_TOP_BLACK: _NEUTRAL,
    Pattern.BEARISH_HARAMI: _BEAR,
    Pattern.BEARISH_ENGULFING: _BEAR,
    Pattern.DARK_CLOUD_COVER: _BEAR,
    Pattern.HANGING_MAN: _BEAR,
    Pattern.EVENING_STAR: _BEAR,
    Pattern.SHOOTING_STAR: _BEAR,
    Pattern.HARAMI_CROSS_BEARISH: _BEAR,
    Pattern.TWEEZER_TOP: _BEAR,
    Pattern.ABANDONED_BABY_BEARISH: _BEAR,
    Pattern.EVENING_DOJI_STAR: _BEAR,
    Pattern.THREE_BLACK_CROWS: _BEAR,
    Pattern.FALLING_THREE_METHODS: _BEAR,
    Pattern.BULLISH_HARAMI: _BULL,
    Pattern.BULLISH_ENGULFING: _BULL,
    Pattern.PIERCING_LINE: _BULL,
    Pattern.MORNING_STAR: _BULL,
    Pattern.HARAMI_CROSS_BULLISH: _BULL,
    Pattern.TWEEZER_BOTTOM: _BULL,
    Pattern.ABANDONED_BABY_BULLISH: _BULL,
    Pattern.MORNING_DOJI_STAR: _BULL,
    Pattern.THREE_WHITE_SOLDIERS: _BULL,
    Pattern.RISING_THREE_METHODS: _BULL,
}

#: Marker position per pattern. For the original 15, this is Pine's own
#: `location=` (every bearColor pattern plus Doji has no `location=`
#: argument, defaulting to above-bar; every bullColor pattern plus
#: Hammer/Inverted Hammer sets `location=location.belowbar` explicitly).
#: For the 15 ported from All Patterns.ps, this is that script's own label
#: position (PosHigh/style_down = above, PosLow/style_up = below) — except
#: Spinning Top White/Black, moved to above-bar to match Doji (see below).
_ABOVE_BAR = frozenset(
    {
        Pattern.DOJI,
        Pattern.BEARISH_HARAMI,
        Pattern.BEARISH_ENGULFING,
        Pattern.DARK_CLOUD_COVER,
        Pattern.HANGING_MAN,
        Pattern.EVENING_STAR,
        Pattern.SHOOTING_STAR,
        Pattern.HARAMI_CROSS_BEARISH,
        Pattern.TWEEZER_TOP,
        Pattern.ABANDONED_BABY_BEARISH,
        Pattern.EVENING_DOJI_STAR,
        Pattern.THREE_BLACK_CROWS,
        Pattern.FALLING_THREE_METHODS,
        # Not from All Patterns.ps's own PosLow placement — moved to match
        # Doji (also neutral/indecision, also above-bar) per user request,
        # overriding that script's literal marker position for these two.
        Pattern.SPINNING_TOP_WHITE,
        Pattern.SPINNING_TOP_BLACK,
    }
)

#: Doji (`shape.cross`), Hammer/Inverted Hammer (`shape.diamond`), and
#: Spinning Top White/Black (no directional bias — an indecision shape,
#: like Doji) — patterns with no directional arrow equivalent in this
#: app's charting library (Lightweight Charts supports only
#: circle/square/arrowUp/arrowDown), so all fall back to `circle`.
_NEUTRAL_SHAPE = frozenset(
    {
        Pattern.DOJI,
        Pattern.HAMMER,
        Pattern.INVERTED_HAMMER,
        Pattern.SPINNING_TOP_WHITE,
        Pattern.SPINNING_TOP_BLACK,
    }
)

#: Chinese display names — the translation layer `zh_name`/`marker_text`
#: read from. Bearish/bullish Harami share 身懷六甲, and bearish/bullish
#: Engulfing share 穿頭破腳, by design: those are classical idioms naming
#: the *shape*, not the direction, and direction is already carried by
#: `color` (red/green) wherever these are shown, same as the English
#: `.value` pairs weren't distinguished by suffix either.
_ZH: dict[Pattern, str] = {
    Pattern.DOJI: "十字星",
    Pattern.HAMMER: "鎚頭",
    Pattern.INVERTED_HAMMER: "倒轉鎚頭",
    Pattern.HANGING_MAN: "吊頸",
    Pattern.SHOOTING_STAR: "射擊之星",
    Pattern.SPINNING_TOP_WHITE: "陀螺",
    Pattern.SPINNING_TOP_BLACK: "陀螺",
    Pattern.BEARISH_HARAMI: "身懷六甲",
    Pattern.BULLISH_HARAMI: "身懷六甲",
    Pattern.BEARISH_ENGULFING: "穿頭破腳",
    Pattern.BULLISH_ENGULFING: "穿頭破腳",
    Pattern.PIERCING_LINE: "曙光初現",
    Pattern.DARK_CLOUD_COVER: "烏雲蓋頂",
    Pattern.HARAMI_CROSS_BULLISH: "十字胎",
    Pattern.HARAMI_CROSS_BEARISH: "十字胎",
    Pattern.TWEEZER_BOTTOM: "平底",
    Pattern.TWEEZER_TOP: "平頂",
    Pattern.EVENING_STAR: "黃昏之星",
    Pattern.MORNING_STAR: "早晨之星",
    Pattern.ABANDONED_BABY_BULLISH: "棄嬰底",
    Pattern.ABANDONED_BABY_BEARISH: "棄嬰頂",
    Pattern.EVENING_DOJI_STAR: "黃昏十字星",
    Pattern.MORNING_DOJI_STAR: "早晨十字星",
    Pattern.THREE_BLACK_CROWS: "三隻烏鴉",
    Pattern.THREE_WHITE_SOLDIERS: "三個白武士",
    Pattern.RISING_THREE_METHODS: "上升三部曲",
    Pattern.FALLING_THREE_METHODS: "下跌三部曲",
}

#: Chinese marker-text overrides — mirrors Pine's own shorter `text=` for
#: Hammer/Inverted Hammer ("H"/"IH" vs "Hammer"/"Inverted Hammer"); every
#: other pattern's marker text is its full `zh_name`.
_ZH_MARKER_TEXT: dict[Pattern, str] = {
    Pattern.HAMMER: "鎚",
    Pattern.INVERTED_HAMMER: "倒鎚",
}

#: Pine indicator inputs kept as module constants — only "Trend in Bars" is
#: user-adjustable (a parameter on the finder functions below); doji size is
#: not exposed anywhere in the UI.
PINE_TREND_BARS = 5
PINE_DOJI_RATIO = 0.05


@dataclass(frozen=True)
class PatternHit:
    bar: int
    pattern: Pattern

    @property
    def name(self) -> str:
        return self.pattern.value


def _trend_flags(
    opens: Sequence[float],
    n: int,
    mode: str,
    trend_bars: int,
    regime: Optional[Sequence[Direction]],
) -> list[Optional[bool]]:
    """Per-bar uptrend/downtrend context — Pine's `open[trend] < open`
    proxy (mode="5day"), or this app's own swing-structure regime
    (mode="regime").

    ``True`` = uptrend (bearish patterns eligible), ``False`` = downtrend
    (bullish patterns eligible), ``None`` = unknown, matching Pine's `na`
    semantics for a series' first ``trend_bars`` bars — comparisons against
    `na` are always false in Pine, so neither direction's patterns fire,
    rather than silently defaulting to one side.
    """
    if mode == "regime":
        if regime is None:
            raise ValueError("regime is required when mode='regime'")
        return [r is Direction.UP for r in regime]

    out: list[Optional[bool]] = []
    for i in range(n):
        if i < trend_bars:
            out.append(None)
            continue
        older, cur = opens[i - trend_bars], opens[i]
        if older < cur:
            out.append(True)
        elif older > cur:
            out.append(False)
        else:
            out.append(None)
    return out


#: EMA length behind the *All Candlestick Patterns* script's `C_BodyAvg` —
#: shared by every pattern ported from that script (Dark Cloud Cover,
#: Harami Cross, Tweezer Top/Bottom, Three Black Crows/White Soldiers,
#: Evening/Morning Doji Star, Rising/Falling Three Methods) to classify a
#: bar's body as "long" (body > its own trailing average) or "small".
BODY_EMA_LEN = 14


def _body_average(
    opens: Sequence[float], closes: Sequence[float]
) -> list[float]:
    """Causal EMA of candle body size — Pine's `ta.ema(C_Body, 14)`, seeded
    with the first bar's own body (Pine's own EMA seeding; no SMA warm-up)."""
    alpha = 2 / (BODY_EMA_LEN + 1)
    out: list[float] = []
    prev: Optional[float] = None
    for o, c in zip(opens, closes):
        body = abs(c - o)
        prev = body if prev is None else alpha * body + (1 - alpha) * prev
        out.append(prev)
    return out


def find_patterns(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    regime: Optional[Sequence[Direction]] = None,
    *,
    trend_mode: str = "regime",
    trend_bars: int = PINE_TREND_BARS,
) -> list[PatternHit]:
    """單日 — Doji, Hammer, Inverted Hammer, Hanging Man, Shooting Star,
    Spinning Top White/Black. Causal — every lookback stays
    at or before bar i."""
    out: list[PatternHit] = []
    n = len(closes)
    trend = _trend_flags(opens, n, trend_mode, trend_bars, regime)

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        rng = h - l
        body = abs(o - c)
        denom = 0.001 + rng  # Pine's own div-by-zero guard, ported literally

        if body <= PINE_DOJI_RATIO * rng:
            out.append(PatternHit(i, Pattern.DOJI))

        if rng > 3 * body and (c - l) / denom > 0.6 and (o - l) / denom > 0.6:
            out.append(PatternHit(i, Pattern.HAMMER))

        if rng > 3 * body and (h - c) / denom > 0.6 and (h - o) / denom > 0.6:
            out.append(PatternHit(i, Pattern.INVERTED_HAMMER))

        if (
            i >= 2
            and rng > 4 * body
            and (c - l) / denom >= 0.75
            and (o - l) / denom >= 0.75
            and trend[i] is True
            and highs[i - 1] < o
            and highs[i - 2] < o
        ):
            out.append(PatternHit(i, Pattern.HANGING_MAN))

        if (
            i >= 1
            and opens[i - 1] < closes[i - 1]
            and o > closes[i - 1]
            and h - max(o, c) >= body * 3
            and min(c, o) - l <= body
        ):
            out.append(PatternHit(i, Pattern.SHOOTING_STAR))

        up_shadow = h - max(o, c)
        dn_shadow = min(o, c) - l
        is_doji = body <= PINE_DOJI_RATIO * rng
        is_spinning_top = dn_shadow >= rng * 0.34 and up_shadow >= rng * 0.34 and not is_doji

        if is_spinning_top and c > o:
            out.append(PatternHit(i, Pattern.SPINNING_TOP_WHITE))

        if is_spinning_top and o > c:
            out.append(PatternHit(i, Pattern.SPINNING_TOP_BLACK))

    return out


def find_two_bar_patterns(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    regime: Optional[Sequence[Direction]] = None,
    *,
    trend_mode: str = "regime",
    trend_bars: int = PINE_TREND_BARS,
    volumes: Optional[Sequence[float]] = None,
) -> list[PatternHit]:
    """雙日 — Bearish/Bullish Harami, Bearish/Bullish Engulfing, Piercing
    Line, Dark Cloud Cover, Harami Cross Bullish/Bearish, Tweezer
    Top/Bottom. Reported on the second (completing) bar. The original 6
    patterns are trend-gated on the *current* bar; five of them (all but
    Engulfing) are not volume/body-average gated (the source script they're
    ported from never references volume). Bearish/Bullish Engulfing (穿頭破腳)
    additionally require the completing bar's volume to exceed the first
    bar's — an engulfing candle on shrinking volume reads as exhaustion, not
    a genuine reversal — when `volumes` is supplied; omitting `volumes`
    (the default) skips that check entirely, matching every other pattern
    here. Dark Cloud Cover, Harami Cross, and Tweezer Top/Bottom are ported
    from a second, larger Pine script ("*All Candlestick Patterns*") and
    follow that script's own conventions instead: trend-gated on the
    *prior* bar (`trend[j]`, not `trend[i]`), and gated on `_body_average`
    to classify a bar's body as long/small."""
    out: list[PatternHit] = []
    n = len(closes)
    trend = _trend_flags(opens, n, trend_mode, trend_bars, regime)
    body_avg = _body_average(opens, closes)

    for i in range(1, n):
        j = i - 1
        oj, cj = opens[j], closes[j]
        oi, ci = opens[i], closes[i]
        bj, bi = abs(cj - oj), abs(ci - oi)
        up = trend[i]
        vol_rising = volumes is None or volumes[i] > volumes[j]

        if (
            cj > oj
            and oi > ci
            and oi <= cj
            and oj <= ci
            and bi < bj
            and up is True
        ):
            out.append(PatternHit(i, Pattern.BEARISH_HARAMI))

        if (
            oj > cj
            and ci > oi
            and ci <= oj
            and cj <= oi
            and bi < bj
            and up is False
        ):
            out.append(PatternHit(i, Pattern.BULLISH_HARAMI))

        if (
            cj > oj
            and oi > ci
            and oi >= cj
            and oj >= ci
            and bi > bj
            and up is True
            and vol_rising
        ):
            out.append(PatternHit(i, Pattern.BEARISH_ENGULFING))

        if (
            oj > cj
            and ci > oi
            and ci >= oj
            and cj >= oi
            and bi > bj
            and up is False
            and vol_rising
        ):
            out.append(PatternHit(i, Pattern.BULLISH_ENGULFING))

        if (
            cj < oj
            and oi < lows[j]
            and ci > cj + (oj - cj) / 2
            and ci < oj
            and up is False
        ):
            out.append(PatternHit(i, Pattern.PIERCING_LINE))

        if (
            trend[j] is True
            and cj > oj
            and bj > body_avg[j]
            and oi > ci
            and oi >= highs[j]
            and ci < oj + (cj - oj) / 2
            and ci > oj
        ):
            out.append(PatternHit(i, Pattern.DARK_CLOUD_COVER))

        prior_body_hi, prior_body_lo = max(oj, cj), min(oj, cj)
        is_doji_i = bi <= PINE_DOJI_RATIO * (highs[i] - lows[i])

        if (
            bj > body_avg[j]
            and oj > cj
            and trend[j] is False
            and is_doji_i
            and highs[i] <= prior_body_hi
            and lows[i] >= prior_body_lo
        ):
            out.append(PatternHit(i, Pattern.HARAMI_CROSS_BULLISH))

        if (
            bj > body_avg[j]
            and cj > oj
            and trend[j] is True
            and is_doji_i
            and highs[i] <= prior_body_hi
            and lows[i] >= prior_body_lo
        ):
            out.append(PatternHit(i, Pattern.HARAMI_CROSS_BEARISH))

        up_shadow_i = highs[i] - max(oi, ci)
        dn_shadow_i = min(oi, ci) - lows[i]
        has_up_i = up_shadow_i > 0.05 * bi
        has_dn_i = dn_shadow_i > 0.05 * bi
        not_bare_doji_i = not is_doji_i or (has_up_i and has_dn_i)

        if (
            trend[j] is True
            and not_bare_doji_i
            and abs(highs[i] - highs[j]) <= body_avg[i] * 0.05
            and cj > oj
            and oi > ci
            and bj > body_avg[j]
        ):
            out.append(PatternHit(i, Pattern.TWEEZER_TOP))

        if (
            trend[j] is False
            and not_bare_doji_i
            and abs(lows[i] - lows[j]) <= body_avg[i] * 0.05
            and oj > cj
            and ci > oi
            and bj > body_avg[j]
        ):
            out.append(PatternHit(i, Pattern.TWEEZER_BOTTOM))

    return out


def find_three_bar_patterns(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    regime: Optional[Sequence[Direction]] = None,
    *,
    trend_mode: str = "regime",
    trend_bars: int = PINE_TREND_BARS,
) -> list[PatternHit]:
    """三日 — Evening Star, Morning Star, Abandoned Baby Bullish/Bearish, and
    Evening/Morning Doji Star are all trend-gated on the FIRST bar (i-2) —
    read before the reversal candles exist, so the gate can't be
    contaminated by the pattern's own move. Only Abandoned Baby's first-bar
    gate is the literal source-script convention (from the second, larger
    "*All Candlestick Patterns*" script); Evening/Morning Star (never
    gated at all in `Indentified Patterns.ps`) and Evening/Morning Doji
    Star (gated on the completing bar i in `All Patterns.ps`) both use
    first-bar as a deliberate deviation from their respective source
    scripts. Three Black Crows/White Soldiers (also from `All Patterns.ps`)
    have no trend gate at all — only body-average-gated. Reported on the
    third (completing) bar."""
    out: list[PatternHit] = []
    n = len(closes)
    trend = _trend_flags(opens, n, trend_mode, trend_bars, regime)
    body_avg = _body_average(opens, closes)

    for i in range(2, n):
        i2, i1 = i - 2, i - 1
        o2, c2 = opens[i2], closes[i2]
        o1, c1 = opens[i1], closes[i1]
        oi, ci = opens[i], closes[i]
        mid1_lo, mid1_hi = min(o1, c1), max(o1, c1)

        if c2 > o2 and mid1_lo > c2 and oi < mid1_lo and ci < oi and trend[i2] is True:
            out.append(PatternHit(i, Pattern.EVENING_STAR))

        if c2 < o2 and mid1_hi < c2 and oi > mid1_hi and ci > oi and trend[i2] is False:
            out.append(PatternHit(i, Pattern.MORNING_STAR))

        body2 = abs(c2 - o2)
        body1 = abs(c1 - o1)
        bi = abs(ci - oi)
        range1 = highs[i1] - lows[i1]
        is_doji1 = body1 <= PINE_DOJI_RATIO * range1

        if (
            trend[i2] is False
            and o2 > c2
            and is_doji1
            and lows[i2] > highs[i1]
            and ci > oi
            and highs[i1] < lows[i]
        ):
            out.append(PatternHit(i, Pattern.ABANDONED_BABY_BULLISH))

        if (
            trend[i2] is True
            and c2 > o2
            and is_doji1
            and highs[i2] < lows[i1]
            and oi > ci
            and lows[i1] > highs[i]
        ):
            out.append(PatternHit(i, Pattern.ABANDONED_BABY_BEARISH))

        body2_lo, body2_hi = min(o2, c2), max(o2, c2)
        body2_mid = body2 / 2 + body2_lo
        body1_lo, body1_hi = min(o1, c1), max(o1, c1)
        bodyi_lo, bodyi_hi = min(oi, ci), max(oi, ci)

        if (
            body2 > body_avg[i2]
            and is_doji1
            and bi > body_avg[i]
            and trend[i2] is True
            and c2 > o2
            and body1_lo > body2_hi
            and oi > ci
            and bodyi_lo <= body2_mid
            and bodyi_lo > body2_lo
            and body1_lo > bodyi_hi
        ):
            out.append(PatternHit(i, Pattern.EVENING_DOJI_STAR))

        if (
            body2 > body_avg[i2]
            and is_doji1
            and bi > body_avg[i]
            and trend[i2] is False
            and o2 > c2
            and body1_hi < body2_lo
            and oi < ci
            and bodyi_hi >= body2_mid
            and bodyi_hi < body2_hi
            and body1_hi < bodyi_lo
        ):
            out.append(PatternHit(i, Pattern.MORNING_DOJI_STAR))

        range_i = highs[i] - lows[i]
        up_shadow_i = highs[i] - bodyi_hi
        up_shadow_1 = highs[i1] - body1_hi
        up_shadow_2 = highs[i2] - body2_hi
        dn_shadow_i = bodyi_lo - lows[i]
        dn_shadow_1 = body1_lo - lows[i1]
        dn_shadow_2 = body2_lo - lows[i2]

        if (
            bi > body_avg[i]
            and body1 > body_avg[i1]
            and body2 > body_avg[i2]
            and ci > oi
            and c1 > o1
            and c2 > o2
            and ci > c1 > c2
            and o1 < oi < c1
            and o2 < o1 < c2
            and range_i * 0.05 > up_shadow_i
            and range1 * 0.05 > up_shadow_1
            and (highs[i2] - lows[i2]) * 0.05 > up_shadow_2
        ):
            out.append(PatternHit(i, Pattern.THREE_WHITE_SOLDIERS))

        if (
            bi > body_avg[i]
            and body1 > body_avg[i1]
            and body2 > body_avg[i2]
            and oi > ci
            and o1 > c1
            and o2 > c2
            and ci < c1 < c2
            and c1 < oi < o1
            and c2 < o1 < o2
            and range_i * 0.05 > dn_shadow_i
            and range1 * 0.05 > dn_shadow_1
            and (highs[i2] - lows[i2]) * 0.05 > dn_shadow_2
        ):
            out.append(PatternHit(i, Pattern.THREE_BLACK_CROWS))

    return out


def find_five_bar_patterns(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    regime: Optional[Sequence[Direction]] = None,
    *,
    trend_mode: str = "regime",
    trend_bars: int = PINE_TREND_BARS,
) -> list[PatternHit]:
    """五日 — Rising/Falling Three Methods, ported from the "*All
    Candlestick Patterns*" script. Reported on the fifth (completing) bar,
    trend-gated on the *first* bar of the five (`trend[i-4]`, matching that
    script's own `C_UpTrend[4]`/`C_DownTrend[4]`)."""
    out: list[PatternHit] = []
    n = len(closes)
    trend = _trend_flags(opens, n, trend_mode, trend_bars, regime)
    body_avg = _body_average(opens, closes)

    for i in range(4, n):
        i4, i3, i2, i1 = i - 4, i - 3, i - 2, i - 1
        o4, c4, h4, l4 = opens[i4], closes[i4], highs[i4], lows[i4]
        o3, c3 = opens[i3], closes[i3]
        o2, c2 = opens[i2], closes[i2]
        o1, c1 = opens[i1], closes[i1]
        oi, ci = opens[i], closes[i]
        b4 = abs(c4 - o4)
        b3 = abs(c3 - o3)
        b2 = abs(c2 - o2)
        b1 = abs(c1 - o1)
        bi = abs(ci - oi)

        if (
            trend[i4] is True
            and b4 > body_avg[i4]
            and c4 > o4
            and b3 < body_avg[i3]
            and o3 > c3
            and o3 < h4
            and c3 > l4
            and b2 < body_avg[i2]
            and o2 > c2
            and o2 < h4
            and c2 > l4
            and b1 < body_avg[i1]
            and o1 > c1
            and o1 < h4
            and c1 > l4
            and bi > body_avg[i]
            and ci > oi
            and ci > c4
        ):
            out.append(PatternHit(i, Pattern.RISING_THREE_METHODS))

        if (
            trend[i4] is False
            and b4 > body_avg[i4]
            and o4 > c4
            and b3 < body_avg[i3]
            and c3 > o3
            and o3 > l4
            and c3 < h4
            and b2 < body_avg[i2]
            and c2 > o2
            and o2 > l4
            and c2 < h4
            and b1 < body_avg[i1]
            and c1 > o1
            and o1 > l4
            and c1 < h4
            and bi > body_avg[i]
            and oi > ci
            and ci < c4
        ):
            out.append(PatternHit(i, Pattern.FALLING_THREE_METHODS))

    return out


#: 好友反攻 — 「錘頭 + 大量」. How many bars the volume average spans. 50 matches
#: the MAVOL50 overlay on the hand-marked reference chart this signal was
#: calibrated against.
RALLY_VOLUME_MA = 50

#: 好友反攻's own shape thresholds — unrelated to the Pine-script pattern set
#: above, kept as this signal's own constants rather than reusing Pine's.
_RALLY_SHADOW_BODY_RATIO = 2.0
_RALLY_OPPOSITE_SHADOW_RATIO = 0.5


def find_rally_attacks(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Optional[Sequence[float]],
    ma_window: int = RALLY_VOLUME_MA,
) -> list[int]:
    """好友反攻 — a 錘頭 shape carrying 大量. Returns the bar indices.

    Keyed on the raw SHAPE alone, independent of trend — the bulls
    counter-attacking does not care which 段 it happens in.

    「大量」 is measured against the trailing ``ma_window`` average rather than
    against yesterday. Yesterday's volume is far too noisy a base: 388.HK's
    2026-07-14 beat the prior day by 10% while trading a third BELOW its 50-day
    average, which is not 大量 by any reading.

    Causal — the average looks only backwards, and a hit is known on its own bar.
    """
    if volumes is None:
        return []

    out: list[int] = []
    for i in range(len(closes)):
        rng = highs[i] - lows[i]
        body = abs(closes[i] - opens[i])
        if rng <= 0 or body <= 0:
            continue

        upper = highs[i] - max(opens[i], closes[i])
        lower = min(opens[i], closes[i]) - lows[i]
        if not (
            lower >= _RALLY_SHADOW_BODY_RATIO * body
            and upper <= _RALLY_OPPOSITE_SHADOW_RATIO * body
        ):
            continue

        # Expanding average until ``ma_window`` bars exist, so the signal is not
        # simply dead for the first 50 bars of every series.
        window = volumes[max(0, i - ma_window + 1) : i + 1]
        if volumes[i] > sum(window) / len(window):
            out.append(i)

    return out

"""RSI, MACD and DMI — pure pandas, no TA library.

Wilder's smoothing throughout (RSI, DMI), matching what 通達信 / 同花順 and
TradingView produce, so values line up with the charts the rules were taught on.

All periods are parameters; the defaults are the ones used in the course:

    RSI    9, with SMA(6) of the RSI
    MACD   12, 26, 9
    DMI    DI 6, ADX 14
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — an EMA with alpha = 1/period.

    Distinct from a standard EMA (alpha = 2/(period+1)); using the wrong one
    shifts RSI by several points and makes ADX visibly different.
    """
    return series.ewm(alpha=1 / period, adjust=False).mean()


@dataclass(frozen=True)
class RSIResult:
    rsi: pd.Series
    signal: pd.Series  # SMA of the RSI


def rsi(
    closes: pd.Series, period: int = 9, signal_period: int = 6
) -> RSIResult:
    """RSI with a simple moving average of itself as the signal line."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = wilder(gain, period)
    avg_loss = wilder(loss, period)

    # All-gain stretches divide by zero; RSI is 100 there by definition.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out.where(avg_loss == 0, 0.0))

    return RSIResult(rsi=out, signal=out.rolling(signal_period).mean())


@dataclass(frozen=True)
class MACDResult:
    dif: pd.Series  # fast EMA - slow EMA
    dea: pd.Series  # EMA of DIF — the signal line
    hist: pd.Series  # (DIF - DEA), doubled as on Chinese platforms


def macd(
    closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> MACDResult:
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()

    # 通達信 plots 2 x (DIF - DEA); TradingView plots the raw difference. The
    # doubled form is what the course charts show.
    return MACDResult(dif=dif, dea=dea, hist=(dif - dea) * 2)


@dataclass(frozen=True)
class DMIResult:
    pdi: pd.Series  # +DI
    mdi: pd.Series  # -DI
    adx: pd.Series


def dmi(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    di_period: int = 6,
    adx_period: int = 14,
) -> DMIResult:
    """Directional Movement Index.

    ``di_period`` smooths the directional movement and true range; ``adx_period``
    smooths DX into ADX.
    """
    up = highs.diff()
    down = -lows.diff()

    # Only the larger of the two moves counts, and only when positive.
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=highs.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=highs.index
    )

    prev_close = closes.shift(1)
    true_range = pd.concat(
        [
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = wilder(true_range, di_period).replace(0, np.nan)
    pdi = 100 * wilder(plus_dm, di_period) / atr
    mdi = 100 * wilder(minus_dm, di_period) / atr

    total = (pdi + mdi).replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / total

    return DMIResult(pdi=pdi, mdi=mdi, adx=wilder(dx.fillna(0), adx_period))

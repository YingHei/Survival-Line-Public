"""OHLCV loading for US equities and 388.HK.

Spec §5. The engine is bar-sequential, so a single bad bar corrupts everything
downstream — the handling here is defensive by design.

停牌 / halts
    Dropped, never forward-filled. A filled bar has ``high == low == prev_close``,
    which classifies as 內困K under R1 and silently corrupts the R2 deferral
    chain. yfinance already omits non-trading days; this module additionally
    drops any zero-volume or zero-range bar that slips through.

Splits / 供股
    ``auto_adjust=True`` by default (``load(..., adjusted=True)``) — an
    unadjusted level makes ``current_stop`` meaningless across a split.
    ``adjusted=False`` uses yfinance's own ``auto_adjust=False``, which is
    NOT truly raw: Yahoo's underlying "unadjusted" Close is already
    split-adjusted at the source (confirmed against AAPL's real 2020-08-28
    close of $499.23 vs. the $124.81 yfinance returns for that date with
    auto_adjust=False — exactly $499.23 / 4, the split ratio three days
    later) and only dividend adjustment is toggled by the flag. That's
    deliberately what this mode is for: TradingView's own default view is
    split-adjusted-but-not-dividend-adjusted, and matching it needs no
    custom adjustment math, just the flag — yfinance has no parameter for
    truly un-split-adjusted prices at all. Cached separately from the
    adjusted series (``{symbol}_raw.csv``) since the two are different bar
    values, not a slice of the same series.

Half-day sessions
    Volume is structurally ~50%, so R6's 「vs yesterday's volume」 test misfires
    on the half day *and* the day after. Callers pass those dates to
    :func:`half_day_bar_indices` and hand the result to the engine.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

COLUMNS = ["open", "high", "low", "close", "volume"]


def load(
    symbol: str,
    start: str,
    end: str,
    *,
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Fetch daily bars, gap-filled against a per-symbol CSV cache.

    The cache holds the full known history for the symbol. A request fully
    inside the cached range (with ``end`` strictly before the cached max
    date) is served from disk with no network call. A request that extends
    past the cached max or precedes the cached min fetches only the missing
    sub-range(s) and merges them in. A backfill attempt that finds nothing
    (the symbol's real history starts later than ``start`` — an IPO/listing
    boundary) is remembered in a ``{symbol}.min_checked`` sidecar so it is
    never re-attempted for the same or a later ``start``. The extend step
    always re-fetches the cached max date itself, not just the days after
    it, so a bar cached while the market was still open (partial OHLC)
    self-corrects once the real close is available. ``refresh=True``
    force-fetches exactly ``[start, end]`` and overwrites those dates in
    the cache unconditionally — the only way to pick up a stock split's
    retroactive adjustment to already-cached rows.

    ``adjusted`` selects which of two independent per-symbol caches to read
    and write — see the module docstring's "Splits / 供股" section. It is
    part of the cache identity, not a post-fetch transform: the two modes
    hold genuinely different bar values, so mixing them into one cache file
    would silently serve the wrong series depending on fetch order.

    Returns a frame indexed by date with lowercase OHLCV columns, sliced to
    ``[start, end]``.
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if adjusted else "_raw"
    path = cache_dir / f"{symbol.replace('.', '_')}{suffix}.csv"
    checked_path = cache_dir / f"{symbol.replace('.', '_')}{suffix}.min_checked"
    cached = _read_cache(path) if path.exists() else None

    if refresh:
        fresh = _fetch(symbol, start, end, adjusted)
        merged = _merge(cached, fresh) if cached is not None else fresh
        _write_csv(merged, path)
        return _slice(merged, start, end)

    if cached is None or cached.empty:
        fresh = _fetch(symbol, start, end, adjusted)
        _write_csv(fresh, path)
        _write_checked_since(checked_path, start)
        return _slice(fresh, start, end)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cached_min, cached_max = cached.index.min(), cached.index.max()
    checked_since = _read_checked_since(checked_path)

    fetched = []
    # A symbol's real history often starts later than `start` (IPO, listing
    # date) — that gap is permanent, and without this check every call would
    # re-attempt the same doomed multi-year fetch forever. Only re-check once
    # a caller asks for something earlier than what's already been ruled out.
    if start_ts < cached_min and (checked_since is None or start_ts < checked_since):
        backfill_end = (cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fetched.append(_fetch(symbol, start, backfill_end, adjusted))
        _write_checked_since(checked_path, start)
    if end_ts >= cached_max:
        # Inclusive of cached_max: a bar cached mid-session carries a
        # live/partial close, and this is the only point that ever
        # revisits that date once it's no longer the newest.
        extend_start = cached_max.strftime("%Y-%m-%d")
        fetched.append(_fetch(symbol, extend_start, end, adjusted))

    if fetched:
        merged = _merge(cached, pd.concat(fetched))
        _write_csv(merged, path)
    else:
        merged = cached

    return _slice(merged, start, end)


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Like `read_csv`, but a corrupt/truncated file is treated as no cache
    rather than crashing the caller. `to_csv` truncates its target on open,
    so a process killed mid-write (this project's own dev workflow kills the
    server with `kill -9` routinely, see CLAUDE.md) can leave 0 bytes or a
    cut-off last row on disk — either way pandas can't parse it, and the
    original content is already gone, so there's nothing to recover but a
    refetch."""
    try:
        return read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic replace: write to a sibling temp file, then rename into place.
    `os.replace` is atomic on POSIX and Windows, so a kill mid-write leaves
    at worst an orphaned `.tmp` file — the live cache path is never observed
    half-written, which is what let `_read_cache` need to defend against in
    the first place.

    The tmp filename includes a random token so two concurrent writers to
    the same `path` never share one — FastAPI runs sync endpoints in a
    threadpool, so two overlapping requests for the same symbol/adjusted
    pair (e.g. a rapid-fire UI toggle re-firing the same fetch) are a real,
    reachable case: with a shared tmp name, whichever thread's `.replace()`
    runs second finds the first thread already consumed it and raises
    `FileNotFoundError` — surfaced as a 500 all the way to the browser.
    Confirmed via a real rapid-toggle browser stress test.

    Trade-off: a *fixed* tmp name meant at most one orphan could ever
    accumulate per cache path (each retry just overwrote it via `to_csv`
    truncating on open) — with a random name per write, repeated kills
    during the same routine dev workflow `_read_cache` already accounts for
    could leave several distinct orphans behind over time, since nothing
    else ever reuses or overwrites an old one. See `cleanup_orphaned_tmp_files`."""
    tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    df.to_csv(tmp_path)
    tmp_path.replace(path)


def cleanup_orphaned_tmp_files(cache_dir: Optional[Path] = None) -> int:
    """Delete every `*.tmp` file left behind by a `_write_csv` that never
    reached its `.replace()` — a process killed between the two (this
    project's dev workflow kills the server with `kill -9` routinely, see
    CLAUDE.md). Safe only as a startup-time sweep, never mid-request: at
    startup nothing else is writing yet, so every `.tmp` file found is
    unambiguously a leftover, not one another thread is actively producing.
    Returns the count removed, purely for a startup log line."""
    cache_dir = cache_dir or CACHE_DIR
    if not cache_dir.exists():
        return 0
    removed = 0
    for tmp_file in cache_dir.glob("*.tmp"):
        tmp_file.unlink()
        removed += 1
    return removed


def _read_checked_since(path: Path) -> Optional[pd.Timestamp]:
    """The earliest `start` already ruled out by a prior backfill attempt."""
    if not path.exists():
        return None
    text = path.read_text().strip()
    return pd.Timestamp(text) if text else None


def _write_checked_since(path: Path, start: str) -> None:
    path.write_text(start + "\n")


def _fetch(symbol: str, start: str, end: str, adjusted: bool = True) -> pd.DataFrame:
    import yfinance as yf

    # yfinance's own `end` is exclusive of the given date, but every caller
    # in this module treats `end` as inclusive (``_slice`` included) — pad
    # by a day so a bar dated exactly `end` is actually reachable. Mirrors
    # the same inclusive-end padding `webull.get_bars` already does
    # (`+ 86_399_000` ms) for the same underlying exclusive-API reason.
    fetch_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.Ticker(symbol).history(
        start=start, end=fetch_end, interval="1d", auto_adjust=adjusted
    )
    return _normalise(raw)


def fetch_company_name(symbol: str) -> Optional[str]:
    """Best-effort company name lookup, or ``None`` on any failure.

    A separate yfinance call from ``_fetch``/``.history()`` — ``.info`` hits
    a different endpoint (``quoteSummary``), so this is never free. Callers
    (watchlist add, the name-backfill script) treat ``None`` as "no name
    available" rather than an error — a missing name must never block a
    symbol from being added.
    """
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).info
    except Exception:  # noqa: BLE001 — any failure here degrades to "no name"
        return None
    return info.get("longName") or info.get("shortName") or None


def _merge(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union cached and freshly-fetched rows, fresh wins on a shared date."""
    frames = [df for df in (cached, fresh) if not df.empty]
    if not frames:
        return cached
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def read_csv(path: Path) -> pd.DataFrame:
    """Load a cached/fixture CSV back into the canonical frame shape."""
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return df[COLUMNS]


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower)[COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"

    df = df.dropna(subset=["high", "low", "close"])

    # 停牌 residue: zero volume or a zero-range bar carries no information and
    # would classify as 內困K, corrupting the R2 chain. Drop rather than fill.
    df = df[(df["volume"] > 0) & (df["high"] > df["low"])]

    return df


def half_day_bar_indices(
    df: pd.DataFrame, half_days: Iterable[str]
) -> set[int]:
    """Bar indices where R6's volume rule must not fire.

    Returns the half-day bars *and* the bars immediately after them: a half day
    suppresses the test on the day itself and produces a false positive the day
    after, when volume returns to normal against a halved baseline.
    """
    wanted = {pd.Timestamp(d).normalize() for d in half_days}
    positions = {i for i, ts in enumerate(df.index) if ts in wanted}
    return positions | {i + 1 for i in positions if i + 1 < len(df)}

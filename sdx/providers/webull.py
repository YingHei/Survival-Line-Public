"""Webull OHLCV loading — REST historical bars, at a caller-chosen interval.

Spec: openspec/changes/webull-streaming-data. Kept entirely separate from
sdx/data.py (yfinance): that module's cache/backfill pipeline is
daily-only and yfinance-specific; this one adds an `interval` dimension and
talks to a different provider. sdx.serve branches between the two rather
than either module knowing about the other.

Symbol format note: this app's tickers are yfinance-style (bare US tickers,
``"388.HK"`` for HK) — ``to_webull_symbol`` translates those into Webull's
``(symbol, Category)`` pairs. Confirmed against the real API via
``scripts/webull_smoketest.py``: HK symbols are 5-digit zero-padded
(``"388.HK"`` -> ``"00388"``), and a ``PaperTrading`` app key authenticates
against the sandbox REST host (``api.sandbox.webull.hk``), not production —
set via ``WEBULL_API_ENDPOINT``.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from ..data import COLUMNS

#: Exchange local timezone per Webull Category, keyed by ``category.name``
#: (matches how ``get_bars`` already passes it to ``get_history_bar``) — see
#: ``_normalise``'s docstring for why this matters for daily bars.
_EXCHANGE_TZ = {
    "US_STOCK": "America/New_York",
    "HK_STOCK": "Asia/Hong_Kong",
}

#: App-facing interval strings (matches the chart header's interval picker)
#: to the SDK's Timespan enum member names.
_TIMESPAN_NAMES = {
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "M60",
    "4h": "M240",
    "D": "D",
    "M": "M",
    "Y": "Y",
}

#: Intervals with no time-of-day component — bucketed and displayed as a
#: bare date (matching yfinance's daily-only convention). Independent of
#: ladder eligibility below: 4h is ladder-eligible but still intraday
#: (UNIX-timestamp) formatted.
DATE_ONLY_INTERVALS = frozenset({"D", "M", "Y"})

#: Intervals the ladder engine runs over — decided explicitly per interval,
#: not derived from DATE_ONLY_INTERVALS (4h is ladder-eligible but not
#: date-only; 5m/15m/30m/1h are neither). Its rules (內困K classification,
#: 停牌/half-day handling) were only ever validated at these granularities.
LADDER_INTERVALS = frozenset({"D", "4h", "M", "Y"})

#: Live streaming isn't offered for these — bucketing a "currently forming"
#: bar correctly needs the last-trading-day-of-period anchor Webull itself
#: uses for Month/Year (confirmed empirically: a July monthly bar lands on
#: 2026-07-30, not 2026-07-01), which requires a trading calendar this app
#: doesn't have. Historical fetch + the ladder engine both still work for
#: them; only the live WebSocket leg is unavailable.
NO_LIVE_INTERVALS = frozenset({"M", "Y"})

#: Webull's per-call cap (see MarketData.get_history_bar's docstring).
_MAX_COUNT = 1200


class WebullNotConfigured(RuntimeError):
    """WEBULL_APP_KEY/WEBULL_APP_SECRET are not set."""


_data_client = None  # lazily constructed singleton; see _get_data_client()


def to_webull_symbol(sdx_symbol: str):
    """Map this app's yfinance-style ticker to Webull's ``(symbol, Category)``.

    ``"388.HK"`` -> ``("00388", Category.HK_STOCK)`` (5-digit zero-padded).
    A bare ticker (``"AAPL"``) -> ``(symbol, Category.US_STOCK)`` unchanged.
    """
    from webull.data.common.category import Category

    if sdx_symbol.upper().endswith(".HK"):
        code = sdx_symbol[: -len(".HK")]
        return code.zfill(5), Category.HK_STOCK
    return sdx_symbol, Category.US_STOCK


def _get_data_client():
    """Lazily construct the shared DataClient. Credentials are read here,
    not at import time, so importing this module never requires them."""
    global _data_client
    if _data_client is not None:
        return _data_client

    app_key = os.environ.get("WEBULL_APP_KEY")
    app_secret = os.environ.get("WEBULL_APP_SECRET")
    if not app_key or not app_secret:
        raise WebullNotConfigured(
            "Webull data source not configured — set WEBULL_APP_KEY and "
            "WEBULL_APP_SECRET (see .env.example)."
        )

    from webull.core.client import ApiClient
    from webull.data.data_client import DataClient

    region = os.environ.get("WEBULL_REGION", "hk")
    api_client = ApiClient(app_key, app_secret, region)
    endpoint = os.environ.get("WEBULL_API_ENDPOINT")
    if endpoint:
        api_client.add_endpoint(region, endpoint)

    _data_client = DataClient(api_client)
    return _data_client


def get_bars(
    symbol: str, interval: str, start: str, end: str, *, data_client=None
) -> pd.DataFrame:
    """Fetch historical bars from Webull at ``interval``, normalized to the
    same ``[open, high, low, close, volume]`` column shape ``sdx.data``
    already produces, sliced to ``[start, end]``.

    ``interval`` is one of the app-facing strings in ``_TIMESPAN_NAMES``
    (``"1m"``, ``"5m"``, ``"15m"``, ``"30m"``, ``"1h"``, ``"D"``).
    """
    from webull.data.common.timespan import Timespan

    client = data_client or _get_data_client()
    webull_symbol, category = to_webull_symbol(symbol)
    timespan = getattr(Timespan, _TIMESPAN_NAMES[interval])

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) + 86_399_000

    response = client.market_data.get_history_bar(
        webull_symbol,
        category.name,
        timespan.name,
        count=str(_MAX_COUNT),
        start_time=start_ms,
        end_time=end_ms,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Webull get_history_bar failed for {symbol}: "
            f"HTTP {response.status_code}: {response.text[:500]}"
        )

    return _normalise(
        response.json(),
        daily=(interval in DATE_ONLY_INTERVALS),
        exchange_tz=_EXCHANGE_TZ[category.name],
    )


def _normalise(bars: list[dict], *, daily: bool, exchange_tz: str = "UTC") -> pd.DataFrame:
    """Webull returns newest-first bars with string-typed OHLCV fields and
    an ISO-8601 ``time``. Sort ascending, type-convert, and index by date
    (daily) or full timestamp (intraday) — matching sdx.data's frame shape
    exactly for the daily case, so downstream code needs no special-casing.

    A daily/M/Y bar's ``time`` is anchored at the EXCHANGE's local midnight,
    not UTC midnight — confirmed against the raw API response for 0388.HK's
    2021-01-04 HK trading session (a Monday; Jan 1 was a holiday, Jan 2-3 a
    weekend): Webull returns it timestamped ``"2021-01-03T16:00:00+0000"``,
    i.e. exactly HK midnight (UTC+8) expressed in UTC. Converting to UTC and
    taking ``.date()`` (the old behaviour, ``exchange_tz`` defaulting unused)
    therefore silently shifted every HK daily bar's label one calendar day
    earlier than the real trading date — a bar dated "2021-01-03" with real
    volume, even though that date is a Sunday HKEX was never open on.
    Localizing to the exchange's own timezone first before taking ``.date()``
    is what fixes it; intraday bars are unaffected and keep the UTC instant,
    since chart bucketing there is timezone-agnostic by design.
    """
    if not bars:
        index = pd.DatetimeIndex([], name="date")
        return pd.DataFrame(columns=COLUMNS, index=index)

    tz = ZoneInfo(exchange_tz)
    rows = []
    for bar in bars:
        ts = datetime.fromisoformat(bar["time"].replace("Z", "+00:00"))
        if daily:
            date_value = pd.Timestamp(ts.astimezone(tz).date())
        else:
            date_value = pd.Timestamp(ts.astimezone(timezone.utc).replace(tzinfo=None))
        rows.append(
            {
                "date": date_value,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": int(float(bar["volume"])),
            }
        )

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.index.name = "date"
    return df[COLUMNS]


# --- streaming: one shared MQTT connection, folding TICK trades into bars ---
#
# Only TICK (individual trades: price/volume/time) drives OHLCV aggregation.
# SNAPSHOT is subscribed to per the spec (cheaper entitlement than QUOTE) but
# not used for bar-folding: the installed SDK's SnapshotResult.close is a
# known bug (bound to pb_snapshot.open, not .close), and a trade-by-trade
# TICK stream is what actually belongs in an OHLCV bar anyway.

#: App-facing interval -> bucket width in seconds. D is included for daily
#: live-updating (a whole trading day as one fixed-width bucket is a fine
#: approximation — unlike Month/Year, it doesn't drift against a real
#: calendar). M/Y have no entry: see NO_LIVE_INTERVALS.
_INTERVAL_SECONDS = {
    "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "D": 86400,
}


class _BarBucket:
    """Mutable OHLCV accumulator for one open interval bucket."""

    __slots__ = ("start", "open", "high", "low", "close", "volume")

    def __init__(self, start: pd.Timestamp, price: float, volume: int):
        self.start = start
        self.open = self.high = self.low = self.close = price
        self.volume = volume

    def fold(self, price: float, volume: int) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume

    def as_update(self, daily: bool) -> dict:
        time_value = self.start.strftime("%Y-%m-%d") if daily else int(self.start.timestamp())
        return {
            "time": time_value,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": int(self.volume),
        }

    @classmethod
    def from_seed(cls, seed: dict, *, daily: bool) -> "_BarBucket":
        """Rebuild a bucket from a REST-fetched bar (``as_update``'s own
        shape) instead of a single tick price — otherwise the first tick
        after every (re)subscribe would seed open/high/low from whatever
        price happens to be trading *right then*, discarding however much
        of the real session already elapsed before this stream connected.
        The inverse of ``as_update``: same ``time`` encoding both ways."""
        start = (
            pd.Timestamp(seed["time"])
            if daily
            else pd.Timestamp(seed["time"], unit="s")
        )
        bucket = cls(start, seed["open"], seed["volume"])
        bucket.high = seed["high"]
        bucket.low = seed["low"]
        bucket.close = seed["close"]
        return bucket


def _bucket_start(ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    """The start of the interval bucket a UTC timestamp falls into."""
    if interval in NO_LIVE_INTERVALS:
        raise ValueError(
            f"Live streaming is not supported for interval={interval!r} "
            "(Month/Year bars only support historical fetch)"
        )
    seconds = _INTERVAL_SECONDS[interval]
    epoch = int(ts.timestamp())
    return pd.Timestamp((epoch - epoch % seconds), unit="s", tz="UTC").tz_localize(None)


class WebullStream:
    """Process-lifetime singleton: one shared MQTT connection, folding TICK
    messages into the current bar for every subscribed (symbol, interval).

    Subscriptions are wire-level per (webull_symbol, category) — shared
    across every interval subscribed for that symbol — but bucketed and
    delivered independently per (webull_symbol, interval), each with its own
    set of consumer queues.
    """

    def __init__(self) -> None:
        self._client = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], _BarBucket] = {}
        self._consumers: dict[tuple[str, str], set] = {}
        self._wire_subs: dict[str, object] = {}  # webull_symbol -> Category
        self._connected = threading.Event()

    # -- public API -----------------------------------------------------

    def subscribe(
        self, symbol: str, category, interval: str, seed: Optional[dict] = None
    ) -> "asyncio.Queue":
        """Register a consumer for (symbol, interval); returns a queue that
        receives a bar-update dict each time that bucket changes. Ensures
        the shared MQTT connection exists and the symbol is subscribed on
        the wire, without blocking for the connection to complete.

        ``seed`` is the current bar as already known from a REST fetch (same
        shape as ``_BarBucket.as_update``) — applied only when no bucket
        exists yet for this key, so the first tick folds into the real
        session's open/high/low instead of starting a fresh bucket from
        whatever price happens to be trading at subscribe time. A later
        subscribe() for a key that's already streaming must never clobber
        the bucket it has already accumulated with a now-stale seed.
        """
        if interval in NO_LIVE_INTERVALS:
            raise ValueError(
                f"Live streaming is not supported for interval={interval!r} "
                "(Month/Year bars only support historical fetch)"
            )
        self._loop = self._loop or asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        with self._lock:
            key = (symbol, interval)
            self._consumers.setdefault(key, set()).add(queue)
            already_wired = symbol in self._wire_subs
            self._wire_subs[symbol] = category
            if seed is not None and key not in self._buckets:
                daily = interval in DATE_ONLY_INTERVALS
                self._buckets[key] = _BarBucket.from_seed(seed, daily=daily)

        self._ensure_client()
        if not already_wired and self._connected.is_set():
            self._wire_subscribe(symbol, category)
        return queue

    def unsubscribe(self, symbol: str, interval: str, queue: "asyncio.Queue") -> None:
        """Drop one consumer. Tears down the wire subscription for `symbol`
        only once no interval for it has any consumer left — the shared
        connection itself is never closed as a result."""
        with self._lock:
            key = (symbol, interval)
            consumers = self._consumers.get(key)
            if consumers:
                consumers.discard(queue)
                if not consumers:
                    del self._consumers[key]
                    self._buckets.pop(key, None)
            still_used = any(k[0] == symbol for k in self._consumers)
            if still_used or symbol not in self._wire_subs:
                return
            category = self._wire_subs.pop(symbol)

        if self._client is not None:
            try:
                self._client.unsubscribe(
                    symbols=[symbol], category=category.name,
                    sub_types=["SNAPSHOT", "TICK"],
                )
            except Exception:  # noqa: BLE001 — best-effort; connection may be down
                pass

    # -- connection lifecycle --------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return

        app_key = os.environ.get("WEBULL_APP_KEY")
        app_secret = os.environ.get("WEBULL_APP_SECRET")
        if not app_key or not app_secret:
            raise WebullNotConfigured(
                "Webull data source not configured — set WEBULL_APP_KEY and "
                "WEBULL_APP_SECRET (see .env.example)."
            )

        from webull.data.data_streaming_client import DataStreamingClient

        region = os.environ.get("WEBULL_REGION", "hk")
        http_host = os.environ.get("WEBULL_API_ENDPOINT") or None
        mqtt_host = os.environ.get("WEBULL_MQTT_ENDPOINT") or None
        session_id = f"sdx-{uuid.uuid4().hex[:12]}"

        client = DataStreamingClient(
            app_key, app_secret, region, session_id,
            http_host=http_host, mqtt_host=mqtt_host,
        )
        client.on_connect_success = self._on_connect
        client.on_quotes_message = self._on_message
        self._client = client
        self._thread = threading.Thread(
            target=self._run_forever, name="webull-stream", daemon=True
        )
        self._thread.start()

    def _run_forever(self) -> None:
        """`connect_and_loop_forever` already retries internally with
        backoff (DefaultQuotesRetryPolicy) and only returns/raises once that
        policy gives up or the socket closes cleanly; the outer loop here is
        a second layer so an exhausted retry policy still reconnects rather
        than leaving the stream permanently dead."""
        while True:
            self._connected.clear()
            try:
                self._client.connect_and_loop_forever()
            except Exception:  # noqa: BLE001 — reconnect regardless of cause
                pass
            self._connected.clear()
            time.sleep(2)

    def _on_connect(self, client, api_client, session_id) -> None:
        self._connected.set()
        with self._lock:
            wire_subs = dict(self._wire_subs)
        for webull_symbol, category in wire_subs.items():
            self._wire_subscribe(webull_symbol, category)

    def _wire_subscribe(self, webull_symbol: str, category) -> None:
        from webull.data.common.subscribe_type import SubscribeType

        self._client.subscribe(
            [webull_symbol], category.name,
            [SubscribeType.SNAPSHOT.name, SubscribeType.TICK.name],
        )

    # -- message handling / bar folding -----------------------------------

    def _on_message(self, client, topic: str, payload) -> None:
        if topic != "tick" or payload.price is None:
            return
        symbol = payload.basic.get_symbol()
        price = float(payload.price)
        volume = int(payload.volume) if payload.volume else 0
        ts = pd.Timestamp(payload.basic.get_timestamp_as_utc())

        for key, update, queues in self._fold_tick(symbol, price, volume, ts):
            for queue in queues:
                self._loop.call_soon_threadsafe(queue.put_nowait, update)

    def _fold_tick(self, symbol: str, price: float, volume: int, ts: pd.Timestamp):
        """Pure(ish) folding step, independent of threads/queues/MQTT so
        it's directly unit-testable. Returns a list of
        (key, update_dict, queues_snapshot) for every (symbol, interval)
        subscription this tick applies to."""
        results = []
        with self._lock:
            keys = [k for k in self._consumers if k[0] == symbol]
            for key in keys:
                _, interval = key
                bucket_start = _bucket_start(ts, interval)
                bucket = self._buckets.get(key)
                if bucket is None or bucket.start != bucket_start:
                    bucket = _BarBucket(bucket_start, price, volume)
                    self._buckets[key] = bucket
                else:
                    bucket.fold(price, volume)
                update = bucket.as_update(daily=(interval in DATE_ONLY_INTERVALS))
                results.append((key, update, set(self._consumers[key])))
        return results


_stream: Optional[WebullStream] = None


def get_stream() -> WebullStream:
    global _stream
    if _stream is None:
        _stream = WebullStream()
    return _stream

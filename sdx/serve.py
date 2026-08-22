"""Local server so watchlists can be edited from the chart.

    python -m sdx.serve            # http://127.0.0.1:8765

The static ``out/index.html`` stays fully usable on its own — it just cannot
add symbols, because a ``file://`` page can neither fetch bars nor write
``watchlists.json``. Served, the page gets a ``+`` button and per-ticker
removal; everything else behaves identically.

Single-user and local by design: bound to 127.0.0.1, no auth, no CORS.
"""

from __future__ import annotations

import argparse
import asyncio
import webbrowser
from datetime import date
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import alerts_log
from . import watchlist as wl
from . import watchlist_layout as wll
from .candles import PINE_TREND_BARS
from .data import cleanup_orphaned_tmp_files, fetch_company_name, load
from .engine import OUTSIDE_BAR_CLOSE_FRACTION, run
from .providers import webull as webull_provider
from .viz import (
    BUNDLE,
    DEFAULT_START,
    LADDER_KEYS,
    PATTERN_CATALOG,
    _TEMPLATE,
    _ladders,
    build_payload,
    indicator_params_out,
    patterns_for_trend_bars,
)

app = FastAPI(title="生死線")

#: favicon.io's standard bundle — see the <link>/<meta> tags in _TEMPLATE's
#: <head>. Root-relative paths (not /static/...) to match what favicon.io
#: itself generates and what every browser requests by convention
#: (/favicon.ico with no link tag at all). The static out/index.html export
#: has no server to answer these — that page just gets no favicon, same
#: graceful-degradation as any other served-only affordance (§ module
#: docstring above).
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

#: Set from main(); the render window and indicator periods.
SETTINGS: dict = {}

_CACHE: dict[str, dict] = {}


class AddRequest(BaseModel):
    symbols: list[str]


class TagRequest(BaseModel):
    held: bool = False
    special: bool = False
    strategies: list[str] = []
    stages: list[str] = []
    patterns: list[str] = []


class TickerLayoutEntry(BaseModel):
    type: Literal["ticker"]
    symbol: str


class SectionLayoutEntry(BaseModel):
    type: Literal["section"]
    # Optional so the frontend can add a section without minting an id
    # itself — api_put_layout() assigns one server-side (wll.new_section_id())
    # for any entry that omits it, per watchlist-layout's requirement that
    # section ids are generated server-side.
    id: Optional[str] = None
    name: str
    collapsed: bool = False


LayoutEntry = Union[TickerLayoutEntry, SectionLayoutEntry]


class AlertLogKey(BaseModel):
    symbol: str
    date: str
    condition: str
    tier: Literal["confirmed", "provisional"]


class AlertAckRequest(BaseModel):
    keys: list[AlertLogKey]
    acked: bool


def payload_for(symbol: str, *, refresh: bool = False) -> dict:
    """Compute a symbol's chart payload, memoised for the process lifetime.

    Always raw (``adjusted=False``) — the app's default Price mode, and the
    server has no way to know a client's stored toggle preference at SSR
    time anyway. The frontend's own default (``adjustedFor()`` in the
    ``_TEMPLATE`` script) and ``YF_PAYLOADS``'s unsuffixed-key convention
    (``yfKey``) both agree with this, so a held symbol's default view is
    zero-fetch; only switching to Adjusted costs a round trip.
    """
    if symbol in _CACHE and not refresh:
        return _CACHE[symbol]

    df = load(symbol, SETTINGS["start"], SETTINGS["end"], refresh=refresh, adjusted=False)
    if df.empty:
        raise HTTPException(404, f"{symbol}: no bars returned")

    result, alts = _ladders(df)
    _CACHE[symbol] = build_payload(
        df, result, symbol, SETTINGS.get("down_arrows", True),
        SETTINGS["params"], alts,
    )
    return _CACHE[symbol]


#: yfinance's own interval choice (independent of Webull's M/Y, which stay
#: exactly as Webull returns them — see bars_payload_for). Maps the app-facing
#: interval string to the pandas resample frequency alias.
_YF_RESAMPLE_FREQ = {"M": "ME", "Y": "YE"}


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample daily OHLCV up to ``freq`` ('ME'/'YE'), anchored on each
    period's LAST TRADING DAY rather than the calendar period-end — matches
    Webull's own M/Y bar convention (confirmed empirically against the live
    API: a July monthly bar lands on the last trading day of July, not
    July 1st or a calendar-end date that may not even be a trading day),
    so a symbol's M/Y shape doesn't visibly jump when switching source.
    """
    grouped = df.groupby(pd.Grouper(freq=freq))
    out = grouped.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    last_dates = grouped.apply(lambda g: g.index.max())
    out.index = last_dates.loc[out.index].values
    out.index.name = "date"
    return out


def bars_payload_for(symbol: str, source: str, interval: str, adjusted: bool = False) -> dict:
    """Chart payload for one symbol from either provider.

    yfinance always runs the ladder engine (unchanged, daily-or-coarser
    only — D, or M/Y via ``_resample_ohlcv``). Webull runs it only for
    ``webull_provider.LADDER_INTERVALS`` (D, 4h, M, Y) — the engine's rules
    are daily-or-coarser-specific (內困K classification, 停牌/half-day
    handling) and were never validated at finer granularity; any other
    interval returns candles/indicators only, via
    ``build_payload(..., result=None)``. Candle/volume/indicator time format
    is a separate axis (``webull_provider.DATE_ONLY_INTERVALS``) — 4h is
    ladder-eligible but still UNIX-timestamp formatted, not a bare date.

    ``D`` is always seeded from yfinance, even when ``source="webull"`` —
    same reasoning as ``_live_seed``: Webull's own REST history
    (``webull_provider.get_bars``) caps at ``_MAX_COUNT`` (1200) most-recent
    bars regardless of the requested ``start``, silently ignoring
    ``SETTINGS["start"]``/``DEFAULT_START`` and stranding the chart a few
    years back instead of the configured full history.

    ``source="yfinance"`` with ``interval`` in ``M``/``Y`` resamples
    yfinance's own daily bars rather than reusing Webull's M/Y — Webull's
    SDK only ever returns split-and-dividend-adjusted bars at D-and-coarser
    granularity (confirmed straight from ``MarketData.get_history_bar``'s
    own docstring: "only the K-line with the previous weight is provided
    for the daily K-line and above"), so there is no way to get a Webull M/Y
    bar on the same basis as yfinance's Raw mode. Selecting Webull as the
    source keeps its own M/Y bars entirely unchanged — this only ever
    substitutes when the user has explicitly chosen yfinance as the source.

    ``adjusted`` only affects the yfinance path (D, or resampled M/Y; see
    ``sdx.data``'s module docstring) — Webull's own REST history has no such
    toggle, so it's ignored there.
    """
    if source == "yfinance" or interval == "D":
        df = load(symbol, SETTINGS["start"], SETTINGS["end"], adjusted=adjusted)
        if source == "yfinance" and interval in _YF_RESAMPLE_FREQ:
            df = _resample_ohlcv(df, _YF_RESAMPLE_FREQ[interval])
        if df.empty:
            raise HTTPException(404, f"{symbol}: no bars returned")
        result, alts = _ladders(df)
        return build_payload(
            df, result, symbol, SETTINGS.get("down_arrows", True),
            SETTINGS["params"], alts,
        )

    try:
        df = webull_provider.get_bars(symbol, interval, SETTINGS["start"], SETTINGS["end"])
    except webull_provider.WebullNotConfigured as exc:
        raise HTTPException(400, str(exc)) from exc
    if df.empty:
        raise HTTPException(404, f"{symbol}: no bars returned")

    run_ladder = interval in webull_provider.LADDER_INTERVALS
    date_only = interval in webull_provider.DATE_ONLY_INTERVALS
    result, alts = (_ladders(df) if run_ladder else (None, None))
    return build_payload(
        df, result, symbol, SETTINGS.get("down_arrows", True),
        SETTINGS["params"], alts, daily=date_only,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """No symbol payload is fetched here at all — the page ships with an
    empty `symbols` map and renders its shell (sidebar, chart frame)
    immediately. The frontend fetches the current symbol via the same
    lazy GET /api/bars path it already uses for any non-preloaded symbol
    (see `select()`/`activateSymbol()` in `_TEMPLATE`), then, once that's
    on screen, background-fetches every 持有/特別關注 symbol so the Alerts
    panel fills in without blocking the initial paint on all of them.
    `defaultParams` substitutes for what used to be read off a preloaded
    symbol's own payload (`ALL.symbols[current].params`) — the indicator
    settings are shared across every symbol regardless, so there's no
    need to wait on any particular one's fetch to know them.
    """
    import json

    watchlist = wl.load()
    layout = wll.load(watchlist)
    data = {
        "symbols": {},
        "watchlist": watchlist,
        "layout": layout,
        "live": True,
        "defaultParams": indicator_params_out(SETTINGS["params"]),
    }
    return (
        _TEMPLATE.replace("__BUNDLE__", BUNDLE.read_text(encoding="utf-8"))
        .replace("__DATA__", json.dumps(data))
        .replace("__PATTERNS__", json.dumps(PATTERN_CATALOG))
    )


@app.get("/favicon.ico")
def favicon_ico() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.get("/favicon-32x32.png")
def favicon_32() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon-32x32.png")


@app.get("/favicon-16x16.png")
def favicon_16() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon-16x16.png")


@app.get("/apple-touch-icon.png")
def apple_touch_icon() -> FileResponse:
    return FileResponse(STATIC_DIR / "apple-touch-icon.png")


@app.get("/android-chrome-192x192.png")
def android_chrome_192() -> FileResponse:
    return FileResponse(STATIC_DIR / "android-chrome-192x192.png")


@app.get("/android-chrome-512x512.png")
def android_chrome_512() -> FileResponse:
    return FileResponse(STATIC_DIR / "android-chrome-512x512.png")


@app.get("/site.webmanifest")
def site_webmanifest() -> FileResponse:
    return FileResponse(STATIC_DIR / "site.webmanifest")


@app.get("/api/watchlist")
def api_watchlist() -> dict:
    return wl.load()


@app.post("/api/watchlist")
def api_add(req: AddRequest) -> dict:
    """Add one or many symbols, reporting each independently.

    A bad ticker in the middle of a paste must not discard the good ones, so
    failures are collected rather than raised. Added symbols start with
    every tag empty; tags are set afterward via PATCH.
    """
    watchlist = wl.load()
    added, failed, dirty = [], [], False

    for raw in req.symbols:
        symbol = wl.normalize_symbol(raw.strip())
        if not symbol:
            continue
        if symbol in watchlist:
            failed.append({"symbol": symbol, "error": "already in watchlist"})
            continue
        try:
            payload = payload_for(symbol, refresh=True)  # validates
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", None) or str(exc)
            failed.append({"symbol": symbol, "error": str(detail)})
            continue

        entry = {"held": False, "special": False, "strategies": [], "stages": [], "patterns": []}
        name = fetch_company_name(symbol)  # best-effort — never blocks the add
        if name:
            entry["name"] = name
        watchlist[symbol] = entry
        dirty = True
        added.append({"symbol": symbol, "payload": payload})

    if dirty:
        wl.save(watchlist)

    return {"added": added, "failed": failed}


@app.patch("/api/watchlist/{symbol:path}")
def api_tag(symbol: str, req: TagRequest) -> dict:
    watchlist = wl.load()
    if symbol not in watchlist:
        raise HTTPException(404, f"{symbol} has no entry")
    # TagRequest only carries held/strategies/stages/patterns — the tag-edit
    # UI never sends `name`, so replacing the entry outright would silently
    # wipe out a previously fetched company name on every tag edit. Carry it
    # over from the entry being replaced instead.
    name = watchlist[symbol].get("name")
    entry = req.model_dump()
    if name:
        entry["name"] = name
    watchlist[symbol] = entry
    wl.save(watchlist)
    return {"symbol": symbol, "tags": watchlist[symbol]}


@app.delete("/api/watchlist/{symbol:path}")
def api_remove(symbol: str) -> dict:
    watchlist = wl.load()
    if symbol not in watchlist:
        raise HTTPException(404, f"{symbol} has no entry")
    del watchlist[symbol]
    wl.save(watchlist)
    return {"removed": symbol}


@app.get("/api/watchlist/layout")
def api_get_layout() -> list[dict]:
    return wll.load(wl.load())


@app.put("/api/watchlist/layout")
def api_put_layout(layout: list[LayoutEntry]) -> list[dict]:
    watchlist = wl.load()
    entries = [entry.model_dump() for entry in layout]

    for entry in entries:
        if entry["type"] == "section" and not entry.get("id"):
            entry["id"] = wll.new_section_id()

    unknown = {e["symbol"] for e in entries if e["type"] == "ticker"} - watchlist.keys()
    if unknown:
        raise HTTPException(400, f"unknown symbol(s): {', '.join(sorted(unknown))}")

    section_ids = [e["id"] for e in entries if e["type"] == "section"]
    if len(section_ids) != len(set(section_ids)):
        raise HTTPException(400, "duplicate section id")

    wll.save(entries)
    return entries


@app.get("/api/alerts/log")
def api_alerts_log() -> list[dict]:
    return alerts_log.load()


@app.post("/api/alerts/log")
def api_alerts_log_append(entries: list[AlertLogKey]) -> list[dict]:
    """Append newly-seen (symbol, date, condition, tier) occurrences — a
    no-op for any the client sends that are already logged (see
    alerts_log.append_new's dedup). The client sends only what it thinks
    is new, but this is the authoritative check, not just an optimization
    to skip the round trip."""
    return alerts_log.append_new([e.model_dump() for e in entries])


@app.patch("/api/alerts/log/ack")
def api_alerts_log_ack(req: AlertAckRequest) -> list[dict]:
    return alerts_log.set_acked([k.model_dump() for k in req.keys], req.acked)


@app.post("/api/refresh/{symbol:path}")
def api_refresh_one(symbol: str) -> dict:
    """Force-refetch one symbol — the escape hatch for a stock split's
    retroactive price adjustment, which the gap-fill cache never revisits
    on its own."""
    payload = payload_for(symbol, refresh=True)
    return {"symbol": symbol, "payload": payload}


@app.post("/api/refresh")
def api_refresh_all() -> dict:
    """Force-refetch every watchlist symbol; one dead ticker must not stop
    the rest, same tolerance as the index page's per-symbol rendering."""
    watchlist = wl.load()
    refreshed, failed = [], []

    for symbol in watchlist:
        try:
            payload = payload_for(symbol, refresh=True)
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", None) or str(exc)
            failed.append({"symbol": symbol, "error": str(detail)})
            continue
        refreshed.append({"symbol": symbol, "payload": payload})

    return {"refreshed": refreshed, "failed": failed}


@app.get("/api/bars/{symbol:path}")
def api_bars(
    symbol: str, source: str = "yfinance", interval: str = "D", adjusted: bool = False
) -> dict:
    """Historical bars from either provider. For ``source="yfinance"``,
    ``interval`` selects D (default), M, or Y — anything else (intraday)
    isn't offered there, since ``sdx.data``'s cache pipeline is daily-only;
    M/Y resample from it (see ``bars_payload_for``). ``adjusted`` picks
    between the two independent yfinance caches (see ``sdx.data.load``) —
    split-adjusted-only (default, matches TradingView) vs. split- and
    dividend-adjusted (matches Futu/Webull)."""
    payload = bars_payload_for(symbol, source, interval, adjusted)
    return {"symbol": symbol, "payload": payload}


@app.get("/api/patterns/{symbol:path}")
def api_patterns(symbol: str, trend_bars: int = PINE_TREND_BARS) -> dict:
    """陰陽燭形態 recomputed under 5-day trend mode for a custom `trend_bars`.

    The round trip the "Trend in Bars" control triggers when its value moves
    away from the default already folded into the main payload
    (``patterns5day``/``patternAnchor5day``) — switching TO 5-day mode at
    the default needs no request at all.
    """
    df = load(symbol, SETTINGS["start"], SETTINGS["end"])
    if df.empty:
        raise HTTPException(404, f"{symbol}: no bars returned")
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    return patterns_for_trend_bars(df, dates, trend_bars)


@app.get("/api/ladder/{symbol:path}")
def api_ladder(
    symbol: str,
    bearish: bool = True,
    bullish: bool = False,
    close_fraction: float = OUTSIDE_BAR_CLOSE_FRACTION,
) -> dict:
    """外擴K ladder recomputed for a custom 收市比例 (body_ok's close fraction).

    The round trip the 收市比例 control triggers when its value moves away
    from the module default already folded into the four ladders
    ``_ladders()`` precomputes (see ``build_payload``'s ``alt`` mechanism) —
    those cover every 陰燭/陽燭 sub-toggle combination, but only at the
    default fraction, so an arbitrary value has to be recomputed here.
    """
    if not 0 <= close_fraction <= 1:
        raise HTTPException(400, "close_fraction must be between 0 and 1")
    df = load(symbol, SETTINGS["start"], SETTINGS["end"])
    if df.empty:
        raise HTTPException(404, f"{symbol}: no bars returned")
    result = run(
        df["high"].tolist(), df["low"].tolist(), df["close"].tolist(),
        df["volume"].tolist(), df["open"].tolist(),
        outside_bar_bearish=bearish, outside_bar_bullish=bullish,
        outside_bar_close_fraction=close_fraction,
    )
    payload = build_payload(
        df, result, symbol, SETTINGS.get("down_arrows", True), SETTINGS["params"],
    )
    return {k: v for k, v in payload.items() if k in LADDER_KEYS}


def _live_seed(symbol: str, interval: str) -> Optional[dict]:
    """The current bar as already known from REST, in ``_BarBucket.as_update``
    shape — passed to ``WebullStream.subscribe`` so the first live tick folds
    into the real session's open/high/low instead of a fresh stream seeding
    a bucket from whatever price happens to be trading right at subscribe
    time (see that function's docstring).

    ``D`` is seeded from yfinance (``sdx.data.load``), not Webull's own REST
    history: confirmed live against the sandbox this app is configured
    against (``WEBULL_API_ENDPOINT=api.sandbox.webull.hk``) that
    ``get_history_bar`` has no row at all yet for the still-forming trading
    day — its last row is days stale. A seed built from that is silently
    discarded the instant the first live tick's bucket fails to match it
    (``WebullStream._fold_tick``'s boundary-rollover path), reproducing
    exactly the bug this seeding exists to fix. yfinance's daily bar for
    `today` — a plain REST fetch, not tied to this stream at all — is
    already correct, so seed from that and let Webull's live ticks extend
    it from there. Other intervals still seed from Webull's own history,
    which does carry a forming bar on this sandbox at finer granularity.

    Best-effort throughout: any failure here just falls back to the old
    start-from-first-tick behavior rather than blocking the stream —
    ``subscribe`` itself still raises the real error (e.g.
    ``WebullNotConfigured``) once it tries to open the connection.
    """
    try:
        if interval == "D":
            df = load(symbol, SETTINGS["start"], SETTINGS["end"])
        else:
            df = webull_provider.get_bars(symbol, interval, SETTINGS["start"], SETTINGS["end"])
    except Exception:  # noqa: BLE001 — seeding is best-effort, never fatal
        return None
    if df.empty:
        return None
    last = df.iloc[-1]
    daily = interval in webull_provider.DATE_ONLY_INTERVALS
    time_value = (
        df.index[-1].strftime("%Y-%m-%d") if daily else int(df.index[-1].timestamp())
    )
    return {
        "time": time_value,
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": int(last["volume"]),
    }


@app.websocket("/ws/bars/{symbol:path}")
async def ws_bars(websocket: WebSocket, symbol: str, interval: str = "5m") -> None:
    """Live bar updates for a Webull-sourced symbol. No yfinance equivalent —
    the client only opens this when ``source=webull``. Month/Year never open
    this (see webull_provider.NO_LIVE_INTERVALS) — the frontend doesn't
    attempt it, but a direct/non-UI request still fails clearly rather than
    with a raw KeyError deep in the bar-folding logic."""
    await websocket.accept()

    try:
        webull_symbol, category = webull_provider.to_webull_symbol(symbol)
        stream = webull_provider.get_stream()
        seed = await asyncio.to_thread(_live_seed, symbol, interval)
        queue = stream.subscribe(webull_symbol, category, interval, seed=seed)
    except webull_provider.WebullNotConfigured as exc:
        await websocket.close(code=4400, reason=str(exc))
        return
    except ValueError as exc:
        await websocket.close(code=4400, reason=str(exc))
        return

    try:
        while True:
            update = await queue.get()
            await websocket.send_json(update)
    except WebSocketDisconnect:
        pass
    finally:
        stream.unsubscribe(webull_symbol, interval, queue)


def main(argv: Optional[list[str]] = None) -> int:
    import uvicorn
    from dotenv import load_dotenv

    # Loads .env (WEBULL_APP_KEY etc.) if present; a no-op otherwise, so
    # yfinance-only usage is unaffected. Only the live server entrypoint
    # needs this — tests set env vars directly, and the smoke test script
    # loads its own .env since it doesn't go through main().
    load_dotenv()

    ap = argparse.ArgumentParser(description="Serve the 生死線 chart.")
    # --end tracks today rather than a pinned date: a hardcoded end silently
    # stops the chart at whatever day it was written, and the staleness is
    # invisible because the page still renders perfectly.
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-down-arrows", action="store_false", dest="down_arrows")
    ap.add_argument("--rsi", type=int, default=9)
    ap.add_argument("--rsi-signal", type=int, default=6)
    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)
    ap.add_argument("--di", type=int, default=6)
    ap.add_argument("--adx", type=int, default=14)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    SETTINGS.update(
        start=args.start,
        end=args.end,
        down_arrows=args.down_arrows,
        params={
            "rsi": args.rsi,
            "rsi_signal": args.rsi_signal,
            "macd_fast": args.macd_fast,
            "macd_slow": args.macd_slow,
            "macd_signal": args.macd_signal,
            "di": args.di,
            "adx": args.adx,
        },
    )

    # Only safe here, before uvicorn starts accepting requests — a leftover
    # .tmp is unambiguously orphaned at this single-threaded point, unlike
    # mid-request where another thread could still be writing one.
    removed = cleanup_orphaned_tmp_files()
    if removed:
        print(f"生死線: cleaned up {removed} orphaned .tmp cache file(s)")

    url = f"http://127.0.0.1:{args.port}"
    print(f"生死線 → {url}")
    if not args.no_open:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

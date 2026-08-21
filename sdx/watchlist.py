"""Manage watchlists.json from the command line.

    python -m sdx.watchlist ls
    python -m sdx.watchlist add 0700.HK
    python -m sdx.watchlist add 0700.HK --held --strategy MACD金叉
    python -m sdx.watchlist rm AAPL
    python -m sdx.watchlist tag 0388.HK --stage 等量增即攻

``add`` fetches the symbol before writing it, so a typo fails here rather than
part-way through the next render. ``tag`` edits an existing entry's tags
without a data fetch.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "watchlists.json"

TAG_FIELDS = ("strategies", "stages", "patterns")

#: Matches a bare HK code with an optional leading zero and/or "." before
#: "HK" — "388.HK", "0388HK", and "388HK" all normalize the same way.
_HK_SYMBOL_RE = re.compile(r"^(\d{1,5})\.?HK$", re.IGNORECASE)


def normalize_symbol(raw: str) -> str:
    """Canonicalize an HK ticker to yfinance's ``NNNN.HK`` shape.

    The add-symbol input accepts "388.HK" (missing leading zero), "0388HK"
    (missing the dot), or "388HK" (both at once) interchangeably — all three
    normalize to "0388.HK". Anything that doesn't look like a bare HK code
    (US tickers, an already-correct HK ticker) passes through unchanged.
    """
    match = _HK_SYMBOL_RE.match(raw)
    if not match:
        return raw
    return f"{match.group(1).zfill(4)}.HK"


def _empty_entry() -> dict:
    # Entries may also carry an optional "name" (company name) key, but it's
    # never set here — only the server's add flow (sdx/serve.py:api_add) and
    # the one-time backfill script fetch it, both via a best-effort yfinance
    # lookup. Leaving it out of the CLI/migration default keeps those paths
    # network-free, and a missing "name" is a normal, valid entry shape.
    return {"held": False, "special": False, "strategies": [], "stages": [], "patterns": []}


def _is_old_shape(data: dict) -> bool:
    """Old files map group names to symbol lists; new files map symbols to tag dicts."""
    return all(isinstance(v, list) for v in data.values())


def _migrate(groups: dict[str, list[str]]) -> dict[str, dict]:
    """Convert `{group: [symbols]}` to the flat `{symbol: {tags}}` shape.

    Every symbol from every old group is collected into the flat map. A
    symbol that was in `持倉` gets `held: true`; every other field starts
    empty regardless of which old group(s) the symbol came from.
    """
    flat: dict[str, dict] = {}
    for group, symbols in groups.items():
        for symbol in symbols:
            entry = flat.setdefault(symbol, _empty_entry())
            if group == "持倉":
                entry["held"] = True
    return flat


def load(path: Path = CONFIG) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return {}
    if _is_old_shape(data):
        return _migrate(data)
    return data


def save(watchlist: dict[str, dict], path: Path = CONFIG) -> None:
    path.write_text(
        json.dumps(watchlist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify(symbol: str) -> int:
    """Fetch the symbol and return the bar count. Raises if it has no data."""
    from .data import load as load_bars

    df = load_bars(symbol, "2026-01-01", "2026-08-02", refresh=True)
    if df.empty:
        raise ValueError(f"{symbol!r} returned no bars — check the ticker")
    return len(df)


def _show(watchlist: dict[str, dict]) -> None:
    if not watchlist:
        print("(empty)")
        return
    for symbol, tags in watchlist.items():
        parts = ["持有"] if tags.get("held") else []
        for field in TAG_FIELDS:
            parts.extend(tags.get(field, []))
        print(f"{symbol}: {', '.join(parts) if parts else '—'}")


def _apply_tag_flags(entry: dict, args) -> None:
    if args.held is not None:
        entry["held"] = args.held
    for field, flag in (
        ("strategies", "strategy"),
        ("stages", "stage"),
        ("patterns", "pattern"),
    ):
        values = getattr(args, flag)
        if values:
            for v in values:
                if v not in entry[field]:
                    entry[field].append(v)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Manage watchlists.json")
    ap.add_argument("--config", type=Path, default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls", help="show the watchlist")

    add_p = sub.add_parser("add", help="add a symbol")
    add_p.add_argument("symbol")
    add_p.add_argument("--no-verify", action="store_true", help="skip the data fetch (offline)")

    rm_p = sub.add_parser("rm", help="remove a symbol")
    rm_p.add_argument("symbol")

    tag_p = sub.add_parser("tag", help="edit an existing symbol's tags")
    tag_p.add_argument("symbol")

    for p in (add_p, tag_p):
        held = p.add_mutually_exclusive_group()
        held.add_argument("--held", dest="held", action="store_true", default=None)
        held.add_argument("--no-held", dest="held", action="store_false")
        p.add_argument("--strategy", action="append", default=[])
        p.add_argument("--stage", action="append", default=[])
        p.add_argument("--pattern", action="append", default=[])

    args = ap.parse_args(argv)
    watchlist = load(args.config)

    if args.cmd == "ls":
        _show(watchlist)
        return 0

    symbol = args.symbol
    if args.cmd == "add":
        symbol = normalize_symbol(symbol)

    if args.cmd == "rm":
        if symbol not in watchlist:
            print(f"{symbol} has no entry")
            return 1
        del watchlist[symbol]
        save(watchlist, args.config)
        print(f"removed {symbol}")
        _show(watchlist)
        return 0

    if args.cmd == "tag":
        if symbol not in watchlist:
            print(f"{symbol} has no entry")
            return 1
        _apply_tag_flags(watchlist[symbol], args)
        save(watchlist, args.config)
        _show(watchlist)
        return 0

    if args.cmd == "add":
        if symbol in watchlist:
            print(f"{symbol} already has an entry")
            return 0
        if not args.no_verify:
            try:
                bars = verify(symbol)
            except Exception as exc:  # noqa: BLE001 — surface any fetch failure
                print(f"could not fetch {symbol}: {exc}")
                return 1
            print(f"{symbol}: {bars} bars")
        entry = _empty_entry()
        _apply_tag_flags(entry, args)
        watchlist[symbol] = entry
        save(watchlist, args.config)
        _show(watchlist)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

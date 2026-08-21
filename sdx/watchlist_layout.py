"""Manage watchlist_layout.json — the watchlist sidebar's display order and
section grouping, kept separate from watchlists.json's tag data so none of
the existing watchlist CRUD (sdx.watchlist, /api/watchlist...) needs to
change shape.

An entry is either ``{"type": "ticker", "symbol": ...}`` or
``{"type": "section", "id": ..., "name": ..., "collapsed": ...}``, in
display order. A section is a standalone line, not a wrapper — nothing
nests.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "watchlist_layout.json"


def new_section_id() -> str:
    return "sec_" + secrets.token_hex(4)


def load(watchlist: dict[str, dict], path: Path = CONFIG) -> list[dict]:
    """Reconcile the stored layout against `watchlist`'s current symbols.

    Any symbol in `watchlist` missing from the stored layout is appended at
    the end, in `watchlist`'s own (dict) order. Any stored ticker entry
    whose symbol is no longer in `watchlist` is dropped. Section entries
    are never dropped. This only reconciles the in-memory result — it does
    not write anything back to `path`.
    """
    if not path.exists():
        stored: list[dict] = []
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        stored = data if isinstance(data, list) else []

    seen: set[str] = set()
    reconciled: list[dict] = []
    for entry in stored:
        if entry.get("type") == "ticker":
            symbol = entry.get("symbol")
            if symbol not in watchlist:
                continue
            seen.add(symbol)
        reconciled.append(entry)

    for symbol in watchlist:
        if symbol not in seen:
            reconciled.append({"type": "ticker", "symbol": symbol})

    return reconciled


def save(layout: list[dict], path: Path = CONFIG) -> None:
    path.write_text(
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

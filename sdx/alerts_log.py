"""Persisted history of watchlist alert occurrences.

sdx/viz.py's Alerts panel writes here so a signal's acknowledgment
survives page reloads without the staleness a single per-symbol flag
would have: each row is a distinct (symbol, date, condition, tier)
occurrence, not a per-symbol bit, so a signal on a new date is always a
new row needing its own acknowledgment — nothing to invalidate or clear
manually. It also builds up a real history that can be reviewed later
(the file is plain JSON, list of records — inspect or query directly).

    python -m sdx.alerts_log ls
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "alerts_log.json"


def _key(entry: dict) -> tuple:
    return (entry["symbol"], entry["date"], entry["condition"], entry["tier"])


def load(path: Path = CONFIG) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data or []


def save(entries: list[dict], path: Path = CONFIG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_new(new_entries: list[dict], path: Path = CONFIG) -> list[dict]:
    """Append only entries whose (symbol, date, condition, tier) key isn't
    already present. The client also dedupes locally against its own
    in-memory copy of the log before ever calling this — that's purely an
    optimization to skip the round trip for something already known; this
    is the authoritative dedup, since two browser tabs (or a stale client
    copy after a long-open session) could otherwise both send the same
    occurrence.
    """
    existing = load(path)
    seen = {_key(e) for e in existing}
    added = False
    for e in new_entries:
        entry = {
            "symbol": e["symbol"],
            "date": e["date"],
            "condition": e["condition"],
            "tier": e["tier"],
            "acked": False,
        }
        k = _key(entry)
        if k not in seen:
            existing.append(entry)
            seen.add(k)
            added = True
    if added:
        save(existing, path)
    return existing


def set_acked(keys: list[dict], acked: bool, path: Path = CONFIG) -> list[dict]:
    """``keys``: list of {symbol, date, condition, tier} identifying which
    existing rows to update. Keys with no matching row are silently
    ignored — the client always ack()s rows it just confirmed exist via
    append_new(), but a race (two tabs) could still send a stale key."""
    existing = load(path)
    want = {(k["symbol"], k["date"], k["condition"], k["tier"]) for k in keys}
    changed = False
    for e in existing:
        if _key(e) in want and e["acked"] != acked:
            e["acked"] = acked
            changed = True
    if changed:
        save(existing, path)
    return existing


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sdx.alerts_log")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls")
    args = ap.parse_args(argv)

    if args.cmd == "ls":
        for e in load():
            mark = "x" if e["acked"] else " "
            print(f"[{mark}] {e['date']}  {e['symbol']:<10} {e['tier']:<11} {e['condition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

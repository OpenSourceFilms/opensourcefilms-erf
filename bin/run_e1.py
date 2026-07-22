#!/usr/bin/env python3
"""One-time batch runner for Stage E1 (event narrative-signal re-tag).

Processes all 173 media files that have catalog/events/ JSONL files.
Skips media_ids that already have a signals file unless --force is passed.

Usage:
    PYTHONPATH=src venvs/embed/bin/python bin/run_e1.py [--force] [--workers N] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from erf.e1_signals import run as e1_run
from erf import outcomes

_CATALOG = _ROOT / "catalog"
_JOURNAL_DB = _ROOT / "state" / "journal.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage E1 batch runner")
    parser.add_argument("--force", action="store_true", help="Overwrite existing signals files")
    parser.add_argument("--workers", type=int, default=8, help="Parallel Flash calls per media (default 8)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N media (0=all)")
    args = parser.parse_args()

    events_dir = _CATALOG / "events"
    signals_dir = _CATALOG / "event_signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    media_ids = sorted(p.stem for p in events_dir.glob("*.jsonl"))
    print(f"Found {len(media_ids)} media with events", flush=True)

    if args.limit:
        media_ids = media_ids[: args.limit]
        print(f"Limiting to {args.limit} media", flush=True)

    ok_count = skipped_count = error_count = 0
    total_records = 0
    total_parse_errors = 0
    t0 = time.monotonic()

    for i, mid in enumerate(media_ids, 1):
        signals_path = signals_dir / f"{mid}.jsonl"
        if signals_path.exists() and not args.force:
            skipped_count += 1
            continue
        # Gate on E0's explicit has_events flag (cross-stage awareness), not file presence.
        if outcomes.flag(mid, "has_events") is False:
            outcomes.record(mid, "E1", "skip", reason="no_events")
            outcomes.set_flag(mid, "has_event_signals", False, reason="no_events")
            skipped_count += 1
            continue

        try:
            result = e1_run(
                mid,
                catalog=_CATALOG,
                journal_db=_JOURNAL_DB,
                max_workers=args.workers,
            )
            ok_count += 1
            total_records += result["records_written"]
            total_parse_errors += result["parse_errors"]
            outcomes.record(mid, "E1", "ok", counts={"signals": result["records_written"]})
            outcomes.set_flag(mid, "has_event_signals", result["records_written"] > 0)
            elapsed = time.monotonic() - t0
            print(
                f"[{i}/{len(media_ids)}] {mid}: {result['records_written']} signals"
                f" ({result['parse_errors']} parse_errors) [{elapsed:.1f}s]",
                flush=True,
            )
        except Exception as exc:
            error_count += 1
            outcomes.record(mid, "E1", "error", reason=str(exc)[:200])
            print(f"[{i}/{len(media_ids)}] ERROR {mid}: {exc}", file=sys.stderr, flush=True)

    elapsed = time.monotonic() - t0
    print(
        f"\nDone in {elapsed:.1f}s — ok={ok_count} skipped={skipped_count} errors={error_count} "
        f"total_signals={total_records} parse_errors={total_parse_errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()

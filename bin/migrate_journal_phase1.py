#!/usr/bin/env python3
"""One-time Phase 1 migration runner (D-Log-04, docs/OBSERVABILITY_REDESIGN.md §4).

Idempotent — safe to re-run; no-ops if state/journal.db is already migrated
(events already has the new §3.1 shape). Delegates entirely to
erf.outcomes.migrate_phase1(), which:
  1. Renames the pre-redesign `events` table (event_type/payload shape) to
     `events_legacy` (frozen, verbatim, never written to again).
  2. Creates the new `events` (outcome-only, §3.1) and `events_trace`
     (operational trace, old column shape) tables.
  3. Backfills `events` from every events_legacy row whose event_type is a
     disposition (ok/skip/empty/error/partial) — start/warn/namespaced
     sub-events stay in events_legacy only.
  4. Folds status_projection from the new events (erf.outcomes.rebuild_projection).

Usage: PYTHONPATH=src venvs/control/bin/python bin/migrate_journal_phase1.py [--db PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/workspace/erf")
sys.path.insert(0, str(ROOT / "src"))

from erf import outcomes as oc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(ROOT / "state" / "journal.db"))
    args = ap.parse_args()

    db = Path(args.db)
    print(f"migrating {db} ...")
    stats = oc.migrate_phase1(db)
    if stats.get("already_migrated"):
        print("already migrated — no-op.")
        return
    if stats.get("fresh"):
        print("no pre-existing events table — laid down the new schema fresh.")
        return
    print(f"events_legacy rows (frozen, verbatim): {stats['legacy_rows']}")
    print(f"backfilled into new events (disposition rows only): {stats['backfilled_rows']}")
    print(f"status_projection rows after fold: {stats['projection_rows']}")


if __name__ == "__main__":
    main()

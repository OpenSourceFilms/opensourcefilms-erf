#!/usr/bin/env python3
"""Disposable example: explicit stage outcomes backed by an SQLite event journal.

No media, models, network requests or existing ERF state are accessed.
Run from a clone: python3 examples/explicit_outcomes_demo.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="erf_explicit_outcome_demo_") as temp:
        work = Path(temp)
        # erf.outcomes binds its paths at import time: set both overrides first.
        for key, path in (("ERF_CATALOG_ROOT", work / "catalog"),
                          ("ERF_JOURNAL_DB", work / "journal.db")):
            os.environ[key] = str(path)

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from erf import outcomes

        media_id = "m_111111111111"  # synthetic ID, no original photography
        outcomes.migrate_phase1()  # initialise fresh events / projection schema
        outcomes.record(media_id, "A", "ok", metrics={"words": 4})
        outcomes.record(media_id, "B1", "empty", reason="no synthetic matches")
        outcomes.set_flag(media_id, "safe_for_example", True)

        # An empty result is explicit completion, not a missing result.
        assert outcomes.status_of(media_id, "A") == "ok"
        assert outcomes.status_of(media_id, "B1") == "empty"
        assert outcomes.is_complete(media_id, "B1")
        assert outcomes.status_of(media_id, "C") is None
        assert outcomes.flag(media_id, "safe_for_example") is True
        with sqlite3.connect(outcomes.DB) as database:
            events = database.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            projection = database.execute(
                "SELECT COUNT(*) FROM status_projection").fetchone()[0]
        assert (events, projection) == (2, 2)

        print(json.dumps({
            "schema": "osf.erf.outcome_demo.v1",
            "event_count": events,
            "projected_stages": projection,
            "stage_a": outcomes.status_of(media_id, "A"),
            "stage_b1": outcomes.status_of(media_id, "B1"),
            "unattempted_stage": outcomes.status_of(media_id, "C"),
            "synthetic_flag": outcomes.flag(media_id, "safe_for_example"),
            "temporary_state_removed_on_exit": True,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

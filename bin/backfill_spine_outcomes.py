#!/usr/bin/env python3
"""One-shot: backfill explicit outcomes for pre-existing spine artifacts (W/RG/RP/DX).

These four stages gained outcome recording on 2026-07-05 (MASTER §5.2.1/§5.2.2).
Historical media already have their artifacts on disk but no recorded outcome, so
the auditor (reconcile_status.py) would read them as unrecorded/silent until the
next full rebuild re-runs the stages. This records `ok` (derived from disk) ONLY
where no outcome exists yet, using the SAME predicates as reconcile_status so the
two never diverge. Idempotent: re-running records nothing once outcomes exist.

Run: venvs/control/bin/python bin/backfill_spine_outcomes.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bin"))
from erf import outcomes as oc            # noqa: E402
from reconcile_status import STAGE_ARTIFACTS, _media_ids  # noqa: E402

STAGES = ["W", "RG", "RP", "DX"]
RUN_ID = "backfill_from_disk_2026-07-05"


def main() -> None:
    media = _media_ids()
    for st in STAGES:
        pred = STAGE_ARTIFACTS[st]
        n = skipped = 0
        for m in media:
            if not pred(m):
                continue
            if oc.get(m, st) is not None:
                skipped += 1
                continue
            oc.record(m, st, "ok", counts={"backfill": True},
                      run_id=RUN_ID, producer="backfill_spine_outcomes.py")
            n += 1
        print(f"{st}: backfilled {n}, already-recorded {skipped}")


if __name__ == "__main__":
    main()

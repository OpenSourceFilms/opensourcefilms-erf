#!/usr/bin/env python3
"""Stale-outcome detector — the failure mode the outcomes spine cannot see.

The spine enforces "absence is never a signal": every stage records ok/skip/empty/
error. It does NOT detect STALENESS — a stage carrying a completed `ok` outcome whose
INPUT artifacts are now absent (e.g. wiped by a catalog move-aside, or never produced
this epoch). Example this catches: EV recorded `ok` while visual_grounding/ +
event_signals/ (its inputs) hold zero files, so the evidence packages are orphaned.

Read-only; exits 1 if any stale outcome is found (CI-friendly).
Usage: PYTHONPATH=src venvs/embed/bin/python bin/check_stale_outcomes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erf import outcomes

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"


def _present(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


# Consumer stage -> the input artifacts it reads to produce its output. An `ok`
# outcome asserts "I produced output FROM these inputs"; if an input is now absent
# the output is stale/orphaned. (skip/empty outcomes legitimately have absent inputs,
# so only `ok` outcomes are checked.)
STAGE_INPUTS = {
    "V0":    lambda m: [CAT / f"keyframes/{m}.jsonl"],
    "E1":    lambda m: [CAT / f"events/{m}.jsonl"],
    "EV":    lambda m: [CAT / f"visual_grounding/{m}.json", CAT / f"event_signals/{m}.jsonl"],
    "LABEL": lambda m: [CAT / f"evidence/{m}.json"],
}


def main() -> None:
    mids = sorted(p.stem for p in (CAT / "media_manifest").glob("m_*.json"))
    by_stage: dict[str, list] = {}
    for mid in mids:
        for stage, inputs in STAGE_INPUTS.items():
            if outcomes.status_of(mid, stage) != "ok":
                continue
            missing = [p.name for p in inputs(mid) if not _present(p)]
            if missing:
                by_stage.setdefault(stage, []).append((mid, missing))

    total = sum(len(v) for v in by_stage.values())
    if total == 0:
        print("OK — every `ok` outcome's declared inputs are present on disk.")
        return
    print(f"STALE: {total} `ok` outcomes whose inputs are missing (output orphaned)\n")
    for stage, items in sorted(by_stage.items()):
        print(f"  {stage}: {len(items)} media")
        for mid, missing in items[:5]:
            print(f"    {mid}: missing {missing}")
        if len(items) > 5:
            print(f"    ... and {len(items) - 5} more")
    sys.exit(1)


if __name__ == "__main__":
    main()

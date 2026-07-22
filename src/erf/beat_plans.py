"""Beat plan catalog — append-only JSONL storage per §9.5.

Each POST to /shorts/assemble appends a new revision.
revision field increments; supersedes_revision points to prior.
"""
from __future__ import annotations

import json
from pathlib import Path

from .atomicio import read_jsonl, write_tmp_rename


def append_beat_plan(bp: dict, catalog: Path) -> None:
    """Append beat_plan as new JSONL line; sets revision and supersedes_revision."""
    bp_id = bp["bp_id"]
    path = catalog / "beat_plans" / f"{bp_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = read_jsonl(path)
    last_rev = existing[-1].get("revision", -1) if existing else -1
    bp["revision"] = last_rev + 1
    bp["supersedes_revision"] = last_rev if existing else None

    # Atomic append: read all + append + write_tmp_rename
    lines = [json.dumps(r, ensure_ascii=False) for r in existing]
    lines.append(json.dumps(bp, ensure_ascii=False))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    write_tmp_rename(path, data)


def latest_beat_plan(bp_id: str, catalog: Path) -> dict | None:
    """Return the latest revision of a beat_plan, or None."""
    path = catalog / "beat_plans" / f"{bp_id}.jsonl"
    records = read_jsonl(path)
    return records[-1] if records else None

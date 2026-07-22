"""Review queue — CRUD for review_queue and review_plan_diffs tables."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .beat_plans import append_beat_plan, latest_beat_plan
from .export import export

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
}

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _make_review_id() -> str:
    ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    ts_enc = ""
    for _ in range(10):
        ts_enc = _CROCKFORD[ms & 31] + ts_enc
        ms >>= 5
    rand_enc = ""
    for _ in range(16):
        rand_enc = _CROCKFORD[rand & 31] + rand_enc
        rand >>= 5
    return "rev_" + ts_enc + rand_enc


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
    return cfg


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = TRUNCATE")
    conn.row_factory = sqlite3.Row
    return conn


def _emit_journal(state_dir: Path, review_id: str, action: str, extra: dict | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps({"action": action, "review_id": review_id, "outputs_produced": [review_id], **(extra or {})})
    try:
        conn = sqlite3.connect(str(state_dir / "journal.db"))
        conn.execute("PRAGMA journal_mode = TRUNCATE")
        conn.execute(
            # Phase 1 (D-Log-04): review-action trace -> events_trace (§3.1).
            "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) VALUES (?, ?, 'erf', 'review', 'ok', ?, ?)",
            (now, f"erf:review:{review_id}", review_id, payload),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # journal is best-effort


# ── Public API ────────────────────────────────────────────────────────────────

def get_review(review_id: str, *, config_path: str | Path = "state/config.toml") -> dict | None:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    conn = _connect(Path(cfg["state"]) / "erf.db")
    row = conn.execute("SELECT * FROM review_queue WHERE review_id = ?", (review_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_reviews(
    *,
    state: str | None = None,
    platform: str | None = None,
    config_path: str | Path = "state/config.toml",
) -> list[dict]:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    conn = _connect(Path(cfg["state"]) / "erf.db")
    sql = "SELECT * FROM review_queue WHERE 1=1"
    params: list = []
    if state is not None:
        sql += " AND state = ?"
        params.append(state)
    if platform is not None:
        sql += " AND platform = ?"
        params.append(platform)
    sql += " ORDER BY created_at_utc DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def approve_review(review_id: str, reviewed_by: str, *, config_path: str | Path = "state/config.toml") -> dict:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    state_dir = Path(cfg["state"])
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(state_dir / "erf.db")
    cur = conn.execute(
        "UPDATE review_queue SET state='approved', reviewed_at_utc=?, reviewed_by=? WHERE review_id=?",
        (now, reviewed_by, review_id),
    )
    if cur.rowcount == 0:
        conn.close()
        raise ValueError(f"review_id '{review_id}' not found")
    conn.commit()
    row = conn.execute("SELECT * FROM review_queue WHERE review_id=?", (review_id,)).fetchone()
    conn.close()
    _emit_journal(state_dir, review_id, "approve", {"reviewed_by": reviewed_by})
    return dict(row)


def reject_review(
    review_id: str, reviewed_by: str, notes: str, *, config_path: str | Path = "state/config.toml"
) -> dict:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    state_dir = Path(cfg["state"])
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(state_dir / "erf.db")
    cur = conn.execute(
        "UPDATE review_queue SET state='rejected', reviewed_at_utc=?, reviewed_by=?, notes=? WHERE review_id=?",
        (now, reviewed_by, notes, review_id),
    )
    if cur.rowcount == 0:
        conn.close()
        raise ValueError(f"review_id '{review_id}' not found")
    conn.commit()
    row = conn.execute("SELECT * FROM review_queue WHERE review_id=?", (review_id,)).fetchone()
    conn.close()
    _emit_journal(state_dir, review_id, "reject", {"reviewed_by": reviewed_by, "notes": notes})
    return dict(row)


def edit_review(
    review_id: str,
    new_beats: list,
    reviewed_by: str,
    *,
    diff_summary: str | None = None,
    config_path: str | Path = "state/config.toml",
) -> dict:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    state_dir = Path(cfg["state"])
    catalog = Path(cfg["catalog"])

    existing = get_review(review_id, config_path=config_path)
    if existing is None:
        raise ValueError(f"review_id '{review_id}' not found")

    beat_plan_id = existing["beat_plan_id"]
    current_bp = latest_beat_plan(beat_plan_id, catalog)
    if current_bp is None:
        raise ValueError(f"beat_plan '{beat_plan_id}' not found in catalog")
    original_revision = current_bp.get("revision", 0)

    # Build new beat_plan, replacing beats
    new_bp = {k: v for k, v in current_bp.items() if k not in ("revision", "supersedes_revision")}
    new_bp["beats"] = new_beats
    new_bp["schema_version"] = "1.1.0"
    append_beat_plan(new_bp, catalog)  # sets revision + supersedes_revision on new_bp in-place
    new_revision = new_bp["revision"]

    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(state_dir / "erf.db")
    conn.execute(
        "INSERT INTO review_plan_diffs (review_id, original_beat_plan_revision, edited_beat_plan_revision, diff_summary, created_at_utc) VALUES (?, ?, ?, ?, ?)",
        (review_id, original_revision, new_revision, diff_summary, now),
    )
    conn.execute(
        "UPDATE review_queue SET state='edited', beat_plan_revision=? WHERE review_id=?",
        (new_revision, review_id),
    )
    conn.commit()
    conn.close()

    _emit_journal(state_dir, review_id, "edit", {"beat_plan_id": beat_plan_id, "new_revision": new_revision, "reviewed_by": reviewed_by})
    return {"review_id": review_id, "beat_plan_id": beat_plan_id, "new_beat_plan_revision": new_revision}


def reassemble_review(
    review_id: str, *, config_path: str | Path = "state/config.toml", fmt: str = "otio"
) -> dict:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    state_dir = Path(cfg["state"])
    catalog = Path(cfg["catalog"])

    existing = get_review(review_id, config_path=config_path)
    if existing is None:
        raise ValueError(f"review_id '{review_id}' not found")

    beat_plan_id = existing["beat_plan_id"]
    bp = latest_beat_plan(beat_plan_id, catalog)
    if bp is None or not bp.get("beats"):
        raise ValueError(f"beat_plan '{beat_plan_id}' has no beats")

    clips = [
        {"media_id": b["media_id"], "in_ms": b["in_ms"], "out_ms": b["out_ms"], "label": b.get("edit_role", "")}
        for b in bp["beats"]
    ]
    manifest = export(clips, fmt=fmt, query=bp.get("topic", ""), config_path=config_path)

    conn = _connect(state_dir / "erf.db")
    conn.execute(
        "UPDATE review_queue SET output_otio_relpath=?, export_id=? WHERE review_id=?",
        (manifest["output_relpath"], manifest["export_id"], review_id),
    )
    conn.commit()
    conn.close()

    _emit_journal(state_dir, review_id, "reassemble", {"export_id": manifest["export_id"]})
    return {"export_id": manifest["export_id"], "output_otio_relpath": manifest["output_relpath"]}

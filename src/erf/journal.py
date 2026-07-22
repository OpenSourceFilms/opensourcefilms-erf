"""Event journal and status projection for ERF.

journal.db is an event log + status projection (SQLite, TRUNCATE mode).
It does NOT touch erf.db. Status catalog files are written via atomicio.

Phase 1 (D-Log-04, docs/OBSERVABILITY_REDESIGN.md): two surfaces now live in
journal.db — `events` (the ONE outcome surface, §3.1: status/reason/metrics/...,
written only via `erf.outcomes.record()`) and `events_trace` (operational trace —
start/warn/sub-events, old event_type/payload shape, written via
`erf.outcomes.trace()`). `emit_start` below writes trace only; `emit_ok`/
`emit_error`/`reap_manifest` funnel into `outcomes.record()`. `status_projection`
is a materialized view folded from `events` (see `outcomes.rebuild_projection`).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .atomicio import write_json
from .manifest import StageManifest

# NOTE: this DDL must stay in sync with erf.outcomes._NEW_EVENTS_DDL, which lays
# down the identical `events`/`events_trace` shape when migrating an existing
# (pre-Phase-1) journal.db in place. Duplicated rather than imported to keep
# journal.py and outcomes.py free of a module-level import of each other.
_SCHEMA = """
PRAGMA journal_mode = TRUNCATE;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id      TEXT    NOT NULL,
    stage         TEXT    NOT NULL,
    status        TEXT    NOT NULL,   -- ok|skip|empty|partial|degraded|stale|error
    reason        TEXT,
    run_id        TEXT,
    producer_sha  TEXT,
    config_fp     TEXT,
    metrics       TEXT,               -- json
    health        TEXT,               -- json
    inputs        TEXT,               -- json
    artifacts     TEXT,               -- json
    at_utc        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_events_media  ON events(media_id, stage);
CREATE INDEX IF NOT EXISTS ix_events_status ON events(status);

-- Operational trace (start / warn / sub-events / namespaced diagnostics). NOT an
-- outcome surface — never folded into status_projection/health. Old events
-- column shape, preserved so the ~30 pre-existing trace call sites work as-is.
CREATE TABLE IF NOT EXISTS events_trace (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at_utc      TEXT    NOT NULL,
    job_id      TEXT    NOT NULL,
    media_id    TEXT    NOT NULL,
    stage       TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    run_id      TEXT,
    payload     TEXT
);

CREATE INDEX IF NOT EXISTS ix_events_trace_job   ON events_trace(job_id);
CREATE INDEX IF NOT EXISTS ix_events_trace_media ON events_trace(media_id, stage);

CREATE TABLE IF NOT EXISTS status_projection (
    media_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    ok          INTEGER,
    run_id      TEXT,
    at_utc      TEXT,
    error       TEXT,
    PRIMARY KEY (media_id, stage)
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    # journal.db is SQLite on a MooseFS/FUSE volume. Concurrency (Phase 7 fans writes
    # across threads) is handled by SQLite's own file locking + busy_timeout: each
    # thread uses its own short-lived connection and WAITS (not errors) if another is
    # mid-write. We deliberately do NOT hold a process-wide lock across connect/commit —
    # a single slow FUSE op under such a lock freezes the whole pool. synchronous=NORMAL
    # avoids per-commit fsync stalls on MooseFS; journal.db is reconstructable from
    # canonical artifacts via bin/reconcile_status.py, and the canonical per-media
    # outcome files (catalog/outcomes/) are contention-free, so this is safe.
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = TRUNCATE")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_journal(db_path: Path) -> None:
    """Create journal.db with the events + status_projection schema if not exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def emit_start(
    db_path: Path,
    media_id: str,
    stage: str,
    run_id: str,
    *,
    payload: Optional[dict] = None,
) -> None:
    """Record that a stage has started. Trace-only (Phase 1): writes events_trace,
    never the outcome surface / status_projection — a stage that has merely
    started has no disposition yet, so there is nothing to fold. Best-effort,
    same as before (a lost start breadcrumb must not break a stage)."""
    from . import outcomes
    outcomes.trace(media_id, stage, "start", payload, run_id=run_id, db=db_path)


def emit_ok(db_path: Path, manifest: StageManifest) -> None:
    """Record successful stage completion from a StageManifest, via the single
    outcome writer (erf.outcomes.record)."""
    from . import outcomes
    outcomes.record(
        manifest.media_id, manifest.stage, "ok",
        counts=manifest.counts or {}, artifacts=list(manifest.outputs or []),
        producer=manifest.run_id, run_id=manifest.run_id, db=db_path,
    )


def emit_error(
    db_path: Path,
    media_id: str,
    stage: str,
    run_id: str,
    error: str,
) -> None:
    """Record stage failure, via the single outcome writer (erf.outcomes.record)."""
    from . import outcomes
    outcomes.record(media_id, stage, "error", reason=error, run_id=run_id, db=db_path)


def reap_manifest(
    db_path: Path,
    manifest: StageManifest,
    catalog_status_dir: Path,
) -> None:
    """Reap a stage manifest: update journal then rebuild the status catalog file.

    This is the control-plane reaper path. Workers write StageManifests to their
    done.jsonl; the control plane reads them and calls this function.

    Phase 1: emit_ok/emit_error above already funnel into erf.outcomes.record()
    (the single outcome writer), so this no longer makes a second, redundant
    record() call — one manifest outcome, one write.
    """
    if manifest.ok:
        emit_ok(db_path, manifest)
    else:
        emit_error(db_path, manifest.media_id, manifest.stage, manifest.run_id, manifest.error or "unknown error")
    rebuild_status(db_path, manifest.media_id, catalog_status_dir, manifest_extra=manifest)


def rebuild_status(
    db_path: Path,
    media_id: str,
    catalog_status_dir: Path,
    manifest_extra: Optional[StageManifest] = None,
) -> dict[str, Any]:
    """Rebuild catalog/status/{media_id}.json from status_projection.

    manifest_extra: if provided, its tool_version/counts/error override the projection row
    for its stage (used immediately after reap when projection may not carry those fields).
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT stage, ok, run_id, at_utc, error FROM status_projection WHERE media_id = ?",
            (media_id,),
        ).fetchall()
    finally:
        conn.close()

    stages: dict[str, Any] = {}
    for row in rows:
        stage = row["stage"]
        stages[stage] = {
            "at_utc": row["at_utc"],
            "ok": bool(row["ok"]) if row["ok"] is not None else None,
            "run_id": row["run_id"],
            "tool_version": None,
            "counts": None,
            "error": row["error"],
        }

    if manifest_extra is not None and manifest_extra.media_id == media_id:
        s = manifest_extra.stage
        if s not in stages:
            stages[s] = {
                "at_utc": manifest_extra.at_utc,
                "ok": manifest_extra.ok,
                "run_id": manifest_extra.run_id,
                "tool_version": manifest_extra.tool_version,
                "counts": manifest_extra.counts,
                "error": manifest_extra.error,
            }
        else:
            stages[s]["tool_version"] = manifest_extra.tool_version
            stages[s]["counts"] = manifest_extra.counts
            if manifest_extra.error:
                stages[s]["error"] = manifest_extra.error

    status = {
        "schema_version": "1.0.0",
        "media_id": media_id,
        "stages": stages,
    }
    out_path = catalog_status_dir / f"{media_id}.json"
    write_json(out_path, status, indent=2)
    return status

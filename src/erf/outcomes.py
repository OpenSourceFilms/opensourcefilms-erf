"""Explicit-outcome spine for ERF (MASTER Invariant 3).

Every stage records what it attempted, what it produced, and — when it produced
nothing — WHY. Absence is never a signal. Outcomes are first-class, structured,
queryable facts written to three surfaces:

  1. journal.db `events` — the ONE append-only outcome log (D-Log-04 §3.1),
     written ONLY by `record()` below. `status_projection` is a materialized
     view folded from it (`rebuild_projection()`) — never independently
     asserted (see reconcile_status.py / pipeline_status.py).
  2. journal.db `events_trace` — operational trace (start/warn/sub-events),
     written ONLY by `trace()` below. NOT an outcome surface: never read for
     status/health. Best-effort (a lost trace breadcrumb must not break a
     stage); `record()`, by contrast, is a hard dependency (P4 — no silent
     swallow, a write failure raises).
  3. catalog/status/{media_id}.json — the per-media label surface any stage can
     read to know what every other stage did (backward/forward/current awareness),
     plus media-level flags (e.g. has_speech=False).

A stage NEVER infers another stage's result from file presence/size; it calls
`get()`/`status_of()`/`flag()` here. New stages inherit this by calling `record()`
on every exit path (ok / skip / empty / error) — the contract is uniform, so
system awareness is baked in and futureproofed.

Phase 1 migration (D-Log-04, docs/OBSERVABILITY_REDESIGN.md §4): `migrate_phase1()`
renames the pre-redesign `events` table (event_type/payload shape) to
`events_legacy` (frozen, verbatim, never written to again), lays down the new
`events`/`events_trace` tables, and backfills disposition rows. Idempotent.

Phase 3 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §3.2): `record()` now
accepts a `health` dict (the §3.1 shape — see `erf.stages.evaluate_health()`)
and DERIVES `degraded` from `status="ok"` + any failing predicate, the
"green-but-wrong killer". `metrics`/`inputs` are the other two Phase 3 fields —
`inputs` carries upstream-artifact fingerprints a stage consumed this run
(no staleness-marking logic reads it yet; that is a later, read-side phase).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # portable checkout root, same as erf.paths.ROOT
# D-Log-35 seam: shadow bake-off runs must not write the production spine.
# ERF_CATALOG_ROOT redirects the outcome/flag label surface alongside the rest
# of the catalog; ERF_JOURNAL_DB redirects the journal itself. Unset = unchanged.
CAT = Path(os.environ.get("ERF_CATALOG_ROOT") or (ROOT / "catalog"))
# Dedicated, single-writer surface for explicit outcomes + media flags. Kept separate
# from the legacy catalog/status/ (which is rebuilt wholesale by journal.rebuild_status)
# so there is exactly one writer here and no read-modify-write conflict.
STATUS_DIR = CAT / "outcomes"
DB = Path(os.environ.get("ERF_JOURNAL_DB") or (ROOT / "state" / "journal.db"))

# degraded/stale are legal dispositions from Phase 1 on (§3.1) but nothing emits
# them until Phase 3 (semantic health predicates / input fingerprinting).
VALID_STATUS = {"ok", "skip", "empty", "error", "partial", "degraded", "stale"}
# Statuses meaning the stage RAN to a definite conclusion (completion, not failure/missing):
COMPLETED = {"ok", "skip", "empty"}
NOT_GREEN = {"partial", "degraded", "stale", "error"}
_REASON_REQUIRED = {"skip", "empty", "error", "degraded", "stale"}

_log = logging.getLogger("erf.outcomes")
_file_lock = threading.Lock()  # serialise same-process status-file read-modify-writes

# NOTE: kept in sync with erf.journal._SCHEMA's events/events_trace block (see the
# comment there) — duplicated, not imported, to avoid a module-level import cycle
# between journal.py and outcomes.py.
_NEW_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id      TEXT    NOT NULL,
    stage         TEXT    NOT NULL,
    status        TEXT    NOT NULL,
    reason        TEXT,
    run_id        TEXT,
    producer_sha  TEXT,
    config_fp     TEXT,
    metrics       TEXT,
    health        TEXT,
    inputs        TEXT,
    artifacts     TEXT,
    at_utc        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_media  ON events(media_id, stage);
CREATE INDEX IF NOT EXISTS ix_events_status ON events(status);

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

-- Derived view (folded from events by rebuild_projection(), maintained
-- incrementally by record()) — but the TABLE must exist for either writer.
-- Included here so migrate_phase1()'s fresh-db branch lays down a complete,
-- immediately-usable schema (W2 checkpoint review finding, D-Log-12: the
-- fresh branch previously created events/events_trace only, and the first
-- record() against a fresh db crashed on the missing table).
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

# events_legacy event_type values that represent a stage disposition (as opposed
# to trace-only start/warn/namespaced sub-events, which stay in events_legacy only).
_LEGACY_DISPOSITION_TYPES = ("ok", "skip", "empty", "error", "partial")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write obj as JSON via tmp+rename. Unique tmp name (no cross-writer collision).
    Never produces a zero-byte file (json.dumps of a dict is always non-empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=1)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _write_event(media_id: str, stage: str, status: str, reason: str | None,
                  metrics: dict | None, health: dict | None, inputs: dict | None,
                  artifacts: list | None, producer: str | None, config_fp: str | None,
                  run_id: str, at_utc: str, db: Path) -> None:
    """The one write to the outcome surface: `events` + `status_projection`, same
    connection/transaction. P4 — no silent swallow: a DB write failure raises. An
    unloggable stage outcome IS a failed stage (docs/OBSERVABILITY_REDESIGN.md §2)."""
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=TRUNCATE")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            "INSERT INTO events (media_id, stage, status, reason, run_id, producer_sha, "
            "config_fp, metrics, health, inputs, artifacts, at_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (media_id, stage, status, reason, run_id, producer, config_fp,
             json.dumps(metrics if metrics is not None else {}),
             json.dumps(health) if health is not None else None,
             json.dumps(inputs) if inputs is not None else None,
             json.dumps(artifacts if artifacts is not None else []),
             at_utc),
        )
        # status_projection: ok=1 means the stage reached a definite conclusion.
        ok = 1 if status in COMPLETED else 0
        err = reason[:1000] if (status == "error" and reason) else None
        conn.execute(
            "INSERT OR REPLACE INTO status_projection (media_id, stage, ok, run_id, at_utc, error) "
            "VALUES (?,?,?,?,?,?)",
            (media_id, stage, ok, run_id, at_utc, err),
        )
        conn.commit()
    finally:
        conn.close()


def trace(media_id: str, stage: str, event_type: str, payload: dict | None = None, *,
          run_id: str | None = None, job_id: str | None = None, db: Path = DB) -> str:
    """Operational trace: start / warn / sub-events / namespaced diagnostics.
    Writes `events_trace` ONLY — never the outcome surface (`events` /
    `status_projection`). Best-effort (same contract as the pre-Phase-1 writers
    it replaces): a lost trace breadcrumb must never break a stage. Returns
    run_id (generated if not given), matching the old emit_start/log_event
    return-a-run_id convenience."""
    run_id = run_id or f"{stage}-{uuid.uuid4().hex[:8]}"
    job_id = job_id or f"{media_id}:{stage}"
    try:
        conn = sqlite3.connect(str(db), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=TRUNCATE")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (_now(), job_id, media_id, stage, event_type, run_id,
                 json.dumps(payload) if payload is not None else None),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # trace writes are best-effort by design
        _log.warning("trace write failed for %s/%s/%s: %s", media_id, stage, event_type, e)
    return run_id


def _merge_status_file(media_id: str, mutate) -> None:
    path = STATUS_DIR / f"{media_id}.json"
    with _file_lock:
        doc: dict = {}
        if path.exists():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                _log.warning("corrupt status file %s, rewriting: %s", path, e)
                doc = {}
        doc["schema_version"] = "1.0.0"
        doc["media_id"] = media_id
        mutate(doc)
        _atomic_write_json(path, doc)


# ── write API ─────────────────────────────────────────────────────────────────

def record(media_id: str, stage: str, status: str, *, reason: str | None = None,
           metrics: dict | None = None, counts: dict | None = None,
           health: dict | None = None, inputs: dict | None = None,
           artifacts: list | None = None, producer: str | None = None,
           config_fp: str | None = None, run_id: str | None = None, db: Path = DB) -> None:
    """Record an explicit (media, stage) outcome — the ONE outcome writer
    (D-Log-04 §3.2). `reason` is REQUIRED for skip/empty/error/degraded/stale
    (raises ValueError). `counts` is a backward-compat alias for `metrics`
    (existing ~50 call sites are unchanged); `metrics` wins if both given.

    P4: no silent swallow — a DB write failure raises. The pipeline must treat
    an unloggable stage as a failed stage, not a silently-green one.

    Phase 3 (D-Log-04 §3.2/§4, docs/OBSERVABILITY_REDESIGN.md): `record()`
    DERIVES `degraded` — if `status="ok"` and `health` contains any predicate
    with `verdict="fail"` (the §3.1 shape, e.g. from `stages.evaluate_health()`),
    the row is written as `status="degraded"` with `reason` auto-set to the
    failing predicate name(s). This only fires for `status="ok"`; a caller MAY
    also pass `status="degraded"` explicitly (its own `reason` is then required
    and kept, as normal — no derivation runs)."""
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status {status!r}; expected one of {sorted(VALID_STATUS)}")
    if status == "ok" and health:
        failing = sorted(
            name for name, v in health.items()
            if isinstance(v, dict) and v.get("verdict") == "fail"
        )
        if failing:
            status = "degraded"
            reason = "health predicate failed: " + ", ".join(failing)
    if status in _REASON_REQUIRED and not reason:
        raise ValueError(f"status={status!r} requires a reason")
    metrics = metrics if metrics is not None else counts
    run_id = run_id or f"{stage}-{uuid.uuid4().hex[:8]}"
    at_utc = _now()
    _write_event(media_id, stage, status, reason, metrics, health, inputs,
                 artifacts, producer, config_fp, run_id, at_utc, db)
    rec = {
        "status": status,
        "reason": reason,
        # "counts" kept verbatim for the ~50 existing catalog/outcomes readers;
        # "metrics" added as the forward-looking alias (§3.1 field name) — same
        # value, no reader relies on it yet, but it lets Phase 3+ read either.
        "counts": metrics or {},
        "metrics": metrics or {},
        "artifacts": artifacts or [],
        "at_utc": at_utc,
        "run_id": run_id,
        "producer": producer,
    }
    _merge_status_file(media_id, lambda d: d.setdefault("stages", {}).__setitem__(stage, rec))


def set_flag(media_id: str, name: str, value, *, reason: str | None = None) -> None:
    """Set an explicit media-level label (e.g. has_speech=False) in the status file."""
    entry = {"value": value, "reason": reason, "at_utc": _now()}
    _merge_status_file(media_id, lambda d: d.setdefault("flags", {}).__setitem__(name, entry))


# ── derived views (D-Log-04 §3.4: recomputed from `events`, never asserted) ────

def rebuild_projection(db: Path = DB) -> int:
    """Recompute status_projection wholesale from `events` (latest row per
    (media_id, stage), by id — the append-only insertion order). `record()`
    already maintains the projection incrementally per-row for cheap read-after-
    write; this is the AUTHORITY fold, and the only thing that may legitimately
    rewrite status_projection (reconcile_status.py calls this before comparing).
    Returns the number of (media,stage) rows in the rebuilt projection."""
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=TRUNCATE")
        conn.execute("PRAGMA busy_timeout=30000")
        rows = conn.execute(
            "SELECT e.media_id, e.stage, e.status, e.run_id, e.at_utc, e.reason "
            "FROM events e "
            "JOIN (SELECT media_id, stage, MAX(id) AS mid FROM events GROUP BY media_id, stage) latest "
            "ON e.media_id = latest.media_id AND e.stage = latest.stage AND e.id = latest.mid"
        ).fetchall()
        conn.execute("DELETE FROM status_projection")
        for media_id, stage, status, run_id, at_utc, reason in rows:
            ok = 1 if status in COMPLETED else 0
            err = reason[:1000] if (status == "error" and reason) else None
            conn.execute(
                "INSERT INTO status_projection (media_id, stage, ok, run_id, at_utc, error) "
                "VALUES (?,?,?,?,?,?)",
                (media_id, stage, ok, run_id, at_utc, err),
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def rebuild_status_files(db: Path = DB, catalog_status_dir: Path | None = None) -> int:
    """Derive catalog/status/{media_id}.json for every real media_id, from
    status_projection (itself derived from `events`). Replaces the old
    `pipeline_status.py --project` writer, which asserted the same surface
    independently (root causes #2/#3 in the redesign doc). Real media only
    (m_* excluding m_unknown* synthetic pre-id records), matching the old writer.
    Returns the number of media files (re)written."""
    catalog_status_dir = catalog_status_dir or (CAT / "status")
    rebuild_projection(db)
    from . import journal as _journal_mod  # local import: avoid a module-level cycle
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        rows = conn.execute("SELECT DISTINCT media_id FROM status_projection").fetchall()
    finally:
        conn.close()
    n = 0
    for (media_id,) in rows:
        if not media_id.startswith("m_") or media_id.startswith("m_unknown"):
            continue
        _journal_mod.rebuild_status(db, media_id, catalog_status_dir)
        n += 1
    return n


def _extract_legacy_fields(payload_str: str | None, event_type: str):
    """Best-effort extraction from a pre-Phase-1 events.payload JSON blob. Tolerant
    of the several historical payload shapes (outcomes.record's {status,reason,
    counts,artifacts,...}, StageManifest.to_done_entry's {counts,error,outputs},
    journal.emit_error's {"error":...}, e1_signals' {"reason":...}); fields not
    present in a given shape are left None — "where present", not invented."""
    reason = metrics = artifacts = producer = None
    if payload_str:
        try:
            payload = json.loads(payload_str)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            reason = payload.get("reason")
            if reason is None and event_type == "error":
                reason = payload.get("error")
            m = payload.get("counts")
            if isinstance(m, dict):
                metrics = m
            a = payload.get("artifacts")
            if not isinstance(a, list):
                a = payload.get("outputs")
            if isinstance(a, list):
                artifacts = a
            producer = payload.get("producer")
    return reason, metrics, artifacts, producer


def migrate_phase1(db: Path = DB) -> dict:
    """Idempotent Phase 1 migration (D-Log-04 §4). On a pre-redesign journal.db:
    renames `events` (event_type/payload shape) -> `events_legacy` (frozen,
    verbatim, never written to again), lays down the new `events` (§3.1) and
    `events_trace` tables, backfills `events` from every events_legacy row whose
    event_type is a disposition (ok/skip/empty/error/partial — start/warn/
    namespaced sub-events stay in events_legacy only), then folds
    status_projection from the new events. No-ops if already migrated (detected
    via `events` column shape). Returns a stats dict."""
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=TRUNCATE")
        conn.execute("PRAGMA busy_timeout=30000")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
        if "status" in cols and "event_type" not in cols:
            conn.executescript(_NEW_EVENTS_DDL)  # in case events_trace is still missing
            conn.commit()
            return {"already_migrated": True, "legacy_rows": None, "backfilled_rows": None}
        if not cols:
            # No events table at all (brand-new db) — just lay down the new schema.
            conn.executescript(_NEW_EVENTS_DDL)
            conn.commit()
            return {"already_migrated": False, "fresh": True, "legacy_rows": 0, "backfilled_rows": 0}
        if "event_type" not in cols:
            raise RuntimeError(f"unrecognised events table shape: columns={sorted(cols)}")

        conn.execute("ALTER TABLE events RENAME TO events_legacy")
        conn.executescript(_NEW_EVENTS_DDL)

        legacy_rows = conn.execute("SELECT COUNT(*) FROM events_legacy").fetchone()[0]
        qmarks = ",".join("?" * len(_LEGACY_DISPOSITION_TYPES))
        cur = conn.execute(
            f"SELECT media_id, stage, event_type, run_id, payload, at_utc "
            f"FROM events_legacy WHERE event_type IN ({qmarks}) ORDER BY id",
            _LEGACY_DISPOSITION_TYPES,
        )
        insert_sql = (
            "INSERT INTO events (media_id, stage, status, reason, run_id, producer_sha, "
            "config_fp, metrics, health, inputs, artifacts, at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        batch = []
        for media_id, stage, event_type, run_id, payload, at_utc in cur:
            reason, metrics, artifacts, producer = _extract_legacy_fields(payload, event_type)
            batch.append((
                media_id, stage, event_type, reason, run_id, producer, None,
                json.dumps(metrics or {}), None, None, json.dumps(artifacts or []), at_utc,
            ))
        conn.executemany(insert_sql, batch)
        conn.commit()
    finally:
        conn.close()
    n_proj = rebuild_projection(db)
    return {
        "already_migrated": False,
        "legacy_rows": legacy_rows,
        "backfilled_rows": len(batch),
        "projection_rows": n_proj,
    }


# ── query API (any stage can consult any other) ───────────────────────────────

def for_media(media_id: str) -> dict:
    path = STATUS_DIR / f"{media_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get(media_id: str, stage: str) -> dict | None:
    return for_media(media_id).get("stages", {}).get(stage)


def status_of(media_id: str, stage: str) -> str | None:
    o = get(media_id, stage)
    return o.get("status") if o else None


def is_complete(media_id: str, stage: str) -> bool:
    """Explicit completion check — reads the recorded outcome, never the filesystem."""
    return status_of(media_id, stage) in COMPLETED


def flag(media_id: str, name: str):
    return for_media(media_id).get("flags", {}).get(name, {}).get("value")


def _scan(predicate) -> list[str]:
    out: list[str] = []
    if not STATUS_DIR.exists():
        return out
    for f in sorted(STATUS_DIR.glob("m_*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = doc.get("media_id")
        if mid and predicate(doc):
            out.append(mid)
    return sorted(out)


def media_by_status(stage: str, status: str) -> list[str]:
    return _scan(lambda d: d.get("stages", {}).get(stage, {}).get("status") == status)


def media_by_flag(name: str, value=True) -> list[str]:
    return _scan(lambda d: d.get("flags", {}).get(name, {}).get("value") == value)

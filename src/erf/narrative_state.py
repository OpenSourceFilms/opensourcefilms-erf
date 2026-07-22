"""Narrative pipeline state machine — §5 orchestrator backing store.

Lifecycle per thread:
  pending → pass1_running → pass1_done → pass2_running → pass2_done
         → gate_running → gate_passed                         (terminal ✓)
                       → gate_failed → pass2_running (regen)
                                    → discarded               (terminal ✗)
  pass1_running → pass1_failed                                (terminal ✗)
  pass2_running → pass2_failed                                (terminal ✗)

Persistence: state/narrative_pipeline.db (WAL, isolated from journal.db).
All multi-worker state claims use BEGIN IMMEDIATE transactions.
Crash recovery: artifact-existence check on startup for in-flight states.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DDL = """
CREATE TABLE IF NOT EXISTS narr_pipeline_state (
    thread_id       TEXT PRIMARY KEY,
    state           TEXT NOT NULL,
    pass1_attempt   INTEGER NOT NULL DEFAULT 0,
    pass2_attempt   INTEGER NOT NULL DEFAULT 0,
    brief_path      TEXT,
    bp_id           TEXT,
    gate_id         TEXT,
    failure_reasons TEXT,
    error_info      TEXT,
    worker_pid      INTEGER,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nps_state ON narr_pipeline_state(state);
"""

TERMINAL_STATES = frozenset(["gate_passed", "pass1_failed", "pass2_failed", "discarded"])
IN_FLIGHT_STATES = frozenset(["pass1_running", "pass2_running", "gate_running"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    conn.executescript(_DDL)
    conn.commit()
    conn.close()


def seed_pending(db_path: Path, thread_ids: list[str], *, force: bool = False) -> int:
    """Insert threads as 'pending' if not already present (unless force resets them).

    force=True resets all listed threads to pending regardless of current state.
    Returns count of newly inserted/reset rows.
    """
    conn = _connect(db_path)
    now = _now()
    inserted = 0
    try:
        for tid in thread_ids:
            if force:
                conn.execute(
                    "INSERT OR REPLACE INTO narr_pipeline_state "
                    "(thread_id, state, created_at, updated_at) VALUES (?,?,?,?)",
                    (tid, "pending", now, now),
                )
                inserted += 1
            else:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO narr_pipeline_state "
                    "(thread_id, state, created_at, updated_at) VALUES (?,?,?,?)",
                    (tid, "pending", now, now),
                )
                inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


def claim(
    db_path: Path,
    thread_id: str,
    from_state: str,
    to_state: str,
    **fields,
) -> bool:
    """Atomically transition thread_id from from_state → to_state.

    Uses BEGIN IMMEDIATE so only one worker claims each thread.
    Returns True on success, False if another worker already claimed it.
    """
    field_sets: dict = {"state": to_state, "updated_at": _now(), "worker_pid": os.getpid()}
    field_sets.update(fields)
    set_clause = ", ".join(f"{k}=?" for k in field_sets)
    values = list(field_sets.values()) + [thread_id, from_state]

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            f"UPDATE narr_pipeline_state SET {set_clause} "
            f"WHERE thread_id=? AND state=?",
            values,
        )
        changed = cur.rowcount == 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return changed


def transition(db_path: Path, thread_id: str, new_state: str, **fields) -> None:
    """Update state without a from_state guard (for post-claim completions)."""
    field_sets: dict = {"state": new_state, "updated_at": _now(), "worker_pid": os.getpid()}
    field_sets.update(fields)
    set_clause = ", ".join(f"{k}=?" for k in field_sets)
    values = list(field_sets.values()) + [thread_id]

    conn = _connect(db_path)
    try:
        conn.execute(
            f"UPDATE narr_pipeline_state SET {set_clause} WHERE thread_id=?",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def get_thread_state(db_path: Path, thread_id: str) -> dict | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM narr_pipeline_state WHERE thread_id=?", (thread_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_by_state(db_path: Path, state: str) -> list[dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM narr_pipeline_state WHERE state=? ORDER BY updated_at",
            (state,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_summary(db_path: Path) -> dict[str, int]:
    """Returns {state: count} across all threads."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT state, COUNT(*) as n FROM narr_pipeline_state GROUP BY state"
        ).fetchall()
        return {r["state"]: r["n"] for r in rows}
    finally:
        conn.close()


def dispatch_regen(db_path: Path, thread_id: str) -> bool:
    """gate_failed → pass2_running (regen), guarded by budget check.

    Does NOT increment pass2_attempt here — that happens only on terminal
    gate2 outcome (see commit_regen_outcome). This ensures a crash before
    gate2 does not consume the regen budget.

    Returns True if regen was dispatched, False if budget is exhausted
    (caller should then discard).
    """
    conn = _connect(db_path)
    now = _now()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT state, pass2_attempt FROM narr_pipeline_state WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        if not row or row["state"] != "gate_failed":
            conn.rollback()
            return False
        if row["pass2_attempt"] >= 1:
            # Budget exhausted — caller must discard
            conn.rollback()
            return False
        # Transition to pass2_running WITHOUT incrementing pass2_attempt
        conn.execute(
            "UPDATE narr_pipeline_state SET state='pass2_running', "
            "updated_at=?, worker_pid=? WHERE thread_id=?",
            (now, os.getpid(), thread_id),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def commit_regen_outcome(db_path: Path, thread_id: str, passed: bool, **fields) -> None:
    """Record gate2 terminal outcome and consume the regen attempt budget.

    Called after gate2 resolves — increments pass2_attempt to 1 and sets
    the final state (gate_passed or discarded).
    """
    new_state = "gate_passed" if passed else "discarded"
    field_sets: dict = {
        "state": new_state,
        "pass2_attempt": 1,
        "updated_at": _now(),
        "worker_pid": os.getpid(),
    }
    field_sets.update(fields)
    set_clause = ", ".join(f"{k}=?" for k in field_sets)
    values = list(field_sets.values()) + [thread_id]

    conn = _connect(db_path)
    try:
        conn.execute(
            f"UPDATE narr_pipeline_state SET {set_clause} WHERE thread_id=?",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def bootstrap_from_artifacts(db_path: Path, catalog: Path) -> dict[str, int]:
    """Pre-populate state for threads whose artifacts already exist on disk.

    Used when Pass 1/2 and §4 gate ran OUTSIDE the orchestrator (e.g. earlier
    batch scripts). For each thread currently in 'pending' state, advances it
    to the furthest stage whose artifact exists.

    Priority: gate result > pass2_done > pass1_done > pending (unchanged).
    Also reads the latest gate run output to pick up PASS/FAIL verdicts.

    Returns {new_state: count} for logging.
    """
    # Load the latest gate results (if any)
    gate_results: dict[str, str] = {}  # thread_id → verdict
    gate_dir = catalog / "narrative_gate"
    latest_path = gate_dir / "latest.json"
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            gate_id = latest.get("gate_id")
            if gate_id:
                gate_file = gate_dir / f"{gate_id}.json"
                if gate_file.exists():
                    gate_record = json.loads(gate_file.read_text(encoding="utf-8"))
                    for r in gate_record.get("results", []):
                        gate_results[r["thread_id"]] = r["verdict"]
        except Exception:
            pass

    # Load beat plan index
    index_path = catalog / "beat_plans" / "_thread_index.json"
    thread_index: dict = {}
    if index_path.exists():
        try:
            thread_index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    conn = _connect(db_path)
    now = _now()
    counts: dict[str, int] = {}
    try:
        rows = conn.execute(
            "SELECT thread_id FROM narr_pipeline_state WHERE state='pending'"
        ).fetchall()
        for row in rows:
            tid = row["thread_id"]
            brief_path = catalog / "narrative_briefs" / f"{tid}.json"
            bp_id = thread_index.get(tid)
            has_brief = brief_path.exists()
            has_bp = bp_id and (catalog / "beat_plans" / f"{bp_id}.jsonl").exists()

            if has_bp and tid in gate_results:
                verdict = gate_results[tid]
                new_state = "gate_passed" if verdict == "PASS" else "gate_failed"
                conn.execute(
                    "UPDATE narr_pipeline_state SET state=?, bp_id=?, brief_path=?, "
                    "updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (new_state, bp_id, str(brief_path), now, tid),
                )
                counts[new_state] = counts.get(new_state, 0) + 1
            elif has_bp:
                conn.execute(
                    "UPDATE narr_pipeline_state SET state='pass2_done', bp_id=?, "
                    "brief_path=?, updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (bp_id, str(brief_path), now, tid),
                )
                counts["pass2_done"] = counts.get("pass2_done", 0) + 1
            elif has_brief:
                conn.execute(
                    "UPDATE narr_pipeline_state SET state='pass1_done', "
                    "brief_path=?, updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (str(brief_path), now, tid),
                )
                counts["pass1_done"] = counts.get("pass1_done", 0) + 1

        conn.commit()
    finally:
        conn.close()
    return counts


def recover_inflight(db_path: Path, catalog: Path) -> dict[str, int]:
    """On process restart: check artifacts for in-flight states and recover.

    Rule: an in-flight state is resolved by checking whether its side-effect
    artifact already exists. If it does, advance the state. If not, rewind to
    the previous stable state so the worker re-dispatches.

    Returns {original_state: count_recovered} for logging.
    """
    conn = _connect(db_path)
    now = _now()
    counts: dict[str, int] = {}
    try:
        # --- pass1_running ---
        rows = conn.execute(
            "SELECT thread_id FROM narr_pipeline_state WHERE state='pass1_running'"
        ).fetchall()
        for row in rows:
            tid = row["thread_id"]
            brief_path = catalog / "narrative_briefs" / f"{tid}.json"
            if brief_path.exists():
                conn.execute(
                    "UPDATE narr_pipeline_state SET state='pass1_done', "
                    "brief_path=?, updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (str(brief_path), now, tid),
                )
            else:
                conn.execute(
                    "UPDATE narr_pipeline_state SET state='pending', "
                    "updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (now, tid),
                )
        counts["pass1_running"] = len(rows)

        # --- pass2_running ---
        index_path = catalog / "beat_plans" / "_thread_index.json"
        thread_index: dict = {}
        if index_path.exists():
            try:
                thread_index = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        rows = conn.execute(
            "SELECT thread_id, bp_id, failure_reasons FROM narr_pipeline_state "
            "WHERE state='pass2_running'"
        ).fetchall()
        for row in rows:
            tid = row["thread_id"]
            state_bp_id = row["bp_id"]         # bp_id from last completed Pass 2
            is_regen = row["failure_reasons"] is not None  # set only after gate_failed

            index_bp_id = thread_index.get(tid)
            new_bp_exists = index_bp_id and (catalog / "beat_plans" / f"{index_bp_id}.jsonl").exists()

            if new_bp_exists and index_bp_id != state_bp_id:
                # A new bp was written (initial or regen completed) — advance to pass2_done
                conn.execute(
                    "UPDATE narr_pipeline_state SET state='pass2_done', "
                    "bp_id=?, updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (index_bp_id, now, tid),
                )
            elif is_regen:
                # Regen crashed before writing new bp — rewind to gate_failed.
                # Budget NOT consumed (pass2_attempt stays 0) so regen can retry.
                conn.execute(
                    "UPDATE narr_pipeline_state SET state='gate_failed', "
                    "updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (now, tid),
                )
            else:
                # Initial Pass 2 crashed before writing — rewind to pass1_done
                conn.execute(
                    "UPDATE narr_pipeline_state SET state='pass1_done', "
                    "updated_at=?, worker_pid=NULL WHERE thread_id=?",
                    (now, tid),
                )
        counts["pass2_running"] = len(rows)

        # --- gate_running --- (gate is fast; just rewind to pass2_done)
        rows = conn.execute(
            "SELECT thread_id FROM narr_pipeline_state WHERE state='gate_running'"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE narr_pipeline_state SET state='pass2_done', "
                "updated_at=?, worker_pid=NULL WHERE thread_id=?",
                (now, row["thread_id"]),
            )
        counts["gate_running"] = len(rows)

        # --- pass2_failed during regen ---
        # If failure_reasons is set, this pass2_failed occurred during a regen
        # attempt (Pass 2 itself crashed). Budget was NOT consumed (pass2_attempt=0),
        # so rewind to gate_failed so the regen can be retried.
        # pass2_failed with failure_reasons=NULL is a true terminal failure (initial Pass 2).
        rows = conn.execute(
            "SELECT thread_id FROM narr_pipeline_state "
            "WHERE state='pass2_failed' AND failure_reasons IS NOT NULL AND pass2_attempt=0"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE narr_pipeline_state SET state='gate_failed', "
                "updated_at=?, worker_pid=NULL WHERE thread_id=?",
                (now, row["thread_id"]),
            )
        counts["pass2_failed_regen"] = len(rows)

        conn.commit()
    finally:
        conn.close()
    return counts

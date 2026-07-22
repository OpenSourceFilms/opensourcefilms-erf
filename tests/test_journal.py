"""Journal round-trip tests: emit start+ok, reap, confirm status projection."""
import sqlite3
import pytest
from pathlib import Path
from erf.journal import init_journal, emit_start, emit_ok, emit_error, reap_manifest, rebuild_status
from erf.manifest import StageManifest


@pytest.fixture
def tmp_state(tmp_path):
    db = tmp_path / "journal.db"
    init_journal(db)
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    return db, status_dir


MID = "m_9f3a1c7be002"


def _manifest(ok=True, stage="A", error=None):
    return StageManifest(
        media_id=MID,
        stage=stage,
        run_id="run_test_001",
        ok=ok,
        at_utc="2026-06-01T10:05:00+00:00",
        tool_version="faster-whisper-1.2.1",
        counts={"words": 1842, "segments": 63},
        error=error,
        outputs=[f"/workspace/erf/catalog/asr/{MID}.words.jsonl"],
    )


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_creates_tables(tmp_state):
    db, _ = tmp_state
    con = sqlite3.connect(str(db))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close()
    assert "events" in tables
    assert "status_projection" in tables


# ── emit_start ────────────────────────────────────────────────────────────────
# Phase 1 (D-Log-04 P1): emit_start is now trace-only (erf.outcomes.trace()) —
# it writes events_trace, not the outcome-only `events` table, and never
# touches status_projection (a stage that has merely started has no disposition
# yet, so there is nothing for the fold to reflect — absence, not a NULL-ok row,
# is how "not yet concluded" is represented now).

def test_emit_start_inserts_trace_event(tmp_state):
    db, _ = tmp_state
    emit_start(db, MID, "A", "run_001")
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT * FROM events_trace WHERE media_id=? AND stage='A'", (MID,)).fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][5] == "start"  # event_type column (events_trace shape: id,at_utc,job_id,media_id,stage,event_type,run_id,payload)
    # and NOT in the outcome-only events table
    con = sqlite3.connect(str(db))
    outcome_rows = con.execute("SELECT * FROM events WHERE media_id=? AND stage='A'", (MID,)).fetchall()
    con.close()
    assert outcome_rows == []


def test_emit_start_does_not_touch_projection(tmp_state):
    db, _ = tmp_state
    emit_start(db, MID, "A", "run_001")
    con = sqlite3.connect(str(db))
    row = con.execute("SELECT ok FROM status_projection WHERE media_id=? AND stage='A'", (MID,)).fetchone()
    con.close()
    assert row is None  # no disposition recorded yet -> no projection row at all


# ── emit_ok ───────────────────────────────────────────────────────────────────

def test_emit_ok_sets_projection(tmp_state):
    db, _ = tmp_state
    m = _manifest(ok=True)
    emit_start(db, MID, "A", m.run_id)
    emit_ok(db, m)
    con = sqlite3.connect(str(db))
    row = con.execute("SELECT ok FROM status_projection WHERE media_id=? AND stage='A'", (MID,)).fetchone()
    con.close()
    assert row[0] == 1


def test_emit_error_sets_projection(tmp_state):
    db, _ = tmp_state
    emit_start(db, MID, "A", "run_fail")
    emit_error(db, MID, "A", "run_fail", "ffprobe returned non-zero")
    con = sqlite3.connect(str(db))
    row = con.execute("SELECT ok, error FROM status_projection WHERE media_id=? AND stage='A'", (MID,)).fetchone()
    con.close()
    assert row[0] == 0
    assert "ffprobe" in row[1]


# ── reap_manifest ─────────────────────────────────────────────────────────────

def test_reap_ok_writes_status_file(tmp_state):
    db, status_dir = tmp_state
    m = _manifest(ok=True)
    reap_manifest(db, m, status_dir)
    out = status_dir / f"{MID}.json"
    assert out.exists()
    import json
    status = json.loads(out.read_text())
    assert status["schema_version"] == "1.0.0"
    assert status["media_id"] == MID
    assert "A" in status["stages"]
    assert status["stages"]["A"]["ok"] is True
    assert status["stages"]["A"]["run_id"] == "run_test_001"


def test_reap_error_writes_status_file(tmp_state):
    db, status_dir = tmp_state
    m = _manifest(ok=False, error="GPU OOM")
    reap_manifest(db, m, status_dir)
    import json
    status = json.loads((status_dir / f"{MID}.json").read_text())
    assert status["stages"]["A"]["ok"] is False
    assert status["stages"]["A"]["error"] == "GPU OOM"


# ── rebuild_status ────────────────────────────────────────────────────────────

def test_rebuild_status_multiple_stages(tmp_state):
    db, status_dir = tmp_state
    for stage in ["A", "B0", "B1"]:
        m = _manifest(ok=True, stage=stage)
        reap_manifest(db, m, status_dir)

    status = rebuild_status(db, MID, status_dir)
    assert set(status["stages"].keys()) == {"A", "B0", "B1"}
    assert all(s["ok"] is True for s in status["stages"].values())


def test_rebuild_status_manifest_extra_overrides(tmp_state):
    db, status_dir = tmp_state
    m = _manifest(ok=True, stage="A")
    reap_manifest(db, m, status_dir)

    m2 = StageManifest(
        media_id=MID, stage="A", run_id="run_002", ok=True,
        at_utc="2026-06-01T11:00:00+00:00",
        tool_version="faster-whisper-1.2.2",
        counts={"words": 9999},
        error=None, outputs=[],
    )
    status = rebuild_status(db, MID, status_dir, manifest_extra=m2)
    assert status["stages"]["A"]["tool_version"] == "faster-whisper-1.2.2"
    assert status["stages"]["A"]["counts"]["words"] == 9999

"""D-Log-20 — the read side of the `inputs=` fingerprint groundwork.

MASTER §5.2 Phase 3 (D-Log-04 §4) had SEV2/T0 populate `inputs=
{"identity_map": <fp>}` on every record() call, but its own docstring
(`stages.py`) says plainly: "no staleness-marking logic reads `inputs` yet."
`bin/reconcile_status.py`'s `_current_fp()`/`_stale_media()` (added this
session) are that reader, wired into `verify()`'s per-stage audit alongside
the existing drift/degraded columns.

Decisive proof (mirrors the Phase 5 degraded-proof pattern): a media whose
recorded `inputs` fingerprint matches its upstream artifact's CURRENT content
must not be flagged; the moment that upstream artifact changes without the
consuming stage re-running, it must flip to stale. Both directions matter
(missing -> present, present -> changed, present -> missing) since the write
side omits the key entirely rather than writing a placeholder for an absent
file (see `_identity_map_fp`'s docstring in `build_speech_events_v2.py`).
"""
from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from erf import journal, outcomes

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load_bin_module("rs_staleness", "reconcile_status.py")


def _fp(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


@pytest.fixture
def tmp_journal(tmp_path):
    db = tmp_path / "journal.db"
    journal.init_journal(db)
    return db


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """Same isolation discipline as test_degraded_proof.py: reconcile_status's
    ROOT is a hardcoded module global that `_current_fp`/`_media_ids` resolve
    artifact paths against — monkeypatch it so this file never touches the
    real corpus, even transiently. outcomes.STATUS_DIR is a SEPARATE module
    global (erf.outcomes, not reconcile_status) that `outcomes.record()`'s
    `_merge_status_file` writes to regardless of the `db=` kwarg passed below
    — patch it too, or every outcomes.record() call in this file writes a
    real catalog/outcomes/{mid}.json into production (found in F3 hygiene
    pass, D-Log-36)."""
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    monkeypatch.setattr(outcomes, "STATUS_DIR", tmp_path / "catalog" / "outcomes")
    return tmp_path


# ════════════════════════════════════════════════════════════════════════════
# _current_fp — the raw fingerprint recompute
# ════════════════════════════════════════════════════════════════════════════

def test_current_fp_matches_write_side_algorithm(isolated_root):
    """Must be bit-identical to build_speech_events_v2._identity_map_fp /
    build_transcript._identity_map_fp (sha256(content).hexdigest()[:16]) or
    every real recorded fingerprint would spuriously read as stale."""
    mid = "m_test_fp"
    content = b'{"media_id": "m_test_fp", "speakers": {}}'
    p = isolated_root / "catalog" / "identity_map" / f"{mid}.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(content)

    assert rs._current_fp(mid, "catalog/identity_map/{mid}.json") == _fp(content)


def test_current_fp_absent_file_is_none(isolated_root):
    """Matches the write side's convention: an absent upstream artifact omits
    the key entirely rather than fingerprinting nothing/empty."""
    assert rs._current_fp("m_does_not_exist", "catalog/identity_map/{mid}.json") is None


# ════════════════════════════════════════════════════════════════════════════
# _stale_media — the decisive proof
# ════════════════════════════════════════════════════════════════════════════

def test_stale_media_no_fingerprint_inputs_returns_empty_without_querying(isolated_root, tmp_journal):
    """Stage A declares no fingerprint_inputs (Phase 0 default) — must short-
    circuit to empty rather than querying events for a stage that never wrote
    `inputs` at all."""
    conn = sqlite3.connect(str(tmp_journal))
    assert rs._stale_media(conn, "A", {"m_1", "m_2"}) == set()
    conn.close()


def test_stale_media_matching_fingerprint_is_not_stale(isolated_root, tmp_journal):
    mid = "m_test_fresh"
    content = b'{"speakers": {"SPEAKER_00": {"person_id": "p_x", "source": "gallery"}}}'
    p = isolated_root / "catalog" / "identity_map" / f"{mid}.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(content)

    outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": 1.0},
                     inputs={"identity_map": _fp(content)}, db=tmp_journal)

    conn = sqlite3.connect(str(tmp_journal))
    assert rs._stale_media(conn, "SEV2", {mid}) == set()
    conn.close()


def test_stale_media_changed_upstream_flips_to_stale(isolated_root, tmp_journal):
    """The decisive case: SEV2 ran and fingerprinted identity_map v1; v1 is
    since replaced by v2 (a re-resolution) without SEV2 re-running. The
    recorded outcome and disk still fully agree (SEV2's own artifact is
    untouched) — this is exactly the blind spot `inputs=` was built to close."""
    mid = "m_test_stale"
    v1 = b'{"speakers": {"SPEAKER_00": {"person_id": "p_x", "gallery_score": 0.5}}}'
    p = isolated_root / "catalog" / "identity_map" / f"{mid}.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(v1)

    outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": 1.0},
                     inputs={"identity_map": _fp(v1)}, db=tmp_journal)

    conn = sqlite3.connect(str(tmp_journal))
    assert rs._stale_media(conn, "SEV2", {mid}) == set()  # sanity: not stale yet

    v2 = b'{"speakers": {"SPEAKER_00": {"person_id": "p_x", "gallery_score": 0.9}}}'
    p.write_bytes(v2)
    assert rs._stale_media(conn, "SEV2", {mid}) == {mid}  # NOW stale
    conn.close()


def test_stale_media_upstream_appeared_since_is_stale(isolated_root, tmp_journal):
    """Write-side convention: identity_map absent when SEV2 ran -> inputs={}
    (key omitted, not null). If identity_map now EXISTS, that's new upstream
    data this run never consumed — must be stale, not silently treated as a
    match just because the recorded dict has no key to compare."""
    mid = "m_test_appeared"
    outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": 0.0},
                     inputs={}, db=tmp_journal)

    conn = sqlite3.connect(str(tmp_journal))
    assert rs._stale_media(conn, "SEV2", {mid}) == set()  # still absent -> not stale

    p = isolated_root / "catalog" / "identity_map" / f"{mid}.json"
    p.parent.mkdir(parents=True)
    p.write_bytes(b'{"speakers": {}}')
    assert rs._stale_media(conn, "SEV2", {mid}) == {mid}  # appeared -> stale
    conn.close()


def test_stale_media_not_a_candidate_is_never_checked(isolated_root, tmp_journal):
    """candidates is the caller's (verify()'s) responsibility to scope to
    journal-disk-agreeing media; _stale_media must not silently expand beyond
    it even if other media have events rows."""
    mid_a, mid_b = "m_in", "m_out"
    for mid in (mid_a, mid_b):
        content = f'{{"m": "{mid}"}}'.encode()
        p = isolated_root / "catalog" / "identity_map" / f"{mid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": 1.0},
                         inputs={"identity_map": "deadbeefdeadbeef"}, db=tmp_journal)  # wrong on purpose

    conn = sqlite3.connect(str(tmp_journal))
    # both would report stale (fingerprint deliberately wrong) if checked --
    # only mid_a is in candidates, so only it may appear.
    result = rs._stale_media(conn, "SEV2", {mid_a})
    assert result == {mid_a}
    assert mid_b not in result
    conn.close()

"""W3a new-pod resume checklist item 4 (GAMEPLAN.md, 2026-07-12): tags.py,
faces.py (F0), and bin/rediarize.py (REDIARIZE) each had the same drift-308
phantom-outcome bug — recording a bare "ok" when the stage ran successfully
but produced no artifact (no chunks / no faces / no diarization turns), so a
green outcome had nothing behind it and downstream consumers (F1/identity,
CLUSTERS) would trust data that doesn't exist. Fixed by recording the honest
terminal disposition (skip/empty) with a reason, after the reap bridge's
initial "ok" write.

All tests spy on the real `erf.outcomes` module (monkeypatch — same pattern as
tests/test_phase4_isolation.py's `_rg_spy`) so nothing here touches the live
`catalog/outcomes/*.json` (STATUS_DIR is a hardcoded global outcomes.record
can't be redirected away from — see tests/test_semantic_health.py's
`real_status_cleanup` docstring) or `state/journal.db`. The journal writes
that do happen (outcomes.trace via emit_start) are already best-effort/silent
against a missing db file per outcomes.trace's own docstring, so no journal
fixture is needed.

bin/ is not a package — rediarize.py is loaded via importlib.util, matching
tests/test_phase4_isolation.py and tests/test_semantic_health.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from erf import faces, journal, outcomes, tags

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rdz = _load_bin_module("rdz_isolation", "rediarize.py")


def _record_spy(monkeypatch):
    calls = []

    def fake_record(media_id, stage, status, **kw):
        calls.append((media_id, stage, status, kw))

    monkeypatch.setattr(outcomes, "record", fake_record)
    return calls


def _write_config(tmp_path: Path, catalog: Path, state: Path, erf_root: Path | None = None) -> Path:
    cfg_path = tmp_path / "config.toml"
    lines = ["[paths]", f'catalog = "{catalog}"', f'state = "{state}"']
    if erf_root is not None:
        lines.append(f'erf_root = "{erf_root}"')
    cfg_path.write_text("\n".join(lines), encoding="utf-8")
    return cfg_path


# ──────────────────────────────────────────────────────────────────────────
# tags — skip:no_chunks
# ──────────────────────────────────────────────────────────────────────────

def test_tags_skip_no_chunks_when_chunks_file_absent(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    state = tmp_path / "state"
    catalog.mkdir()
    state.mkdir()
    journal.init_journal(state / "journal.db")
    cfg_path = _write_config(tmp_path, catalog, state)

    calls = _record_spy(monkeypatch)
    monkeypatch.setattr(outcomes, "flag", lambda mid, name: None)  # unset — realistic default

    manifest = tags.run("m_nochunks", config_path=cfg_path)

    assert manifest.ok is True  # no exception — structural success, disposition is the override
    skip_calls = [c for c in calls if c[1] == "tags" and c[2] == "skip"]
    assert len(skip_calls) == 1
    assert skip_calls[0][3]["reason"] == "no_chunks"
    assert not (catalog / "tags" / "m_nochunks.jsonl").exists()


def test_tags_skip_no_chunks_gated_by_has_chunks_flag_false(monkeypatch, tmp_path):
    """has_chunks=False must short-circuit even if a stray chunks file exists
    (cross-stage flag takes priority over probing the file directly)."""
    catalog = tmp_path / "catalog"
    state = tmp_path / "state"
    (catalog / "chunks").mkdir(parents=True)
    state.mkdir()
    journal.init_journal(state / "journal.db")
    (catalog / "chunks" / "m_x.jsonl").write_text('{"chunk_id":"c1","text":"hi"}\n', encoding="utf-8")
    cfg_path = _write_config(tmp_path, catalog, state)

    calls = _record_spy(monkeypatch)
    monkeypatch.setattr(outcomes, "flag", lambda mid, name: False if name == "has_chunks" else None)

    tags.run("m_x", config_path=cfg_path)

    skip_calls = [c for c in calls if c[1] == "tags" and c[2] == "skip"]
    assert len(skip_calls) == 1
    assert skip_calls[0][3]["reason"] == "no_chunks"


def test_tags_still_scores_when_chunks_present(monkeypatch, tmp_path):
    """Regression guard: the gate must not swallow the real scoring path."""
    catalog = tmp_path / "catalog"
    state = tmp_path / "state"
    (catalog / "chunks").mkdir(parents=True)
    state.mkdir()
    journal.init_journal(state / "journal.db")
    (catalog / "chunks" / "m_y.jsonl").write_text(
        '{"chunk_id":"c1","text":"hello world","start_ms":0,"end_ms":2000}\n', encoding="utf-8"
    )
    cfg_path = _write_config(tmp_path, catalog, state)

    calls = _record_spy(monkeypatch)
    monkeypatch.setattr(outcomes, "flag", lambda mid, name: True if name == "has_chunks" else None)

    manifest = tags.run("m_y", config_path=cfg_path)

    assert manifest.ok is True
    assert not any(c[1] == "tags" and c[2] == "skip" for c in calls)
    tags_path = catalog / "tags" / "m_y.jsonl"
    assert tags_path.exists()
    assert tags_path.read_text().strip() != ""


# ──────────────────────────────────────────────────────────────────────────
# F0 (faces.py) — empty:no_faces
# ──────────────────────────────────────────────────────────────────────────

class _FakeCap:
    def get(self, prop):
        return 25.0

    def read(self):
        return (False, None)  # no frames — track_count stays 0, no GPU/model needed

    def release(self):
        pass


def _install_fake_cv2_insightface(monkeypatch):
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CAP_PROP_FPS = 5
    fake_cv2.VideoCapture = lambda path: _FakeCap()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    class _FakeFaceAnalysis:
        def __init__(self, name=None, root=None):
            pass

        def prepare(self, ctx_id=0, det_size=None):
            pass

        def get(self, frame):
            return []

    fake_app = types.ModuleType("insightface.app")
    fake_app.FaceAnalysis = _FakeFaceAnalysis
    fake_insightface = types.ModuleType("insightface")
    fake_insightface.app = fake_app
    monkeypatch.setitem(sys.modules, "insightface", fake_insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", fake_app)


def test_f0_empty_no_faces_when_zero_tracks(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    state = tmp_path / "state"
    erf_root = tmp_path / "root"
    proxy = erf_root / "media" / "proxies"
    proxy.mkdir(parents=True)
    (proxy / "m_noface.mp4").write_bytes(b"")  # existence only — cv2 is faked
    catalog.mkdir()
    state.mkdir()
    journal.init_journal(state / "journal.db")
    cfg_path = _write_config(tmp_path, catalog, state, erf_root=erf_root)

    _install_fake_cv2_insightface(monkeypatch)
    calls = _record_spy(monkeypatch)
    flag_calls = []
    monkeypatch.setattr(outcomes, "set_flag", lambda mid, name, value, **kw: flag_calls.append((mid, name, value, kw)))

    manifest = faces.run("m_noface", config_path=cfg_path)

    assert manifest.ok is True
    assert manifest.counts["tracks"] == 0
    empty_calls = [c for c in calls if c[1] == "F0" and c[2] == "empty"]
    assert len(empty_calls) == 1
    assert empty_calls[0][3]["reason"] == "no_faces"
    assert not (catalog / "face_tracks" / "m_noface.jsonl").exists()

    has_faces_calls = [c for c in flag_calls if c[1] == "has_faces"]
    assert has_faces_calls[-1][2] is False
    assert has_faces_calls[-1][3]["reason"] == "no_faces"


# ──────────────────────────────────────────────────────────────────────────
# REDIARIZE (bin/rediarize.py) — empty:no_turns
# ──────────────────────────────────────────────────────────────────────────

def test_rediarize_empty_no_turns_records_accurate_disposition_and_skips_rebuild(monkeypatch):
    calls = []
    monkeypatch.setattr(rdz.outcomes, "record", lambda mid, stage, status, **kw: calls.append((mid, stage, status, kw)))
    flag_calls = []
    monkeypatch.setattr(rdz.outcomes, "set_flag", lambda mid, name, value, **kw: flag_calls.append((mid, name, value, kw)))
    rebuild_calls = []
    monkeypatch.setattr(rdz, "_rebuild", lambda mid: rebuild_calls.append(mid))
    monkeypatch.setattr(rdz, "rediarize_one", lambda mid, cfg: {"ok": True, "turns": 0, "speakers": 0})
    monkeypatch.setattr(rdz, "heartbeat", None)
    monkeypatch.setattr(rdz, "log_event", None)
    monkeypatch.setattr(sys, "argv", ["rediarize.py", "--media-ids", "m_empty", "--no-rebuild"])

    rdz.main()  # must not raise

    empty_calls = [c for c in calls if c[1] == "REDIARIZE" and c[2] == "empty"]
    assert len(empty_calls) == 1
    assert empty_calls[0][3]["reason"] == "no_turns"
    assert rebuild_calls == []  # _rebuild must NOT run for a 0-turn result

    embeddings_flag_calls = [c for c in flag_calls if c[1] == "has_diar_embeddings"]
    assert embeddings_flag_calls[-1][2] is False
    assert embeddings_flag_calls[-1][3]["reason"] == "no_turns"


def test_rediarize_ok_path_unaffected_by_empty_check(monkeypatch):
    """Regression guard: a real result (turns > 0) still records ok and runs
    the rebuild — the empty-check must not have swallowed the happy path."""
    calls = []
    monkeypatch.setattr(rdz.outcomes, "record", lambda mid, stage, status, **kw: calls.append((mid, stage, status, kw)))
    monkeypatch.setattr(rdz.outcomes, "set_flag", lambda mid, name, value, **kw: None)
    rebuild_calls = []
    monkeypatch.setattr(rdz, "_rebuild", lambda mid: rebuild_calls.append(mid))
    monkeypatch.setattr(rdz, "rediarize_one", lambda mid, cfg: {"ok": True, "turns": 3, "speakers": 2})
    monkeypatch.setattr(rdz, "heartbeat", None)
    monkeypatch.setattr(rdz, "log_event", None)
    monkeypatch.setattr(sys, "argv", ["rediarize.py", "--media-ids", "m_ok"])  # no --no-rebuild

    rdz.main()

    ok_calls = [c for c in calls if c[1] == "REDIARIZE" and c[2] == "ok"]
    assert len(ok_calls) == 1
    assert ok_calls[0][3]["counts"] == {"turns": 3, "speakers": 2}
    assert rebuild_calls == ["m_ok"]

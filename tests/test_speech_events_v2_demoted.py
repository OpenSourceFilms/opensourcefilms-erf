"""D-Log-31/32 (worklist item 1): stratum-D per-window voiceprint scoring
wired into build_speech_events_v2.py's identity resolution, at word
granularity (D-Log-32 fixed an event-level bleed bug — see
_word_demoted_override's docstring). bin/ is not a package — loaded via
importlib.util, matching tests/test_stage_empty_dispositions.py.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sev2 = _load_bin_module("build_speech_events_v2_test", "build_speech_events_v2.py")


def test_load_demoted_windows_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sev2, "CAT", tmp_path)
    assert sev2._load_demoted_windows("m_absent") == {}


def test_load_demoted_windows_groups_by_speaker_label(tmp_path, monkeypatch):
    d = tmp_path / "demoted_window_identity"
    d.mkdir()
    (d / "m_x.json").write_text(json.dumps({
        "media_id": "m_x",
        "windows": [
            {"speaker_label": "SPEAKER_00", "start_ms": 0, "end_ms": 800,
             "person_id": "p_alice", "score": 0.55},
            {"speaker_label": "SPEAKER_00", "start_ms": 900, "end_ms": 1700,
             "person_id": None, "score": 0.2},
        ],
    }))
    monkeypatch.setattr(sev2, "CAT", tmp_path)
    out = sev2._load_demoted_windows("m_x")
    assert out == {"SPEAKER_00": [(0, 800, "p_alice", 0.55), (900, 1700, None, 0.2)]}


def test_resolve_identity_gallery_source():
    idmap = {"SPEAKER_00": {"source": "gallery", "person_id": "p_bob", "gallery_score": 0.9}}
    pid, suid, conf = sev2._resolve_identity("SPEAKER_00", idmap)
    assert (pid, suid, conf) == ("p_bob", None, 0.9)


def test_resolve_identity_cluster_unknown_source():
    idmap = {"SPEAKER_00": {"source": "cluster_unknown", "cluster_id": "c_123"}}
    pid, suid, conf = sev2._resolve_identity("SPEAKER_00", idmap)
    assert (pid, suid, conf) == (None, "c_123", 0.0)


def test_resolve_identity_demoted_source_no_longer_scores_windows():
    """_resolve_identity is now the pure per-slot fallback only — window
    scoring moved to _word_demoted_override, called before event grouping."""
    idmap = {"SPEAKER_00": {"source": "demoted_overmerge", "cluster_id": "c_unk_1"}}
    pid, suid, conf = sev2._resolve_identity("SPEAKER_00", idmap)
    assert (pid, suid, conf) == (None, "c_unk_1", 0.0)


def test_word_override_no_demoted_windows_returns_none():
    idmap = {"SPEAKER_00": {"source": "demoted_overmerge"}}
    pid, score = sev2._word_demoted_override("SPEAKER_00", idmap, 100, 900, {})
    assert (pid, score) == (None, 0.0)


def test_word_override_non_demoted_source_returns_none():
    idmap = {"SPEAKER_00": {"source": "gallery", "person_id": "p_bob"}}
    demoted_windows = {"SPEAKER_00": [(0, 1000, "p_alice", 0.55)]}
    pid, score = sev2._word_demoted_override("SPEAKER_00", idmap, 100, 900, demoted_windows)
    assert (pid, score) == (None, 0.0)


def test_word_override_matches_covering_window():
    idmap = {"SPEAKER_00": {"source": "demoted_overmerge"}}
    demoted_windows = {"SPEAKER_00": [(0, 1000, "p_alice", 0.55), (5000, 6000, "p_bob", 0.60)]}
    pid, score = sev2._word_demoted_override("SPEAKER_00", idmap, 100, 300, demoted_windows)
    assert (pid, score) == ("p_alice", 0.55)


def test_word_override_no_covering_window_returns_none():
    """The core D-Log-32 fix: a word OUTSIDE any scored window must not
    inherit a nearby window's verdict, even if it shares the same diar label
    and is part of the same continuous turn."""
    idmap = {"SPEAKER_00": {"source": "demoted_overmerge"}}
    demoted_windows = {"SPEAKER_00": [(0, 1000, "p_alice", 0.55)]}
    pid, score = sev2._word_demoted_override("SPEAKER_00", idmap, 5000, 5300, demoted_windows)
    assert (pid, score) == (None, 0.0)


def test_word_override_unresolved_window_returns_none():
    idmap = {"SPEAKER_00": {"source": "demoted_overmerge"}}
    demoted_windows = {"SPEAKER_00": [(0, 1000, None, 0.2)]}
    pid, score = sev2._word_demoted_override("SPEAKER_00", idmap, 100, 900, demoted_windows)
    assert (pid, score) == (None, 0.0)


def test_word_override_end_to_end_no_bleed_across_speaker_change(tmp_path, monkeypatch):
    """Reproduces the exact D-Log-32 bug on a minimal fixture: one continuous
    SPEAKER_00 diar turn actually contains two real speakers. Only the words
    inside the scored window may resolve; the rest must stay anonymous."""
    monkeypatch.setattr(sev2, "CAT", tmp_path)
    d = tmp_path / "demoted_window_identity"
    d.mkdir()
    (d / "m_y.json").write_text(json.dumps({
        "media_id": "m_y",
        "windows": [{"speaker_label": "SPEAKER_00", "start_ms": 1000, "end_ms": 2000,
                     "person_id": "p_test_subject", "score": 0.55}],
    }))
    demoted_windows = sev2._load_demoted_windows("m_y")
    idmap = {"SPEAKER_00": {"source": "demoted_overmerge", "cluster_id": "c_unk_1"}}

    covered = sev2._word_demoted_override("SPEAKER_00", idmap, 1200, 1500, demoted_windows)
    uncovered = sev2._word_demoted_override("SPEAKER_00", idmap, 3000, 3300, demoted_windows)
    assert covered[0] == "p_test_subject"
    assert uncovered[0] is None

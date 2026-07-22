"""D-Log-31 (worklist item 1): stratum-D per-window voiceprint scoring.
bin/ is not a package — build_demoted_window_identity.py is loaded via
importlib.util, matching tests/test_stage_empty_dispositions.py. Module-
level path constants are monkeypatched to tmp_path fixtures rather than
touching real catalog/ data.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dwi = _load_bin_module("build_demoted_window_identity_test", "build_demoted_window_identity.py")


def test_l2_normalizes():
    v = dwi._l2([3.0, 4.0])
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_l2_zero_vector_is_safe():
    v = dwi._l2([0.0, 0.0])
    assert list(v) == [0.0, 0.0]


def test_score_windows_above_threshold_resolves():
    gallery_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    names = ["p_alice", "p_bob"]
    # exact match to p_alice's direction -> cosine sim 1.0, clears 0.45
    results = dwi.score_windows(gallery_vecs, names, [[1.0, 0.0]])
    assert results == [("p_alice", 1.0)]


def test_score_windows_below_threshold_stays_none():
    gallery_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    names = ["p_alice", "p_bob"]
    # points away from both gallery directions -> negative similarity to each,
    # well below GALLERY_THR
    results = dwi.score_windows(gallery_vecs, names, [[-1.0, -1.0]])
    assert results[0][0] is None


def test_score_windows_empty_gallery_returns_none_for_all():
    results = dwi.score_windows(np.zeros((0, 2)), [], [[1.0, 0.0], [0.0, 1.0]])
    assert results == [(None, 0.0), (None, 0.0)]


def test_load_gallery_reads_centroids(tmp_path, monkeypatch):
    gdir = tmp_path / "gallery_clean"
    gdir.mkdir()
    (gdir / "p_alice.json").write_text(json.dumps({
        "person_id": "p_alice", "centroid": [1.0] + [0.0] * 255,
    }))
    monkeypatch.setattr(dwi, "GALLERY_DIR", gdir)
    vecs, names = dwi.load_gallery()
    assert names == ["p_alice"]
    assert vecs.shape == (1, 256)


def test_load_gallery_skips_malformed_centroid(tmp_path, monkeypatch):
    gdir = tmp_path / "gallery_clean"
    gdir.mkdir()
    (gdir / "p_bad.json").write_text(json.dumps({"person_id": "p_bad", "centroid": [1.0, 2.0]}))
    monkeypatch.setattr(dwi, "GALLERY_DIR", gdir)
    vecs, names = dwi.load_gallery()
    assert names == []


def test_load_embeddings_skips_mismatched_chunks_and_windows(tmp_path, monkeypatch):
    emb_path = tmp_path / "wespeaker.jsonl"
    emb_path.write_text(
        json.dumps({
            "media_id": "m_x", "speaker_label": "SPEAKER_00",
            "chunks": [[1.0, 0.0]], "chunk_windows": [],  # mismatched -> skipped
        }) + "\n" +
        json.dumps({
            "media_id": "m_y", "speaker_label": "SPEAKER_00",
            "chunks": [[1.0, 0.0]], "chunk_windows": [[0, 800]],
        }) + "\n"
    )
    monkeypatch.setattr(dwi, "EMBEDDINGS", emb_path)
    out = dwi.load_embeddings()
    assert ("m_x", "SPEAKER_00") not in out
    assert ("m_y", "SPEAKER_00") in out


def test_find_demoted_slots_filters_by_source(tmp_path, monkeypatch):
    idir = tmp_path / "identity_map"
    idir.mkdir()
    (idir / "m_z.json").write_text(json.dumps({
        "media_id": "m_z",
        "speakers": {
            "SPEAKER_00": {"source": "demoted_overmerge", "person_id": "p_unk_1"},
            "SPEAKER_01": {"source": "gallery", "person_id": "p_alice"},
        },
    }))
    monkeypatch.setattr(dwi, "IDENTITY_MAP_DIR", idir)
    out = dwi.find_demoted_slots()
    assert out == {"m_z": ["SPEAKER_00"]}


def test_main_writes_per_window_verdicts_end_to_end(tmp_path, monkeypatch):
    gdir = tmp_path / "gallery_clean"
    gdir.mkdir()
    (gdir / "p_alice.json").write_text(json.dumps({
        "person_id": "p_alice", "centroid": [1.0] + [0.0] * 255,
    }))

    idir = tmp_path / "identity_map"
    idir.mkdir()
    (idir / "m_z.json").write_text(json.dumps({
        "media_id": "m_z",
        "speakers": {"SPEAKER_00": {"source": "demoted_overmerge", "person_id": "p_unk_1"}},
    }))

    emb_path = tmp_path / "wespeaker.jsonl"
    emb_path.write_text(json.dumps({
        "media_id": "m_z", "speaker_label": "SPEAKER_00",
        "chunks": [[1.0] + [0.0] * 255, [0.0] * 256],
        "chunk_windows": [[0, 800], [900, 1700]],
    }) + "\n")

    out_dir = tmp_path / "demoted_window_identity"

    monkeypatch.setattr(dwi, "GALLERY_DIR", gdir)
    monkeypatch.setattr(dwi, "IDENTITY_MAP_DIR", idir)
    monkeypatch.setattr(dwi, "EMBEDDINGS", emb_path)
    monkeypatch.setattr(dwi, "OUT_DIR", out_dir)

    dwi.main()

    out_path = out_dir / "m_z.json"
    assert out_path.exists()
    result = json.loads(out_path.read_text())
    assert result["media_id"] == "m_z"
    windows = result["windows"]
    assert len(windows) == 2
    assert windows[0]["person_id"] == "p_alice"  # exact-match chunk resolves
    assert windows[1]["person_id"] is None       # zero-vector chunk stays anonymous

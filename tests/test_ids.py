"""ID generator tests — determinism, patterns, and edge cases."""
import pytest
from datetime import datetime, timezone
from erf import ids


MID = "m_9f3a1c7be002"


# ── media_id ─────────────────────────────────────────────────────────────────

def test_media_id_pattern():
    mid = ids.media_id("day12/A007.mov", 1234567, 60000)
    assert ids.matches(mid, "media_id"), f"bad pattern: {mid!r}"


def test_media_id_deterministic():
    a = ids.media_id("day12/A007.mov", 1234567, 60000)
    b = ids.media_id("day12/A007.mov", 1234567, 60000)
    assert a == b


def test_media_id_distinct_for_different_inputs():
    a = ids.media_id("day12/A007.mov", 1234567, 60000)
    b = ids.media_id("day12/A007.mov", 1234568, 60000)  # filesize differs
    assert a != b


# ── ASR IDs ──────────────────────────────────────────────────────────────────

def test_word_id():
    wid = ids.word_id(MID, 482)
    assert wid == f"{MID}:w:000482"
    assert ids.matches(wid, "word_id")


def test_segment_id():
    sid = ids.segment_id(MID, 31)
    assert sid == f"{MID}:s:00031"
    assert ids.matches(sid, "segment_id")


# ── Chunk, scene, keyframe ────────────────────────────────────────────────────

def test_chunk_id():
    cid = ids.chunk_id(MID, 42120, 49880)
    assert cid == f"{MID}:c:42120-49880"
    assert ids.matches(cid, "chunk_id")


def test_scene_id():
    scid = ids.scene_id(MID, 7)
    assert scid == f"{MID}:sc:0007"
    assert ids.matches(scid, "scene_id")


def test_keyframe_id():
    kfid = ids.keyframe_id(MID, 45003)
    assert kfid == f"{MID}:kf:45003"
    assert ids.matches(kfid, "keyframe_id")


# ── Embedding ─────────────────────────────────────────────────────────────────

def test_embedding_id():
    eid = ids.embedding_id(f"{MID}:c:42120-49880", "qwen3-8b")
    assert eid == f"emb:{MID}:c:42120-49880:qwen3-8b"
    assert ids.matches(eid, "embedding_id")


# ── Phase 3 ──────────────────────────────────────────────────────────────────

def test_detection_id():
    kfid = ids.keyframe_id(MID, 45003)
    did = ids.detection_id(kfid, 2)
    assert did == f"{MID}:kf:45003:det:002"
    assert ids.matches(did, "detection_id")


def test_face_track_id():
    ftid = ids.face_track_id(MID, 3)
    assert ftid == f"{MID}:ft:0003"
    assert ids.matches(ftid, "face_track_id")


def test_person_id_valid():
    pid = ids.person_id("jane_doe")
    assert pid == "p_jane_doe"
    assert ids.matches(pid, "person_id")


def test_person_id_rejects_spaces():
    with pytest.raises(ValueError):
        ids.person_id("Jane Doe")


def test_person_id_normalises_case():
    pid = ids.person_id("JANE_DOE")
    assert pid == "p_jane_doe"


# ── Events, entity nodes ──────────────────────────────────────────────────────

def test_event_id():
    evid = ids.event_id(MID, 12)
    assert evid == f"{MID}:ev:0012"
    assert ids.matches(evid, "event_id")


def test_entity_node_id():
    nid = ids.entity_node_id("loc", "warehouse")
    assert nid == "n_loc_warehouse"


# ── Export ───────────────────────────────────────────────────────────────────

def test_export_id_pattern():
    at = datetime(2026, 6, 2, 3, 14, 21, tzinfo=timezone.utc)
    eid = ids.export_id("a1b2c3dead", at=at)
    assert eid == "exp_20260602T031421Z_a1b2c3"
    assert ids.matches(eid, "export_id")


def test_export_id_deterministic_for_same_inputs():
    at = datetime(2026, 6, 2, 3, 14, 21, tzinfo=timezone.utc)
    a = ids.export_id("abc123", at=at)
    b = ids.export_id("abc123", at=at)
    assert a == b

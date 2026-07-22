"""Contract validator tests — all 15 schemas against synthetic records."""
import pytest
import jsonschema
from erf.contracts import validate, validate_all, schema_names


# ── Synthetic valid records ───────────────────────────────────────────────────

VALID = {
    "media_manifest": {
        "schema_version": "1.0.0",
        "media_id": "m_9f3a1c7be002",
        "relpath": "day12/A007C031.mov",
        "filesize_bytes": 1234567890,
        "duration_ms": 180000,
        "content_blake2b": "abcdef1234567890abcdef1234567890",
        "container": "mov",
        "video": {
            "codec": "prores",
            "width": 3840, "height": 2160,
            "fps": 23.976, "fps_rational": "24000/1001",
            "bit_depth": 10,
            "color_primaries": "bt2020",
            "transfer": "smpte2084"
        },
        "audio": {"codec": "pcm_s24le", "sample_rate": 48000, "channels": 2},
        "ingested_at_utc": "2026-06-01T10:00:00+00:00",
        "source_note": "Day 12 A-cam",
    },
    "asr_word": {
        "schema_version": "1.0.0",
        "word_id": "m_9f3a1c7be002:w:000482",
        "media_id": "m_9f3a1c7be002",
        "segment_id": "m_9f3a1c7be002:s:00031",
        "index": 482,
        "text": "warehouse",
        "start_ms": 42120, "end_ms": 42800,
        "prob": 0.97,
        "speaker_label": "SPEAKER_00",
        "speaker_person_id": None,
    },
    "asr_segment": {
        "schema_version": "1.0.0",
        "segment_id": "m_9f3a1c7be002:s:00031",
        "media_id": "m_9f3a1c7be002",
        "index": 31,
        "text": "you said you'd never come back here",
        "start_ms": 42120, "end_ms": 49880,
        "avg_logprob": -0.23,
        "no_speech_prob": 0.01,
        "speaker_label": "SPEAKER_00",
        "speaker_person_id": None,
    },
    "asr_diar": {
        "schema_version": "1.0.0",
        "media_id": "m_9f3a1c7be002",
        "turn_index": 7,
        "speaker_label": "SPEAKER_00",
        "start_ms": 40000,
        "end_ms": 52000,
    },
    "chunk": {
        "schema_version": "1.0.0",
        "chunk_id": "m_9f3a1c7be002:c:42120-49880",
        "media_id": "m_9f3a1c7be002",
        "start_ms": 42120, "end_ms": 49880,
        "text": "you said you'd never come back here",
        "word_ids": ["m_9f3a1c7be002:w:000482"],
        "speaker_label": "SPEAKER_00",
        "speaker_person_id": None,
        "emotional_beat": None,
        "b_roll_flag": None,
    },
    "embedding": {
        "schema_version": "1.0.0",
        "embedding_id": "emb:m_9f3a1c7be002:c:42120-49880:qwen3-8b",
        "source_id": "m_9f3a1c7be002:c:42120-49880",
        "source_type": "chunk",
        "media_id": "m_9f3a1c7be002",
        "modality": "text",
        "model_tag": "qwen3-8b",
        "dim": 4096,
        "vector": [0.1] * 4096,
        "normalized": True,
    },
    "scene": {
        "schema_version": "1.0.0",
        "scene_id": "m_9f3a1c7be002:sc:0007",
        "media_id": "m_9f3a1c7be002",
        "index": 7,
        "start_ms": 40000, "end_ms": 55000,
        "start_frame": 961, "end_frame": 1321,
        "detector": "pyscenedetect-content",
    },
    "keyframe": {
        "schema_version": "1.0.0",
        "keyframe_id": "m_9f3a1c7be002:kf:45003",
        "media_id": "m_9f3a1c7be002",
        "scene_id": "m_9f3a1c7be002:sc:0007",
        "time_ms": 45003, "frame": 1080,
        "image_relpath": "media/keyframes/m_9f3a1c7be002/m_9f3a1c7be002:kf:45003.jpg",
        "thumb_relpath": None,
        "width": 3840, "height": 2160,
        "visual_quality_score": None,
        "person_ids": None,
        "b_roll_flag": None,
    },
    "detection": {
        "schema_version": "1.0.0",
        "detection_id": "m_9f3a1c7be002:kf:45003:det:002",
        "keyframe_id": "m_9f3a1c7be002:kf:45003",
        "media_id": "m_9f3a1c7be002",
        "label": "person",
        "score": 0.91,
        "bbox": [0.1, 0.2, 0.3, 0.5],
        "detector": "groundingdino",
        "is_face": False,
        "face_track_id": None,
    },
    "face_track": {
        "schema_version": "1.0.0",
        "face_track_id": "m_9f3a1c7be002:ft:0003",
        "media_id": "m_9f3a1c7be002",
        "start_ms": 42000, "end_ms": 50000,
        "keyframe_ids": ["m_9f3a1c7be002:kf:45003"],
        "embedding": [0.5] * 512,
        "person_id": None,
        "match_score": None,
    },
    "person": {
        "schema_version": "1.0.0",
        "person_id": "p_jane_doe",
        "display_name": "Jane Doe",
        "created_at_utc": "2026-06-01T09:00:00+00:00",
        "consent": True,
        "face_enrolled": False,
        "voice_enrolled": False,
        "face_gallery_relpath": None,
        "voice_gallery_relpath": None,
    },
    "event": {
        "schema_version": "1.1.0",  # bumped for additive edit_role/format_fit/publicability fields
        "event_id": "m_9f3a1c7be002:ev:0012",
        "media_id": "m_9f3a1c7be002",
        "start_ms": 40000, "end_ms": 55000,
        "kind": "scene_event",
        "chunk_ids": ["m_9f3a1c7be002:c:42120-49880"],
        "scene_ids": ["m_9f3a1c7be002:sc:0007"],
        "summary": None,
        "emotional_beat": None,
        "person_ids": None,
        "b_roll_flag": None,
        "visual_quality_score": None,
    },
    "entity_node": {
        "schema_version": "1.0.0",
        "entity_node_id": "n_loc_warehouse",
        "type": "location",
        "label": "warehouse",
        "person_id": None,
        "aliases": ["the warehouse", "storage unit"],
        "mention_refs": ["m_9f3a1c7be002:c:42120-49880"],
    },
    "status": {
        "schema_version": "1.0.0",
        "media_id": "m_9f3a1c7be002",
        "stages": {
            "A": {
                "at_utc": "2026-06-01T10:05:00+00:00",
                "ok": True,
                "run_id": "run_20260601T100500Z",
                "tool_version": "faster-whisper-1.2.1",
                "counts": {"words": 1842, "segments": 63},
                "error": None,
            }
        },
    },
    "export_manifest": {
        "schema_version": "1.0.0",
        "export_id": "exp_20260602T031421Z_a1b2c3",
        "created_at_utc": "2026-06-02T03:14:21+00:00",
        "query": "warehouse argument at night",
        "format": "fcpxml",
        "frame_rate": 23.976,
        "clips": [
            {
                "media_id": "m_9f3a1c7be002",
                "source_relpath": "day12/A007C031.mov",
                "in_frame": 1013,
                "out_frame": 1198,
                "label": "warehouse argument",
            }
        ],
        "output_relpath": "catalog/exports/exp_20260602T031421Z_a1b2c3.fcpxml",
    },
}


# ── Tests: valid records ──────────────────────────────────────────────────────

@pytest.mark.parametrize("schema_name", list(VALID.keys()))
def test_valid_record(schema_name):
    validate(VALID[schema_name], schema_name)


def test_all_schema_files_present():
    names = schema_names()
    for expected in ["media_manifest","asr_word","chunk","embedding","scene","keyframe",
                     "detection","face_track","person","event","entity_node","status","export_manifest"]:
        assert expected in names, f"Schema file missing: {expected}.schema.json"


# ── Tests: invalid records (selected) ────────────────────────────────────────

def test_media_id_pattern_rejected():
    bad = {**VALID["media_manifest"], "media_id": "x_badhash"}
    with pytest.raises(jsonschema.ValidationError):
        validate(bad, "media_manifest")


def test_additional_property_rejected():
    bad = {**VALID["media_manifest"], "unknown_field": "oops"}
    with pytest.raises(jsonschema.ValidationError):
        validate(bad, "media_manifest")


def test_embedding_wrong_modality_model():
    bad = {**VALID["embedding"], "modality": "text", "model_tag": "qwen3vl-2b"}
    with pytest.raises(jsonschema.ValidationError):
        validate(bad, "embedding")


def test_embedding_dim_vector_mismatch():
    bad = {**VALID["embedding"], "vector": [0.1] * 100}  # dim=4096 but only 100 elements
    with pytest.raises(jsonschema.ValidationError):
        validate(bad, "embedding")


def test_visual_embedding_valid():
    rec = {
        **VALID["embedding"],
        "embedding_id": "emb:m_9f3a1c7be002:kf:45003:qwen3vl-2b",
        "source_id": "m_9f3a1c7be002:kf:45003",
        "source_type": "keyframe",
        "modality": "visual",
        "model_tag": "qwen3vl-2b",
        "dim": 2048,
        "vector": [0.1] * 2048,
    }
    validate(rec, "embedding")


def test_chunk_invalid_emotional_beat():
    bad = {**VALID["chunk"], "emotional_beat": "NOT_A_VALID_BEAT"}
    with pytest.raises(jsonschema.ValidationError):
        validate(bad, "chunk")


def test_person_id_pattern():
    bad = {**VALID["person"], "person_id": "Jane Doe"}  # spaces not allowed
    with pytest.raises(jsonschema.ValidationError):
        validate(bad, "person")


def test_validate_all_returns_errors():
    records = [VALID["asr_word"], {**VALID["asr_word"], "prob": 1.5}]  # prob > 1
    errs = validate_all(records, "asr_word")
    assert len(errs) == 1
    assert "record[1]" in errs[0]

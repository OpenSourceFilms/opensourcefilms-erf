import json
from pathlib import Path
import pytest
import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "identity_overlay.schema.json"
with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)


def minimal_overlay():
    return {
        "schema_version": "1.0.0",
        "media_id": "test",
        "produced_by": "run1",
        "stage_version": "v1",
        "face_assignments": [],
        "speaker_assignments": [],
        "chunk_speaker": [],
        "keyframe_persons": [],
        "event_persons": [],
    }


def test_minimal_valid():
    jsonschema.validate(minimal_overlay(), SCHEMA)


def test_full_valid():
    data = minimal_overlay()
    data["face_assignments"] = [{"face_track_id": "ft1", "person_id": "p1", "match_score": 0.9, "method": "gallery"}]
    data["speaker_assignments"] = [{"speaker_label": "SPEAKER_00", "person_id": "p1", "method": "bootstrap", "confidence": 0.8, "evidence": ["ft1"]}]
    data["chunk_speaker"] = [{"chunk_id": "m_abc:c:0-5000", "person_id": "p1"}]
    data["keyframe_persons"] = [{"keyframe_id": "m_abc:kf:0001", "person_ids": ["p1"]}]
    data["event_persons"] = [{"event_id": "m_abc:ev:0000", "person_ids": ["p1"]}]
    jsonschema.validate(data, SCHEMA)


def test_missing_media_id():
    data = minimal_overlay()
    del data["media_id"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, SCHEMA)


def test_face_assignment_invalid_method():
    data = minimal_overlay()
    data["face_assignments"] = [{"face_track_id": "ft1", "person_id": "p1", "match_score": 0.8, "method": "invalid"}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, SCHEMA)


def test_speaker_null_person_valid():
    data = minimal_overlay()
    data["speaker_assignments"] = [{"speaker_label": "SPEAKER_01", "person_id": None, "method": "none"}]
    jsonschema.validate(data, SCHEMA)


def test_speaker_bootstrap_null_valid():
    data = minimal_overlay()
    data["speaker_assignments"] = [{"speaker_label": "SPEAKER_01", "person_id": None, "method": "bootstrap"}]
    jsonschema.validate(data, SCHEMA)

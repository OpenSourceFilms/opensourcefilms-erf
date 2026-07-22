"""short_template schema validation tests."""
import pytest
import jsonschema
from erf.contracts import validate


VALID_TEMPLATE = {
    "schema_version": "1.0.0",
    "template_id": "tmpl_a1b2c3d4e5f6",
    "name": "BTS Hook + Quote",
    "platform": "reels",
    "duration_target_s": 45.0,
    "caption_style": "centre",
    "tone_guidance": "Warm and direct",
    "beat_structure": [
        {"role": "hook", "duration_target_s": 8.0, "guidance": "Open on something surprising"},
        {"role": "body", "duration_target_s": 30.0, "guidance": "Core story moment"},
        {"role": "quote", "duration_target_s": 7.0, "guidance": "Strong closing line"},
    ],
    "exemplar_beat_plan_ids": [],
    "created_at_utc": "2026-06-06T21:00:00+00:00",
    "created_by": "test",
    "active": True,
}


def test_valid_template():
    validate(VALID_TEMPLATE, "short_template")


def test_minimal_template():
    """Required fields only — no optional fields."""
    rec = {
        "schema_version": "1.0.0",
        "template_id": "tmpl_000000000000",
        "name": "Minimal",
        "platform": "any",
        "beat_structure": [{"role": "body", "duration_target_s": 30.0}],
        "created_at_utc": "2026-06-06T00:00:00+00:00",
        "created_by": "test",
    }
    validate(rec, "short_template")


def test_invalid_platform():
    rec = dict(VALID_TEMPLATE)
    rec["platform"] = "instagram"
    with pytest.raises(jsonschema.ValidationError):
        validate(rec, "short_template")


def test_invalid_caption_style():
    rec = dict(VALID_TEMPLATE)
    rec["caption_style"] = "bottom"
    with pytest.raises(jsonschema.ValidationError):
        validate(rec, "short_template")


def test_invalid_template_id_pattern():
    rec = dict(VALID_TEMPLATE)
    rec["template_id"] = "bp_a1b2c3d4e5f6"  # wrong prefix
    with pytest.raises(jsonschema.ValidationError):
        validate(rec, "short_template")


def test_beat_structure_requires_role_and_duration():
    rec = dict(VALID_TEMPLATE)
    rec = {**VALID_TEMPLATE, "beat_structure": [{"role": "hook"}]}  # missing duration_target_s
    with pytest.raises(jsonschema.ValidationError):
        validate(rec, "short_template")


def test_all_platforms():
    for platform in ["tiktok", "reels", "youtube_shorts", "longform", "any"]:
        rec = {**VALID_TEMPLATE, "platform": platform}
        validate(rec, "short_template")

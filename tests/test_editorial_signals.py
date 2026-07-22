"""Tests for Seam 2 — editorial_signals schema freeze + versioned resolver."""
import glob
import json
import pathlib

from erf import contracts, editorial_signals as es
from erf.editorial import ARCHETYPES, FUNCTIONS, TENSIONS


def _schema_enum(defname: str) -> set:
    return set(contracts._load_validator("editorial_signals").schema["$defs"][defname]["enum"])


def test_schema_enums_mirror_editorial_vocab():
    # The frozen schema must track editorial.py's vocab exactly. If someone edits
    # one without the other, this fails loudly instead of silently rejecting data.
    assert _schema_enum("function") == set(FUNCTIONS)
    assert _schema_enum("tension") == set(TENSIONS)
    assert _schema_enum("archetype") == set(ARCHETYPES)


def test_sanitize_coerces_flash_drift():
    doc = {
        "clip_level": {"central_tension": "bogus_tension",
                       "archetype_candidates": ["THIS_SHOULD_NOT_WORK", "NOPE"]},
        "sentences": [{
            "sentence_index": 0,
            "functions": ["REVELATION", "REACTION", "HOOK"],   # remap, drop, keep
            "tension_tags": ["control_vs_chaos", "practical_test"],  # keep, drop (a FUNCTION)
            "chain_affordances": {"can_start_story": True, "can_escalate_story": False,
                                  "can_payoff_story": False, "needs_followup": False},
            "quote_completeness": {"complete_thought": True, "requires_previous": False,
                                   "requires_next": False},
        }],
    }
    clean, coercions = es.sanitize(doc)
    assert clean["clip_level"]["central_tension"] is None
    assert clean["clip_level"]["archetype_candidates"] == ["THIS_SHOULD_NOT_WORK"]
    s = clean["sentences"][0]
    assert s["functions"] == ["REVEAL", "HOOK"]          # REVELATION→REVEAL, REACTION dropped
    assert s["tension_tags"] == ["control_vs_chaos"]
    assert coercions  # drift was recorded


def test_build_record_validates_against_schema():
    # Editorial records live at catalog/editorial/{media_id}/{editorial_job_id}.json
    # (see erf.editorial_signals.write) -- the old flat catalog/editorial/{mid}.json
    # layout this test targeted is gone (2026-07-08 test-currency fix).
    f = sorted(f for f in glob.glob("catalog/editorial/*/label_*.json"))[0]
    mid = pathlib.Path(f).parent.name
    rec = es.build_record(mid, json.load(open(f)), source_evidence_hash="0" * 64,
                          editorial_job_id="t", model="deepseek-v4-flash", stage_version="LABEL/test")
    contracts.validate(rec, "editorial_signals")  # raises on failure


def test_write_latest_stale_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "EDIR", tmp_path / "editorial")
    rec = es.build_record("m_x", {"clip_level": {}, "sentences": []},
                          source_evidence_hash="abc", editorial_job_id="j1",
                          model="m", stage_version="v")
    p = es.write("m_x", rec, journal=False)
    assert es.latest("m_x")["editorial_job_id"] == "j1"     # manifest-only resolve
    assert es.is_stale("m_x", "abc") is False
    assert es.is_stale("m_x", "different") is True
    assert es.write("m_x", rec, journal=False) == p          # immutable no-op


def test_latest_and_stale_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "EDIR", tmp_path / "editorial")
    assert es.latest("m_absent") is None
    assert es.is_stale("m_absent", "x") is True


def test_source_evidence_hash_is_stable_and_order_independent():
    a = {"media_id": "m", "sentences": [{"i": 0}, {"i": 1}]}
    b = {"sentences": [{"i": 0}, {"i": 1}], "media_id": "m"}  # same content, keys reordered
    assert es.source_evidence_hash(a) == es.source_evidence_hash(b)

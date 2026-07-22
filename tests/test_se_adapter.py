"""Tests for Seam 1 — se_adapter (candidate → beat_plan 1.3.0)."""
import json
import pathlib
import shutil

import pytest

from erf import contracts, se_adapter as A

_MID = "m_f2098b3d8b7a"
_REAL = pathlib.Path("/workspace/erf/catalog")
_HAVE = (_REAL / "evidence" / f"{_MID}.json").exists()


def _b(role, am="sync_dialogue", pt="none"):
    return {"role": role, "audio_mode": am, "payoff_type": pt}


def test_edit_role_precedence():
    assert A._edit_role(_b("HOOK")) == "hook"                       # HOOK wins over sync_dialogue
    assert A._edit_role(_b("BUTTON")) == "call_to_action"
    assert A._edit_role(_b("RESULT", am="sync_dialogue")) == "quote"
    assert A._edit_role(_b("PAYOFF", am="broll_audio", pt="visual_proof")) == "proof"
    assert A._edit_role(_b("OBSTACLE", am="broll_audio")) == "b_roll"
    assert A._edit_role(_b("WANT", am="vo")) == "exposition"
    assert A._edit_role(_b("DECISION", am=None, pt=None)) == "body"


def test_adapt_rejects_before_io():
    # verdict / tier / empty-beats rejects happen before any file IO
    assert A.adapt({"candidate_id": "c", "verdict": "reject"}, journal=False)["ok"] is False
    assert A.adapt({"candidate_id": "c", "verdict": "complete",
                    "confidence_tier": "flash_scored_only"}, journal=False)["ok"] is False
    assert A.adapt({"candidate_id": "c", "verdict": "complete",
                    "confidence_tier": "pro_scored", "beats": []}, journal=False)["ok"] is False


@pytest.mark.skipif(not _HAVE, reason="sample clip not present")
def test_adapt_emits_valid_lane_free_plan(tmp_path, monkeypatch):
    for sub in ("evidence", "chunks", "beat_plans"):
        (tmp_path / sub).mkdir()
    shutil.copy(_REAL / "evidence" / f"{_MID}.json", tmp_path / "evidence" / f"{_MID}.json")
    src_chunks = _REAL / "chunks" / f"{_MID}.jsonl"
    if src_chunks.exists():
        shutil.copy(src_chunks, tmp_path / "chunks" / f"{_MID}.jsonl")
    monkeypatch.setattr(A, "CAT", tmp_path)

    sents = A._load_evidence_sentences(_MID)
    idx = sorted(sents)
    ranges = [(idx[0], idx[3]), (idx[10], idx[13]), (idx[20], idx[24])]
    total = sum(sents[b]["end_ms"] - sents[a]["start_ms"] for a, b in ranges)
    lane = next((k for k, (lo, hi) in A._LANE_BAND_MS.items() if lo <= total <= hi), None)
    cand = {"candidate_id": "c_t", "verdict": "complete", "confidence_tier": "pro_scored",
            "primary_format_lane": lane, "title": "t", "beats": [
                {"role": "HOOK", "clip_id": _MID, "sentence_range": list(ranges[0]), "audio_mode": "broll_audio"},
                {"role": "OBSTACLE", "clip_id": _MID, "sentence_range": list(ranges[1]), "audio_mode": "broll_audio"},
                {"role": "PAYOFF", "clip_id": _MID, "sentence_range": list(ranges[2]),
                 "audio_mode": "broll_audio", "payoff_type": "visual_proof"}]}

    res = A.adapt(cand, journal=False)
    assert res["ok"], res
    bp = json.loads((tmp_path / "beat_plans" / f"{res['bp_id']}.jsonl").read_text().splitlines()[-1])
    contracts.validate(bp, "beat_plan")                       # raises on invalid
    assert bp["schema_version"] == "1.3.0" and len(bp["beats"]) == 3
    # D2: lane never leaks into the plan
    assert "primary_format_lane" not in bp
    assert all("lane" not in b for b in bp["beats"])
    # times are copied byte-exact from the evidence package (no synthesis)
    assert bp["beats"][0]["in_ms"] == sents[idx[0]]["start_ms"]
    assert bp["beats"][2]["edit_role"] == "proof"


@pytest.mark.skipif(not _HAVE, reason="sample clip not present")
def test_adapt_rejects_out_of_band_duration(tmp_path, monkeypatch):
    (tmp_path / "evidence").mkdir()
    shutil.copy(_REAL / "evidence" / f"{_MID}.json", tmp_path / "evidence" / f"{_MID}.json")
    monkeypatch.setattr(A, "CAT", tmp_path)
    sents = A._load_evidence_sentences(_MID)
    idx = sorted(sents)
    cand = {"candidate_id": "c", "verdict": "complete", "confidence_tier": "pro_scored",
            "primary_format_lane": "single_clip_15_25s", "title": "t",
            "beats": [{"role": "BODY", "clip_id": _MID, "sentence_range": [idx[0], idx[-1]],
                       "audio_mode": "broll_audio"}]}  # whole clip ≫ 25s
    res = A.adapt(cand, journal=False)
    assert res["ok"] is False and "outside lane" in res["reason"]

"""Seam 1 — candidate → beat_plan adapter (Story Engine Stage I).

Maps a Story Engine candidate (patch_v3 §2/§3) onto a VALIDATED beat_plan 1.3.0 and
writes it. Deterministic, no model calls. The renderer can't tell an se_adapter
beat_plan from the existing author.py one — both speak the frozen seam.

Contract: STORY_ENGINE_PHASE0_SEAMS.md §1 (this is the authority on the mapping).
- never invents timecodes: in_ms/out_ms are copied from the evidence package's
  per-sentence start_ms/end_ms (keyed by sentence_index).
- quote beats are range-completed from editorial_signals quote_completeness.
- chunk_id resolves by max time-overlap against catalog/chunks (speaker_turn_search).
- lane (D2): used once for a duration-band sanity check, then DROPPED — never written
  to the beat_plan (beat_plan stays 1.3.0, lane-free).
- validates against the real beat_plan schema before writing; reject paths journal and
  never write.
"""
from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from . import contracts, editorial_signals as es, runlog
from .atomicio import jsonl_append

CAT = Path("/workspace/erf/catalog")
_QUOTE_MAX_ADD_MS = 1500  # max ms quote-completion may add per side (caps ASR-fragmentation/giant-segment runaway)
_AUTO_ELIGIBLE = {"pro_scored_tournament_ranked", "pro_scored"}
_LANE_BAND_MS = {
    "single_clip_15_25s": (15000, 25000),
    "mini_cross_media_30_60s": (30000, 60000),
    "long_cross_media_60_120s": (60000, 120000),
}


def _edit_role(beat: dict) -> str:
    """Candidate beat role + audio_mode + payoff_type → beat_plan edit_role enum (§1.3)."""
    role = beat.get("role")
    am = beat.get("audio_mode")
    pt = beat.get("payoff_type")
    if role == "HOOK":
        return "hook"
    if role == "BUTTON":
        return "call_to_action"
    if am == "sync_dialogue":
        return "quote"
    if role == "PAYOFF" and pt in {"visual_proof", "technical_reveal", "before_after"}:
        return "proof"
    if am in {"broll_audio", "music_only", "silent_visual"}:
        return "b_roll"
    if am == "vo":
        return "exposition"
    return "body"


def _load_evidence_sentences(clip_id: str) -> Dict[int, Dict[str, int]]:
    """{sentence_index: {start_ms, end_ms}} from the evidence package. Coerces to int."""
    p = CAT / "evidence" / f"{clip_id}.json"
    if not p.exists():
        raise ValueError(f"no evidence package for {clip_id}")
    ev = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, int]] = {}
    for s in ev.get("sentences") or []:
        try:
            out[int(s["sentence_index"])] = {
                "start_ms": int(s["start_ms"]),
                "end_ms": int(s["end_ms"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _complete_quote(clip_id: str, a: int, b: int, edit_role: str,
                    sents: Dict[int, Dict[str, int]],
                    max_add_ms: int = _QUOTE_MAX_ADD_MS) -> Tuple[int, int]:
    """Extend a quote's sentence range to a complete thought (quote_completeness),
    bounded by ADDED MILLISECONDS per side — not a sentence count.

    The ms bound is what keeps completion sane: heavily-fragmented ASR marks long runs
    as requires_previous/next, and some 'sentences' are giant merged segments. A count-
    only or unbounded walk would swallow the whole clip; an ms bound simply never pulls
    in a neighbour that adds more than max_add_ms.
    """
    if edit_role != "quote" or a not in sents or b not in sents:
        return a, b
    rec = es.latest(clip_id)
    if not rec:
        return a, b
    qc = {int(s["sentence_index"]): (s.get("quote_completeness") or {})
          for s in rec.get("sentences", [])}
    base_in = sents[a]["start_ms"]
    while a > 0 and (a - 1) in sents and qc.get(a, {}).get("requires_previous") \
            and base_in - sents[a - 1]["start_ms"] <= max_add_ms:
        a -= 1
    base_out = sents[b]["end_ms"]
    while (b + 1) in sents and qc.get(b, {}).get("requires_next") \
            and sents[b + 1]["end_ms"] - base_out <= max_add_ms:
        b += 1
    return a, b


def _resolve_chunk_id(clip_id: str, in_ms: int, out_ms: int) -> str:
    """The chunk (speaker_turn_search) with max overlap of [in_ms, out_ms]."""
    p = CAT / "chunks" / f"{clip_id}.jsonl"
    chunks: List[dict] = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    base = [c for c in chunks if c.get("chunk_rule") == "speaker_turn_search"] or chunks
    best, best_ov = None, 0
    for c in base:
        try:
            cs, ce = int(c["start_ms"]), int(c["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        ov = max(0, min(out_ms, ce) - max(in_ms, cs))
        if ov > best_ov:
            best_ov, best = ov, c.get("chunk_id")
    if best:
        return best
    runlog.log_event("SE:I", clip_id, "warn",
                     {"reason": "no covering chunk", "in_ms": in_ms, "out_ms": out_ms})
    return f"{clip_id}:nochunk"


def adapt(candidate: dict, *, topic: str | None = None,
          format_fit: str = "youtube_shorts", journal: bool = True) -> dict:
    """Map a candidate → validated beat_plan 1.3.0 and write it.

    Returns {"ok": bool, "bp_id"?, "path"?, "reason"?}. Never raises for expected
    rejects. Journals stage "SE:I".
    """
    media_id = candidate.get("candidate_id", "_corpus")

    def _reject(reason: str) -> dict:
        if journal:
            runlog.log_event("SE:I", media_id, "skip", {"reason": reason})
        return {"ok": False, "reason": reason}

    if candidate.get("verdict") != "complete":
        return _reject(f"verdict={candidate.get('verdict')}")
    if candidate.get("confidence_tier") not in _AUTO_ELIGIBLE:
        return _reject("tier not auto-eligible")
    beats_in = candidate.get("beats") or []
    if not beats_in:
        return _reject("no beats")

    evidence_cache: Dict[str, Dict[int, Dict[str, int]]] = {}
    built_beats: List[dict] = []
    for i, beat in enumerate(beats_in):
        clip_id = beat["clip_id"]
        if clip_id not in evidence_cache:
            try:
                evidence_cache[clip_id] = _load_evidence_sentences(clip_id)
            except ValueError as e:
                return _reject(str(e))
        sents = evidence_cache[clip_id]

        a, b = int(beat["sentence_range"][0]), int(beat["sentence_range"][1])
        er = _edit_role(beat)
        a, b = _complete_quote(clip_id, a, b, er, sents)
        if a not in sents or b not in sents:
            return _reject(f"beat {i}: sentence idx out of range")
        in_ms, out_ms = sents[a]["start_ms"], sents[b]["end_ms"]
        if out_ms <= in_ms:
            return _reject(f"beat {i}: nonpositive duration")

        built_beats.append({
            "beat_index": i,
            "chunk_id": _resolve_chunk_id(clip_id, in_ms, out_ms),
            "media_id": clip_id,
            "in_ms": in_ms,
            "out_ms": out_ms,
            "edit_role": er,
            "note": beat.get("why_this_beat") or None,
        })

    # lane: duration-band sanity check, then DROPPED (D2) — never written to the plan
    total_ms = sum(o["out_ms"] - o["in_ms"] for o in built_beats)
    lane = candidate.get("primary_format_lane")
    band = _LANE_BAND_MS.get(lane) if lane else None
    if band and not (band[0] <= total_ms <= band[1]):
        return _reject(f"duration {total_ms}ms outside lane {lane} band {band}")

    now = datetime.datetime.now(datetime.timezone.utc)
    bp_id = f"bp_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    bp = {
        "schema_version": "1.3.0",
        "bp_id": bp_id,
        "created_at_utc": now.isoformat(),
        "revision": 0,
        "supersedes_revision": None,
        "topic": topic or candidate.get("title") or candidate.get("central_question") or "untitled",
        "format_fit": format_fit,
        "duration_target_s": max(1, round(total_ms / 1000)),
        "beats": built_beats,
        "total_duration_s": round(total_ms / 1000, 2),
        "assembler_note": f"se_adapter from candidate {candidate.get('candidate_id')}",
        "template_id": None,
    }

    try:
        contracts.validate(bp, "beat_plan")
    except Exception as e:  # jsonschema.ValidationError — our mapping produced an invalid plan
        if journal:
            runlog.log_event("SE:I", media_id, "error", {"reason": f"schema: {e}", "bp_id": bp_id})
        return {"ok": False, "reason": f"schema: {e}"}

    out_path = CAT / "beat_plans" / f"{bp_id}.jsonl"
    jsonl_append(out_path, bp)
    if journal:
        runlog.log_event("SE:I", media_id, "ok",
                         {"bp_id": bp_id, "n_beats": len(built_beats), "total_ms": total_ms, "lane": lane})
    return {"ok": True, "bp_id": bp_id, "path": str(out_path)}

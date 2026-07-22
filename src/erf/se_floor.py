"""SE:C2 hard-floor validator — deterministic, no model calls.

Runs before any Pro critic call. Rejects fabricated/array-index IDs,
range errors, audio contradictions, and lane violations.
"""
from __future__ import annotations

_LANE_BANDS_S: dict[str, tuple[float, float]] = {
    "single_clip_15_25s": (15, 25),
    "mini_cross_media_30_60s": (30, 60),
    "long_cross_media_60_120s": (60, 120),
}
_VALID_VERDICTS = {"complete", "reject", "partial_open_thread"}
_VALID_TIERS = {"pro_scored", "pro_scored_tournament_ranked", "flash_scored_only", "mutation_unchecked"}
_VALID_AUDIO = {"sync_dialogue", "vo", "broll_audio", "music_only", "silent_visual"}


def _is_array_index(cid: object) -> bool:
    """True if clip_id looks like a bare integer index (Pro hallucination pattern)."""
    if isinstance(cid, int):
        return True
    return isinstance(cid, str) and cid.strip().lstrip("-").isdigit()


def build_ev_idx(evidence: dict) -> dict[int, dict]:
    """Build {sentence_index: sentence_dict} from an evidence record."""
    return {s["sentence_index"]: s for s in evidence.get("sentences", [])}


def validate_candidate(
    cand: dict,
    valid_ids: set[str],
    ev_idx: dict[str, dict[int, dict]],
    *,
    ingredient_clip_ids: set[str] | None = None,
) -> list[str]:
    """Return list of validation errors (empty = pass). No model calls."""
    errs: list[str] = []

    for f in ("candidate_id", "verdict", "primary_format_lane", "confidence_tier"):
        if not cand.get(f):
            errs.append(f"missing {f}")

    verdict = cand.get("verdict")
    if verdict not in _VALID_VERDICTS:
        errs.append(f"bad verdict: {verdict!r}")
    lane = cand.get("primary_format_lane")
    if lane not in _LANE_BANDS_S:
        errs.append(f"bad lane: {lane!r}")
    if cand.get("confidence_tier") not in _VALID_TIERS:
        errs.append(f"bad confidence_tier: {cand.get('confidence_tier')!r}")

    if verdict != "complete":
        return errs

    beats = cand.get("beats") or []
    if not beats:
        errs.append("complete but no beats")
        return errs

    lo, hi = _LANE_BANDS_S.get(lane, (0, 9999))
    total_s = 0.0
    clip_ids_used: set[str] = set()

    for i, b in enumerate(beats):
        cid = b.get("clip_id")
        if _is_array_index(cid):
            errs.append(f"beat {i}: array-index-as-ID {cid!r} (Pro must not originate IDs)")
            continue
        if not cid or cid not in valid_ids:
            errs.append(f"beat {i}: fabricated/unknown clip_id {cid!r}")
            continue
        if ingredient_clip_ids is not None and cid not in ingredient_clip_ids:
            errs.append(f"beat {i}: clip_id {cid!r} not in ingredient inputs (Pro-originated ID)")

        clip_ids_used.add(cid)
        sr = b.get("sentence_range")
        if not (isinstance(sr, list) and len(sr) == 2 and all(isinstance(x, int) for x in sr)):
            errs.append(f"beat {i}: sentence_range must be [int,int], got {sr!r}")
            continue
        a, z = sr
        if a > z:
            errs.append(f"beat {i}: sentence_range[0]>{sr[1]}")
            continue
        sent_map = ev_idx.get(cid, {})
        missing = next((idx for idx in range(a, z + 1) if idx not in sent_map), None)
        if missing is not None:
            errs.append(f"beat {i}: sentence {missing} not in evidence for {cid}")
        elif a in sent_map and z in sent_map:
            total_s += (sent_map[z]["end_ms"] - sent_map[a]["start_ms"]) / 1000

        am = b.get("audio_mode")
        if am and am not in _VALID_AUDIO:
            errs.append(f"beat {i}: unknown audio_mode {am!r}")
        # Auto-repair sync_audio_required and dialogue_required from audio_mode (deterministic derivation)
        if am == "sync_dialogue":
            b["sync_audio_required"] = True
            b["dialogue_required"] = True
        elif am in ("silent_visual", "broll_audio", "music_only", "vo"):
            b.setdefault("sync_audio_required", False)
            b.setdefault("dialogue_required", False)
        sar = b.get("sync_audio_required", False)
        dr = b.get("dialogue_required", False)
        if am == "silent_visual" and dr:
            errs.append(f"beat {i}: silent_visual + dialogue_required contradiction")
        if dr and am != "sync_dialogue":
            errs.append(f"beat {i}: dialogue_required=true but audio_mode={am!r}")
        if b.get("payoff_type") == "visual_proof" and not b.get("visual_support"):
            errs.append(f"beat {i}: payoff_type=visual_proof requires visual_support")

    has_id_errs = any("fabricated" in e or "array-index" in e for e in errs)
    if not has_id_errs and lane in _LANE_BANDS_S and not (lo <= total_s <= hi):
        errs.append(f"total {total_s:.1f}s outside {lane} band ({lo}-{hi}s)")

    nc = len(clip_ids_used)
    if lane == "single_clip_15_25s" and nc != 1:
        errs.append(f"single_clip needs exactly 1 clip_id, got {nc}")
    if lane == "mini_cross_media_30_60s" and not (2 <= nc <= 4):
        errs.append(f"mini needs 2-4 clip_ids, got {nc}")
    if lane == "long_cross_media_60_120s" and nc < 3:
        errs.append(f"long needs 3+ clip_ids, got {nc}")

    return errs

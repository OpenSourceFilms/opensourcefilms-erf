#!/usr/bin/env python3
"""
build_beat_plan.py — §4 pacing/assembly layer for ERF storyfirst redesign.

For single-source proposals (≥60% of anchors from one media): sequential
phrase-boundary carving through the dominant media's ASR timeline.

For multi-source proposals: per-beat embedding retrieval with top-K
candidate fallback.
"""
from __future__ import annotations

import sys, json, os, argparse, math, hashlib, datetime
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from erf.scan import load_all_embeddings
from erf.query import embed_query

CATALOG               = Path("/workspace/erf/catalog")
STATE                 = Path("/workspace/erf/state")
RELEVANCE_FLOOR       = 0.40
TEXT_FLOOR_SAME_MEDIA = 0.30   # relaxed floor for same-preferred-media events (Half 2: rescue correct events below text floor)
VISUAL_COHERENCE_FLOOR = 0.35  # cross-media keyframe visual gate (Half 1: reject visually-incompatible candidates)
BOOST_EMOTIONAL       = 0.05
BOOST_SIGNAL          = 0.04
BOOST_SPEAKER         = 0.03
ANCHOR_BOOST          = 0.15
DOMINANT_MEDIA_RATIO  = 0.60
CUT_RHYTHM_FLOOR_S    = 3.0
TOP_K                 = 8


# ── helpers ───────────────────────────────────────────────────────────────────

def load_visual_kf_vecs(catalog: Path) -> Dict[str, List[np.ndarray]]:
    """Load per-media keyframe visual embeddings from catalog/embeddings/{mid}.visual.jsonl.

    Returns {media_id: [unit-normalized kf vec, ...]}. Normalisation applied here
    so downstream dot-products are cosine similarities directly.
    """
    result: Dict[str, List[np.ndarray]] = {}
    emb_dir = catalog / "embeddings"
    if not emb_dir.exists():
        return result
    for fpath in emb_dir.glob("*.visual.jsonl"):
        mid = fpath.stem.replace(".visual", "")
        vecs: List[np.ndarray] = []
        for line in fpath.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            v = row.get("vector", [])
            if v:
                arr = np.array(v, dtype=np.float32)
                n = np.linalg.norm(arr)
                if n > 0:
                    vecs.append(arr / n)
        if vecs:
            result[mid] = vecs
    return result

def load_event_meta(catalog: Path) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    for line in (catalog / "event_digest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        meta[ev["event_id"]] = ev
    return meta


def load_asr_words(media_id: str, catalog: Path) -> List[dict]:
    path = catalog / "asr" / f"{media_id}.words.jsonl"
    if not path.exists():
        return []
    words = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    words.sort(key=lambda w: w["start_ms"])
    return words


def find_containing_event(t_ms: int, media_id: str, event_meta: Dict[str, dict]) -> str:
    """Return event_id whose time range best contains t_ms for this media.
    Falls back to the nearest event if t_ms is before all events (e.g., sub-250ms words).
    """
    best_id, best_dur = "unknown", float("inf")
    nearest_id, nearest_gap = "unknown", float("inf")
    for eid, ev in event_meta.items():
        if ev.get("media_id") != media_id:
            continue
        ev_s = int(ev["t_s"] * 1000)
        ev_e = ev_s + int(ev["duration_s"] * 1000)
        if ev_s <= t_ms < ev_e:
            dur = ev_e - ev_s
            if dur < best_dur:
                best_id, best_dur = eid, dur
        gap = abs(ev_s - t_ms)
        if gap < nearest_gap:
            nearest_id, nearest_gap = eid, gap
    return best_id if best_id != "unknown" else nearest_id


# ── sequential carver (single-source proposals) ───────────────────────────────

def carve_sequential(
    beats: List[dict],
    words: List[dict],
    media_id: str,
    event_meta: Dict[str, dict],
) -> List[dict]:
    """
    Assign each beat a non-overlapping word-boundary-snapped time window by
    walking sequentially through the media's ASR words.

    Cut rule: advance to the first word whose end_ms >= target_end_ms.
    That word's end is the cut point (natural word boundary, never mid-word).
    If the window is still below min_duration_s, keep absorbing words.

    Segment-end snapping is intentionally avoided: some ASR segments are 8s+
    which would blow the beat budget for 4s target beats.
    """
    planned: List[dict] = []
    word_ptr    = 0
    total_words = len(words)

    role_map = {
        "hook": "hook", "setup": "exposition", "body": "body",
        "tension": "body", "reveal": "body", "payoff": "body", "b_roll": "b_roll",
    }

    for beat in beats:
        if word_ptr >= total_words:
            print(f"  SKIP beat {beat['beat_index']}: words exhausted", file=sys.stderr)
            continue

        min_ms    = int(beat.get("min_duration_s", 2.5) * 1000)
        target_ms = int(beat.get("target_duration_s", 4.0) * 1000)

        in_ms         = words[word_ptr]["start_ms"]
        target_end_ms = in_ms + target_ms
        out_ms        = in_ms
        ptr           = word_ptr

        # Greedy: take all words ending at or before target boundary
        while ptr < total_words and words[ptr]["end_ms"] <= target_end_ms:
            out_ms = words[ptr]["end_ms"]
            ptr   += 1

        # If nothing consumed (first word already past target), take it
        if out_ms == in_ms and ptr < total_words:
            out_ms = words[ptr]["end_ms"]
            ptr   += 1

        # Extend for minimum duration if still short
        while (out_ms - in_ms) < min_ms and ptr < total_words:
            out_ms = words[ptr]["end_ms"]
            ptr   += 1

        dur_s     = (out_ms - in_ms) / 1000.0
        eid       = find_containing_event(in_ms, media_id, event_meta)
        ev_meta   = event_meta.get(eid, {})
        summary   = (ev_meta.get("summary") or "")[:60]
        edit_role = role_map.get(beat.get("role", "body"), "body")

        planned.append({
            "beat_index":           beat["beat_index"],
            "chunk_id":             f"{media_id}:c:{in_ms}-{out_ms}",
            "media_id":             media_id,
            "in_ms":                in_ms,
            "out_ms":               out_ms,
            "edit_role":            edit_role,
            "note":                 f"{eid} (sequential) | {summary}",
            "hook_score":           None,
            "quote_score":          None,
            "proof_score":          None,
            "suggested_padding_ms": None,
            "duration_s":           round(dur_s, 3),
            "b_roll_candidates":    None,
            "b_roll_selected":      None,
            "broll":                {"enabled": False, "keyframe_id": None, "reason": None},
            "_ptr":                 ptr,
        })
        word_ptr = ptr

    return planned


# ── retrieval helpers (multi-source proposals) ───────────────────────────────

def fused(base: float, ev_meta: dict, spec: dict) -> float:
    score = base
    eb_target = (spec.get("emotional_beat_target") or "").lower()
    if eb_target and eb_target in (ev_meta.get("emotional_beat") or "").lower():
        score += BOOST_EMOTIONAL
    if spec.get("want_hook") and float(ev_meta.get("hook_potential") or 0) >= 0.7:
        score += BOOST_SIGNAL
    if spec.get("want_quote") and float(ev_meta.get("quote_potential") or 0) >= 0.7:
        score += BOOST_SIGNAL
    hint = (spec.get("speaker_hint") or "").lower()
    if hint and hint in (ev_meta.get("speaker_name") or "").lower():
        score += BOOST_SPEAKER
    return score


def retrieve_topk(
    beat: dict,
    all_vecs: Dict[str, np.ndarray],
    event_meta: Dict[str, dict],
    state: Path,
    anchor_ids: List[str],
    k: int = TOP_K,
    preferred_media: Optional[str] = None,
    visual_kf_vecs: Optional[Dict[str, List[np.ndarray]]] = None,
) -> List[Tuple[str, float]]:
    """Score and filter event candidates for a single beat.

    Dual-admission criteria when preferred_media is set:
      Same-media candidates: admitted at TEXT_FLOOR_SAME_MEDIA (0.30) — rescues visually
        grounded events whose text embedding is below the standard floor because the
        visual signal (what they show) dominates their relevance, not their text summary.
      Cross-media candidates: must clear BOTH RELEVANCE_FLOOR (0.40) AND
        VISUAL_COHERENCE_FLOOR (0.35) as max-cosine against preferred_media keyframes —
        rejects candidates whose text summary shares vocabulary with the query but whose
        footage is from an incompatible visual context.

    Decision audit trail logged to stderr per candidate.
    """
    spec  = beat.get("retrieval_spec") or {}
    query = spec.get("semantic_query") or ""
    if not query:
        return []
    intent = beat.get("narrative_intent") or ""
    explicit_anchors = {eid for eid in anchor_ids if eid in intent}
    qvec = np.array(embed_query(query, state), dtype=np.float32)
    n = np.linalg.norm(qvec)
    if n > 0:
        qvec /= n

    scored = []
    for eid, evec in all_vecs.items():
        base  = float(np.dot(qvec, evec))
        score = fused(base, event_meta.get(eid, {}), spec)
        if eid in explicit_anchors:
            score += ANCHOR_BOOST
        scored.append((eid, score))
    scored.sort(key=lambda x: -x[1])

    # Pre-fetch reference keyframes for visual coherence check
    ref_kfs: List[np.ndarray] = (visual_kf_vecs or {}).get(preferred_media, []) if preferred_media else []

    admitted: List[Tuple[str, float]] = []
    for eid, score in scored:
        mid = event_meta.get(eid, {}).get("media_id", "")
        is_same = bool(preferred_media and mid == preferred_media)

        if is_same:
            # Half 2: relaxed floor admits same-media events the text floor would exclude
            if score >= TEXT_FLOOR_SAME_MEDIA:
                admitted.append((eid, score))
            # (events below TEXT_FLOOR_SAME_MEDIA are silently skipped — true noise)
        elif preferred_media and ref_kfs:
            # Half 1: cross-media must clear text floor AND visual coherence floor
            if score < RELEVANCE_FLOOR:
                continue
            cand_kfs: List[np.ndarray] = (visual_kf_vecs or {}).get(mid, [])
            if cand_kfs:
                vc = max(float(np.dot(a, b)) for a in ref_kfs for b in cand_kfs)
                if vc < VISUAL_COHERENCE_FLOOR:
                    print(
                        f"    REJECT {eid} text={score:.3f} visual_coherence={vc:.3f}"
                        f" < {VISUAL_COHERENCE_FLOOR} (cross-media visual gate)",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"    ADMIT  {eid} text={score:.3f} visual_coherence={vc:.3f} (cross-media)",
                    file=sys.stderr,
                )
            else:
                # No kf data for candidate — fall back to text-only admission
                print(
                    f"    ADMIT  {eid} text={score:.3f} visual_coherence=N/A no-kf (cross-media)",
                    file=sys.stderr,
                )
            admitted.append((eid, score))
        else:
            # No preferred_media or no ref kfs: text floor only (original behaviour)
            if score >= RELEVANCE_FLOOR:
                admitted.append((eid, score))

    if not admitted:
        return []

    # preferred-media candidates first, cross-media as last resort
    if preferred_media:
        pref  = [(eid, sc) for eid, sc in admitted if event_meta.get(eid, {}).get("media_id") == preferred_media]
        other = [(eid, sc) for eid, sc in admitted if event_meta.get(eid, {}).get("media_id") != preferred_media]
        return (pref + other)[:k]
    return admitted[:k]


def snap_phrase_boundary(
    words: List[dict], ev_end_ms: int, in_ms: int, target_ms: int, min_ms: int,
) -> Tuple[int, int]:
    candidates = [w for w in words if w["start_ms"] >= in_ms and w["start_ms"] < ev_end_ms]
    if not candidates:
        out = min(in_ms + target_ms, ev_end_ms)
        return (in_ms, out) if out > in_ms else (0, 0)
    in_ms = candidates[0]["start_ms"]
    target_end = in_ms + target_ms
    last = None
    for w in candidates:
        if w["end_ms"] <= target_end:
            last = w
        else:
            break
    out_ms = last["end_ms"] if last is not None else candidates[0]["end_ms"]
    if (out_ms - in_ms) < min_ms:
        idx = candidates.index(last) if last is not None else 0
        for w in candidates[idx + 1:]:
            if w["end_ms"] <= ev_end_ms:
                out_ms = w["end_ms"]
                if (out_ms - in_ms) >= min_ms:
                    break
    out_ms = min(out_ms, ev_end_ms)
    return (in_ms, out_ms) if out_ms > in_ms else (0, 0)


def carve_retrieval(
    beats: List[dict],
    all_vecs: Dict[str, np.ndarray],
    event_meta: Dict[str, dict],
    state: Path,
    anchor_ids: List[str],
    catalog: Path,
    preferred_media: Optional[str] = None,
    visual_kf_vecs: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Tuple[List[dict], List[dict]]:
    role_map = {
        "hook": "hook", "setup": "exposition", "body": "body",
        "tension": "body", "reveal": "body", "payoff": "body", "b_roll": "b_roll",
    }
    media_words: Dict[str, List[dict]] = {}
    cursor: Dict[str, int] = {}
    planned: List[dict] = []
    skipped: List[dict] = []

    for beat in beats:
        candidates = retrieve_topk(beat, all_vecs, event_meta, state, anchor_ids,
                                    preferred_media=preferred_media,
                                    visual_kf_vecs=visual_kf_vecs)
        if not candidates:
            print(f"  SKIP beat {beat['beat_index']}: no candidate above floor", file=sys.stderr)
            skipped.append({"beat_index": beat["beat_index"], "reason": "no_candidate_above_floor"})
            continue
        min_ms    = int(beat.get("min_duration_s", 2.5) * 1000)
        target_ms = int(beat.get("target_duration_s", 4.0) * 1000)
        assigned  = False
        for eid, score in candidates:
            ev  = event_meta[eid]
            mid = ev["media_id"]
            ev_s = int(ev["t_s"] * 1000)
            ev_e = ev_s + int(ev["duration_s"] * 1000)
            if mid not in media_words:
                media_words[mid] = load_asr_words(mid, catalog)
            in_ms = max(ev_s, cursor.get(mid, 0))
            if in_ms >= ev_e:
                continue
            # Try within the declared event boundary first
            snap_in, snap_out = snap_phrase_boundary(media_words[mid], ev_e, in_ms, target_ms, min_ms)
            # If event is shorter than min_ms, extend forward through the media's
            # word list past ev_e. The event is the semantic anchor (WHERE to start
            # cutting); minimum duration may require content beyond its boundary.
            if snap_out <= snap_in or (snap_out - snap_in) < min_ms:
                words_after = media_words[mid]
                ptr = next(
                    (i for i, w in enumerate(words_after) if w["start_ms"] >= (snap_in if snap_in > in_ms else in_ms)),
                    len(words_after),
                )
                snap_in = words_after[ptr]["start_ms"] if ptr < len(words_after) else in_ms
                snap_out = snap_in
                # Walk words forward until min_ms met or target reached.
                # Guard: if the next available word starts beyond the target window,
                # stop — don't leap over silence gaps larger than the target duration.
                tgt_end = snap_in + target_ms
                while ptr < len(words_after):
                    w = words_after[ptr]
                    # Don't cross another media cursor (would create overlap)
                    if w["start_ms"] < cursor.get(mid, 0):
                        ptr += 1
                        continue
                    # Don't leap over a silence gap wider than the target window
                    if w["start_ms"] > tgt_end and snap_out > snap_in:
                        break
                    snap_out = w["end_ms"]
                    ptr += 1
                    if w["end_ms"] >= tgt_end:
                        break
                # Ensure minimum — only through words that start within target window
                while (snap_out - snap_in) < min_ms and ptr < len(words_after) and words_after[ptr]["start_ms"] <= tgt_end:
                    snap_out = words_after[ptr]["end_ms"]
                    ptr += 1
            if snap_out <= snap_in or (snap_out - snap_in) < min_ms:
                continue
            cursor[mid] = snap_out
            dur_s = (snap_out - snap_in) / 1000.0
            planned.append({
                "beat_index":        beat["beat_index"],
                "chunk_id":          f"{mid}:c:{snap_in}-{snap_out}",
                "media_id":          mid,
                "in_ms":             snap_in,
                "out_ms":            snap_out,
                "edit_role":         role_map.get(beat.get("role", "body"), "body"),
                "note":              f"{eid} score={score:.3f} | {(ev.get('summary') or '')[:60]}",
                "hook_score":        None, "quote_score": None, "proof_score": None,
                "suggested_padding_ms": None,
                "duration_s":        round(dur_s, 3),
                "b_roll_candidates": None, "b_roll_selected": None,
                "broll":             {"enabled": False, "keyframe_id": None, "reason": None},
                "_eid":              eid,
            })
            assigned = True
            break
        if not assigned:
            print(f"  SKIP beat {beat['beat_index']}: all candidates exhausted", file=sys.stderr)
            skipped.append({"beat_index": beat["beat_index"], "reason": "all_candidates_exhausted"})

    return planned, skipped


# ── cut-rhythm enforcement ────────────────────────────────────────────────────

def enforce_cut_rhythm(
    planned: List[dict],
    event_meta: Dict[str, dict],
    media_words: Dict[str, List[dict]],
) -> int:
    fixes = 0
    i = 0
    while i < len(planned):
        j = i
        while j < len(planned) and planned[j]["duration_s"] < CUT_RHYTHM_FLOOR_S:
            j += 1
        if (j - i) >= 3:
            viol = planned[i + 2]
            eid  = viol.get("_eid") or viol["note"].split()[0]
            ev   = event_meta.get(eid, {})
            ev_e = int(ev.get("t_s", 0) * 1000) + int(ev.get("duration_s", 0) * 1000)
            words = media_words.get(viol["media_id"], [])
            needed = viol["in_ms"] + int(CUT_RHYTHM_FLOOR_S * 1000)
            words_after = [w for w in words if w["start_ms"] >= viol["out_ms"] and w["end_ms"] <= ev_e]
            new_out = viol["out_ms"]
            for w in words_after:
                new_out = w["end_ms"]
                if new_out - viol["in_ms"] >= int(CUT_RHYTHM_FLOOR_S * 1000):
                    break
            if new_out > viol["out_ms"]:
                viol["out_ms"]    = new_out
                viol["duration_s"] = round((new_out - viol["in_ms"]) / 1000.0, 3)
                viol["chunk_id"]  = f"{viol['media_id']}:c:{viol['in_ms']}-{new_out}"
                fixes += 1
            i = i + 3
        else:
            i = j + 1 if j > i else i + 1
    return fixes


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-id", default="sto_d26d4fd93137")
    parser.add_argument("--catalog", default=str(CATALOG))
    parser.add_argument("--state",   default=str(STATE))
    args = parser.parse_args()

    catalog = Path(args.catalog)
    state   = Path(args.state)

    # 1. Load proposal
    prop_path = catalog / "story_proposals" / f"{args.proposal_id}.json"
    if not prop_path.exists():
        raise FileNotFoundError(prop_path)
    proposal    = json.loads(prop_path.read_text())
    target_s    = proposal["target_duration_s"]
    p_beats     = proposal["beats"]
    anchor_ids  = proposal.get("anchor_event_ids", [])
    format_fit  = proposal.get("format_fit", "youtube_shorts")
    max_beats   = math.ceil(target_s / 3)

    # 2. Detect dominant media
    anchor_media = [eid.split(":")[0] for eid in anchor_ids]
    dominant_media: Optional[str] = None
    if anchor_media:
        top_mid, top_cnt = Counter(anchor_media).most_common(1)[0]
        if top_cnt / len(anchor_media) >= DOMINANT_MEDIA_RATIO:
            dominant_media = top_mid

    print(f"Proposal : {proposal['title']}")
    print(f"  {len(p_beats)} beats | target {target_s}s | max allowed {max_beats}")
    print(f"  Dominant media: {dominant_media or 'none (multi-source)'}")

    n_beats = min(len(p_beats), max_beats)
    if n_beats < len(p_beats):
        print(f"  WARNING: proposal has {len(p_beats)} beats but max_beats={max_beats} "
              f"(target {target_s}s / {int(CUT_RHYTHM_FLOOR_S)}s rhythm) — "
              f"dropping {len(p_beats) - n_beats} tail beat(s)", file=sys.stderr)
    beats_to_plan = p_beats[:n_beats]

    # 3. Load event metadata
    print("\nLoading event metadata…")
    event_meta = load_event_meta(catalog)
    print(f"  {len(event_meta)} events")

    # 3b. Override sequential if anchors cluster in final 30% of dominant media.
    # Sequential carving distributes beats uniformly — it buries content when the
    # interesting moments are temporally concentrated near the end of the source.
    if dominant_media:
        dom_anchor_eids = [eid for eid in anchor_ids if eid.startswith(dominant_media)]
        if dom_anchor_eids:
            dom_max_ms = max(
                int(ev["t_s"] * 1000) + int(ev["duration_s"] * 1000)
                for ev in event_meta.values()
                if ev.get("media_id") == dominant_media
            )
            threshold_ms = dom_max_ms * 0.70
            in_tail = sum(
                1 for eid in dom_anchor_eids
                if int(event_meta.get(eid, {}).get("t_s", 0) * 1000) >= threshold_ms
            )
            tail_ratio = in_tail / len(dom_anchor_eids)
            print(f"  Anchor tail check: {in_tail}/{len(dom_anchor_eids)} anchors in final 30% of {dom_max_ms}ms")
            if tail_ratio >= 0.50:
                print(f"  → anchors clustered at end — overriding to per-beat retrieval")
                dominant_media = None

    # preferred_media: preserved for retrieval ordering even when sequential is
    # overridden. Keeps cross-media as last resort for single-source proposals.
    preferred_media = dominant_media or (
        Counter(anchor_media).most_common(1)[0][0] if anchor_media else None
    )

    # 4. Branch: sequential carving (single-source) vs retrieval (multi-source)
    if dominant_media:
        print(f"\nMode: sequential carving through {dominant_media}")
        words = load_asr_words(dominant_media, catalog)
        print(f"  {len(words)} ASR words loaded ({words[-1]['end_ms']/1000:.1f}s media)")
        planned = carve_sequential(beats_to_plan, words, dominant_media, event_meta)
        media_words = {dominant_media: words}
        gate_skipped: List[dict] = []
        for b in planned:
            b.pop("_ptr", None)
    else:
        print(f"\nMode: per-beat embedding retrieval  preferred_media={preferred_media}")
        all_vecs = load_all_embeddings(catalog)
        print(f"  {len(all_vecs)} event vectors loaded")
        visual_kf_vecs = load_visual_kf_vecs(catalog)
        print(f"  {len(visual_kf_vecs)} media with visual keyframe embeddings loaded")
        ref_kf_count = len(visual_kf_vecs.get(preferred_media, []))
        print(f"  reference keyframes for {preferred_media}: {ref_kf_count}")
        planned, gate_skipped = carve_retrieval(beats_to_plan, all_vecs, event_meta, state, anchor_ids, catalog,
                                                preferred_media=preferred_media, visual_kf_vecs=visual_kf_vecs)
        media_words = {}  # already loaded inside carve_retrieval

    # 5. Cut-rhythm enforcement
    rhythm_fixes = enforce_cut_rhythm(planned, event_meta, media_words)
    for b in planned:
        b.pop("_eid", None)

    # 6. Build beat_plan
    now    = datetime.datetime.now(datetime.timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")
    hex6   = hashlib.blake2b((args.proposal_id + ts_str).encode(), digest_size=3).hexdigest()
    bp_id  = f"bp_{ts_str}_{hex6}"
    total_s = round(sum(b["duration_s"] for b in planned), 3)

    # Duration-cap pre-check: warn at plan time if a vertical/shorts plan exceeds
    # the render duration ceiling (render.py truncates to it). Surfaces a
    # confidently-wrong duration here rather than after a silent render truncation.
    _VERTICAL_FORMATS = {"tiktok", "reels", "youtube_shorts"}  # mirror render._VERTICAL_PLATFORMS
    cap_s = 60.0
    try:
        import tomllib
        with open(state / "config.toml", "rb") as _fh:
            cap_s = float(tomllib.load(_fh).get("render", {}).get("max_duration_s", 60.0))
    except Exception:
        pass
    if format_fit in _VERTICAL_FORMATS and total_s > cap_s + 0.05:
        print(f"  WARNING: total {total_s}s exceeds {format_fit} render cap {cap_s}s "
              f"— render will TRUNCATE {round(total_s - cap_s, 3)}s of content", file=sys.stderr)

    beat_count_ok = len(planned) <= max_beats
    assembler_note = (
        f"beats: {len(planned)}/{n_beats} | beat_count: {'PASS' if beat_count_ok else 'FAIL'} | "
        f"cut_rhythm_fixes: {rhythm_fixes} | mode: {'sequential' if dominant_media else 'retrieval'} | "
        f"total_s: {total_s}"
    )

    beat_plan = {
        "schema_version":      "1.3.0",
        "bp_id":               bp_id,
        "created_at_utc":      ts_str,
        "revision":            0,
        "supersedes_revision": None,
        "topic":               proposal["title"],
        "format_fit":          format_fit,
        "duration_target_s":   target_s,
        "beats":               planned,
        "total_duration_s":    total_s,
        "assembler_note":      assembler_note,
        "gate_skipped_beats":  gate_skipped,
        "template_id":         None,
    }

    # 7. Atomic write
    out_dir = catalog / "beat_plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bp_id}.jsonl"
    tmp      = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(beat_plan))
    os.replace(tmp, out_path)
    print(f"\nWrote: {out_path}\n")

    # 8. Print per-beat report
    hdr = f"{'#':>3}  {'event_id':<30}  {'in_ms':>7}  {'out_ms':>7}  {'dur':>5}  {'role':<10}  constraint"
    print(hdr)
    print("─" * len(hdr))
    short_run = 0
    min_violations = 0
    rhythm_violations = 0
    for b in planned:
        eid_tok = b["note"].split()[0]
        dur = b["duration_s"]
        short_run = (short_run + 1) if dur < CUT_RHYTHM_FLOOR_S else 0
        min_s = 1.5 if b["edit_role"] in ("hook", "b_roll") else 2.5
        flags = []
        if dur < min_s:
            flags.append(f"!min({min_s:.1f}s)")
            min_violations += 1
        if short_run >= 3:
            flags.append("!rhythm")
            rhythm_violations += 1
        flag_str = "  " + " ".join(flags) if flags else "  ✓"
        print(
            f"{b['beat_index']:3}  {eid_tok:<30}  {b['in_ms']:7}  {b['out_ms']:7}  "
            f"{dur:5.2f}  {b['edit_role']:<10}  {flag_str}"
        )

    print(f"\n{'─'*55}")
    print(f"Total     : {total_s}s  (target {target_s}s, delta {total_s - target_s:+.1f}s)")
    print(f"Beats     : {len(planned)}/{n_beats}  ({'PASS' if beat_count_ok else 'FAIL'}  ≤ ceil({target_s}/3)={max_beats})")
    print(f"Min-dur   : {len(planned) - min_violations}/{len(planned)} pass  ({min_violations} violation(s))")
    print(f"Cut-rhythm: {'PASS' if rhythm_violations == 0 else f'FAIL ({rhythm_violations} run(s) > 2 short beats)'}  ({rhythm_fixes} fix(es) applied)")
    print(f"bp_id     : {bp_id}")


if __name__ == "__main__":
    main()

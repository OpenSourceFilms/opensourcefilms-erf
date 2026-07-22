"""Render candidates from se_complete.py output through se_adapter + render.

Reads candidates JSONL, adapts each via Seam 1, renders via the existing renderer.
SE-3 taste gate: the project lead eyeballs the outputs.

Colour architecture: render_timeline is built from speech_events BEFORE render so the
renderer receives pre-computed speaker→colour mapping (speaker_colors_override).
The renderer is dumb — it never infers colours inline for SE candidates.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAT = ROOT / "catalog"
sys.path.insert(0, str(ROOT / "src"))

from erf.se_adapter import adapt as se_adapt
from erf.render import render as do_render, _PERSON_COLORS, _FALLBACK_SPK_COLORS
from erf.runlog import log_event

_AUTO_ELIGIBLE = {"pro_scored", "pro_scored_tournament_ranked"}
_SENT_END = re.compile(r"[.?!…]$")

# Below this duration a cue flashes too briefly to read and reads as flicker;
# such cues are merged into an adjacent same-colour cue (presentation only).
MIN_CUE_MS = 350


def _smooth_gap_words(events: list[dict]) -> list[dict]:
    """Return a copy of events where a speaker-less word bounded by the SAME
    speaker on both sides inherits that speaker. Kills single-word colour flashes
    (e.g. an unattributed 'Uh,' between two turns of one speaker). Render-only;
    canonical speech_events are untouched."""
    ev = [dict(e) for e in events]
    for i, e in enumerate(ev):
        if e.get("speaker_label"):
            continue
        prev = next((ev[j].get("speaker_label") for j in range(i - 1, -1, -1)
                     if ev[j].get("speaker_label")), None)
        nxt = next((ev[j].get("speaker_label") for j in range(i + 1, len(ev))
                    if ev[j].get("speaker_label")), None)
        if prev is not None and prev == nxt:
            e["speaker_label"] = prev
    return ev


def _merge_micro_cues(cues: list[dict]) -> list[dict]:
    """Remove sub-readable flashing cues. Pass 1 folds a cue shorter than
    MIN_CUE_MS into the previous cue when they share a colour. Pass 2 folds any
    still-too-short cue into its temporally NEAREST same-beat neighbour (adopting
    that neighbour's speaker/colour), so a brief interjection at a speaker
    boundary attaches to whichever turn it abuts rather than flashing alone."""
    def _dur(c):
        return (c.get("cue_end_ms") or 0) - (c.get("cue_start_ms") or 0)

    # Pass 1: same-colour merge into previous.
    p1: list[dict] = []
    for c in cues:
        if (p1 and _dur(c) < MIN_CUE_MS
                and p1[-1].get("beat_index") == c.get("beat_index")
                and p1[-1].get("colour_id") == c.get("colour_id")):
            p = p1[-1]
            p["text"] = (p["text"] + " " + c["text"]).strip()
            p["cue_end_ms"] = c.get("cue_end_ms")
            p["speaker_conflict"] = p.get("speaker_conflict") or c.get("speaker_conflict")
            continue
        p1.append(c)

    # Pass 2: fold any remaining tiny cue into the nearest same-beat neighbour.
    out: list[dict] = []
    i = 0
    while i < len(p1):
        c = p1[i]
        if _dur(c) >= MIN_CUE_MS:
            out.append(c); i += 1; continue
        prev = out[-1] if out and out[-1].get("beat_index") == c.get("beat_index") else None
        nxt = p1[i + 1] if i + 1 < len(p1) and p1[i + 1].get("beat_index") == c.get("beat_index") else None
        gap_prev = (c.get("cue_start_ms") or 0) - (prev.get("cue_end_ms") or 0) if prev else None
        gap_next = (nxt.get("cue_start_ms") or 0) - (c.get("cue_end_ms") or 0) if nxt else None
        if prev is not None and (gap_next is None or (gap_prev is not None and gap_prev <= gap_next)):
            prev["text"] = (prev["text"] + " " + c["text"]).strip()
            prev["cue_end_ms"] = c.get("cue_end_ms")
            prev["speaker_conflict"] = prev.get("speaker_conflict") or c.get("speaker_conflict")
            i += 1
        elif nxt is not None:
            nxt["text"] = (c["text"] + " " + nxt["text"]).strip()
            nxt["cue_start_ms"] = c.get("cue_start_ms")
            nxt["speaker_conflict"] = nxt.get("speaker_conflict") or c.get("speaker_conflict")
            i += 1
        else:
            out.append(c); i += 1
    return out

log = logging.getLogger("se_render")


def _read_last_jsonl(path: Path) -> dict:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1])


def _resolve_colour(colour_id: str | None) -> str:
    """Convert colour_id (person_id or stable_unknown_id) to BGR hex."""
    if colour_id:
        if colour_id in _PERSON_COLORS:
            return _PERSON_COLORS[colour_id]
        return _FALLBACK_SPK_COLORS[abs(hash(colour_id)) % len(_FALLBACK_SPK_COLORS)]
    return _FALLBACK_SPK_COLORS[0]


def _load_speech_events(media_id: str) -> list[dict]:
    """Load per-word events for a media_id.

    Preference order: render_events_v2 (adjudicated stream A + confirmed
    stream B, recall-pass words folded in, person_id pre-resolved by
    build_parallel_streams.py) -> legacy speech_events (word-level,
    speaker_label only, no person_id) -> transcript.jsonl.
    """
    rev_path = CAT / "render_events_v2" / f"{media_id}.jsonl"
    if rev_path.exists():
        out: list[dict] = []
        for line in rev_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            evidence = ev.get("evidence") or {}
            conflict = len(set(evidence.values())) > 1
            for wi, w in enumerate(ev.get("words") or []):
                out.append({
                    "event_id": ev.get("event_id"),
                    "media_id": media_id,
                    "word_id": f"{ev.get('event_id')}:w{wi:03d}",
                    "text": w.get("text", ""),
                    "start_ms": w.get("start_ms", 0),
                    "end_ms": w.get("end_ms", 0),
                    "score": w.get("prob", 0.0),
                    "speaker_label": ev.get("diar_label"),
                    "person_id": ev.get("person_id"),
                    "stable_unknown_id": ev.get("stable_unknown_id"),
                    "evidence": evidence,
                    "confidence": ev.get("confidence", 1.0),
                    "gap_word": ev.get("diar_label") is None,
                    "speaker_conflict": conflict,
                })
        if out:
            out.sort(key=lambda e: e["start_ms"])
            return out
    se_path = CAT / "speech_events" / f"{media_id}.jsonl"
    if se_path.exists():
        return [json.loads(l) for l in se_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Fallback: transcript.jsonl (legacy, has speaker_label baked in from diar)
    tr_path = CAT / "asr" / f"{media_id}.transcript.jsonl"
    if tr_path.exists():
        rows = [json.loads(l) for l in tr_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [
            {
                "event_id": f"{media_id}:e:{i:06d}",
                "media_id": media_id,
                "word_id": r.get("word_id", f"{media_id}:w:{i:06d}"),
                "text": r.get("text", ""),
                "start_ms": r.get("start_ms", 0),
                "end_ms": r.get("end_ms", 0),
                "score": 0.0,
                "speaker_label": r.get("speaker_label"),
                "evidence": {},
                "confidence": 1.0,
                "gap_word": r.get("speaker_label") is None,
                "speaker_conflict": False,
            }
            for i, r in enumerate(rows)
        ]
    return []


def _build_render_timeline(
    candidate_id: str,
    beat_plan: dict,
    clips: list[dict],
    catalog: Path,
) -> list[dict]:
    """Build per-cue timeline from speech_events (pre-render, canonical colour source)."""
    # Cross-corpus voice identity (Phase 2 / build_identity_clusters.py): speaker_label ->
    # person_id, derived from WeSpeaker embedding clusters so it is label-permutation immune
    # and CONSISTENT ACROSS VIDEOS. person_id may be a gallery id (p_*) or a stable cluster
    # id for a recurring unenrolled person (p_unk_*); both give a stable corpus-wide colour.
    # Legacy identity_overlay is used only if the new map is absent for a media.
    identity_map: dict[str, dict[str, str | None]] = {}
    for clip in clips:
        mid = clip["media_id"]
        if mid in identity_map:
            continue
        m: dict[str, str | None] = {}
        im_path = catalog / "identity_map" / f"{mid}.json"
        if im_path.exists():
            try:
                data = json.loads(im_path.read_text())
                m = {lbl: info.get("person_id")
                     for lbl, info in data.get("speakers", {}).items()}
            except Exception:
                m = {}
        else:
            ov_path = catalog / "identity_overlay" / f"{mid}.json"
            if ov_path.exists():
                try:
                    ov = json.loads(ov_path.read_text())
                    m = {a["speaker_label"]: a.get("person_id")
                         for a in ov.get("speaker_assignments", [])
                         if isinstance(a, dict) and a.get("speaker_label")}
                except Exception:
                    m = {}
        identity_map[mid] = m

    cues: list[dict] = []
    for beat_idx, clip in enumerate(clips):
        mid = clip["media_id"]
        in_ms = clip["in_ms"]
        out_ms = clip["out_ms"]

        events = _load_speech_events(mid)
        beat_events = [
            e for e in events
            if e.get("start_ms", 0) >= in_ms and e.get("start_ms", 0) < out_ms
        ]
        if not beat_events:
            continue
        beat_events = _smooth_gap_words(beat_events)

        # Group into runs (break on speaker change, gap > 700ms, sentence end)
        runs: list[list[dict]] = []
        cur: list[dict] = []
        for e in beat_events:
            if cur:
                gap = e.get("start_ms", 0) - cur[-1].get("end_ms", 0)
                prev_spk = cur[-1].get("speaker_label")
                curr_spk = e.get("speaker_label")
                if (prev_spk != curr_spk or gap > 700
                        or _SENT_END.search(cur[-1].get("text", ""))):
                    runs.append(cur)
                    cur = []
            cur.append(e)
        if cur:
            runs.append(cur)

        # Chunk each run into max-5-word cues
        for run in runs:
            spk = run[0].get("speaker_label")
            has_conflict = any(e.get("speaker_conflict") for e in run)
            for i in range(0, len(run), 5):
                chunk = run[i:i + 5]
                text = " ".join(e.get("text", "") for e in chunk)
                # render_events_v2 words carry person_id/stable_unknown_id
                # pre-resolved (build_parallel_streams.py); legacy speech_events
                # / transcript.jsonl words lack the key entirely, so fall back
                # to the identity_map/identity_overlay lookup for those.
                if "person_id" in chunk[0]:
                    person_id = chunk[0].get("person_id")
                    stable_unk = chunk[0].get("stable_unknown_id")
                else:
                    person_id = identity_map.get(mid, {}).get(spk) if spk else None
                    stable_unk = None
                track_id = f"{mid}:{spk}" if spk else f"{mid}:unknown"
                if person_id:
                    stable_unk = None
                    colour_id = person_id
                else:
                    if not stable_unk:
                        stable_unk = f"unk_{abs(hash(track_id)):08x}"[:12]
                    colour_id = stable_unk

                cues.append({
                    "candidate_id": candidate_id,
                    "beat_index": beat_idx,
                    "media_id": mid,
                    "clip_start_ms": in_ms,
                    "clip_end_ms": out_ms,
                    "cue_start_ms": chunk[0].get("start_ms"),
                    "cue_end_ms": chunk[-1].get("end_ms"),
                    "text": text,
                    "speaker_label": spk,
                    "speaker_track_id": track_id,
                    "person_id": person_id,
                    "stable_unknown_id": stable_unk,
                    "colour_id": colour_id,
                    "colour_hex": _resolve_colour(colour_id),
                    "speaker_conflict": has_conflict,
                })
    return _merge_micro_cues(cues)


def _build_beat_colors(timeline: list[dict], clips: list[dict]) -> list[dict[str, str]]:
    """Build per-beat {speaker_label → colour_hex} dicts from render_timeline."""
    n_beats = len(clips)
    result: list[dict[str, str]] = [{} for _ in range(n_beats)]
    for cue in timeline:
        beat_idx = cue.get("beat_index", 0)
        if beat_idx >= n_beats:
            continue
        spk = cue.get("speaker_label")
        colour = cue.get("colour_hex", "")
        if spk and colour:
            result[beat_idx][spk] = colour
    return result


def _qc_render_timeline(timeline: list[dict], candidate_id: str) -> list[str]:
    """Check rendered cue timeline for known quality issues. Returns issue strings."""
    issues: list[str] = []

    # 1. Colour flicker: same speaker_track_id appears with >1 colour_id in the story
    track_colours: dict[str, set] = {}
    for cue in timeline:
        track_colours.setdefault(cue["speaker_track_id"], set()).add(cue["colour_id"])
    for tid, cols in track_colours.items():
        if len(cols) > 1:
            issues.append(f"COLOUR_FLICKER: {tid} → {sorted(cols)}")

    # 2. Speaker conflict: any cue has conflicting diarization evidence
    conflict_cues = [c for c in timeline if c.get("speaker_conflict")]
    if conflict_cues:
        issues.append(f"SPEAKER_CONFLICT: {len(conflict_cues)} cues with conflicting diar evidence")

    # 3. Mid-sentence cut: last cue of a clip doesn't end with sentence-ending punctuation
    beat_last: dict[int, dict] = {}
    for cue in timeline:
        beat_last[cue["beat_index"]] = cue
    for bidx, cue in sorted(beat_last.items()):
        text = (cue.get("text") or "").strip()
        if text and not _SENT_END.search(text):
            issues.append(f"MID_SENTENCE_CUT: beat {bidx} ({cue['media_id']}) ends '{text[-40:]}'")

    return issues


def render_candidate(cand: dict, export_prefix: str, *, dry_run: bool = False) -> dict:
    cid = cand.get("candidate_id", "?")
    lane = cand.get("primary_format_lane", "?")
    log_event("SE:I", cid, "start", {"lane": lane, "export_prefix": export_prefix, "dry_run": dry_run})
    if cand.get("verdict") != "complete":
        reason = f"verdict={cand.get('verdict')}"
        log_event("SE:I", cid, "skip", {"reason": reason})
        return {"ok": False, "reason": reason}
    if cand.get("confidence_tier") not in _AUTO_ELIGIBLE:
        reason = f"tier={cand.get('confidence_tier')} not auto-eligible"
        log_event("SE:I", cid, "skip", {"reason": reason})
        return {"ok": False, "reason": reason}

    result = se_adapt(cand)
    if not result.get("ok"):
        reason = result.get("reason", "adapt_failed")
        log.warning("Adapt failed for %s: %s", cid, reason)
        log_event("SE:I", cid, "error", {"reason": reason, "stage": "adapt"})
        return {"ok": False, "reason": reason}

    if dry_run:
        log.info("  [dry-run] would render bp=%s", result.get("bp_id"))
        log_event("SE:I", cid, "ok", {"dry_run": True, "bp_id": result.get("bp_id")})
        return {"ok": True, "dry_run": True, "bp_id": result.get("bp_id")}

    bp_path = Path(result["path"])
    bp = _read_last_jsonl(bp_path)
    clips = [
        {
            "media_id": b["media_id"], "in_ms": b["in_ms"], "out_ms": b["out_ms"],
            "label": b.get("edit_role", "body"), "_is_broll": False, "_crosstalk_trimmed": False,
        }
        for b in bp["beats"]
    ]

    # Build render_timeline BEFORE render — canonical colour source
    timeline = _build_render_timeline(cid, bp, clips, CAT)
    beat_colors = _build_beat_colors(timeline, clips)
    qc_issues = _qc_render_timeline(timeline, cid)
    if qc_issues:
        for iss in qc_issues:
            log.warning("PRE_RENDER_QC [%s]: %s", cid, iss)

    # Write render_timeline to disk
    tl_dir = CAT / "render_timeline"
    tl_dir.mkdir(exist_ok=True)
    tl_path = tl_dir / f"{cid}.jsonl"
    tl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in timeline) + "\n",
        encoding="utf-8",
    )

    export_id = f"{export_prefix}_{result['bp_id'][-6:]}"
    try:
        rendered = do_render(
            beat_plan=bp, clips=clips, export_id=export_id,
            catalog=CAT, proxy_dir=ROOT / "media" / "proxies",
            output_dir=CAT / "exports",
            journal_db=ROOT / "state" / "journal.db",
            config_path=ROOT / "state" / "config.toml",
            reframe_strategy="active_speaker",
            speaker_colors_override=beat_colors,
        )
    except Exception as exc:
        log_event("SE:I", cid, "error", {"reason": str(exc)[:500], "stage": "render", "export_id": export_id})
        raise

    log.info("  timeline: %d cues, %d QC issues", len(timeline), len(qc_issues))
    log_event("SE:I", cid, "ok", {
        "bp_id": result["bp_id"], "export_id": export_id,
        "output": rendered.get("output_mp4_path", ""),
        "n_qc_issues": len(qc_issues),
    })
    return {"ok": True, "bp_id": result["bp_id"], "export_id": export_id, **rendered}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-file", required=True)
    ap.add_argument("--export-prefix", default="se_spine")
    ap.add_argument("--max-renders", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cands_path = Path(args.candidates_file)
    candidates = [json.loads(l) for l in cands_path.read_text().splitlines() if l.strip()]
    eligible = [c for c in candidates if c.get("verdict") == "complete"
                and c.get("confidence_tier") in _AUTO_ELIGIBLE]
    log.info("%d candidates, %d eligible for render", len(candidates), len(eligible))

    renders = []
    for cand in eligible[:args.max_renders]:
        cid = cand.get("candidate_id", "?")
        lane = cand.get("primary_format_lane", "?")
        log.info("Rendering %s (%s)...", cid, lane)
        result = render_candidate(cand, args.export_prefix, dry_run=args.dry_run)
        renders.append({"candidate_id": cid, "lane": lane, **result})
        if result.get("ok") and not args.dry_run:
            out = result.get("output_mp4_path") or result.get("export_id", "?")
            log.info("  -> %s", out)
            print(f"RENDER: {out}")
        elif not result.get("ok"):
            log.warning("  FAILED: %s", result.get("reason"))

    ok_count = sum(1 for r in renders if r.get("ok"))
    log.info("Done: %d/%d rendered successfully", ok_count, len(renders))
    log_event("SE:I", "_corpus", "render_batch_done",
              {"n_rendered": ok_count, "n_total": len(renders), "dry_run": args.dry_run})


if __name__ == "__main__":
    main()

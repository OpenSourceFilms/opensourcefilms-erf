#!/usr/bin/env python3
"""ASR Recall-Gap Detector (corpus-wide).

Read-only analysis pass. Scans per-media JSONL artifacts under catalog/asr/,
detects four classes of recall/attribution gap, and writes one file per media
under catalog/asr_recall_gaps/{media_id}.jsonl. These windows drive the
targeted plain-Whisper recall pass and word-level attribution — they are NOT a
canonical pipeline surface, just a derived worklist.

Gap types:
  A overlap_gap        >=2 diar speakers active over an overlap window but the
                       canonical words.jsonl is empty OR >=95% one speaker
                       (attribution flattening).
  B monologue_gap      a single diar turn >=1.5s carrying <2 canonical words.
  C low_conf_asr       a segment with low avg_logprob / high no_speech_prob that
                       still emitted words.
  D suspicious_silence a contiguous VAD-speech run >=MIN_SILENCE_RUN_MS with zero
                       words that overlaps diarized speech.

2026-07-08 (D-Recall-01): MIN_SILENCE_RUN_MS lowered 1500ms -> 300ms. The 1.5s
floor was arbitrary and excluded exactly the failure mode this detector exists
for: short interjections ("Yeah." "What?") that production's vad_filter=True
faster-whisper pass (src/erf/asr.py) drops. Silero VAD itself won't report a
run shorter than ~250ms, so 300ms costs almost nothing on the noise side.
Corpus-wide measurement before the fix: 359 VAD-confirmed, diar-overlapping,
zero-word windows sat in the 283-1499ms range and were invisible to this
detector (median 540ms, p90 988ms) -- real short utterances, not VAD blips.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Outcome spine (MASTER §5.2.1): stage RG always writes a per-media file (empty
# when no gaps) → it always records `ok`. Guarded import can never break the stage.
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from erf import outcomes
except Exception:  # pragma: no cover
    outcomes = None

MIN_SILENCE_RUN_MS = 300  # D-Recall-01 2026-07-08; was 1500


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file; skip malformed lines; empty list if absent/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARN: {path.name}: line {lineno} invalid JSON, skipped",
                      file=sys.stderr)
    return rows


def midpoint(w: Dict[str, Any]) -> float:
    return (w["start_ms"] + w["end_ms"]) / 2.0


def intersect_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def contained(outer: tuple, inner: tuple, threshold: float = 0.9) -> bool:
    i_start, i_end = inner
    inner_dur = i_end - i_start
    if inner_dur <= 0:
        return True
    return intersect_ms(i_start, i_end, outer[0], outer[1]) / inner_dur >= threshold


def generate_gaps(media_id: str, catalog: Path) -> List[Dict[str, Any]]:
    asr_dir = catalog / "asr"
    words = load_jsonl(asr_dir / f"{media_id}.words.jsonl")
    diar = load_jsonl(asr_dir / f"{media_id}.diar.jsonl")
    overlaps = load_jsonl(asr_dir / f"{media_id}.overlap.jsonl")
    segments = load_jsonl(asr_dir / f"{media_id}.segments.jsonl")
    vad_rows = load_jsonl(asr_dir / f"{media_id}.vad.jsonl")

    words.sort(key=lambda w: w["start_ms"])
    diar.sort(key=lambda t: t["start_ms"])
    overlaps.sort(key=lambda o: o["start_ms"])
    segments.sort(key=lambda s: s["start_ms"])
    vad_rows.sort(key=lambda v: v["start_ms"])

    word_mid = [(midpoint(w), w) for w in words]

    def words_in(a: int, b: int) -> List[Dict]:
        return [w for m, w in word_mid if a <= m <= b]

    def word_dur_sum(ws: List[Dict]) -> float:
        return sum(w["end_ms"] - w["start_ms"] for w in ws)

    def active(a: int, b: int):
        speakers, speech = set(), 0
        for t in diar:
            inter = intersect_ms(a, b, t["start_ms"], t["end_ms"])
            if inter > 0:
                speakers.add(t["speaker_label"])
                speech += inter
        return sorted(speakers), speech

    def row(a, b, gtype, speakers, speech, ws) -> Dict[str, Any]:
        dur = max(1, b - a)
        return {
            "media_id": media_id, "start_ms": a, "end_ms": b, "gap_type": gtype,
            "diar_speakers": speakers, "diar_speech_ms": speech,
            "canonical_word_count": len(ws),
            "word_coverage_ratio": round(word_dur_sum(ws) / dur, 3),
        }

    gaps: List[Dict[str, Any]] = []

    # A) overlap_gap — attribution flattening
    for ov in overlaps:
        a, b = ov["start_ms"], ov["end_ms"]
        if b - a < 800:
            continue
        speakers, speech = active(a, b)
        if len(speakers) < 2:
            continue
        ws = words_in(a, b)
        if ws:
            labels = Counter(w["speaker_label"] for w in ws
                             if w.get("speaker_label") is not None)
            if not labels or labels.most_common(1)[0][1] / len(ws) < 0.95:
                continue
        gaps.append(row(a, b, "overlap_gap", speakers, speech, ws))

    # B) monologue_gap
    for t in diar:
        a, b = t["start_ms"], t["end_ms"]
        if b - a < 1500:
            continue
        ws = words_in(a, b)
        if len(ws) >= 2:
            continue
        gaps.append(row(a, b, "monologue_gap", [t["speaker_label"]], b - a, ws))

    # C) low_conf_asr
    for s in segments:
        if not (s.get("avg_logprob", 0.0) < -0.9 or s.get("no_speech_prob", 0.0) > 0.6):
            continue
        a, b = s["start_ms"], s["end_ms"]
        ws = words_in(a, b)
        if not ws:
            continue
        speakers, speech = active(a, b)
        gaps.append(row(a, b, "low_conf_asr", speakers, speech, ws))

    # D) suspicious_silence — contiguous VAD speech runs
    runs: List[tuple] = []
    cur = None
    for v in vad_rows:
        if v.get("is_speech"):
            if cur and v["start_ms"] <= cur[1] + 1:
                cur = (cur[0], max(cur[1], v["end_ms"]))
            else:
                if cur and cur[1] - cur[0] >= MIN_SILENCE_RUN_MS:
                    runs.append(cur)
                cur = (v["start_ms"], v["end_ms"])
        else:
            if cur and cur[1] - cur[0] >= MIN_SILENCE_RUN_MS:
                runs.append(cur)
            cur = None
    if cur and cur[1] - cur[0] >= MIN_SILENCE_RUN_MS:
        runs.append(cur)
    for a, b in runs:
        ws = words_in(a, b)
        if ws:
            continue
        if not any(intersect_ms(a, b, t["start_ms"], t["end_ms"]) > 0 for t in diar):
            continue
        speakers, speech = active(a, b)
        gaps.append(row(a, b, "suspicious_silence", speakers, speech, ws))

    # Dedup A/B: sort by dur desc, drop windows >=90% contained in a kept one.
    for gt in ("overlap_gap", "monologue_gap"):
        typed = sorted((g for g in gaps if g["gap_type"] == gt),
                       key=lambda g: g["end_ms"] - g["start_ms"], reverse=True)
        keep: List[Dict] = []
        for g in typed:
            cand = (g["start_ms"], g["end_ms"])
            if not any(contained((k["start_ms"], k["end_ms"]), cand) for k in keep):
                keep.append(g)
        gaps = [g for g in gaps if g["gap_type"] != gt] + keep

    # Priority
    for g in gaps:
        dur = g["end_ms"] - g["start_ms"]
        gt = g["gap_type"]
        if (gt in ("monologue_gap", "suspicious_silence") and dur >= 8000) or \
           (gt == "overlap_gap" and g["canonical_word_count"] >= 8):
            g["priority"] = "high"
        elif dur >= 3000:
            g["priority"] = "medium"
        else:
            g["priority"] = "low"

    gaps.sort(key=lambda g: g["start_ms"])
    return gaps


def main() -> None:
    ap = argparse.ArgumentParser(description="ASR Recall-Gap Detector")
    ap.add_argument("--catalog", default="/workspace/erf/catalog")
    ap.add_argument("--only", dest="only_media", metavar="MID")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    catalog = Path(args.catalog)
    asr_dir = catalog / "asr"
    out_dir = catalog / "asr_recall_gaps"
    if not asr_dir.exists():
        print("ERROR: asr directory not found", file=sys.stderr)
        sys.exit(1)

    media = sorted({p.name.split(".", 1)[0] for p in asr_dir.glob("*.words.jsonl")})
    if args.only_media:
        if args.only_media not in media:
            print(f"ERROR: media '{args.only_media}' not found", file=sys.stderr)
            sys.exit(1)
        media = [args.only_media]

    if not args.report_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    with_gaps = total_rows = 0
    record_failures = 0
    type_counts: Counter = Counter()
    for mid in media:
        # Phase 4 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §2 root cause #6/P4):
        # per-media isolation — main()'s loop used to call generate_gaps() with no
        # try/except, so one media's exception killed the whole batch with no error
        # outcome recorded for it OR any media after it. Only the per-media body is
        # guarded; argument parsing and directory setup above still fail the whole run.
        try:
            gaps = generate_gaps(mid, catalog)
            if gaps:
                with_gaps += 1
                total_rows += len(gaps)
                type_counts.update(g["gap_type"] for g in gaps)
                print(f"{mid}: {len(gaps)} gaps")
            if not args.report_only:
                out_file = out_dir / f"{mid}.jsonl"
                tmp = out_file.with_name(out_file.name + ".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for g in gaps:
                        f.write(json.dumps(g) + "\n")
                os.replace(tmp, out_file)
                # Explicit outcome: RG always writes a file, so always `ok` (counts show
                # zero work when there are no gaps — never `skip`/`empty` per §5.2.1).
                if outcomes:
                    outcomes.record(mid, "RG", "ok", counts={"n_gaps": len(gaps)},
                                    artifacts=[f"catalog/asr_recall_gaps/{mid}.jsonl"],
                                    producer="build_recall_gaps.py")
        except Exception as e:
            print(f"ERROR: {mid}: {type(e).__name__}: {e}", file=sys.stderr)
            # The recovery record() itself can raise (P4: no silent swallow —
            # e.g. journal.db lock contention). Guard it so a double failure
            # doesn't kill the batch (the exact defect this handler exists to
            # fix — W2 checkpoint review finding, D-Log-12); surface it via
            # stderr + a non-zero exit at the end instead.
            if outcomes:
                try:
                    outcomes.record(mid, "RG", "error",
                                    reason=f"exception: {type(e).__name__}: {e}"[:500],
                                    producer="build_recall_gaps.py")
                except Exception as rec_e:
                    record_failures += 1
                    print(f"ERROR: {mid}: RG error outcome could not be recorded: "
                          f"{type(rec_e).__name__}: {rec_e}", file=sys.stderr)
            continue

    type_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    print(f"TOTAL media={len(media)} with_gaps={with_gaps} "
          f"rows={total_rows} by_type={{{type_str}}}")
    if record_failures:
        # P4: an unloggable outcome is a failed run — the batch finished, but
        # not every media's disposition made it to the journal.
        print(f"ERROR: {record_failures} error outcome(s) could not be recorded — "
              f"run is FAILED (unrecorded dispositions)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

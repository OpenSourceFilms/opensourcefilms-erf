#!/usr/bin/env python3
"""EV — Evidence-package assembler.

Fuses every extracted signal for a media into ONE grounded document
(catalog/evidence/{mid}.json) that the grounded editor (DS Pro) reads to make
editorial decisions. This is the DS-facing contract: the model sees the actual
sentence-segmented, speaker-attributed, accurately-timed transcript interleaved
with per-sentence editorial signals, plus media-level visual grounding, AV
quality, identity+consent, silences and crosstalk.

All inputs are best-effort — a missing artifact degrades the package, never fails.

Usage:
  PYTHONPATH=src python bin/build_evidence.py --all [--force]
  PYTHONPATH=src python bin/build_evidence.py --media-ids m_x m_y
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    from erf.runlog import stage_run, log_event
    from erf import outcomes
except Exception:
    stage_run = None
    log_event = None
    outcomes = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_evidence")

_CHUNK_MS_RE = re.compile(r":c:(\d+)-(\d+)")


def _jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _person_names(catalog: Path) -> dict[str, dict]:
    out = {}
    for p in (catalog / "persons").glob("p_*.json"):
        d = _json(p)
        if d.get("person_id"):
            out[d["person_id"]] = {"name": d.get("display_name") or d["person_id"],
                                   "consent": bool(d.get("consent"))}
    return out


def process_media(media_id: str, catalog: Path, force: bool, persons: dict) -> dict | None:
    out_path = catalog / "evidence" / f"{media_id}.json"
    if out_path.exists() and not force:
        if log_event:
            log_event("EV", media_id, "skip", {"reason": "exists"})
        return None

    # Consult T0's explicit outcome rather than inferring "is there a transcript" from
    # the filesystem (Outcome Contract: cross-stage awareness, never inference).
    if outcomes and outcomes.flag(media_id, "has_transcript") is False:
        outcomes.record(media_id, "EV", "skip", reason="no_transcript")
        outcomes.set_flag(media_id, "has_evidence", False, reason="no_transcript")
        log.info("[%s] has_transcript=False (per T0) — skip", media_id)
        return None
    words = _jsonl(catalog / "asr" / f"{media_id}.transcript.jsonl")
    if not words:
        if outcomes:
            outcomes.record(media_id, "EV", "skip", reason="no_transcript")
            outcomes.set_flag(media_id, "has_evidence", False, reason="no_transcript")
        log.warning("[%s] no transcript — skip", media_id)
        return None

    manifest = _json(catalog / "media_manifest" / f"{media_id}.json")
    vg = _json(catalog / "visual_grounding" / f"{media_id}.json")
    card = _json(catalog / "media_cards" / f"{media_id}.json")
    events = _jsonl(catalog / "events" / f"{media_id}.jsonl")
    signals = {s.get("event_id"): s for s in _jsonl(catalog / "event_signals" / f"{media_id}.jsonl")}
    tags = _jsonl(catalog / "tags" / f"{media_id}.jsonl")
    scenes = _jsonl(catalog / "scenes" / f"{media_id}.jsonl")
    quality = _jsonl(catalog / "quality" / f"{media_id}.jsonl")
    vad = _jsonl(catalog / "asr" / f"{media_id}.vad.jsonl")
    overlaps = _jsonl(catalog / "asr" / f"{media_id}.overlap.jsonl")

    # tag chunk time spans
    for t in tags:
        m = _CHUNK_MS_RE.search(t.get("chunk_id", ""))
        if m:
            t["_s"], t["_e"] = int(m.group(1)), int(m.group(2))
        else:
            t["_s"], t["_e"] = None, None

    # group words → sentences
    sentences = []
    cur_idx = None
    bucket: list[dict] = []

    def _flush(b: list[dict]):
        if not b:
            return
        s0 = min(w["start_ms"] for w in b)
        s1 = max(w["end_ms"] for w in b)
        lbl = next((w.get("speaker_label") for w in b if w.get("speaker_label")), None)
        pid = next((w.get("person_id") for w in b if w.get("person_id")), None)
        name = persons.get(pid, {}).get("name") if pid else (lbl or "unknown")
        text = " ".join(w["text"] for w in b).strip()
        # editorial signals: best-overlapping event
        best_ev, best_ov = None, 0
        for ev in events:
            ov = _overlap(s0, s1, ev.get("start_ms", 0), ev.get("end_ms", 0))
            if ov > best_ov:
                best_ov, best_ev = ov, ev
        emo = best_ev.get("emotional_beat") if best_ev else None
        broll = bool(best_ev.get("b_roll_flag")) if best_ev else False
        hook = quote = 0.0
        if best_ev:
            sig = signals.get(best_ev.get("event_id"))
            if sig:
                hook = max(hook, float(sig.get("hook_potential") or 0))
                quote = max(quote, float(sig.get("quote_potential") or 0))
        for t in tags:
            if t["_s"] is not None and _overlap(s0, s1, t["_s"], t["_e"]) > 0:
                hook = max(hook, float(t.get("hook_score") or 0))
                quote = max(quote, float(t.get("quote_score") or 0))
        sentences.append({
            "sentence_index": b[0]["sentence_index"],
            "start_ms": s0, "end_ms": s1,
            "speaker_label": lbl, "speaker": name,
            "text": text,
            "emotional_beat": emo, "hook": round(hook, 2), "quote": round(quote, 2),
            "b_roll_flag": broll,
        })

    for w in words:
        if w["sentence_index"] != cur_idx:
            _flush(bucket)
            bucket = []
            cur_idx = w["sentence_index"]
        bucket.append(w)
    _flush(bucket)

    # people / unidentified
    overlay = _json(catalog / "identity_overlay" / f"{media_id}.json")
    pids, unident = [], set()
    for a in overlay.get("speaker_assignments", []):
        if a.get("person_id"):
            pids.append(a["person_id"])
        elif a.get("speaker_label"):
            unident.add(a["speaker_label"])
    seen, people = set(), []
    for pid in pids:
        if pid not in seen and pid in persons:
            seen.add(pid)
            people.append({"person_id": pid, **persons[pid]})

    # av quality aggregates
    def _med(key):
        vals = [q[key] for q in quality if isinstance(q.get(key), (int, float))]
        return round(statistics.median(vals), 3) if vals else None
    clip_vals = [q["clipping_ratio"] for q in quality if isinstance(q.get("clipping_ratio"), (int, float))]
    aud_classes = [q.get("audio_top_class") for q in quality if q.get("audio_top_class")]
    av_quality = {
        "speech_clarity_median": _med("speech_clarity"),
        "clipping_max": round(max(clip_vals), 3) if clip_vals else None,
        "aesthetic_median": _med("aesthetic_score"),
        "audio_top_class": max(set(aud_classes), key=aud_classes.count) if aud_classes else None,
    }

    # silences (vad gaps / non-speech >=500ms) and overlaps
    silences = [{"start_ms": s["start_ms"], "end_ms": s["end_ms"]}
                for s in vad if not s.get("is_speech") and (s["end_ms"] - s["start_ms"]) >= 500]
    ov_list = [{"start_ms": o["start_ms"], "end_ms": o["end_ms"]} for o in overlaps]

    pkg = {
        "media_id": media_id,
        "source": manifest.get("relpath"),
        "duration_ms": manifest.get("duration_ms"),
        "footage_type": vg.get("footage_type"),
        "setting": vg.get("setting"),
        "shot_type": vg.get("shot_type"),
        "visual_summary": vg.get("visual_summary"),
        "on_screen_subjects": vg.get("on_screen_subjects"),
        "synopsis": card.get("synopsis"),
        "emotional_tenor": card.get("emotional_tenor"),
        "people": people,
        "unidentified_speakers": sorted(unident),
        "av_quality": av_quality,
        "n_words": len(words),
        "n_sentences": len(sentences),
        "sentences": sentences,
        "scenes": [{"scene_id": s.get("scene_id"), "start_ms": s.get("start_ms"),
                    "end_ms": s.get("end_ms")} for s in scenes],
        "silences": silences,
        "overlaps": ov_list,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), prefix="." + out_path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(pkg, f, ensure_ascii=False, indent=1)
        os.rename(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    if outcomes:
        outcomes.record(media_id, "EV", "ok",
                        counts={"sentences": len(sentences), "people": len(people)})
        outcomes.set_flag(media_id, "has_evidence", True)
    log.info("[%s] sentences=%d footage=%s people=%s", media_id, len(sentences),
             vg.get("footage_type"), [p["name"] for p in people] or unident)
    return pkg


def main() -> None:
    ap = argparse.ArgumentParser(description="EV — evidence-package assembler")
    ap.add_argument("--catalog", default="/workspace/erf/catalog")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--media-ids", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    catalog = Path(args.catalog)
    persons = _person_names(catalog)
    if args.all:
        ids = sorted(p.name[: -len(".transcript.jsonl")]
                     for p in (catalog / "asr").glob("*.transcript.jsonl"))
    else:
        ids = args.media_ids
    log.info("EV: %d media", len(ids))

    ok = skip = err = 0
    for mid in ids:
        try:
            # Single-writer: process_media records outcomes itself; stage_run's
            # unconditional bare ok would overwrite recorded skips in the
            # journal projection (MASTER §5.2.1, fixed 2026-07-03).
            r = process_media(mid, catalog, args.force, persons)
            ok += r is not None
            skip += r is None
        except Exception as e:
            err += 1
            if outcomes:
                try:
                    outcomes.record(mid, "EV", "error", reason=str(e)[:500])
                except Exception:
                    pass
            log.error("[%s] FAILED: %s", mid, e, exc_info=True)
    log.info("Done. ok=%d skip=%d err=%d", ok, skip, err)


if __name__ == "__main__":
    main()

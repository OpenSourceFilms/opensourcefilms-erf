#!/usr/bin/env python3
"""T0 — Unified canonical transcript.

PRIMARY (2026-07-03, spine promotion): a PROJECTION of the adjudicated
speech-events spine — catalog/render_events_v2 (preferred: adjudicated stream A
minus superseded recall fragments, plus voiceprint-confirmed stream-B words) or
catalog/speech_events_v2_adjudicated. Every downstream consumer (evidence
package, editorial labels, story engine, render) inherits the corrected
who-said-what from this one surface.

LEGACY fallback (media the v2 spine has not processed): merge the three ASR
facets — words.jsonl (canonical order/punctuation), words.wx.jsonl (WhisperX
timing), identity_overlay (speaker_label → person_id).

Output: catalog/asr/{mid}.transcript.jsonl, one line per word:
  {word_index, text, start_ms, end_ms, speaker_label, person_id, sentence_index,
   + projected rows add: stable_unknown_id, source, event_id, stream}
Projected rows keep TRUE timings: stream-B (confirmed second-speaker) words
genuinely overlap stream A in time — word start_ms is NOT globally monotonic.

Usage:
  PYTHONPATH=src python bin/build_transcript.py --all [--force]
  PYTHONPATH=src python bin/build_transcript.py --media-ids m_x m_y

Phase 3 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §3.2): stop discarding
`pct_person_id` (§2 root cause #1 — "T0 even computes pct_person_id then
discards it before recording"). Now computed on BOTH the primary (projected)
and legacy paths, always carried in `metrics`, and — only when this media's
identity_map has >=1 gallery-anchored slot — enforced as a `pct_person_id > 0`
health predicate (rando-only media pass vacuously). Also carries
`inputs={"identity_map": <fp>}`, the content fingerprint of the identity_map
consumed this run.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    from erf.runlog import stage_run, log_event
except Exception:  # logging is best-effort
    stage_run = None
    log_event = None
# Phase 4 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md P4): erf.outcomes is a HARD
# dependency (no `if outcomes is not None` escape) — an unloggable T0 run must fail
# loudly, not silently degrade to never-logs-anything (matches CLUSTERS' Phase 3 pattern).
from erf import outcomes
try:
    from erf import stages as _stages_registry
except Exception:
    _stages_registry = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_transcript")

SENT_END = ("." , "?", "!", "…")
PAUSE_SPLIT_MS = 700
_norm_re = re.compile(r"[^a-z0-9]")


def _norm(text: str) -> str:
    return _norm_re.sub("", text.lower())


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _align_timing(canonical: list[dict], wx: list[dict]) -> int:
    """Transfer wx (accurate) timing onto canonical words via sequence alignment.
    Returns count of words whose timing was replaced."""
    c_norm = [_norm(w.get("text", "")) for w in canonical]
    x_norm = [_norm(w.get("text", "")) for w in wx]
    sm = difflib.SequenceMatcher(a=c_norm, b=x_norm, autojunk=False)
    replaced = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for off in range(i2 - i1):
            cw = canonical[i1 + off]
            xw = wx[j1 + off]
            if xw.get("start_ms") is not None and xw.get("end_ms") is not None:
                cw["start_ms"] = int(xw["start_ms"])
                cw["end_ms"] = int(xw["end_ms"])
                replaced += 1
    return replaced


def _enforce_monotonic(words: list[dict]) -> None:
    prev_end = 0
    for w in words:
        s = int(w.get("start_ms") or 0)
        e = int(w.get("end_ms") or s)
        if s < prev_end:
            s = prev_end
        if e < s:
            e = s
        w["start_ms"], w["end_ms"] = s, e
        prev_end = e


def _load_diar(catalog: Path, media_id: str) -> list[dict]:
    p = catalog / "asr" / f"{media_id}.diar.jsonl"
    turns = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    t = json.loads(line)
                    if t.get("start_ms") is not None and t.get("speaker_label"):
                        turns.append(t)
                except json.JSONDecodeError:
                    pass
    return turns


def _assign_diar_labels(words: list[dict], turns: list[dict]) -> None:
    """Re-derive each word's speaker_label from the diar turn it MOST overlaps.
    More robust than the raw ASR word-label (which lags at turn boundaries —
    the first word of a new speaker often keeps the previous speaker's tag)."""
    if not turns:
        return
    for w in words:
        s, e = w["start_ms"], w["end_ms"]
        mid = (s + e) // 2
        best, best_ov, best_span = None, 0, 1 << 62
        for t in turns:
            ov = min(e, t["end_ms"]) - max(s, t["start_ms"])
            span = t["end_ms"] - t["start_ms"]
            # max overlap wins; on ties prefer the SHORTER (more specific) turn — in
            # crosstalk both speakers' turns cover the word, and the brief turn is the
            # interjector who actually owns those words.
            if ov > best_ov or (ov == best_ov and ov > 0 and span < best_span):
                best_ov, best, best_span = ov, t, span
        if best is None:  # fall back to the turn containing the word midpoint
            for t in turns:
                if t["start_ms"] <= mid < t["end_ms"]:
                    best = t
                    break
        if best is not None:
            w["speaker_label"] = best.get("speaker_label")


def _smooth_labels(words: list[dict]) -> None:
    """Reassign isolated single-word label flips to the surrounding run (diar noise)."""
    for i in range(1, len(words) - 1):
        a, b, c = (words[i - 1].get("speaker_label"), words[i].get("speaker_label"),
                   words[i + 1].get("speaker_label"))
        if b != a and a == c and a is not None:
            words[i]["speaker_label"] = a


def _person_map(catalog: Path, media_id: str) -> dict[str, str]:
    path = catalog / "identity_overlay" / f"{media_id}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    best: dict[str, tuple[float, str]] = {}
    for a in data.get("speaker_assignments", []):
        lbl = a.get("speaker_label")
        pid = a.get("person_id")
        conf = float(a.get("confidence") or 0.0)
        if lbl and pid and (lbl not in best or conf > best[lbl][0]):
            best[lbl] = (conf, pid)
    return {lbl: pid for lbl, (_, pid) in best.items()}


def _identity_map_fp(catalog: Path, media_id: str) -> dict:
    """`inputs=` fingerprint (Phase 3, D-Log-04 §4) for the identity_map this
    run consumed: first 16 hex chars of sha256(file content). Absent file ->
    omit the key entirely (not a placeholder null) — nothing was consumed."""
    p = catalog / "identity_map" / f"{media_id}.json"
    if not p.exists():
        return {}
    return {"identity_map": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}


def _has_gallery_slot(catalog: Path, media_id: str) -> bool:
    """True iff identity_map/{media_id}.json anchored >=1 slot to a known
    gallery person — gates the pct_person_id health predicate (Phase 3)."""
    p = catalog / "identity_map" / f"{media_id}.json"
    if not p.exists():
        return False
    try:
        speakers = json.loads(p.read_text(encoding="utf-8")).get("speakers", {})
    except Exception:
        return False
    return any(info.get("source") == "gallery" for info in speakers.values())


def _pct_person_id(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return round(100 * sum(1 for r in rows if r.get("person_id")) / len(rows), 1)


def _sentence_indices(words: list[dict]) -> list[int]:
    idxs = []
    cur = 0
    for i, w in enumerate(words):
        idxs.append(cur)
        text = (w.get("text") or "").strip()
        ends_sentence = text.endswith(SENT_END) or text.endswith("...")
        gap_split = False
        if i + 1 < len(words):
            gap = int(words[i + 1].get("start_ms") or 0) - int(w.get("end_ms") or 0)
            gap_split = gap > PAUSE_SPLIT_MS
        if ends_sentence or gap_split:
            cur += 1
    return idxs


def _sentence_indices_v2(words: list[dict]) -> list[int]:
    """Sentence indices for spine-projected words: same punctuation/pause rules as
    _sentence_indices, plus a hard break at event boundaries where the speaker or
    stream changes (utterance boundary from the adjudicated spine)."""
    idxs: list[int] = []
    cur = 0
    for i, w in enumerate(words):
        idxs.append(cur)
        text = w["text"]
        ends = text.endswith(SENT_END) or text.endswith("...")
        brk = False
        if i + 1 < len(words):
            nxt = words[i + 1]
            gap = nxt["start_ms"] - w["end_ms"]
            spk_change = (nxt.get("event_id") != w.get("event_id") and
                          (nxt.get("speaker_label") != w.get("speaker_label") or
                           nxt.get("stream") != w.get("stream")))
            brk = gap > PAUSE_SPLIT_MS or spk_change
        if ends or brk:
            cur += 1
    return idxs


def _project_from_events(media_id: str, catalog: Path):
    """Project transcript rows from the adjudicated speech-events spine.

    Source preference: render_events_v2 (adjudicated stream A minus superseded
    recall fragments, plus voiceprint-confirmed stream-B words) then
    speech_events_v2_adjudicated. Returns (rows, layer_name); (None, None) when
    the spine is absent or predates word-level carry (caller falls back to the
    legacy merge). NOTE: rows keep TRUE word timings — stream-B words genuinely
    overlap stream A in time; do not enforce monotonicity here."""
    for layer in ("render_events_v2", "speech_events_v2_adjudicated"):
        p = catalog / layer / f"{media_id}.jsonl"
        if not p.exists():
            continue
        events = _load_jsonl(p)
        if not events:
            return [], layer  # explicit empty spine -> explicit empty transcript
        if not all(isinstance(e.get("words"), list) for e in events):
            return None, None  # spine predates word carry
        events.sort(key=lambda e: (e["start_ms"], e["end_ms"], e.get("event_id") or ""))
        words: list[dict] = []
        for e in events:
            amb = e.get("speaker_track_id") == "unknown_overlap_speaker"
            for w in e["words"]:
                t = (w.get("text") or "").strip()
                if not t:
                    continue
                words.append({
                    "text": t,
                    "start_ms": int(w["start_ms"]),
                    "end_ms": int(w["end_ms"]),
                    "speaker_label": None if amb else e.get("diar_label"),
                    "person_id": e.get("person_id"),
                    "stable_unknown_id": e.get("stable_unknown_id"),
                    "source": w.get("source") or e.get("source"),
                    "event_id": e.get("event_id"),
                    "stream": e.get("stream") or "A",
                })
        sent = _sentence_indices_v2(words)
        rows = []
        for i, w in enumerate(words):
            rows.append({"word_index": i, **w, "sentence_index": sent[i]})
        return rows, layer
    return None, None


def _write_rows_atomic(asr: Path, out_path: Path, rows: list[dict]) -> None:
    asr.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(asr), prefix="." + out_path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.rename(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def process_media(media_id: str, catalog: Path, force: bool) -> dict | None:
    asr = catalog / "asr"
    out_path = asr / f"{media_id}.transcript.jsonl"
    if out_path.exists() and not force:
        # P4 hard-dependency path, not the optional runlog import — this is the
        # most-frequently-hit exit in the file (idempotent re-runs), and it was
        # the one outcome-shaped call still gated behind `if log_event:` (silent
        # no-record if erf.runlog failed to import). W2 checkpoint review
        # finding, D-Log-12.
        outcomes.record(media_id, "T0", "skip", reason="output_exists")
        log.info("[%s] skip (exists)", media_id)
        return None

    # Gate on A's explicit has_speech flag (cross-stage awareness), not file presence.
    if outcomes.flag(media_id, "has_speech") is False:
        outcomes.record(media_id, "T0", "skip", reason="no_speech")
        outcomes.set_flag(media_id, "has_transcript", False, reason="no_speech")
        return None

    # PRIMARY (2026-07-03, spine promotion): project from the adjudicated
    # speech-events spine so every consumer inherits the corrected attribution.
    proj_rows, proj_layer = _project_from_events(media_id, catalog)
    if proj_rows is not None:
        _write_rows_atomic(asr, out_path, proj_rows)
        n_words = len(proj_rows)
        n_sent = (proj_rows[-1]["sentence_index"] + 1) if proj_rows else 0
        n_b = sum(1 for r in proj_rows if r.get("stream") == "B")
        n_recall = sum(1 for r in proj_rows if r.get("source") == "recall_pass")
        # Phase 3 (D-Log-04 §4): stop discarding pct_person_id — always in
        # metrics; health predicate only gated on when identity_map anchored a
        # gallery slot for this media (rando-only media pass vacuously).
        pct_person = _pct_person_id(proj_rows)
        has_gallery = _has_gallery_slot(catalog, media_id)
        stats = {"n_words": n_words, "n_sentences": n_sent,
                 "projection_source": proj_layer,
                 "streamB_words": n_b, "recall_words": n_recall,
                 "pct_person_id": pct_person}
        if n_words:
            values = {"pct_person_id": pct_person} if has_gallery else {}
            health = (_stages_registry.evaluate_health("T0", values)
                      if (_stages_registry is not None and values) else None)
            outcomes.record(media_id, "T0", "ok",
                            counts={"words": n_words, "sentences": n_sent,
                                    "streamB_words": n_b, "recall_words": n_recall,
                                    "pct_person_id": pct_person},
                            health=health, inputs=_identity_map_fp(catalog, media_id),
                            artifacts=[f"catalog/asr/{media_id}.transcript.jsonl"],
                            producer=f"build_transcript.py:{proj_layer}")
            outcomes.set_flag(media_id, "has_transcript", True)
        else:
            outcomes.record(media_id, "T0", "empty", reason="empty_speech_events",
                            metrics={"pct_person_id": pct_person},
                            producer=f"build_transcript.py:{proj_layer}")
            outcomes.set_flag(media_id, "has_transcript", False,
                              reason="empty_speech_events")
        log.info("[%s] PROJECTED from %s: words=%d (+%d recall, %d streamB) sentences=%d",
                 media_id, proj_layer, n_words, n_recall, n_b, n_sent)
        return stats

    # LEGACY fallback (media the v2 spine has not processed yet)
    canonical = _load_jsonl(asr / f"{media_id}.words.jsonl")
    if not canonical:
        outcomes.record(media_id, "T0", "skip", reason="no_words")
        outcomes.set_flag(media_id, "has_transcript", False, reason="no_words")
        log.warning("[%s] no words.jsonl — skip", media_id)
        return None

    # normalize fields and timing
    for w in canonical:
        w["text"] = (w.get("text") or "").strip()
        w["start_ms"] = int(w.get("start_ms") or 0)
        w["end_ms"] = int(w.get("end_ms") or w["start_ms"])

    wx = _load_jsonl(asr / f"{media_id}.words.wx.jsonl")
    replaced = _align_timing(canonical, wx) if wx else 0
    _enforce_monotonic(canonical)

    # Re-derive speaker labels from diarisation by overlap, then smooth noise —
    # fixes caption colours bleeding across speaker boundaries.
    _assign_diar_labels(canonical, _load_diar(catalog, media_id))
    _smooth_labels(canonical)

    pmap = _person_map(catalog, media_id)
    for w in canonical:
        w["person_id"] = pmap.get(w.get("speaker_label") or "")

    sent_idx = _sentence_indices(canonical)

    rows = []
    for i, w in enumerate(canonical):
        if not w["text"]:
            continue
        rows.append({
            "word_index": i,
            "text": w["text"],
            "start_ms": w["start_ms"],
            "end_ms": w["end_ms"],
            "speaker_label": w.get("speaker_label"),
            "person_id": w.get("person_id"),
            "sentence_index": sent_idx[i],
        })

    _write_rows_atomic(asr, out_path, rows)

    n_words = len(rows)
    n_sent = (sent_idx[-1] + 1) if sent_idx else 0
    # Phase 3 (D-Log-04 §4, §2 root cause #1): this is the "computes
    # pct_person_id then discards it" line the redesign doc names — pct_person
    # now flows into `counts`/health instead of only into the return value.
    pct_person = _pct_person_id(rows)
    stats = {"n_words": n_words, "n_sentences": n_sent, "wx_timed": replaced,
             "pct_person_id": pct_person, "wx_available": bool(wx)}
    has_gallery = _has_gallery_slot(catalog, media_id)
    values = {"pct_person_id": pct_person} if has_gallery else {}
    health = (_stages_registry.evaluate_health("T0", values)
              if (_stages_registry is not None and values) else None)
    outcomes.record(media_id, "T0", "ok",
                    counts={"words": n_words, "sentences": n_sent, "wx_timed": replaced,
                            "pct_person_id": pct_person},
                    health=health, inputs=_identity_map_fp(catalog, media_id))
    outcomes.set_flag(media_id, "has_transcript", True)
    log.info("[%s] words=%d sentences=%d wx_timed=%d person_id=%.1f%%",
             media_id, n_words, n_sent, replaced, pct_person)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="T0 — unified canonical transcript builder")
    ap.add_argument("--catalog", default="/workspace/erf/catalog")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--media-ids", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    catalog = Path(args.catalog)
    if args.all:
        ids = sorted(p.name[: -len(".words.jsonl")]
                     for p in (catalog / "asr").glob("*.words.jsonl"))
    else:
        ids = args.media_ids
    log.info("T0 transcript: %d media", len(ids))

    ok = skip = err = 0
    for mid in ids:
        try:
            # Single-writer: process_media records every terminal outcome
            # (ok/empty/skip) via erf.outcomes itself. Wrapping in stage_run
            # double-wrote a bare journal ok over each no_speech skip
            # (53 phantom-ok in reconcile, found 2026-07-03).
            r = process_media(mid, catalog, args.force)
            if r is None:
                skip += 1
            else:
                ok += 1
        except Exception as e:
            err += 1
            # Phase 4 (D-Log-04 §4, P4): no silent swallow — outcomes is a hard
            # dependency (line 56) and record() failures must surface, not be
            # caught-and-dropped. A missed batch-level escape from the initial
            # Phase 4 pass.
            outcomes.record(mid, "T0", "error", reason=str(e)[:500])
            log.error("[%s] FAILED: %s", mid, e, exc_info=True)
    log.info("Done. ok=%d skip=%d err=%d", ok, skip, err)


if __name__ == "__main__":
    main()

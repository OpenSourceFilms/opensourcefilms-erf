#!/usr/bin/env python3
"""recall_pass.py — targeted plain-Whisper second pass over ASR recall-gap windows.

Recovers words the canonical WhisperX path dropped. Writes CANDIDATE words only
(catalog/asr_recall_pass/{media_id}.jsonl); never touches canonical artifacts.
Speaker attribution is deliberately NOT done here — a later stage owns it.
"""
import os
import sys
import json
import argparse
import logging
import tempfile
import subprocess
import string
import itertools
from pathlib import Path

# Outcome spine (MASTER §5.2.1): stage RP records an explicit outcome per media at
# each exit path (the main loop only sees a tuple, so recording lives inside
# process_media where wrote-empty vs did-not-write is distinguishable). Guarded.
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from erf import outcomes
except Exception:  # pragma: no cover
    outcomes = None

MODEL_PATH = (
    "/workspace/pipeline/models/whispercpp/models--Systran--faster-whisper-large-v3/"
    "snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
)
os.environ["HF_HUB_OFFLINE"] = "1"
_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        try:
            _model = WhisperModel(MODEL_PATH, device="cuda", compute_type="float16")
        except Exception as e:
            # CPU-only pod (or CUDA init failure): int8 CPU is slower but correct.
            logging.warning("RP: cuda model load failed (%s); falling back to cpu int8", e)
            _model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
    return _model


# ---- hallucination gates (bad recovery is worse than omission) ----
_HALLUCINATION_GENERIC = {"thank you", "thanks for watching", "thank you very much",
                          "please subscribe", "you", "."}
_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    return s.translate(_PUNCT_TRANS).lower().strip()


_NORMALIZED_GENERIC = {_normalize(t) for t in _HALLUCINATION_GENERIC}


def is_repetition_hallucination(text: str) -> bool:
    """Same token repeated >=6x, or a phrase (<=3 words) repeated >=4x consecutively."""
    tokens = text.lower().strip().split()
    if not tokens:
        return False
    if max(len(list(g)) for _, g in itertools.groupby(tokens)) >= 6:
        return True
    for plen in range(1, 4):
        if len(tokens) < plen * 4:
            continue
        for i in range(len(tokens) - plen + 1):
            phrase = tuple(tokens[i:i + plen])
            cnt, j = 1, i + plen
            while j + plen <= len(tokens) and tuple(tokens[j:j + plen]) == phrase:
                cnt += 1
                j += plen
            if cnt >= 4:
                return True
    return False


def should_drop_segment(seg) -> bool:
    # avg_logprob is the reliable confidence signal. no_speech_prob is noisy in
    # faster-whisper (clean speech routinely reports nsp>0.8), so it only counts
    # as evidence in combination with weak logprob. The repetition/generic gates
    # are the real safety net against silence hallucinations.
    if seg.avg_logprob < -1.0:
        return True
    if seg.no_speech_prob > 0.85 and seg.avg_logprob < -0.5:
        return True
    if _normalize(seg.text) in _NORMALIZED_GENERIC:
        return True
    if is_repetition_hallucination(seg.text):
        return True
    return False


# ---- source resolution ----
def build_bts_index(bts_root: Path) -> dict:
    index: dict = {}
    for dirpath, _, filenames in os.walk(bts_root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if fn not in index or len(full) < len(index[fn]):
                index[fn] = full
    return index


def resolve_source(relpath: str, bts_root: Path, bts_index: dict):
    candidate = bts_root / relpath
    if candidate.exists():
        return str(candidate)
    base = os.path.basename(relpath)
    return bts_index.get(base)


def extract_audio(source: str, start_s: float, dur_s: float, wav_path: str) -> bool:
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
           "-ss", f"{start_s:.6f}", "-t", f"{dur_s:.6f}", "-i", source,
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:
        logging.error("ffmpeg failed: %s", exc.stderr.decode(errors="replace"))
        return False


_GAP_TYPES = {"monologue_gap", "suspicious_silence", "overlap_gap"}
_PRIO = {"high": {"high"}, "medium": {"high", "medium"},
         "low": {"high", "medium", "low"}}


def process_media(media_id, catalog: Path, bts_root: Path, bts_index, min_priority):
    gaps_path = catalog / "asr_recall_gaps" / f"{media_id}.jsonl"
    out_path = catalog / "asr_recall_pass" / f"{media_id}.jsonl"
    manifest_path = catalog / "media_manifest" / f"{media_id}.json"

    if not gaps_path.exists():
        logging.warning("MID %s: no gaps file, skipping", media_id)
        if outcomes:
            outcomes.record(media_id, "RP", "skip", reason="no_gaps_file", producer="recall_pass.py")
        return (0, 0, 0)
    allow = _PRIO[min_priority]
    gaps = []
    with open(gaps_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("gap_type") not in _GAP_TYPES:
                continue
            if row.get("priority", "low") not in allow:
                continue
            gaps.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write(words):
        tmp = out_path.with_name(out_path.name + ".tmp")
        with open(tmp, "w") as fw:
            for r in words:
                fw.write(json.dumps(r) + "\n")
        os.replace(tmp, out_path)

    if not gaps:
        write([])  # explicit empty artifact written → `ok` (artifact present, zero work)
        logging.info("MID %s: 0 windows -> 0 recovered (0 segs dropped)", media_id)
        if outcomes:
            outcomes.record(media_id, "RP", "ok", counts={"n_windows": 0, "n_recovered": 0},
                            artifacts=[f"catalog/asr_recall_pass/{media_id}.jsonl"], producer="recall_pass.py")
        return (0, 0, 0)

    if not manifest_path.exists():
        logging.warning("MID %s: manifest missing, skipping", media_id)
        if outcomes:
            outcomes.record(media_id, "RP", "error", reason="missing_dep:manifest", producer="recall_pass.py")
        return (0, 0, 0)
    relpath = json.load(open(manifest_path)).get("relpath", "")
    source = resolve_source(relpath, bts_root, bts_index)
    if source is None:
        logging.warning("MID %s: source not found for '%s', skipping", media_id, relpath)
        if outcomes:
            outcomes.record(media_id, "RP", "skip", reason="source_unavailable", producer="recall_pass.py")
        return (0, 0, 0)

    model = get_model()
    tmp_dir = Path("/tmp/claude-0")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_words, dropped_segs, n_windows = [], 0, 0
    for gi, gap in enumerate(gaps):
        n_windows += 1
        start_ms, end_ms = int(gap["start_ms"]), int(gap["end_ms"])
        dur_s = (end_ms - start_ms) / 1000.0
        if dur_s <= 0:
            continue
        # D-Recall-01 (2026-07-08): pad the crop with acoustic context on both
        # sides. Whisper transcribes short isolated clips (many gaps are now
        # sub-1500ms, see build_recall_gaps.py MIN_SILENCE_RUN_MS) poorly with
        # zero context -- more likely to trip should_drop_segment's confidence
        # gates for no good reason. Padding words are decoded but never
        # emitted (filtered back to the gap's own [start_ms,end_ms] below), so
        # this only helps decode quality, it can't introduce boundary
        # duplicates against already-canonical neighbouring words.
        PAD_MS = 400
        crop_start_ms = max(0, start_ms - PAD_MS)
        crop_end_ms = end_ms + PAD_MS
        crop_dur_s = (crop_end_ms - crop_start_ms) / 1000.0
        tw = tempfile.NamedTemporaryFile(suffix=".wav", dir=str(tmp_dir), delete=False)
        tw.close()
        try:
            if not extract_audio(source, crop_start_ms / 1000.0, crop_dur_s, tw.name):
                continue
            segs, _ = model.transcribe(tw.name, language="en", beam_size=5,
                                       vad_filter=False, word_timestamps=True)
            for seg in segs:
                if should_drop_segment(seg):
                    dropped_segs += 1
                    continue
                for w in (seg.words or []):
                    if w.probability < 0.35:
                        continue
                    w_start_ms = int(round(w.start * 1000)) + crop_start_ms
                    w_end_ms = int(round(w.end * 1000)) + crop_start_ms
                    if (w_start_ms + w_end_ms) / 2 < start_ms or (w_start_ms + w_end_ms) / 2 > end_ms:
                        continue  # padding-only context word, not part of the flagged gap
                    all_words.append({
                        "media_id": media_id,
                        "start_ms": w_start_ms,
                        "end_ms": w_end_ms,
                        "text": w.word.strip(),
                        "prob": round(w.probability, 4),
                        "source": "recall_pass",
                        "gap_type": gap["gap_type"],
                        "gap_start_ms": start_ms, "gap_end_ms": end_ms,
                        "seg_avg_logprob": round(seg.avg_logprob, 4),
                        "seg_no_speech_prob": round(seg.no_speech_prob, 4),
                        "diar_speakers": gap.get("diar_speakers", []),
                    })
        except Exception:
            logging.exception("MID %s gap %d: error", media_id, gi)
        finally:
            os.path.exists(tw.name) and os.unlink(tw.name)

    all_words.sort(key=lambda r: (r["start_ms"], r["end_ms"]))
    write(all_words)
    logging.info("MID %s: %d windows -> %d recovered (%d segs dropped)",
                 media_id, n_windows, len(all_words), dropped_segs)
    if outcomes:
        outcomes.record(media_id, "RP", "ok",
                        counts={"n_windows": n_windows, "n_recovered": len(all_words),
                                "n_segs_dropped": dropped_segs},
                        artifacts=[f"catalog/asr_recall_pass/{media_id}.jsonl"], producer="recall_pass.py")
    return (n_windows, len(all_words), dropped_segs)


def main():
    ap = argparse.ArgumentParser(description="Recall-pass ASR recovery")
    ap.add_argument("--catalog", default="/workspace/erf/catalog")
    ap.add_argument("--bts-root", default="/workspace/BTS")
    ap.add_argument("--only")
    ap.add_argument("--min-priority", choices=["high", "medium", "low"], default="high")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                        stream=sys.stderr)

    catalog, bts_root = Path(args.catalog), Path(args.bts_root)
    bts_index = build_bts_index(bts_root) if bts_root.exists() else {}
    if not bts_root.exists():
        logging.warning("bts-root %s missing; relpath-only resolution", bts_root)

    gaps_dir = catalog / "asr_recall_gaps"
    if args.only:
        media_ids = [args.only]
    else:
        media_ids = sorted(p.name[:-len(".jsonl")] for p in gaps_dir.glob("*.jsonl"))
    if args.limit is not None:
        media_ids = media_ids[:args.limit]

    tm = tw = twd = tdd = 0
    record_failures = 0
    for mid in media_ids:
        # Phase 4 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §2 root cause #6/P4):
        # per-media isolation — process_media() already records explicit outcomes on
        # its normal exit paths (skip/ok/error); this only catches a genuine crash
        # process_media itself didn't handle, so one media can't kill the batch or
        # silently drop every remaining media's outcome.
        try:
            nw, nr, nd = process_media(mid, catalog, bts_root, bts_index, args.min_priority)
        except Exception as e:
            logging.exception("MID %s: uncaught exception in process_media", mid)
            # The recovery record() itself can raise (P4: no silent swallow).
            # Guard it so a double failure doesn't kill the batch (W2 checkpoint
            # review finding, D-Log-12); surface via stderr + non-zero exit.
            if outcomes:
                try:
                    outcomes.record(mid, "RP", "error",
                                    reason=f"exception: {type(e).__name__}: {e}"[:500],
                                    producer="recall_pass.py")
                except Exception as rec_e:
                    record_failures += 1
                    logging.error("MID %s: RP error outcome could not be recorded: %s: %s",
                                  mid, type(rec_e).__name__, rec_e)
            continue
        if nw > 0:
            tm += 1
            tw += nw
            twd += nr
            tdd += nd
    logging.info("Totals: %d media, %d windows -> %d recovered (%d segs dropped)",
                 tm, tw, twd, tdd)
    if record_failures:
        # P4: an unloggable outcome is a failed run — the batch finished, but
        # not every media's disposition made it to the journal.
        logging.error("%d error outcome(s) could not be recorded — run is FAILED "
                      "(unrecorded dispositions)", record_failures)
        sys.exit(1)


if __name__ == "__main__":
    main()

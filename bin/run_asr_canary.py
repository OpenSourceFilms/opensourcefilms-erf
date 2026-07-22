#!/usr/bin/env python3
"""Run NVIDIA Canary-1B Flash ASR on media and write words+segments JSONL.

Bake-off tool (D-Log-39 Part 2 step 1) — NOT a registry stage: stdout logging
only, no outcome-spine writes. Writes to the catalog root resolved by
erf.paths.catalog_root(), so a shadow bake-off arm sets ERF_CATALOG_ROOT and
production catalog/ is never touched.

Requires: venvs/nemo (NVIDIA NeMo, nvidia/canary-1b-flash weights).

Usage:
  ERF_CATALOG_ROOT=<shadow>/catalog venvs/nemo/bin/python \
      bin/run_asr_canary.py m_abc123 [m_def456 ...] [--force]

LONG-AUDIO CHUNKING (found empirically this session, not in the original
D-Log-38/39 prep which only ever tested a 7s fixture): Canary-1B Flash's
plain model.transcribe() silently truncates around ~14-16s of output,
independent of the requested chunk_len_in_secs used by NVIDIA's own
chunked-inference reference script (examples/asr/asr_chunked_inference/aed/
speech_to_text_aed_chunked_infer.py, chunk_len_in_secs default 40.0) --
consistent with an output-token-budget cap (config shows max_tps: 25), not
an audio-length cap. That script confirms NVIDIA's OWN reference approach
for non-overlapping chunking is exactly "chunk the input audio, transcribe
each piece independently, concatenate the results" -- this adapter does
that directly (CHUNK_LEN_SEC below, conservative margin under the observed
truncation point) rather than depending on their FrameBatchMultiTaskAED
buffered-feature class, which needs full hydra/manifest config plumbing
not worth adopting here. Word loss at chunk boundaries (mid-utterance
cuts) is the same accepted tradeoff NVIDIA's own non-overlapping default
makes -- not attempting overlap-based stitching.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, wave
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
from erf.ids import segment_id, word_id

CAT = catalog_root()          # honors ERF_CATALOG_ROOT (D-Log-35 seam)
PROXY_DIR = ROOT / "media" / "proxies"
MODEL_ID = "nvidia/canary-1b-flash"
CHUNK_LEN_SEC = 12.0  # conservative margin under the observed ~14-16s single-call truncation point

# Near-inaudible fixture where abstention (0 words) is the CORRECT answer
# (D-Log-39 P1 root-cause: honest abstention on unintelligible audio, not a
# defect) — must not be counted as an adapter failure.
NEGATIVE_FIXTURES = {"m_01d094906b30"}


def _extract_wav(media_id: str, tmp_wav: Path) -> bool:
    """Extract 16kHz mono PCM WAV from proxy MP4; returns success."""
    proxy = PROXY_DIR / f"{media_id}.mp4"
    if not proxy.is_file():
        print(f"WARN {media_id}: proxy missing at {proxy}")
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(proxy),
           "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_wav)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and tmp_wav.is_file() and tmp_wav.stat().st_size > 0


def _write_jsonl(records: list[dict], out_path: Path) -> None:
    """Atomically write list of dicts as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def _assign_word_to_segment(word_mid_sec: float, segments: list[dict]) -> str:
    """Map a word midpoint (seconds) to the nearest segment_id.

    Prefers the segment that contains `word_mid_sec`. Falls back to the segment
    whose boundaries are closest to `word_mid_sec`. Always returns a valid id.
    """
    best_id = None
    best_dist = float("inf")

    for seg in segments:
        s = seg["start_ms"] / 1000.0
        e = seg["end_ms"] / 1000.0
        if s <= word_mid_sec <= e:
            return seg["segment_id"]   # exact match
        dist = max(0.0, s - word_mid_sec, word_mid_sec - e)
        if dist < best_dist:
            best_dist = dist
            best_id = seg["segment_id"]

    assert best_id is not None, "BUG: no segments available for word assignment"
    return best_id


def _wav_duration_ms(wav_path: Path) -> int:
    """Return duration of a WAV file in whole milliseconds."""
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            return 0
        return int(round(nframes / rate * 1000))


def _slice_wav_chunks(src_wav: Path, chunk_len_sec: float) -> list[tuple[float, Path]]:
    """Split a 16kHz mono PCM16 WAV into non-overlapping chunk files.

    Returns a list of (chunk_start_sec, chunk_wav_path) in order. Slices raw
    PCM frames directly (no ffmpeg re-invocation per chunk -- the source is
    already 16kHz/mono/16-bit from _extract_wav).
    """
    with wave.open(str(src_wav), "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2
        n_frames = wf.getnframes()
        chunk_frames = int(chunk_len_sec * 16000)
        chunks: list[tuple[float, Path]] = []
        pos = 0
        while pos < n_frames:
            n = min(chunk_frames, n_frames - pos)
            wf.setpos(pos)
            data = wf.readframes(n)
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            out_path = Path(tmp)
            with wave.open(str(out_path), "wb") as out_wf:
                out_wf.setnchannels(1)
                out_wf.setsampwidth(2)
                out_wf.setframerate(16000)
                out_wf.writeframes(data)
            chunks.append((pos / 16000.0, out_path))
            pos += n
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("media_ids", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from nemo.collections.asr.models import EncDecMultiTaskModel
    model = EncDecMultiTaskModel.from_pretrained(MODEL_ID)
    model.eval()

    failed = 0
    for mid in args.media_ids:
        words_path = CAT / "asr" / f"{mid}.words.jsonl"
        segments_path = CAT / "asr" / f"{mid}.segments.jsonl"
        if words_path.exists() and segments_path.exists() and not args.force:
            print(f"SKIP {mid}: exists")
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = Path(f.name)
        chunk_paths: list[Path] = []
        try:
            if not _extract_wav(mid, tmp_wav):
                print(f"FAIL {mid}: audio_extract_failed")
                failed += 1
                continue

            wav_dur_ms = _wav_duration_ms(tmp_wav)
            chunks = _slice_wav_chunks(tmp_wav, CHUNK_LEN_SEC)
            chunk_paths = [p for _, p in chunks]

            full_text_parts: list[str] = []
            raw_segments_all: list[dict] = []
            raw_words_all: list[dict] = []
            inference_failed = False
            for chunk_start_sec, chunk_wav in chunks:
                try:
                    out = model.transcribe(
                        [str(chunk_wav)], batch_size=1, timestamps=True,
                        source_lang="en", target_lang="en", pnc="yes", itn="no",
                        timestamp="yes", diarize="no",
                    )
                except Exception as exc:
                    print(f"FAIL {mid}: inference error on chunk @{chunk_start_sec:.1f}s: {exc}")
                    inference_failed = True
                    break

                hyp = out[0]
                chunk_text = hyp.text.strip()
                if chunk_text:
                    full_text_parts.append(chunk_text)

                try:
                    chunk_segments = hyp.timestamp.get("segment", [])
                except (AttributeError, TypeError):
                    chunk_segments = []
                for seg in chunk_segments:
                    txt = seg.get("segment") or seg.get("text") or seg.get("word")
                    if txt is None or seg.get("start") is None or seg.get("end") is None:
                        continue
                    raw_segments_all.append({
                        "text": txt.strip(),
                        "start": seg["start"] + chunk_start_sec,
                        "end": seg["end"] + chunk_start_sec,
                    })

                try:
                    chunk_words = hyp.timestamp.get("word", [])
                except (AttributeError, TypeError):
                    chunk_words = []
                word_probs = None
                if hasattr(hyp, "word_confidence") and hyp.word_confidence is not None:
                    if len(hyp.word_confidence) == len(chunk_words):
                        word_probs = hyp.word_confidence
                for wi, w in enumerate(chunk_words):
                    w_start_sec = w.get("start")
                    w_end_sec = w.get("end")
                    if w_start_sec is None or w_end_sec is None:
                        continue
                    raw_words_all.append({
                        "word": w["word"],
                        "start": w_start_sec + chunk_start_sec,
                        "end": w_end_sec + chunk_start_sec,
                        "prob": word_probs[wi] if word_probs is not None else 0.0,
                    })

            if inference_failed:
                failed += 1
                continue

            full_text = " ".join(full_text_parts)

            # ---------- segments ----------
            segment_records = []
            if not raw_segments_all:
                segment_records = [{
                    "schema_version": "1.0.0",
                    "segment_id": segment_id(mid, 0),
                    "media_id": mid,
                    "index": 0,
                    "text": full_text,
                    "start_ms": 0,
                    "end_ms": wav_dur_ms,
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "speaker_label": None,
                    "speaker_person_id": None,
                }]
            else:
                for seg_idx, seg in enumerate(raw_segments_all):
                    segment_records.append({
                        "schema_version": "1.0.0",
                        "segment_id": segment_id(mid, seg_idx),
                        "media_id": mid,
                        "index": seg_idx,
                        "text": seg["text"],
                        "start_ms": int(round(seg["start"] * 1000)),
                        "end_ms": int(round(seg["end"] * 1000)),
                        "avg_logprob": None,
                        "no_speech_prob": None,
                        "speaker_label": None,
                        "speaker_person_id": None,
                    })

            if not segment_records:
                print(f"FAIL {mid}: no valid segments found")
                failed += 1
                continue

            # ---------- words ----------
            word_records = []
            for global_word_idx, w in enumerate(raw_words_all):
                word_mid_sec = (w["start"] + w["end"]) / 2.0
                seg_id = _assign_word_to_segment(word_mid_sec, segment_records)
                word_records.append({
                    "schema_version": "1.0.0",
                    "word_id": word_id(mid, global_word_idx),
                    "media_id": mid,
                    "segment_id": seg_id,
                    "index": global_word_idx,
                    "text": w["word"],
                    "start_ms": int(round(w["start"] * 1000)),
                    "end_ms": int(round(w["end"] * 1000)),
                    "prob": w["prob"],
                    "speaker_label": None,
                    "speaker_person_id": None,
                })

            # Write outputs atomically.
            _write_jsonl(segment_records, segments_path)
            _write_jsonl(word_records, words_path)

            # ---------- effect-verification ----------
            with words_path.open("r", encoding="utf-8") as f:
                written_words = [json.loads(line) for line in f if line.strip()]
            with segments_path.open("r", encoding="utf-8") as f:
                written_segments = [json.loads(line) for line in f if line.strip()]

            # 1. Must have words — UNLESS this is a known negative fixture where
            #    abstention is the correct, expected result (D-Log-39 P1).
            if len(written_words) == 0:
                if mid in NEGATIVE_FIXTURES:
                    print(f"OK {mid}: EMPTY transcription (abstention) segments={len(written_segments)} words=0")
                    continue
                print(f"FAIL {mid}: no words produced")
                failed += 1
                continue

            # 2. All word timestamps within wav duration ± 50 ms slack.
            upper_bound_ms = wav_dur_ms + 50
            bad_words = []
            for wrec in written_words:
                if wrec["start_ms"] < -50 or wrec["end_ms"] > upper_bound_ms:
                    bad_words.append(wrec["index"])
            if bad_words:
                print(f"FAIL {mid}: words {bad_words} out of bounds (wav dur {wav_dur_ms} ms)")
                failed += 1
                continue

            # 3. Special canary check for the clean-audio text fixture.
            if mid == "m_9caf6ce31df4" and not full_text:
                print(f"FAIL {mid}: canary text empty (expected non-empty transcript)")
                failed += 1
                continue

            n_words = len(written_words)
            n_segs = len(written_segments)
            min_start = min(wrec["start_ms"] for wrec in written_words)
            max_end = max(wrec["end_ms"] for wrec in written_words)
            preview = full_text[:60]
            print(f"OK {mid}: words={n_words} segments={n_segs} span_ms={min_start}-{max_end} text_preview={preview!r}")

        finally:
            tmp_wav.unlink(missing_ok=True)
            for cp in chunk_paths:
                cp.unlink(missing_ok=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

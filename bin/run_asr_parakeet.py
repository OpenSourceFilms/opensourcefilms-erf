#!/usr/bin/env python3
"""Run NVIDIA Parakeet-TDT-0.6b-v3 ASR on media and write hypothesis
words and segments JSONL.

Bake-off tool (D-Log-39 Part 2 step 1) — NOT a registry stage: stdout logging
only, no outcome‑spine writes. Writes to the catalog root resolved by
erf.paths.catalog_root(), so a shadow bake‑off arm sets ERF_CATALOG_ROOT and
production catalog/ is never touched.

Requires: venvs/nemo (NVIDIA NeMo), huggingface_hub (weights cached offline).
Weights: nvidia/parakeet-tdt-0.6b-v3.

Usage:
  source venvs/nemo/bin/activate && bin/run_asr_parakeet.py m_abc123 [m_def456 ...] [--force]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, wave
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
from erf.ids import segment_id, word_id

CAT = catalog_root()          # honours ERF_CATALOG_ROOT (D‑Log‑35 seam)
PROXY_DIR = ROOT / "media" / "proxies"
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
NEGATIVE_FIXTURES = {"m_01d094906b30"}  # near-inaudible; abstention is a legitimate result


def _extract_wav(media_id: str, tmp_wav: Path) -> bool:
    proxy = PROXY_DIR / f"{media_id}.mp4"
    if not proxy.is_file():
        print(f"WARN {media_id}: proxy missing at {proxy}")
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(proxy),
           "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_wav)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and tmp_wav.is_file() and tmp_wav.stat().st_size > 0


def _write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("media_ids", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Load model once
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(MODEL_ID)
    model.eval()

    failed = 0
    for mid in args.media_ids:
        out_words_path = CAT / "asr" / f"{mid}.words.jsonl"
        out_segments_path = CAT / "asr" / f"{mid}.segments.jsonl"
        if out_words_path.exists() and out_segments_path.exists() and not args.force:
            print(f"SKIP {mid}: both word/segment files exist")
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = Path(f.name)

        try:
            if not _extract_wav(mid, tmp_wav):
                print(f"FAIL {mid}: audio_extract_failed")
                failed += 1
                continue

            # Read WAV duration for fallback / verification
            with wave.open(str(tmp_wav), "rb") as wf:
                wav_duration = wf.getnframes() / wf.getframerate()
            wav_duration_ms = int(round(wav_duration * 1000))

            # Transcribe
            try:
                out = model.transcribe([str(tmp_wav)], batch_size=1, timestamps=True)
                hyp = out[0]
            except Exception as exc:
                print(f"FAIL {mid}: inference error: {exc}")
                failed += 1
                continue

            # --- Parse hypotheses -----------------------------------------------------------
            full_text = hyp.text.strip()
            word_list = hyp.timestamp.get("word", [])
            seg_raw = hyp.timestamp.get("segment", [])

            # Word confidence (if available)
            if hasattr(hyp, "word_confidence") and hyp.word_confidence is not None and len(hyp.word_confidence) == len(word_list):
                word_confs = hyp.word_confidence
            else:
                word_confs = [0.0] * len(word_list)

            # Build segment records
            segments: list[dict] = []
            seg_idx = 0
            segs_processed = False
            if seg_raw:
                for entry in seg_raw:
                    # Defensively pick text field
                    text = entry.get("segment") or entry.get("text") or entry.get("word")
                    if text is None:
                        print(f"WARN {mid}: segment entry {seg_idx} has no recognised text key, skipping")
                        continue
                    seg_start = entry.get("start")
                    seg_end = entry.get("end")
                    if seg_start is None or seg_end is None:
                        print(f"WARN {mid}: segment entry {seg_idx} missing start/end, skipping")
                        continue
                    segments.append({
                        "schema_version": "1.0.0",
                        "segment_id": segment_id(mid, seg_idx),
                        "media_id": mid,
                        "index": seg_idx,
                        "text": text.strip(),
                        "start_ms": int(round(seg_start * 1000)),
                        "end_ms": int(round(seg_end * 1000)),
                        "avg_logprob": None,
                        "no_speech_prob": None,
                        "speaker_label": None,
                        "speaker_person_id": None,
                    })
                    seg_idx += 1
                segs_processed = True

            if not segments:
                # Fallback: one whole‑clip segment
                segments.append({
                    "schema_version": "1.0.0",
                    "segment_id": segment_id(mid, 0),
                    "media_id": mid,
                    "index": 0,
                    "text": full_text,
                    "start_ms": 0,
                    "end_ms": wav_duration_ms,
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "speaker_label": None,
                    "speaker_person_id": None,
                })
                seg_idx = 1  # only one

            # --- Word records (global index across all segments) ----------------------------
            words: list[dict] = []
            word_idx = 0
            for w, conf in zip(word_list, word_confs):
                w_start_ms = int(round(w["start"] * 1000))
                w_end_ms = int(round(w["end"] * 1000))

                # Assign to segment by containment (fallback to nearest if needed)
                seg_id = None
                for seg in segments:
                    if seg["start_ms"] <= w_start_ms <= seg["end_ms"]:
                        seg_id = seg["segment_id"]
                        break
                if seg_id is None:
                    # Nearest segment by start time
                    nearest = min(segments, key=lambda s: abs(s["start_ms"] - w_start_ms))
                    seg_id = nearest["segment_id"]
                    print(f"WARN {mid}: word '{w['word']}' (start_ms={w_start_ms}) not inside any segment, assigned to {nearest['index']}")

                words.append({
                    "schema_version": "1.0.0",
                    "word_id": word_id(mid, word_idx),
                    "media_id": mid,
                    "segment_id": seg_id,
                    "index": word_idx,
                    "text": w["word"],
                    "start_ms": w_start_ms,
                    "end_ms": w_end_ms,
                    "prob": conf,
                    "speaker_label": None,
                    "speaker_person_id": None,
                })
                word_idx += 1

            # Write outputs atomically
            _write_jsonl(segments, out_segments_path)
            _write_jsonl(words, out_words_path)

            # --- Effect verification --------------------------------------------------------
            # Re‑read files to verify
            with out_words_path.open("r", encoding="utf-8") as f:
                written_words = [json.loads(line) for line in f]
            with out_segments_path.open("r", encoding="utf-8") as f:
                written_segments = [json.loads(line) for line in f]

            if len(written_words) == 0:
                if mid in NEGATIVE_FIXTURES:
                    print(f"OK {mid}: EMPTY transcription (abstention) segments={len(written_segments)} words=0")
                else:
                    print(f"FAIL {mid}: verification: 0 words written")
                    failed += 1
                continue

            for w in written_words:
                if not (0 - 50 <= w["start_ms"] <= wav_duration_ms + 50 and
                        0 - 50 <= w["end_ms"]   <= wav_duration_ms + 50):
                    print(f"FAIL {mid}: verification: word index {w['index']} out of bounds "
                          f"(start_ms={w['start_ms']}, end_ms={w['end_ms']}, duration_ms={wav_duration_ms})")
                    failed += 1
                    break
            else:
                # Text canary for known clean audio
                if mid == "m_9caf6ce31df4":
                    if full_text == "":
                        print(f"FAIL {mid}: verification: empty transcript for text canary")
                        failed += 1
                        continue

                # OK print
                if written_words:
                    min_start = min(w["start_ms"] for w in written_words)
                    max_end   = max(w["end_ms"]   for w in written_words)
                else:
                    min_start, max_end = 0, 0
                print(f"OK {mid}: words={len(written_words)} segments={len(written_segments)} "
                      f"span_ms={min_start}-{max_end} text_preview={full_text[:60]!r}")

        except Exception as exc:
            print(f"FAIL {mid}: {exc}")
            failed += 1
        finally:
            tmp_wav.unlink(missing_ok=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

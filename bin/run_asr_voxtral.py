#!/usr/bin/env python3
"""Run the Mistral Voxtral-Mini-4B-Realtime ASR adapter on media and write words/segments JSONL.

Bake-off tool (D-Log-39 Part 2 step 1) — NOT a registry stage: stdout logging
only, no outcome-spine writes. Writes to the catalog root resolved by
erf.paths.catalog_root(), so a shadow bake-off arm sets ERF_CATALOG_ROOT and
production catalog/ is never touched.

Voxtral vLLM server is MANUAL-START infra started separately per the D-Log-39
serving runbook — this script only connects as a client via WebSocket.

NOTE: The realtime protocol returns TEXT ONLY — no native word timestamps.
Per D-Log-39 Part 2 decision 7, ALL arms' scored word timing comes from the
existing D0 whisperX forced-alignment stage run afterward on this adapter's
segments.jsonl text — this adapter's own words.jsonl is therefore a coarse,
evenly-time-distributed placeholder (linear interpolation across the segment
span), not real per-word timing, and is expected to be superseded by D0's
.words.wx.jsonl output before scoring.

Requires: pip package websockets (otherwise standard library).
Weights: remote server, no local model weights needed.

Usage:
  ERF_CATALOG_ROOT=<shadow>/catalog python \
      bin/run_asr_voxtral.py m_abc123 [m_def456 ...] [--force] [--url ws://127.0.0.1:8901/v1/realtime]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
from erf.ids import segment_id, word_id

CAT = catalog_root()   # honors ERF_CATALOG_ROOT
PROXY_DIR = ROOT / "media" / "proxies"


async def transcribe(path: str, url: str) -> str:
    """Voxtral Realtime WebSocket transcription coroutine (proven client v2).

    Args:
        path: 16kHz mono 16-bit PCM WAV file.
        url: WebSocket server URL (e.g. ws://127.0.0.1:8901/v1/realtime).

    Returns:
        Transcribed text string, or "<ERROR code: msg>" on server error.
    """
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
    pcm = w.readframes(w.getnframes())
    chunk_bytes = 16000 * 2 * 500 // 1000   # 500 ms chunks
    text_parts = []
    async with websockets.connect(url, max_size=None) as ws:
        try:
            await asyncio.wait_for(ws.recv(), timeout=10)
        except asyncio.TimeoutError:
            pass
        await ws.send(json.dumps({"type": "session.update",
                                  "model": "mistralai/Voxtral-Mini-4B-Realtime-2602"}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        for i in range(0, len(pcm), chunk_bytes):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + chunk_bytes]).decode(),
            }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            t = msg.get("type")
            if t == "transcription.delta":
                text_parts.append(msg["delta"])
            elif t == "transcription.done":
                return msg["text"]
            elif t == "error":
                return f"<ERROR {msg.get('code')}: {msg.get('error')}>"


def _extract_wav(media_id: str, tmp_wav: Path) -> bool:
    """Extract 16kHz mono PCM WAV from proxy file."""
    proxy = PROXY_DIR / f"{media_id}.mp4"
    if not proxy.is_file():
        print(f"WARN {media_id}: proxy missing at {proxy}")
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(proxy),
           "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_wav)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and tmp_wav.is_file() and tmp_wav.stat().st_size > 0


def _write_jsonl(records: list[dict], out_path: Path) -> None:
    """Atomically write list of dicts as JSONL file."""
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
    ap.add_argument("--url", default="ws://127.0.0.1:8901/v1/realtime",
                    help="Voxtral WebSocket server URL")
    args = ap.parse_args()

    failed = 0
    for mid in args.media_ids:
        seg_path = CAT / "asr" / f"{mid}.segments.jsonl"
        word_path = CAT / "asr" / f"{mid}.words.jsonl"
        if seg_path.exists() and word_path.exists() and not args.force:
            print(f"SKIP {mid}: exists")
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = Path(f.name)
        try:
            if not _extract_wav(mid, tmp_wav):
                print(f"FAIL {mid}: audio_extract_failed")
                failed += 1
                continue

            # Total clip duration in ms
            with wave.open(str(tmp_wav)) as wf:
                duration_ms = round(wf.getnframes() / wf.getframerate() * 1000)

            # Run transcription
            try:
                text = asyncio.run(transcribe(str(tmp_wav), args.url))
            except Exception as e:
                print(f"FAIL {mid}: connection error: {e}")
                failed += 1
                continue

            if text.startswith("<ERROR"):
                print(f"FAIL {mid}: server error: {text}")
                failed += 1
                continue

            text_stripped = text.strip()
            tokens = text_stripped.split() if text_stripped else []

            # Build segment record (one segment spanning the whole clip)
            segment_rec = {
                "schema_version": "1.0.0",
                "segment_id": segment_id(mid, 0),
                "media_id": mid,
                "index": 0,
                "text": text_stripped,
                "start_ms": 0,
                "end_ms": duration_ms,
                "avg_logprob": None,
                "no_speech_prob": None,
                "speaker_label": None,
                "speaker_person_id": None,
            }

            # Build word records with linear interpolation (coarse placeholder)
            word_records = []
            n = len(tokens)
            for idx, token in enumerate(tokens):
                w_start = round(idx * duration_ms / n)
                w_end = round((idx + 1) * duration_ms / n)
                word_records.append({
                    "schema_version": "1.0.0",
                    "word_id": word_id(mid, idx),
                    "media_id": mid,
                    "segment_id": segment_id(mid, 0),
                    "index": idx,
                    "text": token,
                    "start_ms": w_start,
                    "end_ms": w_end,
                    "prob": 0.0,
                    "speaker_label": None,
                    "speaker_person_id": None,
                })

            # Write files atomically
            _write_jsonl([segment_rec], seg_path)
            _write_jsonl(word_records, word_path)  # empty list -> empty file

            # Effect verification: re-read and validate
            with seg_path.open("r", encoding="utf-8") as f:
                seg_lines = [json.loads(line) for line in f if line.strip()]
            with word_path.open("r", encoding="utf-8") as f:
                word_lines = [json.loads(line) for line in f if line.strip()]

            verification_failed = False

            if text_stripped:
                # Non-empty transcription must have words
                if len(word_lines) == 0:
                    print(f"FAIL {mid}: expected >0 words but words file empty")
                    verification_failed = True
                else:
                    for w in word_lines:
                        if not (0 <= w["start_ms"] <= duration_ms and 0 <= w["end_ms"] <= duration_ms):
                            print(f"FAIL {mid}: word timing out of bounds "
                                  f"{w['start_ms']}-{w['end_ms']}")
                            verification_failed = True
                            break

            # Canary fixture assertion: known clean audio must yield non-empty text
            if mid == "m_9caf6ce31df4" and text_stripped == "":
                print(f"FAIL {mid}: expected non-empty transcription for canary but got empty")
                verification_failed = True

            if verification_failed:
                failed += 1
                continue

            # Success output
            if text_stripped:
                n_words = len(tokens)
                print(f"OK {mid}: words={n_words} segments=1 "
                      f"span_ms=0-{duration_ms} text_preview={text_stripped[:60]!r}")
            else:
                print(f"OK {mid}: EMPTY transcription (abstention) segments=1 words=0")

        finally:
            tmp_wav.unlink(missing_ok=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

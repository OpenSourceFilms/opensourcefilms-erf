#!/usr/bin/env python3
"""D0 v2 preprocessor — per-media enhanced alignment artifacts.

Produces three new artifacts per media_id:
  catalog/asr/{id}.words.wx.jsonl  — WhisperX phoneme-aligned words
  catalog/asr/{id}.vad.jsonl       — Silero VAD speech/silence segments
  catalog/asr/{id}.overlap.jsonl   — overlap intervals (from diar turns)

render.py prefers .words.wx.jsonl over .words.jsonl when present.
VAD data is used by render.py to snap cut boundaries to actual silence.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("d0_preprocess")

_ROOT = Path(__file__).resolve().parent.parent

import sys as _sys
_sys.path.insert(0, str(_ROOT / "src"))
try:
    from erf import outcomes as _outcomes
except Exception:
    _outcomes = None
from erf.paths import catalog_root as _catalog_root

_CATALOG = _catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam)
_PROXY_DIR = _ROOT / "media" / "proxies"

DEFAULT_MEDIA_IDS = [
    "m_0e13c93baff4","m_1b1d428710f2","m_28474f1d3e90","m_309fe6dc5879",
    "m_6c3e44cad129","m_75219c4b67e1","m_7f2ea69d12df","m_837dab3dc394",
    "m_9627b0013909","m_9c77e0beed25","m_9d86d08c8774","m_d083ebf46a5f",
    "m_d43c96239b99","m_e1b299626700","m_e8c4088868b6","m_f097503e01df",
    "m_f17689b8cfac","m_f2098b3d8b7a","m_f5dd6a33ea3c","m_fb752be369ca",
]

# Module-level model cache
_silero_model = None
_wx_align_model = None
_wx_align_meta = None


def _atomic_write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        os.rename(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _extract_audio(proxy: Path, out_wav: Path) -> None:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(proxy), "-ac", "1", "-ar", "16000",
         "-vn", "-acodec", "pcm_s16le", str(out_wav), "-loglevel", "error"],
        check=True,
    )


def _has_audio_stream(proxy: Path) -> bool:
    """True if the proxy has at least one audio stream. A media with none has
    nothing for D0 to extract/align — ffmpeg -acodec would fail and was being
    mislabelled as a D0 error. Fail-open (return True) if the probe itself errors,
    so a probe glitch never silently skips a media that does have audio."""
    import static_ffmpeg
    static_ffmpeg.add_paths()
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(proxy)],
            capture_output=True, text=True, timeout=30,
        )
        return "audio" in out.stdout
    except Exception:
        return True


def _get_silero():
    global _silero_model
    if _silero_model is None:
        log.info("Loading Silero VAD model…")
        from silero_vad import load_silero_vad
        _silero_model = load_silero_vad()
    return _silero_model


def _get_wx_align(device="cuda"):
    global _wx_align_model, _wx_align_meta
    if _wx_align_model is None:
        log.info("Loading WhisperX alignment model…")
        import whisperx
        _wx_align_model, _wx_align_meta = whisperx.load_align_model("en", device=device)
    return _wx_align_model, _wx_align_meta


def run_vad(media_id: str, audio_wav: Path, force: bool, device: str) -> None:
    out = _CATALOG / "asr" / f"{media_id}.vad.jsonl"
    if not force and out.exists():
        log.info("[%s] vad: exists — skip", media_id)
        return

    import torch
    import soundfile as sf
    from silero_vad import get_speech_timestamps

    model = _get_silero()
    audio_np, sr = sf.read(str(audio_wav))
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    audio_t = torch.tensor(audio_np, dtype=torch.float32)
    total_ms = int(len(audio_t) / 16)

    # get_speech_timestamps returns sample indices (not seconds)
    speech_ts = get_speech_timestamps(audio_t, model, sampling_rate=16000)

    segs = []
    cursor = 0
    for ts in speech_ts:
        s_ms = int(ts["start"] / 16)
        e_ms = int(ts["end"] / 16)
        if cursor < s_ms:
            segs.append({"start_ms": cursor, "end_ms": s_ms, "is_speech": False})
        segs.append({"start_ms": s_ms, "end_ms": e_ms, "is_speech": True})
        cursor = e_ms
    if cursor < total_ms:
        segs.append({"start_ms": cursor, "end_ms": total_ms, "is_speech": False})

    _atomic_write_jsonl(out, segs)
    log.info("[%s] vad: %d segs (%d speech)", media_id, len(segs), len(speech_ts))


def run_whisperx(media_id: str, audio_wav: Path, force: bool, device: str) -> None:
    out = _CATALOG / "asr" / f"{media_id}.words.wx.jsonl"
    if not force and out.exists():
        log.info("[%s] wx: exists — skip", media_id)
        return

    seg_path = _CATALOG / "asr" / f"{media_id}.segments.jsonl"
    if not seg_path.exists():
        log.warning("[%s] wx: no segments.jsonl — skip", media_id)
        return

    import whisperx

    segs_raw = [json.loads(l) for l in seg_path.read_text().splitlines() if l.strip()]
    # whisperx.align needs seconds
    wx_segs = [
        {"start": s["start_ms"] / 1000.0, "end": s["end_ms"] / 1000.0, "text": s["text"]}
        for s in segs_raw
    ]

    audio = whisperx.load_audio(str(audio_wav))
    model_a, meta = _get_wx_align(device)
    result = whisperx.align(wx_segs, model_a, meta, audio, device, return_char_alignments=False)

    words = []
    for ws in result.get("word_segments", []):
        w = ws.get("word", ws.get("text", "")).strip()
        if not w:
            continue
        s_ms = int(ws.get("start", 0) * 1000)
        e_ms = int(ws.get("end", 0) * 1000)
        if e_ms <= s_ms:
            e_ms = s_ms + 50
        words.append({"text": w, "start_ms": s_ms, "end_ms": e_ms,
                       "score": round(float(ws.get("score", 0.0)), 4)})

    if not words:
        log.warning("[%s] wx: produced no words — skip write", media_id)
        return

    _atomic_write_jsonl(out, words)
    log.info("[%s] wx: %d words", media_id, len(words))


def run_overlap(media_id: str, force: bool) -> None:
    out = _CATALOG / "asr" / f"{media_id}.overlap.jsonl"
    if not force and out.exists():
        log.info("[%s] overlap: exists — skip", media_id)
        return

    diar_path = _CATALOG / "asr" / f"{media_id}.diar.jsonl"
    if not diar_path.exists():
        log.info("[%s] overlap: no diar.jsonl — writing empty", media_id)
        _atomic_write_jsonl(out, [])
        return

    turns = [json.loads(l) for l in diar_path.read_text().splitlines() if l.strip()]
    turns.sort(key=lambda t: t["start_ms"])

    # Sweep-line: find regions where ≥2 speaker turns overlap
    events = []
    for t in turns:
        if t["end_ms"] > t["start_ms"]:
            events.append((t["start_ms"], +1))
            events.append((t["end_ms"], -1))
    events.sort(key=lambda x: (x[0], -x[1]))  # starts before ends at same time

    overlaps = []
    active = 0
    ov_start = None
    for ts, delta in events:
        active += delta
        if active >= 2 and ov_start is None:
            ov_start = ts
        elif active < 2 and ov_start is not None:
            if ts > ov_start:
                overlaps.append({"start_ms": ov_start, "end_ms": ts})
            ov_start = None

    # Merge adjacent
    merged = []
    for ov in overlaps:
        if merged and ov["start_ms"] <= merged[-1]["end_ms"]:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], ov["end_ms"])
        else:
            merged.append(dict(ov))

    _atomic_write_jsonl(out, merged)
    log.info("[%s] overlap: %d intervals", media_id, len(merged))


def process(media_id: str, force: bool, vad_only: bool, device: str) -> None:
    proxy = _PROXY_DIR / f"{media_id}.mp4"
    if not proxy.exists():
        log.error("[%s] proxy not found: %s", media_id, proxy)
        return

    if not _has_audio_stream(proxy):
        log.info("[%s] no audio stream — skip D0 (nothing to extract/align)", media_id)
        return

    run_overlap(media_id, force)

    need_audio = force or not (_CATALOG / "asr" / f"{media_id}.vad.jsonl").exists()
    if not vad_only:
        need_audio = need_audio or not (_CATALOG / "asr" / f"{media_id}.words.wx.jsonl").exists()

    if not need_audio:
        log.info("[%s] all audio artifacts present — skip audio extraction", media_id)
        return

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wav = Path(tmp)
    try:
        log.info("[%s] extracting audio…", media_id)
        _extract_audio(proxy, wav)
        run_vad(media_id, wav, force, device)
        if not vad_only:
            run_whisperx(media_id, wav, force, device)
    finally:
        if wav.exists():
            wav.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="D0 v2 — VAD + WhisperX alignment preprocessor")
    ap.add_argument("media_ids", nargs="*", help="Media IDs (default: all 20 active)")
    ap.add_argument("--all", action="store_true", help="Process every media in catalog/media_manifest")
    ap.add_argument("--force", action="store_true", help="Reprocess even if artifacts exist")
    ap.add_argument("--vad-only", action="store_true", help="Skip WhisperX (VAD + overlap only)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.all:
        from pathlib import Path as _P
        ids = sorted(p.stem for p in (_P("/workspace/erf/catalog/media_manifest")).glob("m_*.json"))
    else:
        ids = args.media_ids if args.media_ids else DEFAULT_MEDIA_IDS
    log.info("Processing %d media IDs", len(ids))
    for i, mid in enumerate(ids, 1):
        log.info("===== %s (%d/%d) =====", mid, i, len(ids))
        try:
            # Single-writer: the outcomes.record below covers every terminal path
            # (ok/skip from artifact presence, error in the except). Wrapping in
            # stage_run double-wrote a bare journal ok over each no_speech skip.
            process(mid, force=args.force, vad_only=args.vad_only, device=args.device)
            if _outcomes:
                from pathlib import Path as _P
                _vad = _P("/workspace/erf/catalog/asr") / f"{mid}.vad.jsonl"
                _wx = _P("/workspace/erf/catalog/asr") / f"{mid}.words.wx.jsonl"
                # D-Log-16: VAD always writes vad.jsonl, even for a media with
                # zero speech (a single is_speech:false segment) -- checking
                # mere file existence made _hv True unconditionally, so a
                # genuinely no-speech media (0 real VAD segments, no words.wx)
                # still recorded "ok" here. Same phantom-outcome pattern as
                # D-Log-13's tags/F0/REDIARIZE fixes: check content, not
                # presence. words.wx.jsonl has no such false-positive (it's
                # only ever written when there are real aligned words, see
                # the early-return above), so _hw stays a pure existence check.
                _hv = False
                if _vad.exists():
                    try:
                        _hv = any(json.loads(l).get("is_speech") for l in _vad.read_text().splitlines() if l.strip())
                    except Exception:
                        _hv = True  # unreadable/corrupt vad file -- don't mask it as "skip"
                _hw = _wx.exists()
                # D-Log-17: "ok" must mean the registry deliverable (words.wx)
                # exists. VAD-found-speech with zero alignable words is a real
                # ran-but-produced-nothing outcome — 4 media where a VAD blip
                # never survives alignment recorded "ok" forever (phantom vs
                # the reconcile predicate, the last systemic slice of drift-9).
                # empty is exempt from the artifact check for D0 (not in
                # explicit_empty_stages), and has_vad=True stays readable for
                # consumers that only need vad.jsonl.
                if _hw:
                    _st, _rsn = "ok", None
                elif _hv:
                    _st, _rsn = "empty", "no_alignable_words"
                else:
                    _st, _rsn = "skip", "no_speech"
                _outcomes.record(mid, "D0", _st, reason=_rsn)
                _outcomes.set_flag(mid, "has_vad", _hv)
                _outcomes.set_flag(mid, "has_wx_align", _hw)
        except Exception as e:
            if _outcomes:
                _outcomes.record(mid, "D0", "error", reason=str(e)[:200])
            log.error("[%s] FAILED: %s", mid, e, exc_info=True)
    log.info("Done.")

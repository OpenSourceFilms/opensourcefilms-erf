#!/usr/bin/env python3
"""Audio QC — per-utterance speech loudness + SNR for a clip's dialogue.

Foundation for the dialogue-aware leveler (advisory 2026-06-21): MEASURE + CLASSIFY,
never apply gain here. Emits catalog/audio_qc/{media_id}.json so audio decisions are
objective (not by-ear), and so low-SNR / off-mic speech is FLAGGED ("audio_floor_unusable")
rather than fought with blind normalizers. loudnorm stays the final compliance stage; this
informs the per-segment gain stage that runs before it.

Usage: PYTHONPATH=src venvs/embed/bin/python bin/audio_qc.py <media_id> [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import static_ffmpeg

static_ffmpeg.add_paths()
FFMPEG = shutil.which("ffmpeg")
CAT = Path("/workspace/erf/catalog")
PROXY = Path("/workspace/erf/media/proxies")
SR = 48000
TARGET_DBFS = -23.0   # per-utterance RMS target (proxy for LUFS) before final loudnorm
SNR_FAIL = 6.0
SNR_WARN = 12.0


def _extract_wav(mid: str) -> tuple[np.ndarray, int]:
    src = PROXY / f"{mid}.mp4"
    if not src.exists():
        raise FileNotFoundError(f"proxy not found: {src}")
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="aqc_")
    os.close(fd)
    try:
        subprocess.run([FFMPEG, "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), tmp],
                       capture_output=True, check=True)
        samples, sr = sf.read(tmp, dtype="float32")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-9)


def _load_words(mid: str) -> list[dict]:
    p = CAT / "asr" / f"{mid}.transcript.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def group_utterances(words: list[dict], max_gap_ms: int = 700) -> list[dict]:
    """Contiguous same-speaker words (inter-word gap <= max_gap_ms) → one utterance."""
    ws = sorted(words, key=lambda w: int(w["start_ms"]))
    utts: list[dict] = []
    cur: list[dict] = []
    for w in ws:
        if cur:
            same_spk = w.get("speaker_label") == cur[-1].get("speaker_label")
            gap = int(w["start_ms"]) - int(cur[-1]["end_ms"])
            if not same_spk or gap > max_gap_ms:
                utts.append(_finalise_utt(cur))
                cur = []
        cur.append(w)
    if cur:
        utts.append(_finalise_utt(cur))
    return utts


def _finalise_utt(words: list[dict]) -> dict:
    return {
        "speaker_label": words[0].get("speaker_label"),
        "start_ms": int(words[0]["start_ms"]),
        "end_ms": int(words[-1]["end_ms"]),
        "text": " ".join(w.get("text", "") for w in words).strip(),
        "n_words": len(words),
    }


def noise_floor_db(samples: np.ndarray, sr: int) -> float:
    """Robust room/noise floor = 10th-percentile RMS over 50ms frames."""
    fr = int(0.05 * sr)
    if fr <= 0 or samples.size < fr:
        return _rms_db(samples)
    n = samples.size // fr
    frames = samples[:n * fr].reshape(n, fr)
    rms = np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9
    db = 20.0 * np.log10(rms)
    return float(np.percentile(db, 10))


def _seg(samples: np.ndarray, sr: int, a_ms: int, b_ms: int) -> np.ndarray:
    return samples[int(a_ms / 1000 * sr):int(b_ms / 1000 * sr)]


def classify(snr_db: float) -> tuple[float, str]:
    if snr_db < 6.0:
        return 0.0, "audio_floor_unusable"
    if snr_db < 8.0:
        return 2.0, "risky"
    if snr_db < 12.0:
        return 4.0, "ok_low_snr"
    return 6.0, "ok"


def measure(mid: str) -> dict:
    samples, sr = _extract_wav(mid)
    words = _load_words(mid)
    nf = noise_floor_db(samples, sr)
    utts = group_utterances(words)

    seg_rms: list[float] = []
    by_spk: dict[str, list[float]] = {}
    for u in utts:
        srms = _rms_db(_seg(samples, sr, u["start_ms"], u["end_ms"]))
        snr = srms - nf
        max_boost, status = classify(snr)
        gain = max(-6.0, min(max_boost, TARGET_DBFS - srms))
        u["speech_rms_db"] = round(srms, 1)
        u["snr_db"] = round(snr, 1)
        u["recommended_gain_db"] = round(gain, 1)
        u["status"] = status
        seg_rms.append(srms)
        by_spk.setdefault(u["speaker_label"], []).append(srms)

    try:
        import pyloudnorm as pyln
        integrated = round(float(pyln.Meter(sr).integrated_loudness(samples)), 1)
    except Exception:
        integrated = None

    spk_med = {k: round(float(np.median(v)), 1) for k, v in by_spk.items() if len(v) >= 2}
    delta = round(max(spk_med.values()) - min(spk_med.values()), 1) if len(spk_med) >= 2 else 0.0

    return {
        "media_id": mid,
        "noise_floor_db": round(nf, 1),
        "integrated_lufs": integrated,
        "n_utterances": len(utts),
        "speech_rms_db_median": round(float(np.median(seg_rms)), 1) if seg_rms else None,
        "speech_rms_db_p10": round(float(np.percentile(seg_rms, 10)), 1) if seg_rms else None,
        "speech_rms_db_p90": round(float(np.percentile(seg_rms, 90)), 1) if seg_rms else None,
        "inter_speaker_delta_db": delta,
        "segments_below_snr_warn": sum(1 for u in utts if u["snr_db"] < SNR_WARN),
        "segments_below_snr_fail": sum(1 for u in utts if u["snr_db"] < SNR_FAIL),
        "speakers": {k: {"n": len(v), "median_speech_rms_db": round(float(np.median(v)), 1)}
                     for k, v in by_spk.items()},
        "utterances": utts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media_id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = measure(args.media_id)
    out = Path(args.out) if args.out else CAT / "audio_qc" / f"{args.media_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(f"{args.media_id}: noise_floor={report['noise_floor_db']}dB "
          f"inter_speaker_delta={report['inter_speaker_delta_db']}dB "
          f"fail={report['segments_below_snr_fail']} warn={report['segments_below_snr_warn']} "
          f"lufs={report['integrated_lufs']} -> {out}")


if __name__ == "__main__":
    main()

"""Dialogue-aware segment leveler (D-Audio-Seg-01).

Per-utterance gain on SPEECH regions only — boosts quiet speech (e.g. an off-mic
second speaker) toward a target, SNR-gated, leaving the gaps (silence / room noise) at
unity. That's the decisive difference from global normalizers (speechnorm/dynaudnorm),
which lift the noise floor. Operates on the render's concatenated output timeline; runs
DF3 → [this] → loudnorm. cap_clean=+15 chosen 2026-06-21 by ear. Never raises; on any
error the caller falls back to the un-leveled audio.
"""
from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import soundfile as sf


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-9) if x.size else -120.0


def _noise_floor_db(s: np.ndarray, sr: int) -> float:
    fr = int(0.05 * sr)
    if fr <= 0 or s.size < fr:
        return _rms_db(s)
    n = s.size // fr
    f = s[:n * fr].reshape(n, fr)
    return float(np.percentile(20.0 * np.log10(np.sqrt(np.mean(f ** 2, axis=1)) + 1e-9), 10))


def apply_to_wav(wav_in: str, wav_out: str, utterances: list[dict], *,
                 target_dbfs: float = -23.0, cap_clean: float = 15.0, cap_normal: float = 6.0,
                 cap_risky: float = 2.0, snr_fail: float = 6.0, snr_norm: float = 8.0,
                 snr_clean: float = 12.0, max_cut: float = -6.0,
                 fade_in_ms: int = 60, fade_out_ms: int = 100,
                 stats_out: Optional[str] = None) -> dict:
    """Apply per-utterance gain to wav_in → wav_out. utterances: [{start_ms,end_ms,speaker_label}]
    on the wav's own timeline. Returns stats; optionally writes them to stats_out (QC sidecar)."""
    s, sr = sf.read(wav_in, dtype="float32")
    mono = s if s.ndim == 1 else s.mean(axis=1)
    nf = _noise_floor_db(mono, sr)
    gain = np.ones(mono.shape[0], dtype=np.float32)
    applied: list[dict] = []
    for u in utterances:
        a = max(0, int(u["start_ms"] / 1000 * sr))
        b = min(mono.shape[0], int(u["end_ms"] / 1000 * sr))
        if b <= a:
            continue
        srms = _rms_db(mono[a:b])
        snr = srms - nf
        if snr < snr_fail:
            applied.append({**u, "status": "audio_floor_unusable", "gain_db": 0.0, "snr_db": round(snr, 1)})
            continue
        cap = cap_clean if snr >= snr_clean else (cap_normal if snr >= snr_norm else cap_risky)
        g_db = max(max_cut, min(cap, target_dbfs - srms))
        g = 10.0 ** (g_db / 20.0)
        env = np.full(b - a, g, dtype=np.float32)
        fi = min(int(fade_in_ms / 1000 * sr), (b - a) // 2)
        fo = min(int(fade_out_ms / 1000 * sr), (b - a) // 2)
        if fi > 0:
            env[:fi] = np.linspace(1.0, g, fi)
        if fo > 0:
            env[-fo:] = np.linspace(g, 1.0, fo)
        gain[a:b] = env
        applied.append({**u, "status": "leveled", "gain_db": round(g_db, 1), "snr_db": round(snr, 1)})

    out = s * gain if s.ndim == 1 else s * gain[:, None]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)  # leave headroom; loudnorm sets final true peak
    sf.write(wav_out, out, sr, subtype="PCM_16")

    stats = {"noise_floor_db": round(nf, 1), "n_utterances": len(applied),
             "n_unusable": sum(1 for a in applied if a["status"] == "audio_floor_unusable"),
             "applied": applied}
    if stats_out:
        with open(stats_out, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=1)
    return stats

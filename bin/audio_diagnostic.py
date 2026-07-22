#!/usr/bin/env python3
"""Audio diagnostic harness — exports labeled A/B variants of a 10s input for comparison.

Usage:
    venvs/audio/bin/python bin/audio_diagnostic.py <input_wav> [--out-dir <dir>]

Output files (in --out-dir, default catalog/exports/audio_diag_<basename>):
    raw.wav                  — input as-is
    df3_only.wav             — DF3 denoising only, no loudnorm
    <variant>.wav            — per VARIANTS dict
    blend_XX.wav             — linear blend of df3_atten12+loudnorm and raw+loudnorm
    freqsplit_*.wav          — frequency-split blend (lows from DF3, highs from raw/atten6)

Promotion criteria (must pass ALL before production use):
    1. No audible gating — secondary speakers present in raw must be audible in output
    2. No missing secondary speaker words — spot-check off-axis dialogue
    3. No speaker identity shift — each speaker must sound like themselves
    4. No ASR transcript regression >5% on known dialogue regions
    5. No more than 3 dB short-term energy loss during identity_overlay speech spans

All outputs written atomically. Prints JSON summary to stdout.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import static_ffmpeg

static_ffmpeg.add_paths()
FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    raise RuntimeError("ffmpeg not found after static_ffmpeg.add_paths()")

AUDIO_PYTHON = Path(__file__).parent.parent / "venvs" / "audio" / "bin" / "python"
AUDIO_RESTORE = Path(__file__).parent / "audio_restore.py"

_BASE_CFG = {
    "eq_enabled": True,
    "df3_enabled": True,
    "loudnorm_enabled": True,
    "highpass_hz": 65.0,
    "low_mid_cut_hz": 150.0,
    "low_mid_cut_q": 0.8,
    "low_mid_cut_db": -2.5,
    "loudnorm_target_lufs": -18.0,
    "loudnorm_lra": 11.0,
    "loudnorm_true_peak": -1.5,
}

PRODUCTION_CFG = {**_BASE_CFG}

# ── Shared normalized-source configs ─────────────────────────────────────────

_LOUDNORM_ONLY_CFG = {
    "eq_enabled": False, "df3_enabled": False, "loudnorm_enabled": True,
    "loudnorm_target_lufs": -18.0, "loudnorm_lra": 11.0, "loudnorm_true_peak": -1.5,
}

_DF3_ATTEN12_NORM_CFG = {
    "eq_enabled": False, "df3_enabled": True, "loudnorm_enabled": True,
    "df3_atten_lim_db": 12.0,
    "loudnorm_target_lufs": -18.0, "loudnorm_lra": 11.0, "loudnorm_true_peak": -1.5,
}

_DF3_ATTEN6_NORM_CFG = {
    "eq_enabled": False, "df3_enabled": True, "loudnorm_enabled": True,
    "df3_atten_lim_db": 6.0,
    "loudnorm_target_lufs": -18.0, "loudnorm_lra": 11.0, "loudnorm_true_peak": -1.5,
}

# ── Session-level variant config ──────────────────────────────────────────────

# Standard single-input variants (update per session)
_DF3_ONLY = {"eq_enabled": False, "df3_enabled": True, "loudnorm_enabled": False}
VARIANTS: dict = {}

# Linear blend: (1-r)*df3_atten12_norm + r*raw_norm → peak-norm → loudnorm
BLEND_RATIOS: dict[str, float] = {}

# Frequency-split blend: lows from low_source, highs from high_source.
# Each entry: (low_source_key, high_source_key, crossover_hz, low_hp_hz|None)
# low_hp_hz: optional highpass on the low band before lowpass (trims rumble)
# Source keys: "atten12_norm", "atten6_norm", "raw_norm"
FREQSPLIT_VARIANTS: dict[str, tuple] = {
    # atten12 lows (200Hz-1500Hz bandpass — slightly more bass trim) + atten6 highs
    "freqsplit_atten6_1500_hp200": ("atten12_norm", "atten6_norm", 1500, 200),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ffmpeg_run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={proc.returncode}: {proc.stderr[-300:]}")
    return proc


def _run_audio_restore(wav_in: str, wav_out: str, cfg: dict) -> dict:
    """Call audio_restore.py with given config, return parsed payload."""
    cmd = [
        str(AUDIO_PYTHON), str(AUDIO_RESTORE),
        wav_in, wav_out,
        "--config-json", json.dumps(cfg),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line.strip())
    return {"ok": False, "error": proc.stderr[-200:]}


def _run_blend_variant(
    df3_norm_wav: str, raw_norm_wav: str, ratio: float, out_path: str,
) -> dict:
    """Blend df3_norm and raw_norm: (1-r)*df3 + r*raw, peak-norm, then loudnorm."""
    df3_data, sr = sf.read(df3_norm_wav, dtype="float32", always_2d=False)
    raw_data, _ = sf.read(raw_norm_wav, dtype="float32", always_2d=False)

    min_len = min(len(df3_data), len(raw_data))
    df3_data, raw_data = df3_data[:min_len], raw_data[:min_len]
    if df3_data.ndim == 1 and raw_data.ndim > 1:
        df3_data = df3_data[:, None]
    if raw_data.ndim == 1 and df3_data.ndim > 1:
        raw_data = raw_data[:, None]

    mixed = (1.0 - ratio) * df3_data + ratio * raw_data
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed * (0.891 / peak)

    tmp = out_path + ".premix.wav"
    sf.write(tmp, mixed, sr, format="WAV", subtype="PCM_16")
    result = _run_audio_restore(tmp, out_path, _LOUDNORM_ONLY_CFG)
    try:
        Path(tmp).unlink()
    except OSError:
        pass
    return result


def _run_freqsplit_variant(
    low_source_wav: str, high_source_wav: str, crossover_hz: int, out_path: str,
    low_hp_hz: int | None = None,
) -> dict:
    """Take lows from low_source (lowpass), highs from high_source (highpass),
    sum, peak-normalize, final loudnorm pass.
    low_hp_hz: if set, apply a highpass at this frequency on the low band before lowpass
    (trims sub-N Hz rumble without touching vocal body)."""
    tmp_low = out_path + ".low.wav"
    tmp_high = out_path + ".high.wav"

    low_filter = (f"highpass=f={low_hp_hz}:p=2,lowpass=f={crossover_hz}:p=2"
                  if low_hp_hz else f"lowpass=f={crossover_hz}:p=2")
    _ffmpeg_run([FFMPEG, "-y", "-i", low_source_wav, "-af", low_filter, tmp_low])
    _ffmpeg_run([FFMPEG, "-y", "-i", high_source_wav,
                 "-af", f"highpass=f={crossover_hz}:p=2", tmp_high])

    low_data, sr = sf.read(tmp_low, dtype="float32", always_2d=False)
    high_data, _ = sf.read(tmp_high, dtype="float32", always_2d=False)

    min_len = min(len(low_data), len(high_data))
    low_data, high_data = low_data[:min_len], high_data[:min_len]

    combined = low_data + high_data
    peak = np.max(np.abs(combined))
    if peak > 0:
        combined = combined * (0.891 / peak)

    tmp_combined = out_path + ".premix.wav"
    sf.write(tmp_combined, combined, sr, format="WAV", subtype="PCM_16")
    result = _run_audio_restore(tmp_combined, out_path, _LOUDNORM_ONLY_CFG)

    for p in (tmp_low, tmp_high, tmp_combined):
        try:
            Path(p).unlink()
        except OSError:
            pass
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_wav")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    inp = Path(args.input_wav).resolve()
    if not inp.exists():
        print(json.dumps({"ok": False, "error": f"input not found: {inp}"}))
        sys.exit(1)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path("/workspace/erf/catalog/exports") / f"audio_diag_{inp.stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    results = {}

    # ── raw ───────────────────────────────────────────────────────────────────
    raw_out = str(out_dir / "raw.wav")
    shutil.copy2(str(inp), raw_out)
    results["raw"] = {"ok": True, "path": raw_out}

    # ── df3_only ──────────────────────────────────────────────────────────────
    df3_out = str(out_dir / "df3_only.wav")
    df3_payload = _run_audio_restore(str(inp), df3_out, {
        "eq_enabled": False, "df3_enabled": True, "loudnorm_enabled": False,
    })
    results["df3_only"] = {"ok": df3_payload.get("ok"), "path": df3_out,
                           "stages": df3_payload.get("stages_run")}

    # ── standard variants ─────────────────────────────────────────────────────
    for name, cfg in VARIANTS.items():
        out = str(out_dir / f"{name}.wav")
        payload = _run_audio_restore(str(inp), out, cfg)
        results[name] = {
            "ok": payload.get("ok"), "path": out,
            "stages": payload.get("stages_run"),
            "input_lufs": payload.get("input_lufs"),
            "output_lufs": payload.get("output_lufs"),
            "true_peak": payload.get("true_peak"),
        }

    # ── blend variants ────────────────────────────────────────────────────────
    if BLEND_RATIOS:
        df3_norm_path = str(out_dir / "_df3_atten12_norm.wav")
        raw_norm_path = str(out_dir / "_raw_norm.wav")
        dp = _run_audio_restore(str(inp), df3_norm_path, _DF3_ATTEN12_NORM_CFG)
        rp = _run_audio_restore(str(inp), raw_norm_path, _LOUDNORM_ONLY_CFG)

        if dp.get("ok") and rp.get("ok"):
            for name, ratio in BLEND_RATIOS.items():
                blend_out = str(out_dir / f"{name}.wav")
                try:
                    bp = _run_blend_variant(df3_norm_path, raw_norm_path, ratio, blend_out)
                    results[name] = {
                        "ok": bp.get("ok"), "path": blend_out,
                        "ratio_raw": ratio,
                        "output_lufs": bp.get("output_lufs"),
                        "true_peak": bp.get("true_peak"),
                    }
                except Exception as exc:
                    results[name] = {"ok": False, "error": str(exc)[:200]}
        else:
            for name in BLEND_RATIOS:
                results[name] = {"ok": False, "error": "blend source generation failed"}

        for p in (df3_norm_path, raw_norm_path):
            try:
                Path(p).unlink()
            except OSError:
                pass

    # ── frequency-split variants ──────────────────────────────────────────────
    if FREQSPLIT_VARIANTS:
        # Build only the sources that are actually referenced
        needed_sources = set()
        for spec in FREQSPLIT_VARIANTS.values():
            needed_sources.update((spec[0], spec[1]))

        source_cfgs = {
            "atten12_norm": _DF3_ATTEN12_NORM_CFG,
            "atten6_norm":  _DF3_ATTEN6_NORM_CFG,
            "raw_norm":     _LOUDNORM_ONLY_CFG,
        }
        source_wavs: dict[str, str] = {}
        for key in needed_sources:
            path = str(out_dir / f"_{key}.wav")
            payload = _run_audio_restore(str(inp), path, source_cfgs[key])
            if payload.get("ok"):
                source_wavs[key] = path
            else:
                source_wavs[key] = ""

        for name, spec in FREQSPLIT_VARIANTS.items():
            low_key, high_key, crossover_hz = spec[0], spec[1], spec[2]
            low_hp_hz = spec[3] if len(spec) > 3 else None
            out = str(out_dir / f"{name}.wav")
            low_wav = source_wavs.get(low_key, "")
            high_wav = source_wavs.get(high_key, "")
            if not low_wav or not high_wav:
                results[name] = {"ok": False, "error": f"source missing: {low_key} or {high_key}"}
                continue
            try:
                payload = _run_freqsplit_variant(low_wav, high_wav, crossover_hz, out, low_hp_hz)
                results[name] = {
                    "ok": payload.get("ok"), "path": out,
                    "low_source": low_key, "high_source": high_key,
                    "crossover_hz": crossover_hz,
                    "low_hp_hz": low_hp_hz,
                    "output_lufs": payload.get("output_lufs"),
                    "true_peak": payload.get("true_peak"),
                }
            except Exception as exc:
                results[name] = {"ok": False, "error": str(exc)[:200]}

        for wav in source_wavs.values():
            try:
                Path(wav).unlink()
            except OSError:
                pass

    # ── Future candidates (stubbed) ───────────────────────────────────────────
    # UVR-De-Echo-Normal, MossFormer2_SE_48K, UVR-DeEcho-DeReverb — see promotion criteria above

    print(json.dumps({
        "ok": True,
        "out_dir": str(out_dir),
        "duration_s": round(time.monotonic() - t0, 2),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()

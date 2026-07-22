#!/usr/bin/env python3
"""Audio restoration cascade: EQ → DF3 → loudnorm.

Subprocess entrypoint called from render.py. Runs inside venvs/audio/.

Usage:
    python audio_restore.py <wav_in> <wav_out> [--config-json '<json>']

Prints exactly one JSON line to stdout on exit.
"""
import json
import os
import re
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

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "eq_enabled": True,
    "df3_enabled": True,
    "loudnorm_enabled": True,
    "freqsplit_enabled": True,   # D-AudioRestore-07: DF3 on HP only, LP preserved
    "freqsplit_hp_hz": 200.0,    # crossover; HP = original - LP (perfect complement)
    "highpass_hz": 65.0,
    "low_mid_cut_hz": 150.0,
    "low_mid_cut_q": 0.8,
    "low_mid_cut_db": -2.5,
    "df3_atten_lim_db": None,
    "loudnorm_target_lufs": -18.0,
    "loudnorm_lra": 11.0,
    "loudnorm_true_peak": -1.5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_sf(path: str):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return data, sr


def _save_sf_atomic(path: str, audio: np.ndarray, sr: int) -> None:
    tmp = path + ".writing.wav"
    sf.write(tmp, audio, sr, format="WAV", subtype="PCM_16")
    os.replace(tmp, path)


def _ffmpeg_run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run ffmpeg, raise RuntimeError on non-zero exit."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={proc.returncode}: {proc.stderr[-400:]}")
    return proc


def _extract_json(text: str) -> dict:
    """Extract the first {...} block from text (handles multi-line ffmpeg JSON)."""
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        raise ValueError("No JSON block found in ffmpeg output")
    return json.loads(m.group())


# ---------------------------------------------------------------------------
# LUFS measurement (pre-cascade)
# ---------------------------------------------------------------------------
def measure_lufs(wav_path: str) -> float | None:
    """Measure integrated LUFS via ffmpeg ebur128. Returns float or None."""
    try:
        proc = subprocess.run(
            [FFMPEG, "-i", wav_path, "-af", "ebur128=target=-23", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        # ebur128 summary line: "  I:        -23.0 LUFS"
        for line in proc.stderr.splitlines():
            stripped = line.strip()
            if stripped.startswith("I:"):
                parts = stripped.split()
                # ["I:", "-23.0", "LUFS"]
                if len(parts) >= 2:
                    return float(parts[1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Stage 1 — EQ
# ---------------------------------------------------------------------------
def run_eq(wav_in: str, wav_out: str, cfg: dict) -> None:
    hz = cfg["highpass_hz"]
    f = cfg["low_mid_cut_hz"]
    q = cfg["low_mid_cut_q"]
    db = cfg["low_mid_cut_db"]
    tmp = wav_out + ".writing.wav"
    _ffmpeg_run([
        FFMPEG, "-y", "-i", wav_in,
        "-af", f"highpass=f={hz},equalizer=f={f}:t=q:w={q}:g={db}",
        tmp,
    ])
    os.replace(tmp, wav_out)


# ---------------------------------------------------------------------------
# Stage 2 — DF3
# ---------------------------------------------------------------------------
def run_df3(wav_in: str, wav_out: str, atten_lim_db: float | None = None) -> None:
    from df.enhance import enhance, init_df, load_audio  # noqa: PLC0415

    tmp = wav_out + ".writing.wav"
    model, df_state, _ = init_df()
    audio, _ = load_audio(wav_in, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
    arr = enhanced.cpu().float().numpy().T
    sf.write(tmp, arr, df_state.sr(), format="WAV", subtype="PCM_16")
    os.replace(tmp, wav_out)


# ---------------------------------------------------------------------------
# Stage 2b — freqsplit DF3 (D-AudioRestore-07)
# ---------------------------------------------------------------------------
def run_freqsplit_df3(wav_in: str, wav_out: str, hp_hz: float, atten_lim_db: float | None = None) -> None:
    """DF3 on HP component only; LP preserved unchanged. HP = original − LP (perfect complement).

    Prevents DF3 from colouring bass/room-tone while still denoising the voice register.
    Falls back to full-range DF3 if scipy is unavailable.
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        run_df3(wav_in, wav_out, atten_lim_db=atten_lim_db)
        return

    data, sr = _load_sf(wav_in)
    mono = data.ndim == 1
    if mono:
        data = data[:, np.newaxis]

    # LP via zero-phase 8th-order Butterworth; HP = original − LP (perfect complement, sums back exactly)
    nyq = sr / 2.0
    sos = butter(8, min(float(hp_hz) / nyq, 0.9999), btype="low", output="sos")
    lp = np.zeros_like(data)
    for ch in range(data.shape[1]):
        lp[:, ch] = sosfiltfilt(sos, data[:, ch])
    hp = data - lp

    hp_wav = wav_in + ".fs_hp.wav"
    hp_denoised_wav = wav_in + ".fs_hp_dn.wav"
    try:
        _save_sf_atomic(hp_wav, hp.squeeze() if mono else hp, int(sr))
        run_df3(hp_wav, hp_denoised_wav, atten_lim_db=atten_lim_db)

        hp_d, sr2 = _load_sf(hp_denoised_wav)
        if hp_d.ndim == 1:
            hp_d = hp_d[:, np.newaxis]

        if int(sr2) != int(sr):
            # DF3 resampled — resample LP to match
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr2), int(sr))
            up, down = int(sr2) // g, int(sr) // g
            lp_r = np.stack([resample_poly(lp[:, c], up, down) for c in range(lp.shape[1])], axis=1)
            n = min(len(lp_r), len(hp_d))
            result = lp_r[:n] + hp_d[:n]
            out_sr = int(sr2)
        else:
            n = min(len(lp), len(hp_d))
            result = lp[:n] + hp_d[:n]
            out_sr = int(sr)

        _save_sf_atomic(wav_out, result.squeeze() if mono else result, out_sr)
    finally:
        for f in (hp_wav, hp_denoised_wav):
            try:
                Path(f).unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Stage 3 — loudnorm two-pass
# ---------------------------------------------------------------------------
def run_loudnorm(wav_in: str, wav_out: str, cfg: dict) -> dict:
    """EBU R128 two-pass loudnorm. Returns {input_lufs, output_lufs, output_true_peak}."""
    I = cfg["loudnorm_target_lufs"]
    lra = cfg["loudnorm_lra"]
    tp = cfg["loudnorm_true_peak"]

    # Pass 1 — measure
    p1 = _ffmpeg_run([
        FFMPEG, "-y", "-i", wav_in,
        "-af", f"loudnorm=I={I}:LRA={lra}:TP={tp}:print_format=json",
        "-f", "null", "-",
    ])
    m1 = _extract_json(p1.stderr)
    measured_i = m1["input_i"]
    measured_lra = m1["input_lra"]
    measured_tp = m1["input_tp"]
    measured_thresh = m1["input_thresh"]
    target_offset = m1["target_offset"]

    # Pass 2 — apply (atomic write)
    tmp = wav_out + ".writing.wav"
    p2 = _ffmpeg_run([
        FFMPEG, "-y", "-i", wav_in,
        "-af", (
            f"loudnorm=I={I}:LRA={lra}:TP={tp}:"
            f"measured_I={measured_i}:measured_LRA={measured_lra}:"
            f"measured_TP={measured_tp}:measured_thresh={measured_thresh}:"
            f"offset={target_offset}:linear=true:print_format=json"
        ),
        "-ar", "48000",
        tmp,
    ])
    os.replace(tmp, wav_out)

    m2 = _extract_json(p2.stderr)
    return {
        "input_lufs": float(measured_i),
        "output_lufs": float(m2.get("output_i", measured_i)),
        "output_true_peak": float(m2.get("output_tp", tp)),
    }


def run_speechnorm(wav_in: str, wav_out: str, cfg: dict) -> None:
    """Adaptive speech leveling (ffmpeg speechnorm).

    Brings quiet speech (e.g. an off-mic second speaker) up toward a consistent level and
    gently tames loud speech, so all dialogue sits at a good volume. `threshold` keeps it
    from amplifying near-silence/room noise. Runs AFTER DF3 (noise floor already reduced)
    and BEFORE loudnorm (the EBU pass sets the final overall target + true-peak limit).
    """
    peak = cfg.get("speechnorm_peak", 0.95)
    exp = cfg.get("speechnorm_expansion", 8.0)
    comp = cfg.get("speechnorm_compression", 2.0)
    thr = cfg.get("speechnorm_threshold", 0.02)
    tmp = wav_out + ".writing.wav"
    _ffmpeg_run([
        FFMPEG, "-y", "-i", wav_in,
        "-af", f"speechnorm=p={peak}:e={exp}:c={comp}:t={thr}",
        "-ar", "48000", tmp,
    ])
    os.replace(tmp, wav_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.monotonic()

    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "usage: audio_restore.py <wav_in> <wav_out>", "stages_run": []}))
        sys.exit(1)

    wav_in = str(Path(sys.argv[1]).resolve())
    wav_out = str(Path(sys.argv[2]).resolve())

    cfg = dict(DEFAULTS)
    if "--config-json" in sys.argv:
        idx = sys.argv.index("--config-json")
        try:
            cfg.update(json.loads(sys.argv[idx + 1]))
        except (IndexError, json.JSONDecodeError) as exc:
            print(json.dumps({"ok": False, "error": f"invalid --config-json: {exc}", "stages_run": []}))
            sys.exit(1)

    stages_run: list[str] = []
    warnings: list[str] = []
    input_lufs: float | None = None
    output_lufs: float | None = None
    true_peak: float | None = None

    input_lufs = measure_lufs(wav_in)

    stage1_out = wav_out + ".stage1.wav"
    stage2_out = wav_out + ".stage2.wav"
    stage2b_out = wav_out + ".stage2b.wav"
    best: str = wav_in  # fallback chain — always valid

    # ── Stage 1: EQ ──────────────────────────────────────────────────────────
    if cfg["eq_enabled"]:
        try:
            run_eq(best, stage1_out, cfg)
            stages_run.append("eq")
            best = stage1_out
        except Exception as exc:
            warnings.append(f"eq_error:{str(exc)[:200]}")

    # ── Stage 2: DF3 (full-range or freqsplit per D-AudioRestore-07) ─────────
    if cfg["df3_enabled"]:
        try:
            if cfg.get("freqsplit_enabled") and cfg.get("freqsplit_hp_hz"):
                run_freqsplit_df3(
                    best, stage2_out,
                    hp_hz=float(cfg["freqsplit_hp_hz"]),
                    atten_lim_db=cfg.get("df3_atten_lim_db"),
                )
                stages_run.append("df3+freqsplit")
            else:
                run_df3(best, stage2_out, atten_lim_db=cfg.get("df3_atten_lim_db"))
                stages_run.append("df3")
            best = stage2_out
        except Exception as exc:
            warnings.append(f"df3_error:{str(exc)[:200]}")

    # ── Stage 2b: adaptive speech leveling (D-Render-11) ─────────────────────
    if cfg.get("speechnorm_enabled"):
        try:
            run_speechnorm(best, stage2b_out, cfg)
            stages_run.append("speechnorm")
            best = stage2b_out
        except Exception as exc:
            warnings.append(f"speechnorm_error:{str(exc)[:200]}")

    # ── Stage 3: loudnorm ────────────────────────────────────────────────────
    if cfg["loudnorm_enabled"]:
        try:
            loud = run_loudnorm(best, wav_out + ".loud.wav", cfg)
            os.replace(wav_out + ".loud.wav", wav_out)
            stages_run.append("loudnorm")
            best = wav_out
            output_lufs = loud["output_lufs"]
            true_peak = loud["output_true_peak"]
        except Exception as exc:
            warnings.append(f"loudnorm_error:{str(exc)[:200]}")

    # ── Finalise ─────────────────────────────────────────────────────────────
    if best != wav_out:
        shutil.copy2(best, wav_out)

    for tmp_path in (stage1_out, stage2_out, stage2b_out):
        try:
            if Path(tmp_path).exists() and tmp_path != wav_out:
                Path(tmp_path).unlink()
        except OSError:
            pass

    payload: dict = {
        "ok": True,
        "stages_run": stages_run,
        "input_lufs": input_lufs,
        "output_lufs": output_lufs,
        "true_peak": true_peak,
        "duration_s": round(time.monotonic() - t_start, 2),
    }
    if warnings:
        payload["warnings"] = warnings

    print(json.dumps(payload))
    sys.exit(0)


if __name__ == "__main__":
    main()

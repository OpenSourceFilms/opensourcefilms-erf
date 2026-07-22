"""Stage Q0 — Quality metrics.

Visual (per keyframe): blur, brightness, contrast, saturation, motion_energy,
camera_shake, b_roll_heuristic, palette (top-5 dominant colours via PIL quantize),
phash, phash_distance_prev, is_near_duplicate, aesthetic_score, best_frame.

Audio (per media): speech_clarity, silence_ratio, noise_ratio, clipping_ratio —
ffmpeg volumedetect/silencedetect.  Plus PANNs CNN14 classification (subprocess).

No paid ML models. All nullable — partial results are fine.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import uuid
from pathlib import Path

from .atomicio import read_jsonl, write_tmp_rename
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "erf_root": "/workspace/erf",
}
_BLUR_THRESHOLD = 100.0   # below this → likely blurry
_MOTION_MIN = 5.0
_MOTION_MAX = 80.0
_MUSIC_ADVISORY_DEFAULT = 0.30

# ── Aesthetic scorer ───────────────────────────────────────────────────────────

_AESTHETIC_MODEL_CACHE: dict = {}
# Guards the lazy CLIP load so concurrent workers (Phase 7 thread pool) don't all
# trigger the load at once — a stampede that thrashes MooseFS I/O and freezes the
# pool. One thread loads; the rest wait and reuse the single cached instance.
_AESTHETIC_MODEL_LOCK = threading.Lock()
_AESTHETIC_POSITIVE_PROMPTS = [
    "sharp focus, well-lit portrait, subject centered, professional photography",
    "clear shot with good lighting and sharp details, cinematic quality",
    "beautiful well-composed frame, in focus, correct exposure",
]
_AESTHETIC_NEGATIVE_PROMPTS = [
    "blurry out of focus photo, motion blur, soft focus, low quality",
    "dark underexposed photo, poor lighting, grainy image",
    "overexposed washed out image, blown highlights, no detail",
]


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
        cfg["audio_music_advisory_threshold"] = raw.get("quality", {}).get(
            "audio_music_advisory_threshold", _MUSIC_ADVISORY_DEFAULT
        )
    return cfg


# ── Visual metrics ─────────────────────────────────────────────────────────────

def _visual_metrics(img_path: Path) -> dict:
    """Compute blur, brightness, contrast, saturation, palette for one image."""
    result = {}
    try:
        import cv2
        import numpy as np

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return result

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        result["blur"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result["brightness"] = float(gray.mean())
        result["contrast"] = float(gray.std())

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        result["saturation"] = float(hsv[:, :, 1].mean())

    except ImportError:
        pass

    # Palette via PIL (no cv2 required)
    try:
        from PIL import Image
        img_pil = Image.open(img_path).convert("RGB")
        # Quantize to 5 colours
        small = img_pil.resize((100, 100))
        q = small.quantize(colors=5, method=Image.Quantize.FASTOCTREE)
        palette_raw = q.getpalette()[:5 * 3]
        hex_colours = []
        for i in range(0, len(palette_raw), 3):
            r, g, b = palette_raw[i], palette_raw[i + 1], palette_raw[i + 2]
            hex_colours.append(f"#{r:02x}{g:02x}{b:02x}")
        result["palette"] = hex_colours[:5]
    except Exception:
        pass

    return result


def _motion_between_frames(prev_path: Path, curr_path: Path) -> dict:
    """Estimate motion_energy (mean abs diff) and camera_shake (optical flow var)."""
    result = {}
    try:
        import cv2
        import numpy as np

        p = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
        c = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
        if p is None or c is None:
            return result
        if p.shape != c.shape:
            c = cv2.resize(c, (p.shape[1], p.shape[0]))

        diff = cv2.absdiff(p, c)
        result["motion_energy"] = float(diff.mean())

        flow = cv2.calcOpticalFlowFarneback(
            p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        result["camera_shake"] = float(mag.std())

    except Exception:
        pass
    return result


def _b_roll_heuristic(metrics: dict) -> bool | None:
    blur = metrics.get("blur")
    motion = metrics.get("motion_energy")
    speech = metrics.get("speech_clarity")

    if blur is None:
        return None

    blur_ok = blur > _BLUR_THRESHOLD
    motion_ok = motion is None or (_MOTION_MIN <= motion <= _MOTION_MAX)
    speech_low = speech is None or speech < 0.3

    return blur_ok and motion_ok and speech_low


# ── Audio metrics ──────────────────────────────────────────────────────────────

def _audio_metrics(media_path: Path) -> dict:
    """Run ffmpeg audio filters to estimate speech/silence/noise/music/clipping."""
    result: dict = {}
    try:
        # volumedetect + silencedetect
        cmd = [
            "ffmpeg", "-i", str(media_path),
            "-af", "volumedetect,silencedetect=noise=-30dB:duration=0.5",
            "-f", "null", "-",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = out.stderr

        # Parse mean_volume (dBFS)
        mv = re.search(r"mean_volume:\s*([\-\d.]+)", stderr)
        mean_vol = float(mv.group(1)) if mv else None

        # Parse max_volume
        xv = re.search(r"max_volume:\s*([\-\d.]+)", stderr)
        max_vol = float(xv.group(1)) if xv else None

        # Clipping: if max_volume >= -1 dBFS
        if max_vol is not None:
            result["clipping_ratio"] = 1.0 if max_vol >= -1.0 else 0.0

        # Silence segments
        silence_ends = re.findall(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", stderr)
        total_silence = sum(float(d) for _, d in silence_ends)

        # Probe duration
        dur_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "csv=p=0", str(media_path)]
        dur_out = subprocess.run(dur_cmd, capture_output=True, text=True)
        try:
            duration = float(dur_out.stdout.strip())
        except ValueError:
            duration = 0.0

        if duration > 0:
            result["silence_ratio"] = min(total_silence / duration, 1.0)
            speech_ratio = max(0.0, 1.0 - result.get("silence_ratio", 0.0))
            # Heuristic: speech_clarity from volume + speech presence
            if mean_vol is not None and mean_vol > -50:
                result["speech_clarity"] = round(speech_ratio * min(1.0, (mean_vol + 50) / 30), 3)
            else:
                result["speech_clarity"] = 0.0

        # noise_ratio: active audio that isn't clearly speech (heuristic)
        sc = result.get("speech_clarity", 0.0) or 0.0
        sr_val = result.get("silence_ratio", 0.0) or 0.0
        active = max(0.0, 1.0 - sr_val)
        result["noise_ratio"] = round(active * (1.0 - sc), 3)

    except Exception:
        pass

    return result


def _audio_classify(media_path: Path, erf_root: Path) -> dict:
    """Run CNN14 classifier via audio_classify venv subprocess. Null on any failure.

    Set ERF_SKIP_AUDIO_CLASSIFY=1 to return nulls immediately (for fast batch re-runs).
    """
    import os as _os
    result: dict = {
        "audio_music_prob": None,
        "audio_speech_prob": None,
        "audio_top_class": None,
        "audio_classifier_model": None,
    }
    if _os.environ.get("ERF_SKIP_AUDIO_CLASSIFY"):
        return result
    try:
        venv_py = erf_root / "venvs" / "audio_classify" / "bin" / "python"
        helper = erf_root / "bin" / "audio_classify_helper.py"
        if not venv_py.exists() or not helper.exists():
            return result

        # Extract mono 32kHz WAV (30s max) to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", str(media_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "32000", "-ac", "1",
                "-t", "30", wav_path,
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, timeout=30)

            r = subprocess.run(
                [str(venv_py), str(helper), wav_path],
                capture_output=True, text=True, timeout=120,
            )
            if r.stdout.strip():
                out = json.loads(r.stdout.strip())
                if "error" not in out:
                    result.update(out)
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass
    return result


def _phash_keyframes(keyframes: list, erf_root: Path) -> dict:
    """Compute pHash per keyframe; detect near-duplicates against previous frame."""
    out: dict = {}
    try:
        import imagehash
        from PIL import Image

        prev_kid: str | None = None
        prev_hash = None
        for kf in keyframes:
            kid = kf["keyframe_id"]
            img_path = erf_root / kf["image_relpath"]
            if not img_path.exists():
                out[kid] = {"phash": None, "phash_distance_prev": None, "is_near_duplicate": False}
                prev_kid = kid
                prev_hash = None
                continue

            h = imagehash.phash(Image.open(img_path))
            dist = int(h - prev_hash) if prev_hash is not None else None
            out[kid] = {
                "phash": str(h),
                "phash_distance_prev": dist,
                "is_near_duplicate": dist is not None and dist < 10,
            }
            prev_kid = kid
            prev_hash = h
    except ImportError:
        pass
    except Exception:
        pass
    return out


def _aesthetic_score_keyframes(keyframes: list, erf_root: Path) -> dict:
    """Compute CLIP-based aesthetic score per keyframe. Returns {keyframe_id: float 0-1}.

    Uses sentence-transformers clip-ViT-B-32 cross-modal similarity against reference prompts.
    Set ERF_SKIP_AESTHETIC=1 to return {} immediately (fast batch re-runs).
    """
    import os as _os
    if _os.environ.get("ERF_SKIP_AESTHETIC"):
        return {}

    valid_kfs = [
        (kf, erf_root / kf["image_relpath"])
        for kf in keyframes
        if (erf_root / kf["image_relpath"]).exists()
    ]
    if not valid_kfs:
        return {}

    try:
        if "clip" not in _AESTHETIC_MODEL_CACHE:
            with _AESTHETIC_MODEL_LOCK:
                # double-checked: another worker may have loaded it while we waited
                if "clip" not in _AESTHETIC_MODEL_CACHE:
                    from sentence_transformers import SentenceTransformer
                    _AESTHETIC_MODEL_CACHE["clip"] = SentenceTransformer("clip-ViT-B-32")

        model = _AESTHETIC_MODEL_CACHE["clip"]
        from PIL import Image
        from sentence_transformers.util import cos_sim

        pos_embs = model.encode(_AESTHETIC_POSITIVE_PROMPTS, convert_to_tensor=True, show_progress_bar=False)
        neg_embs = model.encode(_AESTHETIC_NEGATIVE_PROMPTS, convert_to_tensor=True, show_progress_bar=False)

        raw_scores: dict[str, float] = {}
        for kf, img_path in valid_kfs:
            img = Image.open(img_path).convert("RGB")
            img_emb = model.encode([img], convert_to_tensor=True, show_progress_bar=False)
            pos_sim = float(cos_sim(img_emb, pos_embs).mean())
            neg_sim = float(cos_sim(img_emb, neg_embs).mean())
            raw_scores[kf["keyframe_id"]] = pos_sim - neg_sim

        # Normalize to [0,1] within this media batch
        if not raw_scores:
            return {}
        vals = list(raw_scores.values())
        mn, mx = min(vals), max(vals)
        span = mx - mn
        return {
            kid: round((v - mn) / span, 4) if span > 1e-6 else 0.5
            for kid, v in raw_scores.items()
        }

    except Exception:
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    media_id: str,
    media_path: str | Path,
    *,
    config_path: str | Path = "state/config.toml",
) -> StageManifest:
    media_path = Path(media_path)
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    erf_root = Path(cfg["erf_root"])

    run_id = f"Q0-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "Q0", run_id)

    def _journal_event(event_type: str, payload: dict | None = None) -> None:
        try:
            import sqlite3 as _sq
            import datetime as _dt
            # +00:00 offset (not "Z") — matches every other at_utc writer; the
            # "Z" suffix misorders pipeline_status's cross-table TEXT comparison
            # within the same second (W2 checkpoint review finding, D-Log-12).
            ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            with _sq.connect(str(journal_db)) as _con:
                _con.execute(
                    # Phase 1 (D-Log-04): Q0 sub-event trace -> events_trace (§3.1)
                    # (Q0's real disposition is recorded via reap_manifest below).
                    "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (ts, f"{media_id}:Q0", media_id, "Q0", event_type, run_id,
                     json.dumps(payload) if payload is not None else None),
                )
        except Exception:
            pass

    error: str | None = None
    record_count = 0
    outputs: list[str] = []

    try:
        quality_dir = catalog / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        quality_path = quality_dir / f"{media_id}.jsonl"

        # Audio metrics (one per media)
        audio = _audio_metrics(media_path)
        audio.update(_audio_classify(media_path, erf_root))

        # Visual metrics (per keyframe)
        kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
        keyframes = read_jsonl(kf_path) if kf_path.exists() else []

        phash_map = _phash_keyframes(keyframes, erf_root)

        # Aesthetic scoring
        score_map: dict = {}
        best_kid: str | None = None
        if keyframes:
            _journal_event("Q0_aesthetic/start", {"n_keyframes": len(keyframes)})
            try:
                score_map = _aesthetic_score_keyframes(keyframes, erf_root)
                if score_map:
                    best_kid = max(score_map, key=lambda k: score_map[k])
                _journal_event("Q0_aesthetic/ok", {
                    "scored": len(score_map),
                    "best_frame": best_kid,
                })
            except Exception as ae:
                _journal_event("Q0_aesthetic/error", {"error": f"{type(ae).__name__}: {ae}"})

        # Collect all records in memory, then write atomically (Invariant 2)
        records: list[dict] = []
        prev_img: Path | None = None
        for kf in keyframes:
            img_path = erf_root / kf["image_relpath"]
            if not img_path.exists():
                continue

            metrics = _visual_metrics(img_path)
            if prev_img is not None and prev_img.exists():
                motion = _motion_between_frames(prev_img, img_path)
                metrics.update(motion)

            metrics.update(audio)
            metrics["b_roll_heuristic"] = _b_roll_heuristic(metrics)

            ph = phash_map.get(kf["keyframe_id"], {})
            kid = kf["keyframe_id"]
            records.append({
                "schema_version": "1.4.0",
                "media_id": media_id,
                "keyframe_id": kid,
                **{k: metrics.get(k) for k in [
                    "blur", "brightness", "contrast", "saturation",
                    "noise_ratio", "motion_energy", "camera_shake",
                    "b_roll_heuristic", "palette", "speech_clarity",
                    "silence_ratio", "clipping_ratio",
                    "audio_music_prob", "audio_speech_prob",
                    "audio_top_class", "audio_classifier_model",
                ]},
                "phash": ph.get("phash"),
                "phash_distance_prev": ph.get("phash_distance_prev"),
                "is_near_duplicate": ph.get("is_near_duplicate", False),
                "aesthetic_score": score_map.get(kid),
                "best_frame": (kid == best_kid) if best_kid is not None else None,
            })
            prev_img = img_path

        # If no keyframes, write one audio-only record
        if not keyframes:
            records.append({
                "schema_version": "1.4.0",
                "media_id": media_id,
                "keyframe_id": None,
                "blur": None, "brightness": None, "contrast": None, "saturation": None,
                "noise_ratio": audio.get("noise_ratio"),
                "motion_energy": None, "camera_shake": None, "b_roll_heuristic": None,
                "palette": None,
                "speech_clarity": audio.get("speech_clarity"),
                "silence_ratio": audio.get("silence_ratio"),
                "clipping_ratio": audio.get("clipping_ratio"),
                "audio_music_prob": audio.get("audio_music_prob"),
                "audio_speech_prob": audio.get("audio_speech_prob"),
                "audio_top_class": audio.get("audio_top_class"),
                "audio_classifier_model": audio.get("audio_classifier_model"),
                "phash": None, "phash_distance_prev": None, "is_near_duplicate": False,
                "aesthetic_score": None,
                "best_frame": None,
            })

        record_count = len(records)
        data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode("utf-8")
        write_tmp_rename(quality_path, data)

        outputs.append(str(quality_path))

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="Q0",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"records": record_count},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest

"""Stage B3 — Keyframe extraction.

Extracts scene-midpoint jpg + one frame per KEYFRAME_INTERVAL_MS within each scene.
Writes webp thumbs alongside. Emits keyframe JSONL per §7.6.
"""
from __future__ import annotations

import json
import subprocess
import tomllib
import uuid
from pathlib import Path

from .atomicio import jsonl_append, read_jsonl
from .ids import keyframe_id
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "erf_root": "/workspace/erf",
    "keyframe_interval_ms": 5000,
}


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
        cfg["keyframe_interval_ms"] = raw.get("keyframes", {}).get("interval_ms", cfg["keyframe_interval_ms"])
    return cfg


def _extract_frame(
    media_path: Path,
    time_s: float,
    jpg_path: Path,
    webp_path: Path,
    width: int = 1920,
    height: int = 1080,
) -> tuple[int, int]:
    """Extract one frame as jpg and webp. Returns (actual_width, actual_height)."""
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    webp_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_jpg = jpg_path.with_suffix(".tmp.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{time_s:.6f}",
        "-i", str(media_path),
        "-vframes", "1",
        "-q:v", "2",
        str(tmp_jpg),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp_jpg.rename(jpg_path)

    # Webp thumb at 320px wide
    tmp_webp = webp_path.with_suffix(".tmp.webp")
    cmd2 = [
        "ffmpeg", "-y",
        "-i", str(jpg_path),
        "-vf", "scale=320:-1",
        "-c:v", "libwebp", "-quality", "75",
        str(tmp_webp),
    ]
    subprocess.run(cmd2, check=True, capture_output=True)
    tmp_webp.rename(webp_path)

    # Read dimensions with ffprobe
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(jpg_path),
    ]
    try:
        out = subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip()
        parts = out.split(",")
        w, h = int(parts[0]), int(parts[1])
    except Exception:
        w, h = width, height
    return w, h


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
    interval_ms = int(cfg["keyframe_interval_ms"])

    run_id = f"B3-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "B3", run_id)

    error: str | None = None
    kf_count = 0
    outputs: list[str] = []

    try:
        # Load scenes
        scenes_path = catalog / "scenes" / f"{media_id}.jsonl"
        if not scenes_path.exists():
            raise FileNotFoundError(f"Scenes not found: {scenes_path}")
        scenes = read_jsonl(scenes_path)

        # Load fps from manifest for frame index calculation
        mm_path = catalog / "media_manifest" / f"{media_id}.json"
        fps = 25.0
        if mm_path.exists():
            mm = json.loads(mm_path.read_text())
            fps = float((mm.get("video") or {}).get("fps") or 25.0)

        kf_dir = erf_root / "media" / "keyframes" / media_id
        thumb_dir = erf_root / "media" / "thumbs" / media_id
        kf_jsonl = catalog / "keyframes" / f"{media_id}.jsonl"
        kf_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if kf_jsonl.exists():
            kf_jsonl.unlink()

        # Collect (time_ms, scene_id) pairs to extract
        times_to_extract: list[tuple[int, str]] = []
        for sc in scenes:
            sc_id = sc["scene_id"]
            s_ms = sc["start_ms"]
            e_ms = sc["end_ms"]
            span_ms = e_ms - s_ms

            # Scene midpoint
            mid_ms = s_ms + span_ms // 2
            times_to_extract.append((mid_ms, sc_id))

            # Additional frames at interval within the scene (skip first since midpoint covers start)
            t = s_ms + interval_ms
            while t < e_ms - interval_ms // 2:
                # Only add if not too close to midpoint
                if abs(t - mid_ms) > interval_ms // 2:
                    times_to_extract.append((t, sc_id))
                t += interval_ms

        # Deduplicate by time_ms
        seen_ms: set[int] = set()
        unique_times: list[tuple[int, str]] = []
        for tm, sc_id in sorted(times_to_extract):
            if tm not in seen_ms:
                unique_times.append((tm, sc_id))
                seen_ms.add(tm)

        for time_ms, sc_id in unique_times:
            time_s = time_ms / 1000.0
            kf_id = keyframe_id(media_id, time_ms)
            frame_num = int(round(time_s * fps))

            jpg_name = f"{kf_id.replace(':', '_')}.jpg"
            webp_name = f"{kf_id.replace(':', '_')}.webp"
            jpg_path = kf_dir / jpg_name
            webp_path = thumb_dir / webp_name

            image_relpath = f"media/keyframes/{media_id}/{jpg_name}"
            thumb_relpath = f"media/thumbs/{media_id}/{webp_name}"

            try:
                w, h = _extract_frame(media_path, time_s, jpg_path, webp_path)
            except Exception as e:
                # Non-fatal per frame — log and skip
                import warnings
                warnings.warn(f"Frame extract failed at {time_ms}ms: {e}")
                continue

            rec = {
                "schema_version": "1.0.0",
                "keyframe_id": kf_id,
                "media_id": media_id,
                "scene_id": sc_id,
                "time_ms": time_ms,
                "frame": frame_num,
                "image_relpath": image_relpath,
                "thumb_relpath": thumb_relpath,
                "width": w,
                "height": h,
                "visual_quality_score": None,
                "person_ids": None,
                "b_roll_flag": None,
            }
            jsonl_append(kf_jsonl, rec)
            kf_count += 1

        outputs.append(str(kf_jsonl))

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="B3",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"keyframes": kf_count},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest

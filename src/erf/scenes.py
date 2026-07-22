"""Stage B2 — Scene detection.

Primary: TransNetV2 (transnetv2-pytorch 1.0.5) via analyze_video().
Fallback: PySceneDetect 0.6.4 ContentDetector.
Detector used is recorded in each scene record's `detector` field.
"""
from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

from .atomicio import jsonl_append, read_jsonl
from .ids import scene_id
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "erf_root": "/workspace/erf",
}
_TRANSNET_THRESHOLD = 0.5
_PYSCENE_THRESHOLD = 27.0  # ContentDetector default


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
    return cfg


def _load_manifest(media_id: str, catalog: Path) -> dict | None:
    import json
    p = catalog / "media_manifest" / f"{media_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _detect_transnetv2(media_path: Path) -> list[dict] | None:
    """Returns list of {start_frame, end_frame, start_ms, end_ms} or None on failure."""
    try:
        from transnetv2_pytorch import TransNetV2
        model = TransNetV2(device="auto")
        result = model.analyze_video(str(media_path), threshold=_TRANSNET_THRESHOLD, quiet=True)
        scenes = result.get("scenes", [])
        if not scenes:
            return []
        # scenes: list of dicts with keys start_frame, end_frame, start_time, end_time
        out = []
        for s in scenes:
            out.append({
                "start_frame": int(s.get("start_frame", 0)),
                "end_frame": int(s.get("end_frame", 0)),
                "start_ms": int(round(float(s.get("start_time", 0)) * 1000)),
                "end_ms": int(round(float(s.get("end_time", 0)) * 1000)),
            })
        return out
    except Exception:
        return None


def _detect_pyscenedetect(media_path: Path, fps: float) -> list[dict]:
    """PySceneDetect ContentDetector fallback."""
    from scenedetect import open_video, detect, ContentDetector
    video = open_video(str(media_path))
    scene_list = detect(str(media_path), ContentDetector(threshold=_PYSCENE_THRESHOLD))

    fps = fps or 25.0
    out = []
    for start_tc, end_tc in scene_list:
        start_frame = start_tc.get_frames()
        end_frame = end_tc.get_frames()
        out.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_ms": int(round(start_frame / fps * 1000)),
            "end_ms": int(round(end_frame / fps * 1000)),
        })
    return out


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

    run_id = f"B2-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "B2", run_id)

    error: str | None = None
    scene_count = 0
    outputs: list[str] = []
    tool_version: str | None = None

    try:
        # Get FPS from media_manifest for fallback
        mm = _load_manifest(media_id, catalog)
        fps = float((mm or {}).get("video", {}).get("fps") or 25.0)
        duration_ms = int((mm or {}).get("duration_ms", 0))

        # Try TransNetV2 first
        detector_tag = "transnetv2-1.0.5"
        scenes_raw = _detect_transnetv2(media_path)

        if scenes_raw is None:
            # Fallback to PySceneDetect
            detector_tag = "pyscenedetect-content-0.6.4"
            scenes_raw = _detect_pyscenedetect(media_path, fps)

        # If no scenes detected (very short clip), synthesise a single scene
        if not scenes_raw:
            total_frames = int(round(duration_ms / 1000 * fps)) if fps > 0 else 0
            scenes_raw = [{
                "start_frame": 0,
                "end_frame": total_frames,
                "start_ms": 0,
                "end_ms": duration_ms,
            }]

        scenes_path = catalog / "scenes" / f"{media_id}.jsonl"
        scenes_path.parent.mkdir(parents=True, exist_ok=True)
        if scenes_path.exists():
            scenes_path.unlink()

        for idx, s in enumerate(scenes_raw):
            rec = {
                "schema_version": "1.0.0",
                "scene_id": scene_id(media_id, idx),
                "media_id": media_id,
                "index": idx,
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "detector": detector_tag,
            }
            jsonl_append(scenes_path, rec)
            scene_count += 1

        outputs.append(str(scenes_path))
        try:
            import transnetv2_pytorch as _tv2
            tool_version = getattr(_tv2, "__version__", "1.0.5")
        except ImportError:
            import scenedetect as _sd
            tool_version = getattr(_sd, "__version__", "0.6.4")

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="B2",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=tool_version,
        counts={"scenes": scene_count},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest

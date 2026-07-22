"""Stage P — Proxy + thumbnails.

ffmpeg H.264 proxy mp4 (half-res, CRF 23) + webp thumbs from keyframes.
For Phase 1 we generate a single midpoint thumb since B3 (keyframes) is
Phase 2. The thumbnail can be regenerated once B3 runs.
"""
from __future__ import annotations

import subprocess

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass
import tomllib
import uuid
from pathlib import Path

from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "erf_root": "/workspace/erf",
}


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
    return cfg


def _make_proxy(media_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(media_path),
        "-vf", "scale=iw/2:ih/2:flags=lanczos",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.rename(out_path)


def _make_thumb(media_path: Path, out_path: Path, duration_ms: int) -> None:
    """Extract a single webp thumb at the midpoint."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midpoint_s = max(0, (duration_ms / 2) / 1000)
    tmp = out_path.with_suffix(".tmp.webp")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(midpoint_s),
        "-i", str(media_path),
        "-vframes", "1",
        "-vf", "scale=320:-1",
        "-c:v", "libwebp", "-quality", "75",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.rename(out_path)


def run(
    media_id: str,
    media_path: str | Path,
    *,
    config_path: str | Path = "state/config.toml",
    duration_ms: int = 0,
) -> StageManifest:
    media_path = Path(media_path)
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    erf_root = Path(cfg["erf_root"])
    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])

    run_id = f"P-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "P", run_id)

    error: str | None = None
    outputs: list[str] = []

    try:
        proxy_path = erf_root / "media" / "proxies" / f"{media_id}.mp4"
        _make_proxy(media_path, proxy_path)
        outputs.append(str(proxy_path))

        thumb_path = erf_root / "media" / "thumbs" / media_id / f"{media_id}_mid.webp"
        _make_thumb(media_path, thumb_path, duration_ms)
        outputs.append(str(thumb_path))

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="P",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"proxies": 1 if ok else 0, "thumbs": 1 if ok else 0},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest

"""Export — OTIO/FCPXML/EDL/SRT from export_manifest.

Writes to catalog/exports/. Registers in U0 (exports + export_clips tables in erf_fts.db).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tomllib
from pathlib import Path

from .atomicio import write_json
from .contracts import validate
from .ids import export_id as make_export_id


def _load_config(config_path: Path) -> dict:
    cfg = {
        "catalog": "/workspace/erf/catalog",
        "state": "/workspace/erf/state",
        "erf_fts_db": "/workspace/erf/state/erf_fts.db",
        "journal_db": "/workspace/erf/state/journal.db",
    }
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        paths = raw.get("paths", {})
        cfg["catalog"] = paths.get("catalog", cfg["catalog"])
        cfg["state"] = paths.get("state", cfg["state"])
        cfg["erf_fts_db"] = paths.get("erf_fts_db", cfg["erf_fts_db"])
        cfg["journal_db"] = paths.get("journal_db", cfg["journal_db"])
    return cfg


def _load_manifest(media_id: str, catalog: Path) -> dict | None:
    p = catalog / "media_manifest" / f"{media_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fps_from_manifest(manifest: dict) -> float:
    video = manifest.get("video") or {}
    return float(video.get("fps", 25.0)) or 25.0


def _fps_rational_from_manifest(manifest: dict) -> str | None:
    video = manifest.get("video") or {}
    return video.get("fps_rational")


def _ms_to_frames(ms: int, fps: float) -> int:
    return round(ms / 1000 * fps)


def _ensure_u0_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS exports (
            export_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            query TEXT,
            format TEXT NOT NULL,
            output_relpath TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS export_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            export_id TEXT NOT NULL REFERENCES exports(export_id),
            media_id TEXT NOT NULL,
            in_frame INTEGER NOT NULL,
            out_frame INTEGER NOT NULL,
            label TEXT
        );
    """)
    conn.commit()


def _register_export(
    fts_db_path: Path,
    export_manifest: dict,
) -> None:
    conn = sqlite3.connect(str(fts_db_path))
    try:
        _ensure_u0_tables(conn)
        conn.execute(
            "INSERT OR REPLACE INTO exports (export_id, created_at_utc, query, format, output_relpath) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                export_manifest["export_id"],
                export_manifest["created_at_utc"],
                export_manifest.get("query"),
                export_manifest["format"],
                export_manifest["output_relpath"],
            ),
        )
        for clip in export_manifest.get("clips", []):
            conn.execute(
                "INSERT INTO export_clips (export_id, media_id, in_frame, out_frame, label) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    export_manifest["export_id"],
                    clip["media_id"],
                    clip["in_frame"],
                    clip["out_frame"],
                    clip.get("label"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _build_otio_timeline(clips_with_meta: list[dict], frame_rate: float) -> object:
    """Build an opentimelineio Timeline from clip metadata."""
    import opentimelineio as otio

    timeline = otio.schema.Timeline(name="ERF Export")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    timeline.tracks.append(track)

    for clip_meta in clips_with_meta:
        in_frame = clip_meta["in_frame"]
        out_frame = clip_meta["out_frame"]
        source_path = clip_meta.get("source_relpath", "")
        label = clip_meta.get("label") or clip_meta.get("media_id", "")

        duration_frames = out_frame - in_frame
        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(in_frame, frame_rate),
            duration=otio.opentime.RationalTime(duration_frames, frame_rate),
        )
        # available_range required by FCPXML adapter; use double the clip duration as a safe default
        avail_duration = otio.opentime.RationalTime(max(duration_frames * 2, 1), frame_rate)
        media_ref = otio.schema.ExternalReference(
            target_url=f"file://{source_path}",
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, frame_rate),
                duration=avail_duration,
            ),
        )
        clip = otio.schema.Clip(
            name=label,
            media_reference=media_ref,
            source_range=source_range,
        )
        track.append(clip)

    return timeline


def _emit_export_journal(journal_db_path: Path, export_id: str, manifest_relpath: str) -> None:
    """Write a lineage journal event for the export (Invariant 7)."""
    from datetime import datetime, timezone
    import json as _json
    now = datetime.now(timezone.utc).isoformat()
    payload = _json.dumps({"export_id": export_id, "outputs_produced": [manifest_relpath]})
    conn = sqlite3.connect(str(journal_db_path))
    try:
        conn.execute("PRAGMA journal_mode = TRUNCATE")
        conn.execute(
            # Phase 1 (D-Log-04): export-lineage trace -> events_trace (§3.1).
            "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
            "VALUES (?, ?, 'erf', 'export', 'ok', ?, ?)",
            (now, f"erf:export:{export_id}", export_id, payload),
        )
        conn.commit()
    except Exception:
        pass  # journal is best-effort for API operations
    finally:
        conn.close()


def export(
    clips: list[dict],
    *,
    fmt: str = "fcpxml",
    query: str | None = None,
    config_path: str | Path = "state/config.toml",
    frame_rate: float | None = None,
) -> dict:
    """Build and write an export.

    clips: list of {media_id, in_ms, out_ms, label?, source_relpath?}
    fmt: 'fcpxml' | 'edl' | 'otio' | 'srt'
    Returns the export_manifest dict.
    """
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    catalog = Path(cfg["catalog"])
    fts_db_path = Path(cfg["erf_fts_db"])

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    # Build clip records with frame numbers
    clip_records = []
    all_fps: list[float] = []
    for clip in clips:
        mid = clip["media_id"]
        manifest = _load_manifest(mid, catalog)
        fps = frame_rate or (manifest and _fps_from_manifest(manifest)) or 25.0
        fps_rational = manifest and _fps_rational_from_manifest(manifest)
        all_fps.append(fps)

        in_frame = _ms_to_frames(clip["in_ms"], fps)
        out_frame = _ms_to_frames(clip["out_ms"], fps)
        source_relpath = clip.get("source_relpath") or (manifest and manifest.get("relpath")) or mid

        clip_records.append({
            "media_id": mid,
            "source_relpath": source_relpath,
            "in_frame": in_frame,
            "out_frame": out_frame,
            "label": clip.get("label"),
            "fps": fps,
            "fps_rational": fps_rational,
        })

    # Determine timeline frame rate
    tl_fps = frame_rate or (all_fps[0] if all_fps else 25.0)

    # Content hash for export_id
    content_str = json.dumps(clip_records, sort_keys=True)
    content_hash = hashlib.blake2b(content_str.encode(), digest_size=3).hexdigest()
    exp_id = make_export_id(content_hash, now)

    exports_dir = catalog / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    ext_map = {"fcpxml": ".fcpxml", "edl": ".edl", "otio": ".otio", "srt": ".srt"}
    out_filename = f"{exp_id}{ext_map.get(fmt, '.bin')}"
    out_path = exports_dir / out_filename
    output_relpath = str(out_path.relative_to(catalog.parent))

    if fmt in ("fcpxml", "edl", "otio"):
        try:
            timeline = _build_otio_timeline(clip_records, tl_fps)
            import opentimelineio as otio
            adapter_map = {"fcpxml": "fcpx_xml", "edl": "cmx_3600", "otio": "otio_json"}
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            otio.adapters.write_to_file(timeline, str(tmp), adapter_name=adapter_map[fmt])
            tmp.rename(out_path)
        except ImportError:
            # OTIO not installed — write a stub
            out_path.write_text(f"# OTIO not installed; export_id={exp_id}\n")
    elif fmt == "srt":
        _write_srt(clip_records, catalog, out_path)

    export_manifest = {
        "schema_version": "1.0.0",
        "export_id": exp_id,
        "created_at_utc": now.isoformat(),
        "query": query,
        "format": fmt,
        "frame_rate": tl_fps,
        "clips": [
            {
                "media_id": c["media_id"],
                "source_relpath": c["source_relpath"],
                "in_frame": c["in_frame"],
                "out_frame": c["out_frame"],
                "label": c.get("label"),
            }
            for c in clip_records
        ],
        "output_relpath": output_relpath,
    }

    validate(export_manifest, "export_manifest")

    manifest_path = exports_dir / f"{exp_id}.json"
    manifest_relpath = str(manifest_path.relative_to(catalog.parent))
    write_json(manifest_path, export_manifest, indent=2)
    _register_export(fts_db_path, export_manifest)
    _emit_export_journal(Path(cfg["journal_db"]), exp_id, manifest_relpath)

    return export_manifest


def _frames_to_tc(frame: int, fps: float, drop: bool = False) -> str:
    total_s = int(frame // fps)
    ff = int(round(frame % fps))
    hh = total_s // 3600
    mm = (total_s % 3600) // 60
    ss = total_s % 60
    sep = ";" if drop else ":"
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"


def _write_srt(clip_records: list[dict], catalog: Path, out_path: Path) -> None:
    lines = []
    for i, clip in enumerate(clip_records, 1):
        mid = clip["media_id"]
        fps = clip.get("fps", 25.0)
        in_s = clip["in_frame"] / fps
        out_s = clip["out_frame"] / fps

        def _srt_tc(s: float) -> str:
            ms = int((s % 1) * 1000)
            total = int(s)
            hh, rem = divmod(total, 3600)
            mm, ss = divmod(rem, 60)
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

        # Load chunk text for the clip (approximate by time range)
        label = clip.get("label") or mid
        lines.extend([str(i), f"{_srt_tc(in_s)} --> {_srt_tc(out_s)}", label, ""])

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.rename(out_path)

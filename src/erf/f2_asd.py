"""Stage F2 — Active Speaker Detection orchestrator.

Calls bin/asd_worker.py in the isolated asd venv, validates the facet,
writes it atomically to catalog/active_speaker/{media_id}.json.

Does NOT modify F0 or F1 artifacts.
"""
from __future__ import annotations

import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import jsonschema

from .atomicio import write_json
from .journal import emit_error, emit_start, rebuild_status
from .manifest import StageManifest

_log = logging.getLogger("erf.f2_asd")

_ERF_ROOT = Path(__file__).parent.parent.parent
_ASD_PYTHON = _ERF_ROOT / "venvs" / "asd" / "bin" / "python"
_ASD_WORKER = _ERF_ROOT / "bin" / "asd_worker.py"
_SCHEMA_PATH = _ERF_ROOT / "schemas" / "active_speaker.schema.json"

STAGE = "F2"
STAGE_VERSION = "1.0.0"


def _load_schema() -> Optional[dict]:
    if not _SCHEMA_PATH.exists():
        _log.warning("Schema not found at %s — skipping validation", _SCHEMA_PATH)
        return None
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


_SCHEMA = None  # loaded on first use


def run_f2(
    media_id: str,
    *,
    erf_root: Path = _ERF_ROOT,
    asd_threshold: float = 0.5,
    journal_db: Optional[Path] = None,
) -> dict:
    """Run active-speaker detection for one media item.

    Returns: {"ok": bool, "skip_reason": str|None, "error": str|None, "output_path": str|None}
    """
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = _load_schema()

    catalog = erf_root / "catalog"
    state = erf_root / "state"
    _journal_db = journal_db or (state / "journal.db")
    status_dir = catalog / "status"

    run_id = f"F2-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()

    # ------------------------------------------------------------------
    # Guard: gate on F0's explicit has_faces flag (cross-stage awareness), not on
    # probing the face_tracks file. The file is read below as DATA when present.
    # ------------------------------------------------------------------
    face_tracks_path = catalog / "face_tracks" / f"{media_id}.jsonl"
    try:
        from . import outcomes as _oc
        _has_faces = _oc.flag(media_id, "has_faces")
    except Exception:
        _oc, _has_faces = None, None
    if _has_faces is False or (_has_faces is None and not face_tracks_path.exists()):
        _log.info("%s: has_faces=False (per F0) — skip", media_id)
        emit_start(_journal_db, media_id, STAGE, run_id)
        emit_error(_journal_db, media_id, STAGE, run_id, "SKIP:no_face_tracks")
        rebuild_status(_journal_db, media_id, status_dir)
        if _oc:
            _oc.record(media_id, "F2", "skip", reason="no_faces")
            _oc.set_flag(media_id, "has_asd", False, reason="no_faces")
        return {"ok": True, "skip_reason": "no_face_tracks", "error": None, "output_path": None}

    # ------------------------------------------------------------------
    # Optional inputs
    # ------------------------------------------------------------------
    proxy_path = erf_root / "media" / "proxies" / f"{media_id}.mp4"
    det_path = catalog / "detections" / f"{media_id}.face.jsonl"
    diar_path = catalog / "asr" / f"{media_id}.diar.jsonl"

    emit_start(_journal_db, media_id, STAGE, run_id,
               payload={"media_id": media_id, "threshold": asd_threshold})

    # ------------------------------------------------------------------
    # Invoke worker subprocess
    # ------------------------------------------------------------------
    cmd = [
        str(_ASD_PYTHON), str(_ASD_WORKER),
        "--media-id", media_id,
        "--proxy", str(proxy_path),
        "--face-tracks", str(face_tracks_path),
        "--detections", str(det_path) if det_path.exists() else "",
        "--diar", str(diar_path) if diar_path.exists() else "",
        "--weights-dir", str(erf_root / ".." / "weights" / "asd"),
        "--scratch-base", "/tmp/erf_asd",
        "--threshold", str(asd_threshold),
    ]
    _log.info("%s: launching asd worker", media_id)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        _log.error("%s: worker timed out (900s)", media_id)
        emit_error(_journal_db, media_id, STAGE, run_id, "worker_timeout")
        rebuild_status(_journal_db, media_id, status_dir)
        return {"ok": False, "skip_reason": None, "error": "worker_timeout", "output_path": None}

    if proc.stderr:
        level = _log.warning if proc.returncode != 0 else _log.debug
        for line in proc.stderr.strip().split("\n")[-20:]:
            level("worker stderr [%s]: %s", media_id, line)

    # ------------------------------------------------------------------
    # Parse worker output
    # ------------------------------------------------------------------
    stdout_lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
    if not stdout_lines:
        err = f"worker_no_output (exit {proc.returncode})"
        emit_error(_journal_db, media_id, STAGE, run_id, err)
        rebuild_status(_journal_db, media_id, status_dir)
        return {"ok": False, "skip_reason": None, "error": err, "output_path": None}

    try:
        result = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as exc:
        err = f"worker_invalid_json: {exc}"
        emit_error(_journal_db, media_id, STAGE, run_id, err)
        rebuild_status(_journal_db, media_id, status_dir)
        return {"ok": False, "skip_reason": None, "error": err, "output_path": None}

    if not result.get("ok"):
        raw_err = result.get("error", "unknown_worker_error")
        if raw_err.startswith("SKIP:"):
            skip_reason = raw_err[5:]
            _log.info("%s: worker skip (%s)", media_id, skip_reason)
            emit_error(_journal_db, media_id, STAGE, run_id, raw_err)
            rebuild_status(_journal_db, media_id, status_dir)
            return {"ok": True, "skip_reason": skip_reason, "error": None, "output_path": None}
        emit_error(_journal_db, media_id, STAGE, run_id, raw_err)
        rebuild_status(_journal_db, media_id, status_dir)
        return {"ok": False, "skip_reason": None, "error": raw_err, "output_path": None}

    # ------------------------------------------------------------------
    # Inject provenance fields + validate
    # ------------------------------------------------------------------
    facet = result["facet"]
    facet["produced_by"] = run_id
    facet["stage_version"] = STAGE_VERSION

    if _SCHEMA is not None:
        try:
            jsonschema.validate(instance=facet, schema=_SCHEMA)
        except jsonschema.ValidationError as ve:
            err = f"schema_validation_failed: {ve.message[:200]}"
            _log.error("%s: %s", media_id, err)
            emit_error(_journal_db, media_id, STAGE, run_id, err)
            rebuild_status(_journal_db, media_id, status_dir)
            return {"ok": False, "skip_reason": None, "error": err, "output_path": None}

    # ------------------------------------------------------------------
    # Write catalog artifact (atomic)
    # ------------------------------------------------------------------
    out_dir = catalog / "active_speaker"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{media_id}.json"
    write_json(out_path, facet, indent=2)
    _log.info("%s: wrote %s", media_id, out_path)

    # ------------------------------------------------------------------
    # Journal + status
    # ------------------------------------------------------------------
    manifest = StageManifest(
        media_id=media_id,
        stage=STAGE,
        run_id=run_id,
        ok=True,
        at_utc=at_utc,
        tool_version=STAGE_VERSION,
        counts={
            "segments": len(facet["segments"]),
            "turns": len(facet["turn_active_speaker"]),
        },
        outputs=[str(out_path)],
    )
    from .journal import emit_ok
    emit_ok(_journal_db, manifest)
    rebuild_status(_journal_db, media_id, status_dir)

    return {"ok": True, "skip_reason": None, "error": None, "output_path": str(out_path)}

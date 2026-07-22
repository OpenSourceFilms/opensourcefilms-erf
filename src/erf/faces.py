"""Stage F0 — face detect + track + embed (GPU worker).

Decodes proxy at 3fps, runs buffalo_l detect+embed, tracks across frames,
writes catalog/face_tracks/{media_id}.jsonl and catalog/detections/{media_id}.face.jsonl.
person_id and match_score are always null (C2 invariant — F1 does all assignment).
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .atomicio import jsonl_append, read_jsonl
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_INSIGHTFACE_ROOT = "/workspace/weights/insightface"
_FACE_SAMPLE_FPS = 3
_DET_SCORE_MIN = 0.5
_TRACK_IOU_THRESH = 0.3
_TRACK_COSINE_THRESH = 0.50
_TRACK_GAP_MS = 2000


def _set_cudnn_path() -> None:
    cudnn_lib = "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib not in existing:
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{existing}" if existing else cudnn_lib


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _l2_norm(v) -> list[float]:
    import numpy as np
    arr = np.array(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    return (arr / n).tolist() if n > 0 else arr.tolist()


def run(media_id: str, *, config_path: str | Path = "state/config.toml") -> StageManifest:
    config_path = Path(config_path)
    import tomllib
    cfg: dict = {}
    if config_path.exists():
        with open(config_path, "rb") as fh:
            cfg = tomllib.load(fh)

    catalog = Path(cfg.get("paths", {}).get("catalog", "catalog"))
    state = Path(cfg.get("paths", {}).get("state", "state"))
    erf_root = Path(cfg.get("paths", {}).get("erf_root", "."))

    proxy_path = erf_root / "media" / "proxies" / f"{media_id}.mp4"
    tracks_path = catalog / "face_tracks" / f"{media_id}.jsonl"
    detections_path = catalog / "detections" / f"{media_id}.face.jsonl"

    run_id = f"F0-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"
    emit_start(journal_db, media_id, "F0", run_id)

    error: str | None = None
    track_count = 0
    detection_count = 0
    frames_sampled = 0
    empty_no_faces = False

    try:
        if not proxy_path.exists():
            raise FileNotFoundError(f"Proxy not found: {proxy_path} — run Stage P first")

        _set_cudnn_path()

        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", root=_INSIGHTFACE_ROOT)
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Load B3 keyframe times for detection record linkage
        kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
        keyframes = read_jsonl(kf_path) if kf_path.exists() else []
        kf_times = {kf["time_ms"]: kf["keyframe_id"] for kf in keyframes}

        cap = cv2.VideoCapture(str(proxy_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(1, round(fps / _FACE_SAMPLE_FPS))

        # Active tracks: list of dicts with last_ms, last_bbox, embeddings, start_ms
        active_tracks: list[dict] = []
        closed_tracks: list[dict] = []
        track_index = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            time_ms = int(frame_idx / fps * 1000)
            frame_idx += 1
            frames_sampled += 1

            # Close stale tracks
            still_active = []
            for tr in active_tracks:
                if time_ms - tr["last_ms"] > _TRACK_GAP_MS:
                    closed_tracks.append(tr)
                else:
                    still_active.append(tr)
            active_tracks = still_active

            faces = app.get(frame)
            for face in faces:
                if face.det_score < _DET_SCORE_MIN:
                    continue

                emb = _l2_norm(face.embedding)
                bbox = [float(x) for x in face.bbox.tolist()]

                # Find best matching active track
                best_idx = -1
                best_score = -1.0
                for ti, tr in enumerate(active_tracks):
                    iou = _iou(bbox, tr["last_bbox"])
                    cos = _cosine(emb, tr["mean_emb"])
                    score = max(iou if iou > _TRACK_IOU_THRESH else -1,
                                cos if cos > _TRACK_COSINE_THRESH else -1)
                    if score > best_score:
                        best_score = score
                        best_idx = ti

                if best_idx >= 0:
                    tr = active_tracks[best_idx]
                    tr["embeddings"].append(emb)
                    tr["last_bbox"] = bbox
                    tr["last_ms"] = time_ms
                    tr["end_ms"] = time_ms
                    # Update running mean
                    n = len(tr["embeddings"])
                    arr = np.array(tr["embeddings"])
                    mean = arr.mean(axis=0)
                    tr["mean_emb"] = _l2_norm(mean.tolist())
                else:
                    new_track = {
                        "track_id": f"{media_id}:ft:{track_index:04d}",
                        "start_ms": time_ms,
                        "end_ms": time_ms,
                        "last_ms": time_ms,
                        "last_bbox": bbox,
                        "embeddings": [emb],
                        "mean_emb": emb,
                        "det_scores": [float(face.det_score)],
                        "kf_times": [],
                    }
                    active_tracks.append(new_track)
                    track_index += 1

                # Record detection at keyframe times
                # Find nearest keyframe within ±500ms
                for kf_ms, kf_id in kf_times.items():
                    if abs(kf_ms - time_ms) <= (1000 / fps / 2 + 1):
                        # This face is at a keyframe — will be linked to the matching track
                        pass

        cap.release()
        closed_tracks.extend(active_tracks)

        # Write tracks
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        if tracks_path.exists():
            tracks_path.unlink()

        detections_path.parent.mkdir(parents=True, exist_ok=True)
        if detections_path.exists():
            detections_path.unlink()

        for tr in closed_tracks:
            # Link to B3 keyframes whose time_ms falls in track span
            kf_ids = [kf_id for kf_ms, kf_id in kf_times.items()
                      if tr["start_ms"] <= kf_ms <= tr["end_ms"]]

            rec = {
                "schema_version": "1.0.0",
                "face_track_id": tr["track_id"],
                "media_id": media_id,
                "start_ms": tr["start_ms"],
                "end_ms": tr["end_ms"],
                "embedding": tr["mean_emb"],
                "keyframe_ids": kf_ids,
                "person_id": None,
                "match_score": None,
                "produced_by": run_id,
                "stage_version": "1.0.0",
            }
            jsonl_append(tracks_path, rec)
            track_count += 1

            # Write detection records at keyframe times
            for det_idx, kf_id in enumerate(kf_ids):
                kf_ms = next(ms for ms, kid in kf_times.items() if kid == kf_id)
                det_rec = {
                    "schema_version": "1.0.0",
                    "detection_id": f"{media_id}:det:face:{kf_id}:{det_idx:02d}",
                    "media_id": media_id,
                    "keyframe_id": kf_id,
                    "time_ms": kf_ms,
                    "is_face": True,
                    "face_track_id": tr["track_id"],
                    "bbox": tr["last_bbox"],
                    "det_score": float(max(tr["det_scores"])) if tr.get("det_scores") else 0.0,
                    "produced_by": run_id,
                    "stage_version": "1.0.0",
                }
                jsonl_append(detections_path, det_rec)
                detection_count += 1

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    # No faces detected — F0 ran the full detect/track pass but produced no
    # face_tracks/detections artifacts (both were unlinked above and never
    # rewritten). Recording bare "ok" here is exactly the drift-308 phantom
    # pattern (ok outcome, no artifact on disk); F0 is not in
    # reconcile_status.py's explicit_empty_stages, so "empty" is treated the
    # same as "skip" there (no artifact expected) and clears drift cleanly.
    # This is gated on the actual result (track_count), not an upstream flag
    # like tags/B1's has_chunks — a no-speech media can still have visible
    # faces, so F0 must run before it can know it produced nothing.
    if ok and track_count == 0:
        empty_no_faces = True
    manifest = StageManifest(
        media_id=media_id,
        stage="F0",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version="insightface-1.0",
        counts={"tracks": track_count, "detections": detection_count, "frames_sampled": frames_sampled},
        error=error,
        outputs=[str(tracks_path), str(detections_path)],
    )
    status_dir = catalog / "status"
    reap_manifest(journal_db, manifest, status_dir)
    if empty_no_faces:
        try:
            from . import outcomes as _oc2
            # Override the bridge's "ok" with the accurate empty (runs after reap).
            _oc2.record(media_id, "F0", "empty", reason="no_faces")
        except Exception:
            pass
    if ok:  # explicit cross-stage flag for F1/F2/V0 to read instead of probing files
        try:
            from . import outcomes
            outcomes.set_flag(media_id, "has_faces", track_count > 0,
                              reason=None if track_count else "no_faces")
        except Exception:
            pass
    return manifest

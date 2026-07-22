#!/usr/bin/env python3
"""
Active-speaker detection subprocess worker for Stage F2.

Called as:
  venvs/asd/bin/python bin/asd_worker.py \
    --media-id <id> --proxy <path> \
    --face-tracks <jsonl> --detections <jsonl|""> \
    --diar <jsonl|""> --weights-dir <path> \
    --scratch-base /tmp/erf_asd --threshold 0.5

Prints ONE JSON line to stdout on success or failure. Nothing else.
All other output goes to stderr.
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from insightface.app import FaceAnalysis
from python_speech_features import mfcc
from scipy.ndimage import median_filter

from light_asd import ASD_Model, lossAV

_INSIGHTFACE_ROOT = "/workspace/weights/insightface"

log = logging.getLogger("asd_worker")
_h = logging.StreamHandler(sys.stderr)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_h)
log.setLevel(logging.INFO)

FPS = 25
CENTROID_FPS = 5


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_bbox(frame_idx, frame_dets, anchors):
    """Return the InsightFace bbox closest to the nearest sparse anchor, or None."""
    if not anchors or frame_idx not in frame_dets or not frame_dets[frame_idx]:
        return None
    frame_ms = int(frame_idx * 1000 / FPS)
    anchor = min(anchors, key=lambda a: abs(a["time_ms"] - frame_ms))
    ref = anchor["bbox"]
    dets = frame_dets[frame_idx]
    best_iou, best = -1.0, None
    for d in dets:
        i = _iou(ref, d["bbox"])
        if i > best_iou:
            best_iou, best = i, d["bbox"]
    if best_iou > 0.3:
        return best
    # fallback: nearest centre
    rcx = (ref[0] + ref[2]) / 2
    rcy = (ref[1] + ref[3]) / 2
    best_d2, best = float("inf"), None
    for d in dets:
        b = d["bbox"]
        dx = (b[0] + b[2]) / 2 - rcx
        dy = (b[1] + b[3]) / 2 - rcy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2, best = d2, b
    return best


def _sample_centroids(crops, start_ms, end_ms):
    """Sample cx at CENTROID_FPS intervals; crops is [(frame_idx, gray112, cx), ...]."""
    step_ms = 1000 // CENTROID_FPS
    out = []
    t = start_ms
    while t < end_ms:
        nearest = min(range(len(crops)), key=lambda i: abs(crops[i][0] * 1000 // FPS - t))
        out.append({"t_ms": int(t), "cx": float(crops[nearest][2])})
        t += step_ms
    return out


def _build_empty(media_id):
    return {
        "schema_version": "1.0.0",
        "media_id": media_id,
        "analyzed_fps": FPS,
        "centroid_fps": CENTROID_FPS,
        "segments": [],
        "turn_active_speaker": [],
    }


def _ok(facet):
    print(json.dumps({"ok": True, "facet": facet}), flush=True)


def _err(msg):
    print(json.dumps({"ok": False, "error": msg}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media-id", required=True)
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--face-tracks", required=True)
    ap.add_argument("--detections", default="")
    ap.add_argument("--diar", default="")
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--scratch-base", default="/tmp/erf_asd")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    scratch_dir = Path(args.scratch_base) / args.media_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run(args, scratch_dir)
    except Exception as exc:
        log.exception("Unhandled error")
        _err(str(exc)[:400])
        sys.exit(1)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _run(args, scratch_dir):
    proxy = Path(args.proxy)
    if not proxy.exists():
        _err(f"proxy missing: {proxy}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Frame extraction (25fps JPEG → scratch)
    # ------------------------------------------------------------------
    frame_dir = scratch_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(proxy), "-r", str(FPS), "-q:v", "2",
         str(frame_dir / "%06d.jpg"), "-hide_banner", "-loglevel", "error"],
        check=True, capture_output=True,
    )
    frame_files = sorted(frame_dir.glob("*.jpg"))
    n_frames = len(frame_files)
    log.info("Extracted %d frames", n_frames)
    if n_frames == 0:
        _err("ffmpeg produced no frames")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Audio extraction → 16kHz mono WAV → MFCC
    # ------------------------------------------------------------------
    wav_path = scratch_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(proxy), "-ar", "16000", "-ac", "1",
         "-vn", str(wav_path), "-hide_banner", "-loglevel", "error"],
        check=True, capture_output=True,
    )
    audio, _ = sf.read(str(wav_path))
    if audio.dtype in (np.float32, np.float64):
        audio = (audio * 32767).astype(np.int16)
    mfcc_full = mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
    log.info("MFCC shape: %s", mfcc_full.shape)

    # ------------------------------------------------------------------
    # 3. InsightFace det_10g on every frame
    # ------------------------------------------------------------------
    face_app = FaceAnalysis(name="buffalo_l", root=_INSIGHTFACE_ROOT)
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    frame_dets: dict[int, list[dict]] = {}
    log.info("Running InsightFace on %d frames…", n_frames)
    for i, fname in enumerate(frame_files):
        img = cv2.imread(str(fname))
        frame_dets[i] = [{"bbox": f.bbox.astype(int).tolist()} for f in face_app.get(img)]
    log.info("InsightFace done — %d frames with detections", sum(1 for v in frame_dets.values() if v))

    # ------------------------------------------------------------------
    # 4. Load sparse detections (optional — used to anchor tracks)
    # ------------------------------------------------------------------
    sparse_bboxes: dict[str, list[dict]] = {}
    if args.detections and Path(args.detections).exists():
        with open(args.detections) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = obj["face_track_id"]
                sparse_bboxes.setdefault(tid, []).append(
                    {"time_ms": obj["time_ms"], "bbox": obj["bbox"]}
                )
        for tid in sparse_bboxes:
            sparse_bboxes[tid].sort(key=lambda x: x["time_ms"])

    # ------------------------------------------------------------------
    # 5. Load face tracks
    # ------------------------------------------------------------------
    tracks = []
    with open(args.face_tracks) as f:
        for line in f:
            if not line.strip():
                continue
            tracks.append(json.loads(line))
    log.info("Loaded %d face tracks", len(tracks))

    # ------------------------------------------------------------------
    # 6. Build per-track crop sequences
    # ------------------------------------------------------------------
    # crops entry: (frame_idx, gray_112, cx_norm)
    all_track_crops: list[tuple[str, list]] = []
    for trk in tracks:
        tid = trk["face_track_id"]
        sf_frame = int(trk["start_ms"] * FPS / 1000)
        ef_frame = min(int(trk["end_ms"] * FPS / 1000), n_frames)
        if ef_frame - sf_frame < 10:
            log.info("SKIP %s: track_too_short (%d frames)", tid, ef_frame - sf_frame)
            continue
        anchors = sparse_bboxes.get(tid, [])
        crops = []
        if anchors:
            # Anchor-guided matching: use F0 sparse bboxes as spatial anchors
            for fi in range(sf_frame, ef_frame):
                bbox = _match_bbox(fi, frame_dets, anchors)
                if bbox is None:
                    continue
                img = cv2.imread(str(frame_files[fi]))
                h, w = img.shape[:2]
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                crops.append((fi, cv2.resize(gray, (112, 112)), ((x1 + x2) / 2) / w))
        else:
            # No-anchor fallback: temporal continuity tracking
            # Start with highest-score face in first occupied frame, then follow nearest center
            prev_cx, prev_cy = None, None
            for fi in range(sf_frame, ef_frame):
                dets = frame_dets.get(fi, [])
                if not dets:
                    continue
                if prev_cx is None:
                    # First frame: pick most-central face (closest to frame centre)
                    img = cv2.imread(str(frame_files[fi]))
                    h, w = img.shape[:2]
                    best_bbox = min(
                        dets, key=lambda d: abs((d["bbox"][0] + d["bbox"][2]) / 2 - w / 2)
                    )["bbox"]
                else:
                    # Subsequent frames: nearest centre to previous frame
                    best_d2, best_bbox = float("inf"), None
                    for d in dets:
                        b = d["bbox"]
                        dx = (b[0] + b[2]) / 2 - prev_cx
                        dy = (b[1] + b[3]) / 2 - prev_cy
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2, best_bbox = d2, b
                img = cv2.imread(str(frame_files[fi]))
                h, w = img.shape[:2]
                x1, y1 = max(0, int(best_bbox[0])), max(0, int(best_bbox[1]))
                x2, y2 = min(w, int(best_bbox[2])), min(h, int(best_bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                prev_cx = (best_bbox[0] + best_bbox[2]) / 2
                prev_cy = (best_bbox[1] + best_bbox[3]) / 2
                gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                crops.append((fi, cv2.resize(gray, (112, 112)), ((x1 + x2) / 2) / w))
        if len(crops) < 10:
            log.info("SKIP %s: too_few_visible_frames (%d)", tid, len(crops))
            continue
        all_track_crops.append((tid, crops))

    if not all_track_crops:
        log.info("No valid tracks — returning empty facet")
        _ok(_build_empty(args.media_id))
        return

    # ------------------------------------------------------------------
    # 7. Load model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        Path(args.weights_dir) / "finetuning_TalkSet.model",
        map_location="cpu",
        weights_only=False,
    )
    model = ASD_Model()
    model.load_state_dict({k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}, strict=True)
    model.to(device).eval()

    loss_fn = lossAV()
    loss_fn.load_state_dict({k[7:]: v for k, v in ckpt.items() if k.startswith("lossAV.")}, strict=True)
    loss_fn.to(device).eval()

    # ------------------------------------------------------------------
    # 8. ASD inference per track
    # ------------------------------------------------------------------
    segments_all: list[dict] = []
    smoothing_window = max(1, int(0.3 * FPS))

    for tid, crops in all_track_crops:
        T = len(crops)
        f0 = crops[0][0]

        visual_np = np.stack([c[1] for c in crops]).astype(np.float32)
        visual_t = torch.FloatTensor(visual_np).unsqueeze(0).to(device)

        mfcc_start = f0 * 4
        mfcc_chunk = mfcc_full[mfcc_start: mfcc_start + T * 4]
        if mfcc_chunk.shape[0] < T * 4:
            mfcc_chunk = np.pad(mfcc_chunk, ((0, T * 4 - mfcc_chunk.shape[0]), (0, 0)), "wrap")
        audio_t = torch.FloatTensor(mfcc_chunk).unsqueeze(0).to(device)

        with torch.no_grad():
            a_emb = model.forward_audio_frontend(audio_t)
            v_emb = model.forward_visual_frontend(visual_t)
            outs_av = model.forward_audio_visual_backend(a_emb, v_emb)
            logits = loss_fn.FC(outs_av)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        smoothed = median_filter(probs, size=smoothing_window)
        speaking = smoothed >= args.threshold

        in_seg = False
        seg_start_ms = 0
        seg_probs: list[float] = []
        for i, s in enumerate(speaking):
            frame_ms = int(crops[i][0] * 1000 / FPS)
            if s and not in_seg:
                seg_start_ms = frame_ms
                seg_probs = [float(smoothed[i])]
                in_seg = True
            elif s and in_seg:
                seg_probs.append(float(smoothed[i]))
            elif not s and in_seg:
                if frame_ms > seg_start_ms:
                    segments_all.append({
                        "face_track_id": tid,
                        "start_ms": seg_start_ms,
                        "end_ms": frame_ms,
                        "speaking_confidence": float(np.mean(seg_probs)),
                        "centroids": _sample_centroids(crops, seg_start_ms, frame_ms),
                    })
                in_seg = False
        if in_seg:
            seg_end_ms = int(crops[-1][0] * 1000 / FPS)
            if seg_end_ms > seg_start_ms:
                segments_all.append({
                    "face_track_id": tid,
                    "start_ms": seg_start_ms,
                    "end_ms": seg_end_ms,
                    "speaking_confidence": float(np.mean(seg_probs)),
                    "centroids": _sample_centroids(crops, seg_start_ms, seg_end_ms),
                })

    # ------------------------------------------------------------------
    # 9. Turn rollup
    # ------------------------------------------------------------------
    turns: list[dict] = []
    if args.diar and Path(args.diar).exists():
        with open(args.diar) as f:
            for line in f:
                if not line.strip():
                    continue
                turns.append(json.loads(line))

    turn_active_speaker: list[dict] = []
    for turn in turns:
        t_start, t_end = turn["start_ms"], turn["end_ms"]
        dur = t_end - t_start
        if dur <= 0:
            continue
        track_ms: dict[str, float] = {}
        track_conf: dict[str, float] = {}
        for seg in segments_all:
            ov_start = max(seg["start_ms"], t_start)
            ov_end = min(seg["end_ms"], t_end)
            if ov_end <= ov_start:
                continue
            ov = ov_end - ov_start
            tid = seg["face_track_id"]
            track_ms[tid] = track_ms.get(tid, 0.0) + ov
            track_conf[tid] = track_conf.get(tid, 0.0) + seg["speaking_confidence"] * ov
        winner_tid = max(track_ms, key=track_ms.get) if track_ms else None
        if winner_tid and track_ms[winner_tid] > 0.3 * dur:
            conf = track_conf[winner_tid] / track_ms[winner_tid]
        else:
            winner_tid = None
            conf = 0.0
        turn_active_speaker.append({
            "turn_index": turn["turn_index"],
            "speaker_label": turn["speaker_label"],
            "face_track_id": winner_tid,
            "speaking_ms": int(track_ms.get(winner_tid, 0)) if winner_tid else 0,
            "confidence": float(conf),
        })

    # ------------------------------------------------------------------
    # 10. Emit result
    # ------------------------------------------------------------------
    _ok({
        "schema_version": "1.0.0",
        "media_id": args.media_id,
        "analyzed_fps": FPS,
        "centroid_fps": CENTROID_FPS,
        "segments": segments_all,
        "turn_active_speaker": turn_active_speaker,
    })


if __name__ == "__main__":
    main()

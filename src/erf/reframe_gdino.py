"""GroundingDINO-based reframe — Strategy B (DEC-5: CPU, non-blocking).

Samples proxy video at ~sample_fps, runs "person" detection per frame,
applies EMA smoothing identical to face_centroid (α=0.15, ±8% dead-zone).
Falls back to centre-crop if no detections.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

_log = logging.getLogger("erf.reframe_gdino")

_CROP_W = 304
_SRC_W = 960
_MAX_CROP_X = 656        # SRC_W - CROP_W
_CENTRE_CROP_X = 328     # (SRC_W - CROP_W) // 2

_DEFAULT_CONFIG = "/workspace/weights/groundingdino/GroundingDINO_SwinT_OGC.py"
_DEFAULT_WEIGHTS = "/workspace/weights/groundingdino/groundingdino_swint_ogc.pth"


def _gdino_crop_x_from_boxes(boxes) -> Optional[float]:
    """Return normalised x-centre of the highest-confidence 'person' box, or None."""
    if boxes is None or len(boxes) == 0:
        return None
    import torch
    if isinstance(boxes, torch.Tensor) and boxes.shape[0] > 0:
        # boxes are cxcywh normalised; cx = box[i][0]
        return float(boxes[0][0])
    return None


def _ema_crop_x(
    detections: list[tuple[int, float]],   # [(time_ms, cx_normalised), ...]
    alpha: float = 0.15,
    dead_zone: float = 0.08,
) -> list[tuple[int, int]]:
    """EMA + dead-zone smoothing → list of (time_ms, crop_x_pixels)."""
    if not detections:
        return []

    result: list[tuple[int, int]] = []
    smooth = detections[0][1]
    crop_centre = smooth

    for time_ms, cx in detections:
        smooth = alpha * cx + (1 - alpha) * smooth
        if abs(smooth - crop_centre) > dead_zone:
            crop_centre = smooth
        crop_x = max(0, min(_MAX_CROP_X, int(crop_centre * _SRC_W - _CROP_W / 2)))
        result.append((time_ms, crop_x))

    return result


def get_crop_centres_gdino(
    media_id: str,
    start_ms: int,
    end_ms: int,
    proxy_path: Path,
    *,
    sample_fps: float = 3.0,
    alpha: float = 0.15,
    dead_zone: float = 0.08,
    device: Optional[str] = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    config_path: str = _DEFAULT_CONFIG,
    weights_path: str = _DEFAULT_WEIGHTS,
) -> list[tuple[int, int]]:
    """Sample proxy and return EMA-smoothed [(time_ms, crop_x), ...] via GroundingDINO.

    Falls back to [] if model fails or no detections, so caller can fall back to
    face_centroid or centre-crop.
    """
    from groundingdino.util.inference import load_model, predict, preprocess_caption
    import groundingdino.datasets.transforms as T
    import torch
    from PIL import Image
    import numpy as np

    _device = device or "cpu"
    dur_ms = end_ms - start_ms
    if dur_ms <= 0:
        return []

    try:
        model = load_model(config_path, weights_path, device=_device)
    except Exception as exc:
        _log.error("gdino load_model failed: %s", exc)
        return []

    # Build ffmpeg frame-extraction command
    n_frames = max(1, int(dur_ms / 1000.0 * sample_fps))
    step_ms = dur_ms / n_frames
    timestamps_ms = [int(start_ms + i * step_ms) for i in range(n_frames)]

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    caption = preprocess_caption("person")

    raw_detections: list[tuple[int, float]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for t_ms in timestamps_ms:
            frame_path = Path(tmpdir) / f"frame_{t_ms}.jpg"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{t_ms / 1000:.3f}",
                "-i", str(proxy_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(frame_path),
            ]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not frame_path.exists():
                continue

            try:
                img_pil = Image.open(frame_path).convert("RGB")
                img_tensor, _ = transform(img_pil, None)
                with torch.no_grad():
                    boxes, logits, _ = predict(
                        model=model,
                        image=img_tensor,
                        caption=caption,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=_device,
                    )
                cx = _gdino_crop_x_from_boxes(boxes)
                if cx is not None:
                    raw_detections.append((t_ms, cx))
            except Exception as exc:
                _log.debug("gdino frame %d ms failed: %s", t_ms, exc)

    if not raw_detections:
        _log.debug("gdino: no detections for %s %d–%d ms", media_id, start_ms, end_ms)
        return []

    return _ema_crop_x(raw_detections, alpha=alpha, dead_zone=dead_zone)


def gdino_gate_test(
    catalog: Path,
    keyframes_dir: Path,
    config_path: str = _DEFAULT_CONFIG,
    weights_path: str = _DEFAULT_WEIGHTS,
) -> dict:
    """Gate test: load model on CPU, run detection on first available keyframe.

    Returns {"passed": bool, "boxes_found": int, "device": str, "error": str|None}
    """
    from groundingdino.util.inference import load_model, predict, preprocess_caption
    import groundingdino.datasets.transforms as T
    import torch
    from PIL import Image

    device = "cpu"
    try:
        model = load_model(config_path, weights_path, device=device)
    except Exception as exc:
        return {"passed": False, "boxes_found": 0, "device": device, "error": f"load_model: {exc}"}

    # Find a keyframe jpg
    frames_root = Path(keyframes_dir)
    jpgs = sorted(frames_root.rglob("*.jpg"))
    if not jpgs:
        jpgs = sorted(Path(catalog).rglob("frames/**/*.jpg"))
    if not jpgs:
        return {"passed": False, "boxes_found": 0, "device": device,
                "error": "no keyframe jpgs found"}

    frame_path = jpgs[0]
    try:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_pil = Image.open(frame_path).convert("RGB")
        img_tensor, _ = transform(img_pil, None)
        caption = preprocess_caption("person")
        with torch.no_grad():
            boxes, logits, _ = predict(
                model=model,
                image=img_tensor,
                caption=caption,
                box_threshold=0.35,
                text_threshold=0.25,
                device=device,
            )
        n = len(boxes) if boxes is not None else 0
        return {"passed": True, "boxes_found": n, "device": device,
                "error": None, "frame_used": str(frame_path)}
    except Exception as exc:
        return {"passed": False, "boxes_found": 0, "device": device, "error": str(exc)}

#!/usr/bin/env python3
"""Per-frame face detection helper for Stage F2 ASD.

Runs InsightFace on pre-extracted JPEG frames in the EMBED venv
(which has working GPU InsightFace from F0).

Called by asd_worker.py as a subprocess:
  embed/bin/python bin/asd_detect_helper.py <frame_dir> <output_json>

Output JSON: [{"frame_idx": 0, "dets": [{"bbox": [x1,y1,x2,y2]}, ...]}, ...]
"""
import json
import os
import sys
from pathlib import Path

_CUDNN_LIB = "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"
if _CUDNN_LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import cv2
from insightface.app import FaceAnalysis

_INSIGHTFACE_ROOT = "/workspace/weights/insightface"


def main():
    if len(sys.argv) != 3:
        print(json.dumps({"ok": False, "error": "usage: asd_detect_helper.py <frame_dir> <output_json>"}))
        sys.exit(1)

    frame_dir = Path(sys.argv[1])
    output_json = Path(sys.argv[2])

    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        print(json.dumps({"ok": False, "error": "no frames found"}))
        sys.exit(1)

    app = FaceAnalysis(name="buffalo_l", root=_INSIGHTFACE_ROOT)
    app.prepare(ctx_id=0, det_size=(640, 640))

    results = []
    for i, fpath in enumerate(frame_files):
        img = cv2.imread(str(fpath))
        faces = app.get(img)
        dets = [{"bbox": f.bbox.astype(float).tolist()} for f in faces]
        results.append({"frame_idx": i, "dets": dets})

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results))
    print(json.dumps({"ok": True, "n_frames": len(results)}))


if __name__ == "__main__":
    main()

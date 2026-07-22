#!/usr/bin/env python3
"""P1 — content-based proxy orientation fix.

Rotation flags in the source files are unreliable (some correct, some spurious),
so orientation is decided from CONTENT: run face detection at 0/90/180/270 and
keep the rotation that yields the most confident faces. If the proxy is
mis-oriented, re-encode it upright (transpose) and journal the change.

No-face footage (screen recordings) can't be decided this way — left untouched
and reported, to be handled by a VLM orientation pass.

Usage: PYTHONPATH=src venvs/embed/bin/python bin/fix_orientation.py --all | --media-ids ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass
try:
    from erf.runlog import log_event
except Exception:
    log_event = None

ROOT = Path("/workspace/erf")
PROX = ROOT / "media" / "proxies"
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# transpose filter to make the CURRENT proxy upright given detected cw rotation needed
_VF = {90: "transpose=1", 180: "transpose=2,transpose=2", 270: "transpose=2"}
_app = None


def _detector():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_l",
                            providers=["CPUExecutionProvider"],
                            root="/workspace/weights/insightface")
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def _score(img) -> float:
    faces = _detector().get(img)
    return float(sum(f.det_score for f in faces)) if faces else 0.0


def _mid_frame(proxy: Path):
    cap = cv2.VideoCapture(str(proxy))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def detect_rotation(proxy: Path) -> tuple[int, float]:
    frame = _mid_frame(proxy)
    if frame is None:
        return 0, 0.0
    scores = {0: _score(frame)}
    for deg, code in _ROT.items():
        scores[deg] = _score(cv2.rotate(frame, code))
    best = max(scores, key=scores.get)
    return best, scores[best]


def fix_media(mid: str) -> dict:
    proxy = PROX / f"{mid}.mp4"
    if not proxy.exists():
        return {"media_id": mid, "status": "no_proxy"}
    deg, conf = detect_rotation(proxy)
    if conf < 0.5:
        return {"media_id": mid, "status": "no_faces", "best_deg": deg, "conf": round(conf, 2)}
    if deg == 0:
        return {"media_id": mid, "status": "ok", "conf": round(conf, 2)}
    tmp = proxy.with_suffix(".rot.mp4")
    cmd = ["ffmpeg", "-y", "-i", str(proxy), "-vf", _VF[deg], "-c:v", "libx264",
           "-crf", "20", "-preset", "fast", "-c:a", "copy", str(tmp), "-loglevel", "error"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"media_id": mid, "status": "encode_fail", "err": r.stderr[-200:]}
    tmp.replace(proxy)
    if log_event:
        log_event("ORIENT", mid, "ok", {"rotated_cw": deg, "conf": round(conf, 2)})
    return {"media_id": mid, "status": "fixed", "rotated_cw": deg, "conf": round(conf, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--media-ids", nargs="+")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ids = args.media_ids or sorted(p.stem for p in PROX.glob("m_*.mp4"))
    counts: dict[str, int] = {}
    fixed = []
    for mid in ids:
        res = fix_media(mid)
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        if res["status"] == "fixed":
            fixed.append(res)
            print(f"  FIXED {mid} cw={res['rotated_cw']} conf={res['conf']}")
        elif res["status"] == "no_faces":
            print(f"  no-face {mid} (needs VLM)")
    print(f"summary: {counts}")
    (ROOT / "logs.all" / "enrich").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs.all" / "enrich" / "orientation_fixed.json").write_text(json.dumps(fixed, indent=2))


if __name__ == "__main__":
    main()

"""Batch-patch existing Q0 records to add aesthetic_score and best_frame fields.

Reads catalog/quality/{media_id}.jsonl, scores keyframes with CLIP, writes back
atomically. Bumps schema_version from 1.3.0 (or any prior) to 1.4.0.

Usage:
    ERF_SKIP_AUDIO_CLASSIFY=1 PYTHONPATH=src venvs/embed/bin/python bin/patch_aesthetic.py

Set ERF_SKIP_AESTHETIC=1 to null-patch all records (schema bump only).
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.chdir(Path(__file__).parent.parent)

from erf.quality import _aesthetic_score_keyframes
from erf.atomicio import read_jsonl, write_tmp_rename

ERP_ROOT = Path("/workspace/erf")
CATALOG = ERP_ROOT / "catalog"
QUALITY_DIR = CATALOG / "quality"
KF_DIR = CATALOG / "keyframes"


def patch_media(media_id: str) -> dict:
    quality_path = QUALITY_DIR / f"{media_id}.jsonl"
    if not quality_path.exists():
        return {"status": "missing", "media_id": media_id}

    records = read_jsonl(quality_path)
    if not records:
        return {"status": "empty", "media_id": media_id}

    # Load keyframes for aesthetic scoring
    kf_path = KF_DIR / f"{media_id}.jsonl"
    keyframes = read_jsonl(kf_path) if kf_path.exists() else []

    score_map: dict = {}
    best_kid: str | None = None
    if keyframes:
        try:
            score_map = _aesthetic_score_keyframes(keyframes, ERP_ROOT)
            if score_map:
                best_kid = max(score_map, key=lambda k: score_map[k])
        except Exception as e:
            print(f"  [warn] aesthetic scoring failed for {media_id}: {e}")

    # Patch records
    patched = []
    for rec in records:
        kid = rec.get("keyframe_id")
        rec["schema_version"] = "1.4.0"
        if kid is not None:
            rec["aesthetic_score"] = score_map.get(kid)
            rec["best_frame"] = (kid == best_kid) if best_kid is not None else None
        else:
            # Audio-only record
            rec["aesthetic_score"] = None
            rec["best_frame"] = None
        patched.append(rec)

    data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in patched) + "\n").encode("utf-8")
    write_tmp_rename(quality_path, data)
    return {"status": "ok", "media_id": media_id, "records": len(patched), "scored": len(score_map)}


def main():
    media_ids = sorted(p.stem for p in QUALITY_DIR.glob("*.jsonl"))
    print(f"Patching {len(media_ids)} media files...")

    ok = error = skip = 0
    for i, mid in enumerate(media_ids, 1):
        try:
            result = patch_media(mid)
            status = result["status"]
            if status == "ok":
                scored = result.get("scored", 0)
                print(f"[{i:3d}/{len(media_ids)}] {mid} — {result['records']} records, {scored} frames scored")
                ok += 1
            else:
                print(f"[{i:3d}/{len(media_ids)}] {mid} — {status}")
                skip += 1
        except Exception as e:
            print(f"[{i:3d}/{len(media_ids)}] {mid} — ERROR: {e}")
            error += 1

    print(f"\nDone: {ok} ok, {skip} skipped, {error} errors")


if __name__ == "__main__":
    main()

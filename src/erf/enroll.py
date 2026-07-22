"""erf-enroll — cast/crew enrollment CLI for Phase 3b identity."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import tomli_w
from datetime import datetime, timezone
from pathlib import Path

_CONFIG_PATH = Path("state/config.toml")
_CATALOG = Path("catalog")
_ENROLL = Path("enroll")
_LANCE = Path("index/lance")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_person(person_id: str) -> dict | None:
    p = _CATALOG / "persons" / f"{person_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save_person(rec: dict) -> None:
    (_CATALOG / "persons").mkdir(parents=True, exist_ok=True)
    p = _CATALOG / "persons" / f"{rec['person_id']}.json"
    p.write_text(json.dumps(rec, indent=2))


def _persons_table():
    import lancedb
    db = lancedb.connect(str(_LANCE))
    if "persons_v1" in db.list_tables().tables:  # list_tables() returns ListTablesResponse in lancedb 0.33+
        return db.open_table("persons_v1")
    return None


def _load_gallery() -> dict:
    p = _CATALOG / "persons" / "_gallery.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"schema_version": "1.0.0", "face_match_floor": 0.38, "face_match_margin": 0.10, "persons": {}}


def _write_config_threshold(floor: float, margin: float) -> None:
    if not _CONFIG_PATH.exists():
        return
    with open(_CONFIG_PATH, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg.setdefault("identity", {})
    cfg["identity"]["face_match_floor"] = round(floor, 4)
    cfg["identity"]["face_match_margin"] = round(margin, 4)
    with open(_CONFIG_PATH, "wb") as fh:
        tomli_w.dump(cfg, fh)


def _calibrate_gallery(person_ids: list[str]) -> tuple[float, float]:
    """Compute face match floor from enrolled gallery. Returns (floor, margin)."""
    import numpy as np
    tbl = _persons_table()
    if tbl is None or len(person_ids) < 2:
        return 0.38, 0.10

    rows = tbl.search().limit(10000).to_list()
    by_person: dict[str, list] = {}
    for r in rows:
        pid = r["person_id"]
        by_person.setdefault(pid, []).append(r["vector"])

    if len(by_person) < 2:
        return 0.38, 0.10

    means = {pid: np.mean(vecs, axis=0) for pid, vecs in by_person.items()}
    pids = list(means.keys())

    # Cross-person cosines (all pairs)
    cross = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = means[pids[i]], means[pids[j]]
            cross.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))

    # Intra-person cosines
    intra = []
    for pid, vecs in by_person.items():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                a, b = np.array(vecs[i]), np.array(vecs[j])
                intra.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))

    if not cross or not intra:
        return 0.38, 0.10

    crossover = (max(cross) + min(intra)) / 2.0
    floor = max(0.30, min(0.45, crossover))
    return round(floor, 4), 0.10


# ── Subcommands ────────────────────────────────────────────────────────────────

def cmd_new(args: argparse.Namespace) -> int:
    name = args.name
    if args.person_id:
        person_id = args.person_id
    else:
        person_id = "p_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    existing = _load_person(person_id)
    if existing:
        print(f"already exists: {person_id}")
        return 0

    (_ENROLL / person_id / "faces").mkdir(parents=True, exist_ok=True)
    (_ENROLL / person_id / "voice").mkdir(parents=True, exist_ok=True)

    rec = {
        "schema_version": "1.0.0",
        "person_id": person_id,
        "display_name": name,
        "consent": args.consent,
        "face_enrolled": False,
        "face_gallery_relpath": None,
        "voice_enrolled": False,
        "enrolled_at_utc": _now(),
    }
    _save_person(rec)
    print(person_id)
    return 0


def cmd_face(args: argparse.Namespace) -> int:
    person_id = args.person_id
    rec = _load_person(person_id)
    if rec is None:
        print(f"error: unknown person {person_id} — run erf-enroll new first", file=sys.stderr)
        return 1

    face_dir = _ENROLL / person_id / "faces"
    images = sorted(face_dir.glob("*.jpg")) + sorted(face_dir.glob("*.png")) + sorted(face_dir.glob("*.jpeg"))
    if not images:
        print(f"error: no images in {face_dir}", file=sys.stderr)
        return 1

    try:
        import cv2
        import insightface
        from insightface.app import FaceAnalysis
        import numpy as np
        import lancedb
        import pyarrow as pa
    except ImportError as e:
        print(f"error: missing dependency — {e}", file=sys.stderr)
        return 1

    # Ensure cuDNN is on the library path before loading ONNX models
    _cudnn_lib = "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"
    import os as _os
    _existing = _os.environ.get("LD_LIBRARY_PATH", "")
    if _cudnn_lib not in _existing:
        _os.environ["LD_LIBRARY_PATH"] = f"{_cudnn_lib}:{_existing}" if _existing else _cudnn_lib

    app = FaceAnalysis(name="buffalo_l", root="/workspace/weights/insightface")
    app.prepare(ctx_id=0, det_size=(640, 640))

    templates = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"warn: could not read {img_path}", file=sys.stderr)
            continue
        faces = app.get(img)
        for face in faces:
            if face.det_score < 0.6:
                continue
            emb = np.array(face.embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            templates.append({"embedding": emb, "source": str(img_path)})

    if not templates:
        print(f"error: no detectable faces in {face_dir}", file=sys.stderr)
        return 1

    if len(templates) < 3:
        print(f"warn: only {len(templates)} template(s) for {person_id} (recommended ≥3)", file=sys.stderr)

    # Upsert to persons_v1
    db = lancedb.connect(str(_LANCE))
    enrolled_at = _now()

    rows = []
    for i, t in enumerate(templates):
        rows.append({
            "id": f"{person_id}:tmpl:{i:03d}",
            "person_id": person_id,
            "vector": t["embedding"].tolist(),
            "display_name": rec["display_name"],
            "source_image_relpath": t["source"],
            "enrolled_at_utc": enrolled_at,
        })

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("person_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 512)),
        pa.field("display_name", pa.string()),
        pa.field("source_image_relpath", pa.string()),
        pa.field("enrolled_at_utc", pa.string()),
    ])

    if "persons_v1" not in db.list_tables().tables:
        tbl = db.create_table("persons_v1", data=rows, schema=schema)
    else:
        tbl = db.open_table("persons_v1")
        # Remove existing rows for this person
        tbl.delete(f"person_id = '{person_id}'")
        tbl.add(rows)

    # Calibrate
    all_persons = [p.stem for p in (_CATALOG / "persons").glob("*.json") if p.stem != "_gallery"]
    floor, margin = _calibrate_gallery(all_persons)

    gallery = _load_gallery()
    gallery["face_match_floor"] = floor
    gallery["face_match_margin"] = margin
    gallery.setdefault("persons", {})[person_id] = {"template_count": len(templates)}
    (_CATALOG / "persons").mkdir(parents=True, exist_ok=True)
    (_CATALOG / "persons" / "_gallery.json").write_text(json.dumps(gallery, indent=2))

    try:
        _write_config_threshold(floor, margin)
    except Exception:
        pass  # config update is best-effort

    rec["face_enrolled"] = True
    rec["face_gallery_relpath"] = "catalog/persons/_gallery.json"
    _save_person(rec)

    print(f"Enrolled {len(templates)} templates for {person_id} (floor={floor})")
    return 0


def cmd_voice(args: argparse.Namespace) -> int:
    print("voice enrollment not yet implemented")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    persons_dir = _CATALOG / "persons"
    overlay_dir = _CATALOG / "identity_overlay"

    person_files = sorted(p for p in persons_dir.glob("*.json") if p.stem != "_gallery")
    if not person_files:
        print("No enrolled persons.")
    else:
        tbl = _persons_table()
        print(f"{'PERSON_ID':<20} {'NAME':<20} {'FACE':<8} {'VOICE':<8} TEMPLATES")
        print("-" * 65)
        for pf in person_files:
            rec = json.loads(pf.read_text())
            pid = rec["person_id"]
            tmpl_count = 0
            if tbl is not None:
                try:
                    rows = tbl.search().where(f"person_id = '{pid}'").limit(100).to_list()
                    tmpl_count = len(rows)
                except Exception:
                    pass
            print(f"{pid:<20} {rec['display_name']:<20} {'yes' if rec['face_enrolled'] else 'no':<8} "
                  f"{'yes' if rec['voice_enrolled'] else 'no':<8} {tmpl_count}")

    # Unresolved speaker labels
    overlays = []
    if overlay_dir.exists():
        overlays = sorted(overlay_dir.glob("*.json"))
        if args.media_id:
            overlays = [o for o in overlays if o.stem == args.media_id]

    unresolved = []
    for ov_path in overlays:
        try:
            ov = json.loads(ov_path.read_text())
            mid = ov["media_id"]
            for sa in ov.get("speaker_assignments", []):
                if sa["person_id"] is None:
                    unresolved.append((mid, sa["speaker_label"], sa.get("method", "none")))
        except Exception:
            pass

    if unresolved:
        print()
        print(f"{'MEDIA_ID':<22} {'SPEAKER_LABEL':<18} METHOD")
        print("Unresolved speaker labels:")
        print("-" * 55)
        for mid, label, method in unresolved:
            print(f"  {mid:<20} {label:<18} {method}")
        print(f"\n  → use: erf-enroll link <media_id> <speaker_label> <person_id>")
    else:
        print("\nNo unresolved speaker labels.")
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    links_path = _CATALOG / "persons" / "_manual_links.jsonl"
    links_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "media_id": args.media_id,
        "speaker_label": args.speaker_label,
        "person_id": args.person_id,
        "created_at_utc": _now(),
    }
    with open(links_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"Linked {args.speaker_label} in {args.media_id} → {args.person_id}")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    import lancedb
    db = lancedb.connect(str(_LANCE))
    if "persons_v1" in db.list_tables().tables:
        db.drop_table("persons_v1")

    persons_dir = _CATALOG / "persons"
    rebuilt = 0
    for pf in sorted(persons_dir.glob("*.json")):
        if pf.stem.startswith("_"):
            continue
        rec = json.loads(pf.read_text())
        if rec.get("face_enrolled"):
            ns = argparse.Namespace(person_id=rec["person_id"])
            cmd_face(ns)
            rebuilt += 1

    print(f"Rebuilt persons_v1 ({rebuilt} persons)")
    return 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="erf-enroll")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new")
    p_new.add_argument("--name", required=True)
    p_new.add_argument("--person-id", dest="person_id", default=None)
    p_new.add_argument("--consent", type=lambda x: x.lower() != "false", default=True)

    p_face = sub.add_parser("face")
    p_face.add_argument("person_id")

    p_voice = sub.add_parser("voice")
    p_voice.add_argument("person_id")

    p_status = sub.add_parser("status")
    p_status.add_argument("--media-id", dest="media_id", default=None)

    p_link = sub.add_parser("link")
    p_link.add_argument("media_id")
    p_link.add_argument("speaker_label")
    p_link.add_argument("person_id")

    sub.add_parser("rebuild")

    args = parser.parse_args()
    dispatch = {
        "new": cmd_new,
        "face": cmd_face,
        "voice": cmd_voice,
        "status": cmd_status,
        "link": cmd_link,
        "rebuild": cmd_rebuild,
    }
    sys.exit(dispatch[args.cmd](args))


if __name__ == "__main__":
    main()

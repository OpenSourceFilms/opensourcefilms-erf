#!/usr/bin/env python3
"""Backfill source_note for manifests where it is null.

Searches /workspace/BTS/ for the original file by matching filename,
derives source_note from the parent directory name + filename date,
and writes atomically.

Run from /workspace/erf:
  venvs/embed/bin/python bin/backfill_source_note.py [--dry-run]
"""
import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erf.asr import _derive_source_note

BTS_ROOTS = [
    Path("/workspace/BTS"),
    Path("/workspace/erf/media"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

with open("state/config.toml", "rb") as fh:
    cfg = tomllib.load(fh)
catalog = Path(cfg["paths"]["catalog"])

# Build filename → Path index from all known source roots
print("Scanning source directories...")
filename_index: dict[str, Path] = {}
for root in BTS_ROOTS:
    if not root.exists():
        continue
    for ext in ("*.mp4", "*.mov", "*.MP4", "*.MOV"):
        for p in root.rglob(ext):
            if p.name not in filename_index:
                filename_index[p.name] = p
print(f"  Indexed {len(filename_index)} source files\n")

n_set = n_skip_missing = n_skip_no_note = n_already_set = 0

for manifest_path in sorted((catalog / "media_manifest").glob("*.json")):
    d = json.loads(manifest_path.read_text())
    if d.get("source_note") is not None:
        n_already_set += 1
        continue

    fname = Path(d.get("relpath", "")).name
    src_path = filename_index.get(fname)

    if src_path is None:
        print(f"  SKIP {d['media_id']} — source file not found for {fname!r}")
        n_skip_missing += 1
        continue

    note = _derive_source_note(src_path)
    if note is None:
        print(f"  SKIP {d['media_id']} — no derivable note from {src_path.parent.name!r}")
        n_skip_no_note += 1
        continue

    d["source_note"] = note
    if not args.dry_run:
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, manifest_path)
    print(f"  {'DRY' if args.dry_run else 'SET'} {d['media_id']} → {note!r}")
    n_set += 1

print(f"\nDone. set={n_set}  skipped_missing={n_skip_missing}  skipped_no_note={n_skip_no_note}  already_set={n_already_set}")

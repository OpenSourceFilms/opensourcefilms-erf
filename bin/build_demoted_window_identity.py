#!/usr/bin/env python3
"""Stratum-D per-window voiceprint scoring (D-Log-31, worklist item 1).

Every identity_map slot with source == "demoted_overmerge" currently gets
ONE anonymous person_id applied uniformly across its whole (media, label)
time-span (build_identity_clusters.py's physical-exclusivity post-pass).
But the diarizer's over-merge means that span can genuinely contain speech
from several different real people (GAMEPLAN example: m_e1b299626700's
demoted SPEAKER_00 slot contains six different people). This script scores
each of voiceprint_v2_extract.py's per-window chunk embeddings (already
computed as a byproduct of that tool, previously unused) independently
against the gallery, so different windows within one demoted label can
resolve to different people or stay anonymous based on their own local
voice — never worse than today's uniform-anonymous fallback, only ever an
upgrade for whichever windows clear the threshold.

Same gallery + threshold convention as build_identity_clusters.py
(GALLERY_THR=0.45 cosine sim, voiceprint_v2/gallery_clean centroids) so a
window's verdict here is directly comparable to a slot's verdict there.

Writes catalog/demoted_window_identity/{media_id}.json:
  {"media_id": ..., "windows": [
      {"speaker_label", "start_ms", "end_ms", "person_id", "score"}, ...
  ]}
person_id is null when no gallery person clears GALLERY_THR for that
window — downstream falls back to the existing anonymous behavior.

Run: PYTHONPATH=src venvs/control/bin/python bin/build_demoted_window_identity.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root

CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam); unset = ROOT/catalog
GALLERY_DIR = CAT / "voiceprint_v2" / "gallery_clean"
EMBEDDINGS = CAT / "voiceprint_v2" / "embeddings" / "wespeaker.jsonl"
IDENTITY_MAP_DIR = CAT / "identity_map"
OUT_DIR = CAT / "demoted_window_identity"
GALLERY_THR = 0.45  # same floor as build_identity_clusters.py — comparable verdicts


def _l2(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def load_gallery() -> tuple[np.ndarray, list[str]]:
    names, vecs = [], []
    for f in sorted(glob.glob(str(GALLERY_DIR / "*.json"))):
        d = json.loads(Path(f).read_text())
        c = d.get("centroid")
        if isinstance(c, list) and len(c) == 256:
            names.append(d.get("person_id", Path(f).stem))
            vecs.append(_l2(c))
    if not vecs:
        return np.zeros((0, 256)), []
    return np.stack(vecs), names


def load_embeddings() -> dict[tuple[str, str], dict]:
    """(media_id, speaker_label) -> {"chunks": [...], "chunk_windows": [...]}"""
    out: dict[tuple[str, str], dict] = {}
    if not EMBEDDINGS.exists():
        return out
    for line in EMBEDDINGS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        chunks = row.get("chunks") or []
        windows = row.get("chunk_windows") or []
        if not chunks or len(chunks) != len(windows):
            continue  # no usable per-window data for this slot (stale/pre-D-Log-31 row)
        out[(row["media_id"], row["speaker_label"])] = {
            "chunks": chunks, "chunk_windows": windows,
        }
    return out


def find_demoted_slots() -> dict[str, list[str]]:
    """media_id -> [speaker_label, ...] for every demoted_overmerge slot."""
    out: dict[str, list[str]] = {}
    for f in sorted(glob.glob(str(IDENTITY_MAP_DIR / "*.json"))):
        d = json.loads(Path(f).read_text())
        labels = [lab for lab, rec in d.get("speakers", {}).items()
                  if rec.get("source") == "demoted_overmerge"]
        if labels:
            out[d["media_id"]] = labels
    return out


def score_windows(gallery_vecs: np.ndarray, gallery_names: list[str],
                   chunks: list[list[float]]) -> list[tuple[str | None, float]]:
    """Per-chunk (person_id_or_None, best_score)."""
    if gallery_vecs.shape[0] == 0:
        return [(None, 0.0) for _ in chunks]
    results = []
    for c in chunks:
        v = _l2(c)
        sims = gallery_vecs @ v
        j = int(np.argmax(sims))
        best = float(sims[j])
        if best >= GALLERY_THR:
            results.append((gallery_names[j], round(best, 4)))
        else:
            results.append((None, round(best, 4)))
    return results


def main() -> None:
    gallery_vecs, gallery_names = load_gallery()
    embeddings = load_embeddings()
    demoted = find_demoted_slots()

    n_media_written = 0
    n_windows_total = 0
    n_windows_resolved = 0
    n_slots_no_embeddings = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mid, labels in demoted.items():
        windows_out = []
        for label in labels:
            key = (mid, label)
            data = embeddings.get(key)
            if data is None:
                n_slots_no_embeddings += 1
                continue
            verdicts = score_windows(gallery_vecs, gallery_names, data["chunks"])
            for (start_ms, end_ms), (person_id, score) in zip(data["chunk_windows"], verdicts):
                windows_out.append({
                    "speaker_label": label, "start_ms": start_ms, "end_ms": end_ms,
                    "person_id": person_id, "score": score,
                })
                n_windows_total += 1
                if person_id is not None:
                    n_windows_resolved += 1
        if not windows_out:
            continue
        windows_out.sort(key=lambda w: (w["speaker_label"], w["start_ms"]))
        out_path = OUT_DIR / f"{mid}.json"
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"media_id": mid, "windows": windows_out},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, out_path)
        n_media_written += 1

    print(f"demoted slots: {sum(len(v) for v in demoted.values())} across {len(demoted)} media")
    print(f"slots with no per-window embeddings (not yet extracted): {n_slots_no_embeddings}")
    print(f"windows scored: {n_windows_total}, resolved (>= {GALLERY_THR}): {n_windows_resolved}")
    print(f"wrote {n_media_written} files to {OUT_DIR}")


if __name__ == "__main__":
    main()

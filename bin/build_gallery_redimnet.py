#!/usr/bin/env python3
"""Re-embed WeSpeaker gallery_clean source clips with ReDimNet2-b6 (D-Identity-04).

Reads the same confirmed gold labels that built the WeSpeaker gallery_clean,
re-embeds each (media_id, speaker_label) clip using ReDimNet2-b6 (corrected
checkpoint, PalabraAI/redimnet2 b6 lm/vb2+vox2_v0 — supersedes the v1
ReDimNet-b6 gallery), and writes a parallel gallery at 192-dim.

Writes:
  state/identity/gallery_redimnet2_b6/{person_id}.json   (person_id, centroid, n_clips)
  state/identity/gallery_redimnet2_b6/manifest.json      (clip provenance + build metadata)
  state/eval/runs/gallery_redimnet2_report.json          (quality report, also printed)

Run with:
  PYTHONPATH=src TORCH_HOME=/workspace/erf/weights/torchhub \\
    venvs/diarize/bin/python bin/build_gallery_redimnet.py
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure TORCH_HOME is set before erf.embeddings import triggers torch load
os.environ.setdefault("TORCH_HOME", "/workspace/erf/weights/torchhub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from erf.embeddings import cosine, get_embedder  # noqa: E402

CAT = ROOT / "catalog"
PROXY = ROOT / "media" / "proxies"
OUT_GAL = ROOT / "state" / "identity" / "gallery_redimnet2_b6"
EVAL_RUNS = ROOT / "state" / "eval" / "runs"
SUPPLEMENT = CAT / "identity" / "enrollment_supplement_2026-07-04.jsonl"

SR = 16_000
MIN_CHUNK_S = 0.8
MAX_CHUNK_S = 4.0
MAX_CHUNKS_PER_CLIP = 12

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_confirmed_clips() -> dict[str, list[tuple[str, str]]]:
    """Return person_id -> sorted list of unique (media_id, speaker_label) pairs.

    Reads the same three gold-label files used by identity_resolve_v1.py.
    """
    confirmed: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for r in (json.loads(l) for l in open(CAT / "identity/gold_labels.jsonl") if l.strip()):
        if r.get("person_id"):
            confirmed[r["person_id"]].add((r["media_id"], r["speaker"]))

    for r in (json.loads(l) for l in open(CAT / "identity/gold_labels_face_bridged.jsonl") if l.strip()):
        pid = r.get("confirmed_person_id")
        if pid:
            confirmed[pid].add((r["media_id"], r["speaker"]))

    if (CAT / "identity/supplemental_reviewed.jsonl").exists():
        for r in (json.loads(l) for l in open(CAT / "identity/supplemental_reviewed.jsonl") if l.strip()):
            if r.get("person_id"):
                confirmed[r["person_id"]].add((r["media_id"], r["speaker"]))

    return {pid: sorted(clips) for pid, clips in confirmed.items()}


def decode_audio_16k(path: Path) -> np.ndarray | None:
    """Decode the full audio of a proxy file to 16 kHz mono float32 via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        log.warning("ffmpeg failed for %s: %s", path.name, result.stderr.decode()[:200])
        return None
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def get_speaker_turns(media_id: str, speaker_label: str) -> list[tuple[int, int]]:
    """Return sample-indexed (start, end) pairs for speaker turns from diar.jsonl.

    diar.jsonl fields: speaker_label, start_ms, end_ms.
    """
    diar_path = CAT / "asr" / f"{media_id}.diar.jsonl"
    if not diar_path.exists():
        return []
    turns = []
    for line in diar_path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if t.get("speaker_label") == speaker_label:
            s = int(t["start_ms"] * SR / 1000)
            e = int(t["end_ms"] * SR / 1000)
            if e > s:
                turns.append((s, e))
    return turns


def load_supplement_clips() -> dict[str, list[dict]]:
    """Return person_id -> list of supplement rows from enrollment_supplement JSONL.

    Returns an empty dict when the supplement file does not exist.
    Each row is expected to have: person_id, media_id, start_s, end_s, source_clip.
    These ranges are pre-vetted; no further filtering is applied.
    """
    if not SUPPLEMENT.exists():
        return {}
    supplement: dict[str, list[dict]] = defaultdict(list)
    for line in SUPPLEMENT.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        supplement[r["person_id"]].append(r)
    return dict(supplement)


def make_chunks(turns: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Split speaker turn intervals into 0.8–4.0 s chunks (sample indices).

    Returns up to MAX_CHUNKS_PER_CLIP chunks, sorted longest-first.
    """
    mn = int(MIN_CHUNK_S * SR)
    mx = int(MAX_CHUNK_S * SR)
    out: list[tuple[int, int]] = []
    for s, e in turns:
        pos = s
        while e - pos >= mn:
            w = min(mx, e - pos)
            out.append((pos, pos + w))
            pos += w
    out.sort(key=lambda c: c[1] - c[0], reverse=True)
    return out[:MAX_CHUNKS_PER_CLIP]


def embed_clip(
    audio: np.ndarray,
    turns: list[tuple[int, int]],
    embedder,
) -> tuple[np.ndarray | None, int]:
    """Embed one (media, speaker) clip; returns (mean_embedding | None, n_chunks_ok)."""
    chunks = make_chunks(turns)
    vecs: list[np.ndarray] = []
    for s, e in chunks:
        seg = audio[s:e]
        if len(seg) < int(0.5 * SR):
            continue
        try:
            vecs.append(embedder.embed(seg))
        except Exception as exc:
            log.warning("embed error on chunk [%d-%d]: %s", s, e, exc)
    if not vecs:
        return None, 0
    mean = np.stack(vecs, axis=0).mean(axis=0)
    n = np.linalg.norm(mean)
    return (mean / n if n > 1e-9 else mean), len(vecs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_GAL.mkdir(parents=True, exist_ok=True)
    EVAL_RUNS.mkdir(parents=True, exist_ok=True)

    log.info("Loading confirmed clips from gold labels …")
    confirmed = load_confirmed_clips()
    log.info("Persons: %s", sorted(confirmed))

    log.info("Initialising ReDimNet2-b6 embedder …")
    embedder = get_embedder("redimnet2_b6")

    # Collect per-person clip embeddings
    person_clips: dict[str, list[tuple[np.ndarray, dict]]] = defaultdict(list)
    manifest_clips: list[dict] = []

    for pid, clips in sorted(confirmed.items()):
        log.info("Person %s: %d confirmed clip(s)", pid, len(clips))
        for media_id, speaker_label in clips:
            proxy = PROXY / f"{media_id}.mp4"
            if not proxy.exists():
                log.warning("  no proxy for %s, skipping", media_id)
                continue
            audio = decode_audio_16k(proxy)
            if audio is None:
                continue
            turns = get_speaker_turns(media_id, speaker_label)
            if not turns:
                log.warning("  no diar turns for %s/%s, skipping", media_id, speaker_label)
                continue
            total_s = sum(e - s for s, e in turns) / SR
            emb, n_chunks = embed_clip(audio, turns, embedder)
            if emb is None:
                log.warning("  embedding failed for %s/%s", media_id, speaker_label)
                continue
            meta = {
                "person_id": pid,
                "media_id": media_id,
                "source": "gold_label",
                "speaker_label": speaker_label,
                "n_turns": len(turns),
                "total_speech_s": round(total_s, 2),
                "n_chunks_embedded": n_chunks,
            }
            person_clips[pid].append((emb, meta))
            manifest_clips.append(meta)
            log.info("  %s/%s  turns=%d  speech=%.1fs  chunks=%d → OK",
                     media_id, speaker_label, len(turns), total_s, n_chunks)

    # Process enrollment supplement clips (explicit pre-vetted time ranges)
    supplement = load_supplement_clips()
    if supplement:
        log.info("Loading enrollment supplement from %s …", SUPPLEMENT)
        for pid, rows in sorted(supplement.items()):
            log.info("Supplement %s: %d row(s)", pid, len(rows))
            for row in rows:
                media_id = row["media_id"]
                source_clip = row.get("source_clip", "")
                start_s = row["start_s"]
                end_s = row["end_s"]
                proxy = PROXY / f"{media_id}.mp4"
                if not proxy.exists():
                    log.warning("  no proxy for %s (supplement %s), skipping",
                                media_id, source_clip)
                    continue
                audio = decode_audio_16k(proxy)
                if audio is None:
                    continue
                # Convert explicit time range → sample-indexed turn list
                turns_sup = [(int(start_s * SR), int(end_s * SR))]
                total_s_sup = end_s - start_s
                emb, n_chunks = embed_clip(audio, turns_sup, embedder)
                if emb is None:
                    log.warning("  embedding failed for supplement %s/%s [%.2f-%.2f]",
                                media_id, source_clip, start_s, end_s)
                    continue
                meta = {
                    "person_id": pid,
                    "media_id": media_id,
                    "source": "enrollment_supplement",
                    "source_clip": source_clip,
                    "start_s": start_s,
                    "end_s": end_s,
                    "total_speech_s": round(total_s_sup, 2),
                    "n_chunks_embedded": n_chunks,
                }
                person_clips[pid].append((emb, meta))
                manifest_clips.append(meta)
                log.info("  %s/%s [%.2f-%.2f]  chunks=%d → OK",
                         media_id, source_clip, start_s, end_s, n_chunks)
    else:
        log.info("No enrollment supplement found at %s, skipping", SUPPLEMENT)

    # Build centroids and write per-person gallery files
    person_centroids: dict[str, np.ndarray] = {}
    for pid, clip_list in sorted(person_clips.items()):
        vecs = np.stack([e for e, _ in clip_list], axis=0)
        centroid = vecs.mean(axis=0)
        n = np.linalg.norm(centroid)
        centroid = centroid / n if n > 1e-9 else centroid
        person_centroids[pid] = centroid

        entry = {"person_id": pid, "centroid": centroid.tolist(), "n_clips": len(clip_list)}
        out_path = OUT_GAL / f"{pid}.json"
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry))
        os.replace(tmp, out_path)
        log.info("Gallery written: %s (%d clip(s))", pid, len(clip_list))

    # Write manifest
    manifest = {
        "model": "redimnet2_b6",
        "hub": "PalabraAI/redimnet2",
        "train_type": "lm",
        "dataset": "vb2+vox2_v0",
        "dim": embedder.dim,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_persons": len(person_centroids),
        "n_clips": len(manifest_clips),
        "clips": manifest_clips,
    }
    mf_path = OUT_GAL / "manifest.json"
    tmp = mf_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, mf_path)
    log.info("Manifest written: %s", mf_path)

    # Quality report
    persons = sorted(person_clips)
    report_rows: list[dict] = []
    for pid in persons:
        embs = [e for e, _ in person_clips[pid]]
        n = len(embs)

        # Intra-person pairwise cosines
        pairs = [cosine(embs[i], embs[j]) for i in range(n) for j in range(i + 1, n)]
        mean_intra = float(np.mean(pairs)) if pairs else None

        # Max cross-person centroid cosine
        cent = person_centroids[pid]
        cross_scores = [(cosine(cent, person_centroids[other]), other)
                        for other in persons if other != pid]
        if cross_scores:
            max_cross_cos, max_cross_pid = max(cross_scores, key=lambda x: x[0])
        else:
            max_cross_cos, max_cross_pid = None, None

        report_rows.append({
            "person_id": pid,
            "n_clips": n,
            "mean_intra_cosine": mean_intra,
            "max_cross_centroid_cosine": max_cross_cos,
            "max_cross_person": max_cross_pid,
            "flag_cross_too_close": max_cross_cos is not None and max_cross_cos > 0.5,
        })

    # Print table
    print("\n=== ReDimNet2-b6 gallery quality report ===")
    hdr = f"{'Person':<22} {'clips':>5} {'intra_cos':>10} {'max_cross':>10} {'cross_vs':<22} {'flag':>6}"
    print(hdr)
    print("─" * len(hdr))
    for r in report_rows:
        intra = f"{r['mean_intra_cosine']:.3f}" if r["mean_intra_cosine"] is not None else "  n/a"
        cross = f"{r['max_cross_centroid_cosine']:.3f}" if r["max_cross_centroid_cosine"] is not None else "  n/a"
        flag = "!CLOSE" if r["flag_cross_too_close"] else ""
        print(f"{r['person_id']:<22} {r['n_clips']:5d} {intra:>10} {cross:>10} "
              f"{r.get('max_cross_person',''):<22} {flag:>6}")
    flags = [r for r in report_rows if r["flag_cross_too_close"]]
    if flags:
        print(f"\nWARNING: {len(flags)} person pair(s) with cross-centroid cosine > 0.5")

    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "redimnet2_b6",
        "n_persons": len(persons),
        "n_clips_total": sum(r["n_clips"] for r in report_rows),
        "persons": report_rows,
        "flags": flags,
    }
    rep_path = EVAL_RUNS / "gallery_redimnet2_report.json"
    tmp = rep_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(full_report, indent=2))
    os.replace(tmp, rep_path)
    print(f"\nReport saved: {rep_path}")


if __name__ == "__main__":
    main()

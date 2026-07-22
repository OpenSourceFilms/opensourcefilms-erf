#!/usr/bin/env python3
"""Bake-off of speaker embedding models on golden attribution windows."""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- MUST set environment before torch import ---
os.environ["TORCH_HOME"] = "/workspace/erf/weights/torchhub"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import torchaudio
from pyannote.audio import Inference, Model

# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
ROOT = Path("/workspace/erf")
GOLDEN = ROOT / "state/eval/attribution_golden.jsonl"
ADJ_DIR = ROOT / "catalog/speech_events_v2_adjudicated"
PROXIES = ROOT / "media/proxies"
WORK = Path("/tmp/claude-0/-workspace-erf/80181239-cfb4-4f35-a5a6-8733da1bda64/scratchpad/embed_bakeoff")
RUNS = ROOT / "state/eval/runs"

WORK_WAVS = WORK / "wavs"
EMB_CACHE_PAT = "emb_{backend}.npz"

BACKEND_CHOICES = ["wespeaker", "redimnet_b6", "redimnet2_b6", "ecapa2"]
ECAPA2_WEIGHTS = Path("/workspace/erf/weights/ecapa2/ecapa2.pt")

# ----------------------------------------------------------------------
def find_proxy(media_id: str) -> Optional[Path]:
    """Locate proxy file for a media_id inside PROXIES."""
    candidates = list(Path(PROXIES).glob(f"{media_id}*"))
    if not candidates:
        logger.warning(f"No proxy found for media_id {media_id}")
        return None
    # if multiple, first is fine (ffmpeg can handle many formats)
    return candidates[0]


def extract_wav(
    tag: str,
    start_sec: float,
    duration_sec: float,
    proxy: Path,
) -> None:
    """Extract a mono 16 kHz wav segment via ffmpeg."""
    out = WORK_WAVS / f"{tag}.wav"
    if out.exists():
        logger.info(f"Wav {tag} exists, skipping extraction")
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", str(proxy),
        "-t", str(duration_sec),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-f", "wav",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Extracted {tag}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed for {tag}: {e.stderr}")
        raise


def load_golden_entries(path: Path) -> List[Dict]:
    """Parse JSONL golden file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries.append(entry)
    return entries


def classify_windows(
    entries: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split golden entries into named (exactly one assertion with person_id)
    and anon (all assertions have anonymous:true). Others skipped.
    Returns (named_windows, anon_windows) where each window is the full entry dict.
    """
    named = []
    anon = []
    for entry in entries:
        if not entry.get("verified"):
            continue
        expected = entry.get("expected", [])
        if not expected:
            continue

        # named: exactly one assertion with a person_id
        if len(expected) == 1 and "person_id" in expected[0]:
            named.append(entry)
            continue

        # anon: every assertion has anonymous:true
        if all(a.get("anonymous") is True for a in expected):
            anon.append(entry)
            continue

        # otherwise skip (multi-assertion that doesn't fit)
    return named, anon


def build_golden_spans_by_media(
    entries: List[Dict],
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Return dict mapping media_id -> list of (start_ms, end_ms) for all golden windows.
    We assume golden start/end fields are float seconds.
    """
    spans: Dict[str, List[Tuple[float, float]]] = {}
    for e in entries:
        mid = e.get("media_id")
        start = e.get("start_ms")
        end = e.get("end_ms")
        if mid is None or start is None or end is None:
            logger.warning(f"Skipping golden entry without media/start_ms/end_ms: {e.get('golden_id','?')}")
            continue
        spans.setdefault(mid, []).append((float(start), float(end)))
    return spans


def overlaps_any_golden(
    ms_start: float,
    ms_end: float,
    media_id: str,
    golden_spans: Dict[str, List[Tuple[float, float]]],
) -> bool:
    """Return True if [ms_start, ms_end) overlaps any golden span for media_id."""
    if media_id not in golden_spans:
        return False
    for gs, ge in golden_spans[media_id]:
        if max(ms_start, gs) < min(ms_end, ge):  # any overlap
            return True
    return False


def gather_gallery_candidates(
    adj_dir: Path,
    golden_spans: Dict,
    min_conf: float = 0.9,
) -> Dict[str, List[Dict]]:
    """
    Scan all ADJ_DIR/*.jsonl events and collect candidates per person.
    Returns dict: person_id -> list of event dicts (with extra key 'duration_ms').
    """
    person_cands: Dict[str, List[Dict]] = {}
    jsonl_files = list(adj_dir.glob("*.jsonl"))
    logger.info(f"Scanning {len(jsonl_files)} adjudicated files for gallery clips")
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                evt = json.loads(line)
                pid = evt.get("person_id")
                if not pid:
                    continue
                conf = evt.get("confidence", {})
                speaker_conf = conf.get("speaker", 0)
                if speaker_conf < min_conf:
                    continue
                ms_start = evt.get("start_ms")
                ms_end = evt.get("end_ms")
                if ms_start is None or ms_end is None:
                    continue
                dur = ms_end - ms_start
                if not (2000 <= dur <= 20000):
                    continue
                n_words = evt.get("n_words", 0)
                if n_words < 4:
                    continue
                media_id = evt.get("media_id")
                if not media_id:
                    continue

                # leak guard: must not overlap any golden window in same media
                if overlaps_any_golden(ms_start, ms_end, media_id, golden_spans):
                    continue

                # add candidate
                evt["duration_ms"] = dur
                person_cands.setdefault(pid, []).append(evt)

    return person_cands


def select_gallery_clips(
    person_cands: Dict[str, List[Dict]],
    max_per_media: int = 2,
    max_total: int = 12,
) -> Dict[str, List[Dict]]:
    """
    For each person, cap to max_per_media with descending duration,
    then take up to max_total clips overall.
    Returns per-person list of selected event dicts.
    """
    selected: Dict[str, List[Dict]] = {}
    for pid, cands in person_cands.items():
        # group by media_id
        by_media: Dict[str, List[Dict]] = {}
        for evt in cands:
            mid = evt["media_id"]
            by_media.setdefault(mid, []).append(evt)

        per_media_selected = []
        for mid, evts in by_media.items():
            evts.sort(key=lambda e: e["duration_ms"], reverse=True)
            per_media_selected.extend(evts[:max_per_media])

        # sort overall by duration and take top max_total
        per_media_selected.sort(key=lambda e: e["duration_ms"], reverse=True)
        clipped = per_media_selected[:max_total]
        if len(clipped) < 3:
            logger.warning(f"Person {pid}: only {len(clipped)} gallery clips (first 10).")

        selected[pid] = clipped
    return selected


def pad_window(start: float, end: float, min_dur: float = 1.5) -> Tuple[float, float]:
    """Symmetrically pad to at least min_dur seconds, clamp start at 0."""
    dur = end - start
    if dur >= min_dur:
        return start, dur
    extra = (min_dur - dur) / 2
    new_start = max(0.0, start - extra)
    new_duration = end + extra - new_start
    return new_start, new_duration


# ----------------------------------------------------------------------
# Embedding backends
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def load_wespeaker_embedder():
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole", device=torch.device("cuda"))
    def embed(wav_path: Path) -> np.ndarray:
        emb = inference(str(wav_path))
        vec = (emb if isinstance(emb, np.ndarray) else np.asarray(emb.data)).flatten()
        return normalize(vec.astype(np.float32))
    return embed, model, inference


def load_redimnet_b6_embedder():
    model = torch.hub.load(
        "IDRnD/ReDimNet",
        "ReDimNet",
        model_name="b6",
        train_type="ft_lm",
        dataset="vox2",
        trust_repo=True,
    )
    model = model.cuda().eval()

    def embed(wav_path: Path) -> np.ndarray:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        x = waveform.to(torch.float32).cuda()  # [1, T] — model expects (batch, samples)
        with torch.no_grad():
            emb = model(x)
        vec = emb.float().cpu().numpy().reshape(-1)
        return normalize(vec)

    return embed, model


def load_redimnet2_b6_embedder():
    # NOTE: dataset="vox2" was the torch.hub default but is NOT redimnet2's best
    # b6 checkpoint. Verified against PalabraAI/redimnet2 README model-zoo table
    # and GitHub release v1.0.0 assets (2026-07-08): b6 is released in THREE
    # dataset variants — vox2 (Vox1-O EER 0.29%), vb2+vox2_v0 (EER 0.23%, best),
    # and vb2+vox2+cnc2_v0 (EER 0.26%). train_type="lm" is correct (v2 has no
    # "ft_lm" tier — 'ptn'/'lm'/'dis' are the only train_types, 'lm' is the
    # recommended/best of those). Using dataset="vox2" understated v2's true
    # performance in the original bake-off; use the flagship vb2+vox2_v0 mix.
    model = torch.hub.load(
        "PalabraAI/redimnet2",
        "redimnet2",
        model_name="b6",
        train_type="lm",
        dataset="vb2+vox2_v0",
        trust_repo=True,
    )
    model = model.cuda().eval()

    def embed(wav_path: Path) -> np.ndarray:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        x = waveform.to(torch.float32).cuda()  # [1, T] — model expects (batch, samples)
        with torch.no_grad():
            emb = model(x)
        vec = emb.float().cpu().numpy().reshape(-1)
        return normalize(vec)

    return embed, model


def load_ecapa2_embedder():
    """Jenthe/ECAPA2 (HuggingFace, CC-BY-NC-4.0, non-commercial-OK), TorchScript.

    fp32 only — fp16 breaks (cuFFT rejects the non-power-of-two STFT window
    size used internally by ECAPA2's front-end).
    """
    if not ECAPA2_WEIGHTS.exists():
        raise FileNotFoundError(
            f"ECAPA2 weights not found at {ECAPA2_WEIGHTS}. Download via: "
            "huggingface_hub.hf_hub_download(repo_id='Jenthe/ECAPA2', "
            "filename='ecapa2.pt', local_dir='/workspace/erf/weights/ecapa2')"
        )
    model = torch.jit.load(str(ECAPA2_WEIGHTS), map_location="cuda").cuda()

    def embed(wav_path: Path) -> np.ndarray:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        x = waveform.to(torch.float32).cuda()  # [1, T]
        with torch.no_grad():
            emb = model(x)
        vec = emb.float().cpu().numpy().reshape(-1)
        return normalize(vec)

    return embed, model


def embed_all_wavs(
    tags: List[str],
    embed_fn,
    cache_path: Path,
    refresh: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Embed every tag, using and updating a .npz cache.
    Returns dict tag -> normalized embedding array.
    """
    if cache_path.exists() and not refresh:
        cache = dict(np.load(cache_path, allow_pickle=False))
        logger.info(f"Loaded cache with {len(cache)} entries from {cache_path}")
    else:
        cache = {}
        if refresh and cache_path.exists():
            cache_path.unlink()

    to_embed = set(tags) - set(cache.keys())
    if not to_embed:
        logger.info("All embeddings already cached.")
        return cache

    logger.info(f"Embedding {len(to_embed)} new wavs")
    for tag in sorted(to_embed):
        wav_path = WORK_WAVS / f"{tag}.wav"
        if not wav_path.exists():
            logger.warning(f"Wav missing for tag {tag}, skipping embedding")
            continue
        emb = embed_fn(wav_path)
        cache[tag] = emb

    # save cache (overwrite is safe)
    np.savez_compressed(cache_path, **cache)
    logger.info(f"Updated cache at {cache_path} with {len(cache)} embeddings")
    return cache


# ----------------------------------------------------------------------
def evaluate_backend(
    backend_name: str,
    named_tags: List[str],
    anon_tags: List[str],
    tag_to_person: Dict[str, Optional[str]],  # named -> person_id, anon -> None
    gallery_person_tags: Dict[str, List[str]],
    cache_path: Path,
    refresh: bool,
) -> Dict[str, Any]:
    """Full pipeline for one backend: embed, centroids, score, return metrics."""
    # 1. load backend
    logger.info(f"--- Loading backend {backend_name} ---")
    if backend_name == "wespeaker":
        embed_fn, model, inf = load_wespeaker_embedder()
    elif backend_name == "redimnet_b6":
        embed_fn, model = load_redimnet_b6_embedder()
    elif backend_name == "redimnet2_b6":
        embed_fn, model = load_redimnet2_b6_embedder()
    elif backend_name == "ecapa2":
        embed_fn, model = load_ecapa2_embedder()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # 2. embed all needed wavs (gallery + eval)
    all_tags = set()
    for tags in gallery_person_tags.values():
        all_tags.update(tags)
    all_tags.update(named_tags)
    all_tags.update(anon_tags)
    try:
        embeddings = embed_all_wavs(list(all_tags), embed_fn, cache_path, refresh)
    finally:
        # release GPU memory
        del model, embed_fn
        if backend_name == "wespeaker":
            del inf
        torch.cuda.empty_cache()

    # 3. compute centroids (L2 mean + normalize)
    centroids: Dict[str, np.ndarray] = {}
    for pid, tags in gallery_person_tags.items():
        vecs = []
        for t in tags:
            if t in embeddings:
                vecs.append(embeddings[t])
            else:
                logger.warning(f"Missing embedding for gallery tag {t}")
        if not vecs:
            logger.warning(f"No gallery embeddings for {pid}; centroid unavailable")
            continue
        centroid = np.mean(np.stack(vecs, axis=0), axis=0)
        centroids[pid] = normalize(centroid)

    # 4. named windows scoring
    named_preds = []  # list of dict with tag, truth, pred, cos
    for tag in named_tags:
        if tag not in embeddings:
            logger.warning(f"Missing embedding for named window {tag}")
            continue
        emb = embeddings[tag]
        truth = tag_to_person[tag]
        best_pid = None
        best_cos = -1.0
        for pid, cent in centroids.items():
            c = np.dot(emb, cent)
            if c > best_cos:
                best_cos = c
                best_pid = pid
        named_preds.append({
            "tag": tag,
            "truth": truth,
            "pred": best_pid if best_pid is not None else None,
            "cos": best_cos,
        })

    # 5. anon windows scoring
    anon_preds = []  # list of dict with tag, best_cos
    for tag in anon_tags:
        if tag not in embeddings:
            logger.warning(f"Missing embedding for anon window {tag}")
            continue
        emb = embeddings[tag]
        best_cos = -1.0
        for pid, cent in centroids.items():
            c = np.dot(emb, cent)
            if c > best_cos:
                best_cos = c
        anon_preds.append({
            "tag": tag,
            "best_cos": best_cos,
        })

    # 6. threshold sweep
    thetas = np.arange(0.05, 0.96, 0.01)
    best_theta = 0.0
    best_balanced = 0.0
    metrics_at_theta = {}
    for th in thetas:
        named_correct = sum(1 for p in named_preds if p["pred"] == p["truth"] and p["cos"] >= th)
        named_rate = named_correct / len(named_preds) if named_preds else 0.0
        anon_correct = sum(1 for p in anon_preds if p["best_cos"] < th)
        anon_rate = anon_correct / len(anon_preds) if anon_preds else 0.0
        bal = (named_rate + anon_rate) / 2.0
        if bal > best_balanced:
            best_balanced = bal
            best_theta = th
            metrics_at_theta = {
                "named_rate": named_rate,
                "anon_rate": anon_rate,
                "balanced": bal,
            }

    # 7. top-1 accuracy and margin
    n_correct = sum(1 for p in named_preds if p["pred"] == p["truth"])
    top1_acc = n_correct / len(named_preds) if named_preds else 0.0

    margins = []
    for p in named_preds:
        if p["truth"] not in centroids:
            continue
        truth_cos = float(np.dot(embeddings[p["tag"]], centroids[p["truth"]]))
        other_cos = max(
            (float(np.dot(embeddings[p["tag"]], cent))
             for pid, cent in centroids.items() if pid != p["truth"]),
            default=0.0,
        )
        margins.append(truth_cos - other_cos)
    mean_margin = np.mean(margins) if margins else 0.0

    result = {
        "backend": backend_name,
        "n_named": len(named_preds),
        "top1_acc": top1_acc,
        "best_theta": best_theta,
        "named_at_theta": metrics_at_theta.get("named_rate", 0.0),
        "anon_rej_at_theta": metrics_at_theta.get("anon_rate", 0.0),
        "balanced_at_theta": metrics_at_theta.get("balanced", 0.0),
        "mean_margin": mean_margin,
        "per_named": named_preds,
        "per_anon": anon_preds,
    }
    return result


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bake-off speaker embeddings")
    parser.add_argument(
        "--backends",
        default=",".join(BACKEND_CHOICES),
        type=lambda s: [x.strip() for x in s.split(",")],
        help="comma-separated backends to run",
    )
    parser.add_argument("--refresh", action="store_true", help="force recompute all embeddings")
    parser.add_argument("--min-clip-conf", type=float, default=0.9, help="minimum speaker confidence for gallery clips")
    args = parser.parse_args()

    if not GOLDEN.exists():
        logger.error(f"Golden file missing: {GOLDEN}")
        sys.exit(1)

    RUNS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    WORK_WAVS.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 1: selection ----------
    golden_entries = load_golden_entries(GOLDEN)
    logger.info(f"Loaded {len(golden_entries)} golden entries")
    named_windows, anon_windows = classify_windows(golden_entries)
    logger.info(f"Named windows: {len(named_windows)}, anon: {len(anon_windows)}")

    # Build guard spans for all golden entries
    golden_spans = build_golden_spans_by_media(golden_entries)

    # Gallery candidates
    person_cands = gather_gallery_candidates(
        ADJ_DIR, golden_spans, min_conf=args.min_clip_conf
    )
    selected_gallery = select_gallery_clips(person_cands)
    all_persons = set(selected_gallery.keys())
    # also include persons from named windows (they may not have gallery clips -> centroid missing)
    for e in named_windows:
        pid = e["expected"][0]["person_id"]
        all_persons.add(pid)

    # ---------- Extract all wavs ----------
    # Assign tags
    tag_to_meta = {}
    gallery_tags: Dict[str, List[str]] = {}  # pid -> list of tags
    for pid, clips in selected_gallery.items():
        tags = []
        for i, evt in enumerate(clips):
            tag = f"gal_{pid}_{i:03d}"
            tags.append(tag)
            tag_to_meta[tag] = {
                "type": "gallery",
                "person_id": pid,
                "start_ms": evt["start_ms"],
                "end_ms": evt["end_ms"],
                "duration_ms": evt["duration_ms"],
                "media_id": evt["media_id"],
                "tag": tag,
            }
        gallery_tags[pid] = tags

    # Named windows
    named_tags = []
    for idx, e in enumerate(named_windows):
        entry_id = e.get("golden_id", f"eval_{idx}")
        tag = f"eval_{entry_id}"
        named_tags.append(tag)
        tag_to_meta[tag] = {
            "type": "named",
            "truth": e["expected"][0]["person_id"],
            "start_sec": e["start_ms"] / 1000.0,
            "end_sec": e["end_ms"] / 1000.0,
            "media_id": e["media_id"],
            "tag": tag,
        }

    # Anon windows
    anon_tags = []
    for idx, e in enumerate(anon_windows):
        entry_id = e.get("golden_id", f"anon_{idx}")
        tag = f"eval_{entry_id}"
        anon_tags.append(tag)
        tag_to_meta[tag] = {
            "type": "anon",
            "start_sec": e["start_ms"] / 1000.0,
            "end_sec": e["end_ms"] / 1000.0,
            "media_id": e["media_id"],
            "tag": tag,
        }

    # Build tag->person mapping for named (truth) and anon (None)
    tag_to_person: Dict[str, Optional[str]] = {}
    for tag in named_tags:
        meta = tag_to_meta[tag]
        tag_to_person[tag] = meta["truth"]
    for tag in anon_tags:
        tag_to_person[tag] = None

    # Extract wavs for gallery clips
    for meta in tag_to_meta.values():
        if meta["type"] == "gallery":
            proxy = find_proxy(meta["media_id"])
            if not proxy:
                logger.warning(f"Skipping gallery clip {meta['tag']} – no proxy")
                continue
            start_sec = meta["start_ms"] / 1000
            dur_sec = meta["duration_ms"] / 1000
            extract_wav(meta["tag"], start_sec, dur_sec, proxy)

    # Extract eval windows (with padding)
    for meta in tag_to_meta.values():
        if meta["type"] in ("named", "anon"):
            proxy = find_proxy(meta["media_id"])
            if not proxy:
                logger.warning(f"Skipping eval window {meta['tag']} – no proxy")
                continue
            padded_start, padded_dur = pad_window(meta["start_sec"], meta["end_sec"])
            extract_wav(meta["tag"], padded_start, padded_dur, proxy)

    # ---------- Phase 2 & 3 for each backend ----------
    all_backend_results = {}
    for backend in args.backends:
        if backend not in BACKEND_CHOICES:
            logger.error(f"Unknown backend {backend}, skipping")
            continue
        cache_path = WORK / EMB_CACHE_PAT.format(backend=backend)
        res = evaluate_backend(
            backend_name=backend,
            named_tags=named_tags,
            anon_tags=anon_tags,
            tag_to_person=tag_to_person,
            gallery_person_tags=gallery_tags,
            cache_path=cache_path,
            refresh=args.refresh,
        )
        all_backend_results[backend] = res

    # ---------- Reporting ----------
    print("\n" + "=" * 80)
    print("BAKEOFF RESULTS")
    header = f"{'Backend':<20} {'#nam':>5} {'top1':>6} {'bestθ':>7} {'nam@θ':>7} {'ano@θ':>7} {'bal@θ':>7} {'margin':>8}"
    print(header)
    print("-" * len(header))
    for bn, r in all_backend_results.items():
        print(
            f"{bn:<20} {r['n_named']:5d} {r['top1_acc']:6.3f} "
            f"{r['best_theta']:7.2f} {r['named_at_theta']:7.3f} "
            f"{r['anon_rej_at_theta']:7.3f} {r['balanced_at_theta']:7.3f} "
            f"{r['mean_margin']:8.3f}"
        )

    # Worst‑cases: any named window misidentified by ANY backend
    print("\n--- Named windows misidentified by any backend ---")
    mis_set = set()
    for bn, r in all_backend_results.items():
        for pred in r["per_named"]:
            if pred["pred"] != pred["truth"]:
                mis_set.add(pred["tag"])

    if mis_set:
        col_names = ["golden_id", "truth"] + [f"{bn}_pred" for bn in all_backend_results] + [f"{bn}_cos" for bn in all_backend_results]
        print("\t".join(col_names))
        for tag in sorted(mis_set):
            truth = tag_to_person[tag]
            row = [tag, truth]
            for bn, r in all_backend_results.items():
                pred_entry = next((p for p in r["per_named"] if p["tag"] == tag), None)
                if pred_entry:
                    row.append(pred_entry["pred"])
                else:
                    row.append("?")
            for bn, r in all_backend_results.items():
                pred_entry = next((p for p in r["per_named"] if p["tag"] == tag), None)
                if pred_entry:
                    row.append(f"{pred_entry['cos']:.3f}")
                else:
                    row.append("NaN")
            print("\t".join(map(str, row)))
    else:
        print("(none)")

    # ---------- Save run report ----------
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"embed_bakeoff_{utc_now}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "backends": args.backends,
            "min_clip_conf": args.min_clip_conf,
            "refresh": args.refresh,
        },
        "gallery_clips": {
            pid: [tag_to_meta[t] for t in tags]
            for pid, tags in gallery_tags.items()
        },
        "windows": {
            "named": [tag_to_meta[t] for t in named_tags],
            "anon": [tag_to_meta[t] for t in anon_tags],
        },
        "per_backend": all_backend_results,
    }
    tmp_path = run_dir / ".tmp_report.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    os.replace(tmp_path, run_dir / "report.json")
    print(f"\nReport saved to {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()

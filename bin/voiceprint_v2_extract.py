#!/usr/bin/env python3
"""Extract speaker-verification embeddings per (media_id, speaker_label),
filtered to clean speech via Silero VAD, for the identity resolver (Phase 2/3).

Backend goes through the central adapter (src/erf/embeddings.py) — default
"redimnet2_b6" (192-dim, D-Identity-04 adopted 2026-07-08, supersedes
"redimnet_b6" v1 D-Identity-03); "wespeaker" (256-dim) kept for bake-off
comparison only. Writes a DERIVED artifact only:
    catalog/voiceprint_v2/embeddings/{backend}.jsonl   (one row per media,speaker)
    catalog/voiceprint_v2/embeddings/{backend}_manifest.json

Run in venvs/diarize.  Rows are appended+flushed incrementally so scoring can start
on partial output while the run continues.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys, json, time, glob, argparse, subprocess, datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from erf.embeddings import get_embedder  # noqa: E402

from erf.paths import catalog_root  # noqa: E402

ASR = catalog_root() / "asr"  # honors ERF_CATALOG_ROOT (D-Log-35 seam)
PROXY = ROOT / "media" / "proxies"
OUTDIR = catalog_root() / "voiceprint_v2" / "embeddings"
SR = 16000
MIN_CLEAN_INTERVAL_S = 0.5
MIN_SPEAKER_CLEAN_S = 1.0
MIN_CHUNK_S = 0.8
MAX_CHUNK_S = 4.0
MAX_CHUNKS = 12


def log(*a):
    print(*a, flush=True)


def l2(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def load_audio_16k_mono(path: Path):
    """Decode to 16 kHz mono float32 via ffmpeg (robust for mp4)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def intersect(a_ivs, b_ivs):
    """Intersection of two lists of [start,end] sample intervals."""
    out = []
    i = j = 0
    a_ivs, b_ivs = merge_intervals(a_ivs), merge_intervals(b_ivs)
    while i < len(a_ivs) and j < len(b_ivs):
        s = max(a_ivs[i][0], b_ivs[j][0])
        e = min(a_ivs[i][1], b_ivs[j][1])
        if e > s:
            out.append([s, e])
        if a_ivs[i][1] < b_ivs[j][1]:
            i += 1
        else:
            j += 1
    return out


def chunks_from_intervals(clean):
    """Split clean intervals into >=MIN_CHUNK_S, <=MAX_CHUNK_S windows."""
    out = []
    mn, mx = int(MIN_CHUNK_S * SR), int(MAX_CHUNK_S * SR)
    for s, e in clean:
        pos = s
        while e - pos >= mn:
            w = min(mx, e - pos)
            out.append((pos, pos + w))
            pos += w
    # keep the longest MAX_CHUNKS
    out.sort(key=lambda c: c[1] - c[0], reverse=True)
    return out[:MAX_CHUNKS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N media (0=all)")
    ap.add_argument("--media-file", default="", help="newline-delimited media_ids to restrict to")
    ap.add_argument("--backend", default="redimnet2_b6",
                     choices=["redimnet2_b6", "redimnet_b6", "wespeaker"])
    ap.add_argument("--out", default="", help="default: catalog/voiceprint_v2/embeddings/{backend}.jsonl")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUTDIR / f"{args.backend}.jsonl"

    media_ids = sorted(p.name[:-len(".diar.jsonl")] for p in ASR.glob("*.diar.jsonl"))
    if args.media_file:
        keep = {x.strip() for x in Path(args.media_file).read_text().splitlines() if x.strip()}
        media_ids = [m for m in media_ids if m in keep]
    if args.limit:
        media_ids = media_ids[: args.limit]
    log(f"[init] {len(media_ids)} media to process")

    t0 = time.time()
    from silero_vad import load_silero_vad, get_speech_timestamps
    embedder = get_embedder(args.backend)
    vad = load_silero_vad()
    log(f"[init] models loaded in {time.time()-t0:.1f}s")

    started = datetime.datetime.utcnow().isoformat()
    n_media_ok = n_media_skip = n_spk_ok = 0
    skipped, errors = [], []
    fout = open(out_path, "w")  # fresh; append+flush per row

    for idx, mid in enumerate(media_ids):
        if idx % 25 == 0:
            log(f"[{idx}/{len(media_ids)}] {time.time()-t0:.0f}s  spk_ok={n_spk_ok}")
        try:
            apath = PROXY / f"{mid}.mp4"
            if not apath.exists():
                n_media_skip += 1; errors.append({"media_id": mid, "error": "no_proxy"}); continue
            audio = load_audio_16k_mono(apath)
            if audio is None or audio.size < SR:
                n_media_skip += 1; errors.append({"media_id": mid, "error": "audio_decode"}); continue

            sp = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=SR)
            vad_ivs = [[d["start"], d["end"]] for d in sp]

            turns_by_spk = {}
            for line in (ASR / f"{mid}.diar.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                turns_by_spk.setdefault(t["speaker_label"], []).append(
                    [int(t["start_ms"] * SR / 1000), int(t["end_ms"] * SR / 1000)]
                )

            for spk, turns in turns_by_spk.items():
                clean = intersect(turns, vad_ivs)
                clean = [iv for iv in clean if (iv[1] - iv[0]) >= MIN_CLEAN_INTERVAL_S * SR]
                clean_s = sum(iv[1] - iv[0] for iv in clean) / SR
                if clean_s < MIN_SPEAKER_CLEAN_S:
                    skipped.append({"media_id": mid, "speaker_label": spk, "reason": "insufficient_clean_speech"}); continue
                ch = chunks_from_intervals(clean)
                if not ch:
                    skipped.append({"media_id": mid, "speaker_label": spk, "reason": "no_valid_chunks"}); continue
                vecs = [embedder.embed(audio[s:e].astype(np.float32)) for s, e in ch]
                mean = l2(np.mean(vecs, axis=0))
                row = {
                    "media_id": mid, "speaker_label": spk, "n_chunks": len(vecs),
                    "clean_dur_s": round(clean_s, 3),
                    "mean": [round(float(x), 6) for x in mean],
                    "chunks": [[round(float(x), 6) for x in v] for v in vecs],
                    # D-Log-31: per-chunk time range (ms), parallel to "chunks" —
                    # needed to map a window's score back to the words it covers
                    # (stratum-D per-window scoring). Additive; existing consumers
                    # that only read "mean"/"chunks" are unaffected.
                    "chunk_windows": [[round(s / SR * 1000), round(e / SR * 1000)] for s, e in ch],
                }
                fout.write(json.dumps(row) + "\n"); fout.flush()
                n_spk_ok += 1
            n_media_ok += 1
        except Exception as exc:
            n_media_skip += 1; errors.append({"media_id": mid, "error": repr(exc)})

    fout.close()
    manifest = {
        "model": embedder.name, "started": started, "finished": datetime.datetime.utcnow().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "n_media_ok": n_media_ok, "n_media_skipped": n_media_skip,
        "n_speakers": n_spk_ok, "n_skipped_speakers": len(skipped),
        "skipped": skipped, "errors": errors,
    }
    tmp = Path(str(out_path).replace(".jsonl", "_manifest.json")) .with_suffix(".tmp")
    final = Path(str(out_path).replace(".jsonl", "_manifest.json"))
    tmp.write_text(json.dumps(manifest, indent=2)); os.replace(tmp, final)
    log(f"[done] media_ok={n_media_ok} skip={n_media_skip} spk_ok={n_spk_ok} skipped_spk={len(skipped)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

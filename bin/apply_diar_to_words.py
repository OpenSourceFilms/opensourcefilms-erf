#!/usr/bin/env python3
"""Engine-agnostic diarization relabeler for the shadow bake-off (D-Log-35 §7b).

Reads a diar-hypothesis turns file (any engine) from the SHADOW catalog plus
the PRODUCTION canonical words/segments, applies the exact production
post-processing (erf.asr._dedup_speakers → _merge_turns → _assign_speaker with
state/config.toml [asr.diarization] params — the same path bin/rediarize.py
takes), and writes the four canonical diar surfaces into the SHADOW catalog:
asr/{mid}.words.jsonl, .segments.jsonl, .diar.jsonl, .diar_embeddings.json.

Per-label embeddings are extracted with the production WeSpeaker model
(pyannote/wespeaker-voxceleb-resnet34-LM, gallery_clean's space) from the turn
audio — the engine under test need not emit embeddings itself.

REFUSES to run if the target catalog root is the production catalog.
Bake-off tool — stdout logging only, no outcome-spine writes.

Requires: venvs/diarize.

Usage:
  ERF_CATALOG_ROOT=<shadow>/catalog venvs/diarize/bin/python \
      bin/apply_diar_to_words.py m_abc123 [...] --turns-source diarizen
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
from erf.asr import _merge_turns, _assign_speaker, _dedup_speakers
from erf.atomicio import write_tmp_rename

CAT = catalog_root()  # SHADOW target via ERF_CATALOG_ROOT (D-Log-35 seam)
PROXY_DIR = ROOT / "media" / "proxies"
CFG_PATH = ROOT / "state" / "config.toml"
MIN_TURN_S = 0.4  # embedding crop floor — matches reextract_diar_embeddings.py


def _load_cfg() -> dict:
    cfg = {"diar_merge_gap_ms": 500, "diar_min_turn_ms": 200,
           "diar_fallback_ms": 3000, "diar_dedup_threshold": 0.80}
    if CFG_PATH.is_file():
        with CFG_PATH.open("rb") as fh:
            diar = tomllib.load(fh).get("asr", {}).get("diarization", {})
        for k, ck in [("merge_gap_ms", "diar_merge_gap_ms"),
                      ("min_turn_ms", "diar_min_turn_ms"),
                      ("fallback_ms", "diar_fallback_ms")]:
            if k in diar:
                cfg[ck] = int(diar[k])
        if "dedup_threshold" in diar:
            cfg["diar_dedup_threshold"] = float(diar["dedup_threshold"])
    return cfg


def _extract_wav(mid: str, dst: Path) -> bool:
    src = PROXY_DIR / f"{mid}.mp4"
    if not src.is_file():
        print(f"WARN {mid}: proxy missing at {src}")
        return False
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True)
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def _label_embeddings(inf, Segment, wav_path: Path, turns: list[dict]) -> dict:
    """Duration-weighted WeSpeaker centroid per label from turn crops >= MIN_TURN_S
    (pattern: bin/reextract_diar_embeddings.py). Labels with no valid crop are
    simply absent — _dedup_speakers only considers labels it has embeddings for."""
    per: dict[str, list] = {}
    for t in turns:
        a, b = t["start_ms"] / 1000.0, t["end_ms"] / 1000.0
        if b - a < MIN_TURN_S:
            continue
        try:
            v = np.asarray(inf.crop(str(wav_path), Segment(a, b))).reshape(-1)
        except Exception as exc:
            print(f"WARN crop failed {t['speaker_label']} {t['start_ms']}-{t['end_ms']}: {exc}")
            continue
        if v.shape[0] == 256 and np.all(np.isfinite(v)) and float(v @ v) > 1e-9:
            per.setdefault(t["speaker_label"], []).append((v, b - a))
    out = {}
    for lbl, items in per.items():
        vecs = np.stack([v for v, _ in items])
        ws = [w for _, w in items]
        out[lbl] = np.average(vecs, axis=0, weights=ws).astype(float).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("media_ids", nargs="+")
    ap.add_argument("--turns-source", default="diarizen",
                    help="hypothesis filename prefix under diar_hypotheses/{mid}/")
    ap.add_argument("--from-catalog", default="/workspace/erf/catalog",
                    help="production catalog root to READ words/segments from")
    args = ap.parse_args()

    prod = Path("/workspace/erf/catalog").resolve()
    if CAT.resolve() == prod or CAT.resolve() == Path(args.from_catalog).resolve():
        sys.exit("refusing to write the production catalog — set ERF_CATALOG_ROOT to a shadow root")

    cfg = _load_cfg()
    import torch
    from pyannote.audio import Model, Inference
    from pyannote.core import Segment
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inf = Inference(model, window="whole",
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    failed = 0
    for mid in args.media_ids:
        turns_path = CAT / "diar_hypotheses" / mid / f"{args.turns_source}.jsonl"
        if not turns_path.is_file():
            print(f"FAIL {mid}: turns file missing: {turns_path}")
            failed += 1
            continue
        hyp = [json.loads(l) for l in turns_path.read_text().splitlines() if l.strip()]
        if not hyp:
            print(f"FAIL {mid}: turns file empty")
            failed += 1
            continue
        raw = [{"turn_index": i, "speaker_label": t["speaker_label"],
                "start_ms": t["start_ms"], "end_ms": t["end_ms"]}
               for i, t in enumerate(hyp)]

        src_asr = Path(args.from_catalog) / "asr"
        wp, sp = src_asr / f"{mid}.words.jsonl", src_asr / f"{mid}.segments.jsonl"
        if not wp.is_file():
            print(f"FAIL {mid}: production words missing: {wp}")
            failed += 1
            continue
        words = [json.loads(l) for l in wp.read_text().splitlines() if l.strip()]
        segs = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()] if sp.is_file() else []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav = Path(f.name)
        try:
            if not _extract_wav(mid, wav):
                print(f"FAIL {mid}: audio_extract_failed")
                failed += 1
                continue
            emb0 = _label_embeddings(inf, Segment, wav, raw)
            if emb0:
                raw = _dedup_speakers(raw, emb0, cfg["diar_dedup_threshold"])
            turns = _merge_turns(raw, cfg["diar_merge_gap_ms"], cfg["diar_min_turn_ms"])
            final_emb = _label_embeddings(inf, Segment, wav, turns)
        finally:
            wav.unlink(missing_ok=True)

        for w in words:
            w["speaker_label"] = _assign_speaker(w["start_ms"], w["end_ms"], turns,
                                                 cfg["diar_fallback_ms"])
        for s in segs:
            s["speaker_label"] = _assign_speaker(s["start_ms"], s["end_ms"], turns,
                                                 cfg["diar_fallback_ms"])

        out_asr = CAT / "asr"
        out_asr.mkdir(parents=True, exist_ok=True)
        write_tmp_rename(out_asr / f"{mid}.words.jsonl",
                         ("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in words)).encode())
        if segs:
            write_tmp_rename(out_asr / f"{mid}.segments.jsonl",
                             ("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in segs)).encode())
        write_tmp_rename(out_asr / f"{mid}.diar.jsonl",
                         ("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in turns)).encode())
        write_tmp_rename(out_asr / f"{mid}.diar_embeddings.json",
                         json.dumps(final_emb, separators=(",", ":")).encode())

        n_lab = sum(1 for w in words if w.get("speaker_label"))
        print(f"OK {mid}: raw_turns={len(hyp)} final_turns={len(turns)} "
              f"final_labels={len({t['speaker_label'] for t in turns})} "
              f"emb_labels={len(final_emb)} words_labeled={n_lab} words_null={len(words) - n_lab}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

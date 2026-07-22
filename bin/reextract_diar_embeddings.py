#!/usr/bin/env python3
"""Repair degenerate (NaN/zero) per-speaker voiceprints.

asr.py takes per-speaker embeddings straight from community-1's DiarizeOutput;
for some labels that comes back NaN/zero even when the label has real diarized
speech, leaving that speaker unidentifiable. This re-extracts a WeSpeaker
embedding directly from the label's audio turns (pyannote Inference) and writes
it to a SIDECAR so asr.py stays the single writer of diar_embeddings.json.

  reads:  catalog/asr/{mid}.diar_embeddings.json  (to find degenerate slots)
          catalog/asr/{mid}.diar.jsonl            (turns per label)
          media/proxies/{mid}.mp4                  (audio)
  writes: catalog/diar_embeddings_repair/{mid}.json  {label: [256 floats]}

build_identity_clusters.py prefers the repair when the canonical is degenerate.
"""
from __future__ import annotations
import argparse, glob, json, math, os, subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam); unset = ROOT/catalog
PROX = ROOT / "media" / "proxies"
MIN_TURN_S = 0.4
MIN_LABEL_S = 0.5


def _finite_nonzero(vec) -> bool:
    if not (isinstance(vec, list) and len(vec) == 256):
        return False
    if any((not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x)) for x in vec):
        return False
    return sum(x * x for x in vec) >= 1e-12


def _degenerate_repairable() -> dict[str, list[str]]:
    """media_id -> [labels] that are degenerate AND have >= MIN_LABEL_S diar speech."""
    out: dict[str, list[str]] = {}
    for f in glob.glob(str(CAT / "asr" / "*.diar_embeddings.json")):
        mid = os.path.basename(f).split(".")[0]
        emb = json.load(open(f))
        df = CAT / "asr" / f"{mid}.diar.jsonl"
        if not df.exists():
            continue
        turns = [json.loads(l) for l in df.read_text().splitlines() if l.strip()]
        for lbl, vec in emb.items():
            if _finite_nonzero(vec):
                continue
            spk_ms = sum(t["end_ms"] - t["start_ms"] for t in turns if t.get("speaker_label") == lbl)
            if spk_ms >= MIN_LABEL_S * 1000:
                out.setdefault(mid, []).append(lbl)
    return out


def _extract_wav(mid: str, dst: Path) -> bool:
    src = PROX / f"{mid}.mp4"
    if not src.exists():
        return False
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True)
    return r.returncode == 0 and dst.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from pyannote.audio import Model, Inference
    from pyannote.core import Segment

    todo = _degenerate_repairable()
    mids = sorted(todo)
    if args.limit:
        mids = mids[:args.limit]
    print(f"repairable media: {len(mids)}  slots: {sum(len(todo[m]) for m in mids)}")

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf = Inference(model, window="whole", device=dev)
    out_dir = CAT / "diar_embeddings_repair"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        for mid in mids:
            if not _extract_wav(mid, wav):
                print(f"  {mid}: no proxy/wav -> skip"); fail += 1; continue
            turns = [json.loads(l) for l in (CAT / "asr" / f"{mid}.diar.jsonl").read_text().splitlines() if l.strip()]
            repaired: dict[str, list[float]] = {}
            for lbl in todo[mid]:
                vecs, ws = [], []
                for t in [x for x in turns if x.get("speaker_label") == lbl]:
                    a, b = t["start_ms"] / 1000.0, t["end_ms"] / 1000.0
                    if b - a < MIN_TURN_S:
                        continue
                    try:
                        v = np.asarray(inf.crop(str(wav), Segment(a, b))).reshape(-1)
                    except Exception:
                        continue
                    if v.shape[0] == 256 and np.all(np.isfinite(v)) and float(v @ v) > 1e-9:
                        vecs.append(v); ws.append(b - a)
                if vecs:
                    cen = np.average(np.stack(vecs), axis=0, weights=ws)
                    repaired[lbl] = cen.astype(float).tolist()
            if repaired:
                p = out_dir / f"{mid}.json"
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({"media_id": mid, "embeddings": repaired},
                                          ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, p)
                print(f"  {mid}: repaired {list(repaired)}"); ok += 1
            else:
                print(f"  {mid}: no valid embedding extracted"); fail += 1
    print(f"DONE repaired={ok} failed={fail}")


if __name__ == "__main__":
    main()

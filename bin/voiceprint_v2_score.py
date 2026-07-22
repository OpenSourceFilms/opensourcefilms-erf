#!/usr/bin/env python3
"""Bake-off: leave-one-MEDIA-out precision/recall of speaker->person resolution.
A=current diar embedding (catalog/asr/{id}.diar_embeddings.json); C=WeSpeaker
clean-chunk mean (catalog/voiceprint_v2/embeddings/wespeaker.jsonl). Metric =
recall at >=95% accepted precision (wrong person-colour is the failure mode)."""
import json, glob, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
def l2(v):
    v = np.asarray(v, np.float32); n = np.linalg.norm(v); return v/n if n > 1e-9 else v
truth = {}
for f in glob.glob(str(ROOT / "catalog/identity_overlay/m_*.json")):
    o = json.load(open(f)); mid = o.get("media_id")
    for a in o.get("speaker_assignments", []):
        if a.get("method") in ("bootstrap", "manual") and a.get("person_id"):
            truth[(mid, a["speaker_label"])] = a["person_id"]
A = {}
for f in glob.glob(str(ROOT / "catalog/asr/m_*.diar_embeddings.json")):
    mid = Path(f).name[:-len(".diar_embeddings.json")]
    try:
        for spk, v in json.load(open(f)).items():
            vv = l2(v)
            if np.isfinite(vv).all() and vv.any():
                A[(mid, spk)] = vv
    except Exception:
        pass
C = {}
wp = ROOT / "catalog/voiceprint_v2/embeddings/wespeaker.jsonl"
if wp.exists():
    for line in wp.read_text().splitlines():
        if line.strip():
            r = json.loads(line); C[(r["media_id"], r["speaker_label"])] = l2(r["mean"])
def lomo(emb):
    keys = [k for k in truth if k in emb]; rows = []
    for (mid, spk) in keys:
        cen = collections.defaultdict(list)
        for (m2, s2) in keys:
            if m2 != mid:
                cen[truth[(m2, s2)]].append(emb[(m2, s2)])
        if len(cen) < 2:
            continue
        pids = list(cen); G = np.stack([l2(np.mean(cen[p], 0)) for p in pids])
        sc = G @ emb[(mid, spk)]; order = np.argsort(sc)[::-1]
        rows.append((truth[(mid, spk)], pids[order[0]], float(sc[order[0]]), float(sc[order[0]] - sc[order[1]])))
    return rows
def report(name, rows):
    print(f"\n=== {name}  (evaluable LOMO speakers: {len(rows)}) ===")
    if not rows:
        print("  no data yet"); return
    print("  T     M    accepted precision recall")
    best = (0.0, None)
    for T in (0.30, 0.40, 0.50, 0.60, 0.70):
        for M in (0.05, 0.10, 0.15):
            acc = [r for r in rows if r[2] >= T and r[3] >= M]
            if not acc:
                continue
            prec = sum(r[0] == r[1] for r in acc) / len(acc); rec = len(acc) / len(rows)
            print(f"  {T:.2f}  {M:.2f}  {len(acc):5d}    {prec:5.1%}   {rec:5.1%}{'  <-95%' if prec>=0.95 else ''}")
            if prec >= 0.95 and rec > best[0]:
                best = (rec, (T, M, len(acc)))
    print(f"  >> max recall @>=95% precision: {best[0]:.1%} at {best[1]}")
report("A: current diar embedding", lomo(A))
report("C: WeSpeaker clean-chunk", lomo(C))

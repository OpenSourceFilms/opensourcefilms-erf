#!/usr/bin/env python3
"""Forensic diagnostic for gate-empty golden windows in ASR bake-off arms (D-Log-41).

Per probe-media golden window x arm, counts transcript words overlapping the
window at increasing dilations, both raw and assertion-governed (replicating
eval_attribution.py's overlap + governance semantics). Splits gate 'empty'
windows into: label-tied mismatch (raw words present, governance filter rejects),
near-boundary timing skew (governed words recover under small dilation), and
true absence. Read-only; touches no catalog or state surface.
"""
import json
from pathlib import Path

ERF = Path(__file__).resolve().parent.parent
ARMS = ["control", "canary", "parakeet", "voxtral"]
PROBE = {"m_5a8bedaa054a", "m_e1b299626700", "m_0e36ebd54774"}
DIL = [0, 250, 500, 1000, 2000]

def governs(w, a):
    if "stream" in a and w.get("stream", "A") != a["stream"]:
        return False
    if "speaker_label" in a and w.get("speaker_label") != a["speaker_label"]:
        return False
    return True

golden = [e for e in map(json.loads, filter(str.strip, open(ERF / "state/eval/attribution_golden.jsonl")))
          if e.get("verified") and e["media_id"] in PROBE]
print(f"{len(golden)} verified golden windows on probe media")

for arm in ARMS:
    tx = {}
    for mid in PROBE:
        p = ERF / f"state/bakeoff/asr_shadow/{arm}/catalog/asr/{mid}.transcript.jsonl"
        tx[mid] = [json.loads(l) for l in filter(str.strip, open(p))] if p.exists() else []
    cats = {"label_mismatch": [], "timing_near": [], "true_absence": []}
    for g in golden:
        mid, s, e = g["media_id"], g["start_ms"], g["end_ms"]
        A = g["expected"]
        words = tx[mid]
        w0 = [w for w in words if w["start_ms"] < e and w["end_ms"] > s]
        if any(any(governs(w, a) for a in A) for w in w0):
            continue  # window not gate-empty for this arm
        gov1k = any(any(governs(w, a) for a in A)
                    for w in words if w["start_ms"] < e + 1000 and w["end_ms"] > s - 1000)
        tag = f'{g["golden_id"]}/{g["stratum"]}/{mid[:8]} raw@0={len(w0)}'
        if gov1k:
            cats["timing_near"].append(tag)
        elif w0:
            cats["label_mismatch"].append(tag)
        else:
            # no raw overlap at d0; may still be label-tied at larger dilation —
            # bucket edges are approximate, per-window rows are the ground truth
            cats["true_absence"].append(tag)
    print(f"\n{arm}: empty@0={sum(len(v) for v in cats.values())} "
          + " ".join(f"{k}={len(v)}" for k, v in cats.items()))
    for k, v in cats.items():
        for tag in v:
            print(f"    {k:14s} {tag}")

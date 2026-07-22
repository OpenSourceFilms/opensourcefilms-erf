#!/usr/bin/env python3
"""Propose within-media speaker-label MERGES (over-segmentation -> colour instability).
CANDIDATE-ONLY: writes catalog/voiceprint_v2/merge_candidates.jsonl; never rewrites diarization.
Signal = high WeSpeaker voice sim + labels do NOT speak concurrently + no person conflict."""
import os, json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "catalog"
COS_MIN = 0.62           # conservative: well above the ~0.39 different-speaker mean
MAX_TIME_OVERLAP = 0.10  # more overlap than this => concurrent => different people


def l2(v):
    v = np.asarray(v, np.float64); n = np.linalg.norm(v); return v / n if n > 1e-9 else v


def turns(mid):
    d = defaultdict(list); p = CAT / "asr" / f"{mid}.diar.jsonl"
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                t = json.loads(ln); d[t["speaker_label"]].append((t["start_ms"], t["end_ms"]))
    return d


def overlap_frac(a, b):
    tot = sum(e - s for s, e in a) or 1
    return sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b) / tot


def person_of(sa, label):
    for x in sa:
        if x.get("speaker_label") == label and x.get("person_id"):
            return x["person_id"]
    return None


def main():
    bymedia = defaultdict(dict)
    for ln in (CAT / "voiceprint_v2/embeddings/wespeaker.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln); bymedia[r["media_id"]][r["speaker_label"]] = l2(r["mean"])
    out = []
    for mid, labels in bymedia.items():
        if len(labels) < 2:
            continue
        tn = turns(mid)
        ovp = CAT / "identity_overlay" / f"{mid}.json"
        sa = json.loads(ovp.read_text()).get("speaker_assignments", []) if ovp.exists() else []
        ls = sorted(labels)
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                a, b = ls[i], ls[j]
                cos = float(labels[a] @ labels[b])
                if cos < COS_MIN:
                    continue
                tov = max(overlap_frac(tn.get(a, []), tn.get(b, [])), overlap_frac(tn.get(b, []), tn.get(a, [])))
                if tov > MAX_TIME_OVERLAP:
                    continue
                pa, pb = person_of(sa, a), person_of(sa, b)
                if pa and pb and pa != pb:
                    continue
                out.append({"media_id": mid, "label_a": a, "label_b": b, "voice_cosine": round(cos, 4),
                            "time_overlap_frac": round(tov, 4), "person_a": pa, "person_b": pb,
                            "reason": "high_voice_sim_no_time_overlap", "decision": "merge_candidate", "confirmed": False})
    out.sort(key=lambda r: r["voice_cosine"], reverse=True)
    dest = CAT / "voiceprint_v2" / "merge_candidates.jsonl"
    tmp = dest.with_suffix(".jsonl.tmp"); tmp.write_text("".join(json.dumps(r) + "\n" for r in out)); os.replace(tmp, dest)
    print(f"[overseg] {len(out)} merge candidates across {len(set(r['media_id'] for r in out))} media -> {dest}")
    for r in out[:8]:
        print(f"  {r['media_id']} {r['label_a']}+{r['label_b']} cos={r['voice_cosine']} tov={r['time_overlap_frac']} p={r['person_a']}/{r['person_b']}")


if __name__ == "__main__":
    main()

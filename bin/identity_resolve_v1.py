#!/usr/bin/env python3
"""Voice-first identity RESOLVER (non-destructive candidate layer).
For every (media_id, speaker_label) ReDimNet2-b6 embedding (D-Identity-04,
adopted 2026-07-08, supersedes D-Identity-03's ReDimNet-b6 v1), score vs the
clean gallery centroids. Human gold pairs lock (CONFIRMED) or abstain
(REVIEWED_UNKNOWN). Non-gold pairs: best>=THR -> promote (STRONG if
voice-proposed person also on screen in that media's face_assignments, else
CANDIDATE if margin thin); else STABLE_UNKNOWN.
CONFLICT = voice resolves a gold pair to a person different from its confirmed label.
Writes ONLY catalog/voiceprint_v2/{resolver_candidates.jsonl,resolver_summary.json}.
Touches NOTHING in identity_overlay. Apply-to-production is a separate confirmed step."""
import os, json, glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "catalog"
GALLERY_DIR = ROOT / "state" / "identity" / "gallery_redimnet2_b6"
EMBEDDINGS = CAT / "voiceprint_v2/embeddings/redimnet2_b6.jsonl"
# bestθ=0.52 from the corrected redimnet2_b6 embed bake-off (2026-07-08,
# state/eval/runs/embed_bakeoff_20260708T025020Z/report.json, balanced_at_theta
# 0.884). Carried over as the starting point the same way v1's 0.42 was;
# sanity-checked post-resolve-run (Step 4) against promotion/precision shape
# on the confirmed/gold subset -- adjust here if that looks obviously wrong.
THR = 0.52
MARGIN_STRONG = 0.06  # best-vs-second gap above which voice alone is decisive


def l2(v):
    v = np.asarray(v, np.float64); n = np.linalg.norm(v); return v / n if n > 1e-9 else v


# --- gallery centroids ---
GAL = {}
for f in sorted(glob.glob(str(GALLERY_DIR / "*.json"))):
    if Path(f).name == "manifest.json":
        continue
    d = json.load(open(f)); GAL[d["person_id"]] = l2(d["centroid"])
GP = sorted(GAL)

# --- human gold: lock + reviewed-unknown ---
CONFIRMED = {}        # (mid,spk) -> person_id
REVIEWED_UNK = set()  # (mid,spk) user looked and said unknown/mixed
def key(m, s): return (m, s)
for r in (json.loads(l) for l in open(CAT / "identity/gold_labels.jsonl") if l.strip()):
    k = key(r["media_id"], r["speaker"])
    if r.get("person_id"): CONFIRMED[k] = r["person_id"]
    else: REVIEWED_UNK.add(k)
for r in (json.loads(l) for l in open(CAT / "identity/gold_labels_face_bridged.jsonl") if l.strip()):
    k = key(r["media_id"], r["speaker"]); c = r.get("confirmed_person_id")
    if c: CONFIRMED[k] = c
    else: REVIEWED_UNK.add(k)
for r in (json.loads(l) for l in open(CAT / "identity/supplemental_reviewed.jsonl") if l.strip()):
    k = key(r["media_id"], r["speaker"])
    if r.get("person_id"): CONFIRMED[k] = r["person_id"]
    else: REVIEWED_UNK.add(k)

# --- face presence per media (who is on screen) ---
FACE = defaultdict(set)
for f in glob.glob(str(CAT / "identity_overlay/*.json")):
    d = json.load(open(f))
    for a in d.get("face_assignments", []):
        if a.get("person_id"): FACE[d["media_id"]].add(a["person_id"])

# --- merge candidates (within-media over-seg) ---
MERGE = [json.loads(l) for l in open(CAT / "voiceprint_v2/merge_candidates.jsonl") if l.strip()] \
    if (CAT / "voiceprint_v2/merge_candidates.jsonl").exists() else []

# --- score every speaker embedding ---
rows = [json.loads(l) for l in open(EMBEDDINGS) if l.strip()]
out = []
for r in rows:
    mid, spk = r["media_id"], r["speaker_label"]; k = key(mid, spk)
    q = l2(r["mean"])
    sims = sorted(((float(q @ GAL[p]), p) for p in GP), reverse=True)
    best_cos, nearest = sims[0]; second_cos = sims[1][0]
    margin = round(best_cos - second_cos, 4)
    av_present = nearest in FACE.get(mid, set())
    base = {"media_id": mid, "speaker_label": spk, "nearest_person": nearest,
            "best_cos": round(best_cos, 4), "second_cos": round(second_cos, 4),
            "margin": margin, "av_present": av_present, "clean_dur_s": r.get("clean_dur_s")}
    if k in CONFIRMED:
        pid = CONFIRMED[k]
        # validation: does voice agree with the confirmed human label?
        if best_cos >= THR and nearest != pid:
            base.update(status="conflict", person_id=pid, voice_says=nearest)
        else:
            base.update(status="confirmed", person_id=pid)
    elif k in REVIEWED_UNK:
        base.update(status="reviewed_unknown", person_id=None)
    elif best_cos >= THR:
        if av_present or margin >= MARGIN_STRONG:
            base.update(status="promote_strong", person_id=nearest)
        else:
            base.update(status="promote_candidate", person_id=nearest)
    else:
        base.update(status="stable_unknown", person_id=None)
    out.append(base)

# --- write candidate layer (atomic) ---
dest = CAT / "voiceprint_v2/resolver_candidates.jsonl"
tmp = dest.with_suffix(".jsonl.tmp")
tmp.write_text("".join(json.dumps(r) + "\n" for r in out)); os.replace(tmp, dest)

by_status = Counter(r["status"] for r in out)
promo = [r for r in out if r["status"].startswith("promote")]
promo_by_person = Counter(r["person_id"] for r in promo)
conflicts = [r for r in out if r["status"] == "conflict"]
summary = {"total_speakers": len(out), "by_status": dict(by_status),
           "promotions_by_person": dict(promo_by_person),
           "n_conflicts": len(conflicts), "n_merge_candidates": len(MERGE),
           "threshold": THR, "margin_strong": MARGIN_STRONG}
sdest = CAT / "voiceprint_v2/resolver_summary.json"
stmp = sdest.with_suffix(".json.tmp")
stmp.write_text(json.dumps(summary, indent=2)); os.replace(stmp, sdest)

# --- human-readable report ---
print(f"=== identity_resolve_v1 :: {len(out)} (media,speaker) embeddings ===")
for s in ["confirmed", "promote_strong", "promote_candidate", "stable_unknown", "reviewed_unknown", "conflict"]:
    print(f"  {s:18s} {by_status.get(s,0)}")
print(f"\n  NEW promotions by person (non-gold, voice-resolved >= {THR}):")
for p, n in promo_by_person.most_common():
    strong = sum(1 for r in promo if r["person_id"] == p and r["status"] == "promote_strong")
    print(f"    {p:20s} {n:3d}  ({strong} strong / {n-strong} candidate)")
if conflicts:
    print(f"\n  CONFLICTS (voice disagrees with confirmed gold):")
    for r in conflicts:
        print(f"    {r['media_id']} {r['speaker_label']}: gold={r['person_id']} voice={r['voice_says']} cos={r['best_cos']}")
print(f"\n  merge candidates (within-media over-seg): {len(MERGE)}")
for m in MERGE:
    print(f"    {m['media_id']} {m['label_a']}+{m['label_b']} cos={m['voice_cosine']}")
print(f"\n  -> {dest}\n  -> {sdest}")

#!/usr/bin/env python3
"""Apply resolver_candidates.jsonl into identity_overlay speaker_assignments (canonical write).
RESOLVED -> overwrite person_id (gold or voice_resolver_v1). ABSTAIN (stable/reviewed unknown)
-> CLEAR any stale bootstrap person_id to None. Upserts missing speaker_label entries.
Atomic temp-replace per overlay. Reports every action. Run AFTER snapshotting overlays."""
import os, json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]; CAT = ROOT / "catalog"
rows = [json.loads(l) for l in open(CAT / "voiceprint_v2/resolver_candidates.jsonl") if l.strip()]

# --- exact-key dedup guard: (media_id, speaker_label) must be unique. If the
# candidate layer ever double-emits a slot (re-run without truncation, etc.)
# keep only the highest-confidence row instead of silently letting Python
# dict-overwrite order decide. ---
by_key = {}
dup_dropped = 0
for r in rows:
    k = (r["media_id"], r["speaker_label"])
    prev = by_key.get(k)
    if prev is None:
        by_key[k] = r
    else:
        dup_dropped += 1
        if r.get("best_cos", -1) > prev.get("best_cos", -1):
            by_key[k] = r
rows = list(by_key.values())

# --- merge-pair conflict guard: merge_candidates.jsonl marks speaker_labels
# that are the SAME physical clip/person over-segmented by diarization. If
# the resolver independently promotes/confirms the two labels to DIFFERENT
# person_ids, applying both would split one person across two identities in
# identity_overlay. Downgrade both to abstain instead of writing a conflict. ---
MERGE = [json.loads(l) for l in open(CAT / "voiceprint_v2/merge_candidates.jsonl") if l.strip()] \
    if (CAT / "voiceprint_v2/merge_candidates.jsonl").exists() else []
same_clip_conflicts = []
row_by_key = {(r["media_id"], r["speaker_label"]): r for r in rows}
for m in MERGE:
    ka, kb = (m["media_id"], m["label_a"]), (m["media_id"], m["label_b"])
    ra, rb = row_by_key.get(ka), row_by_key.get(kb)
    if ra is None or rb is None:
        continue
    pa, pb = ra.get("person_id"), rb.get("person_id")
    if pa and pb and pa != pb and ra["status"] != "confirmed" and rb["status"] != "confirmed":
        same_clip_conflicts.append((ka, pa, kb, pb, m.get("voice_cosine")))
        for r in (ra, rb):
            r["status"] = "same_clip_conflict"
            r["person_id"] = None

bymedia = defaultdict(list)
for r in rows:
    bymedia[r["media_id"]].append(r)

actions = Counter(); cleared_names = Counter(); set_names = Counter()
for mid, rs in bymedia.items():
    ov = CAT / "identity_overlay" / f"{mid}.json"
    if not ov.exists():
        actions["overlay_missing"] += 1; continue
    d = json.loads(ov.read_text())
    sa = d.get("speaker_assignments", [])
    by_label = {a.get("speaker_label"): a for a in sa}
    changed = False
    for r in rs:
        spk, st, pid = r["speaker_label"], r["status"], r.get("person_id")
        a = by_label.get(spk)
        if a is None:
            a = {"speaker_label": spk}; sa.append(a); by_label[spk] = a
        old = a.get("person_id")
        if st in ("confirmed", "conflict"):
            a.update(person_id=pid, method="human_gold", confidence=1.0,
                     evidence=["gold_label"])
            if st == "conflict":
                a["resolver_note"] = f"voice_nearest={r.get('voice_says')}@{r['best_cos']}"
            if old != pid: set_names[pid] += 1; changed = True
        elif st in ("promote_strong", "promote_candidate"):
            a.update(person_id=pid, method="voice_resolver_v1", confidence=r["best_cos"],
                     evidence=[f"voice_cos={r['best_cos']}", f"margin={r['margin']}",
                               f"av_present={r['av_present']}", f"tier={st}"])
            if old != pid: set_names[pid] += 1; changed = True
        else:  # stable_unknown / reviewed_unknown -> clear stale name
            if old:
                a.update(person_id=None, method="resolver_abstain", confidence=None,
                         evidence=[f"abstain={st}", f"prev={old}", f"best_cos={r['best_cos']}"])
                cleared_names[old] += 1; changed = True
        actions[st] += 1
    if changed:
        d["speaker_assignments"] = sa
        tmp = ov.with_suffix(".json.tmp"); tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, ov); actions["overlays_written"] += 1

print("=== identity_apply_resolver_v1 ===")
if dup_dropped:
    print(f"  DUP GUARD: dropped {dup_dropped} duplicate (media,speaker) candidate row(s), kept highest best_cos")
if same_clip_conflicts:
    print(f"  SAME-CLIP GUARD: {len(same_clip_conflicts)} merge-pair(s) resolved to conflicting persons -> abstained both")
    for ka, pa, kb, pb, vcos in same_clip_conflicts:
        print(f"    {ka[0]} {ka[1]}={pa} vs {kb[1]}={pb}  (merge voice_cosine={vcos})")
for k, v in actions.most_common():
    print(f"  {k:18s} {v}")
print("\n  names SET (person_id written), by person:")
for p, n in set_names.most_common(): print(f"    {p:20s} {n}")
print("\n  stale bootstrap names CLEARED (now abstained), by person:")
for p, n in cleared_names.most_common(): print(f"    {p:20s} {n}")

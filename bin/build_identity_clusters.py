#!/usr/bin/env python3
"""Phase 2 (the missing brain): cross-corpus voice identity clustering.

Every (media_id, speaker_label) slot has a 256-d WeSpeaker centroid in
catalog/asr/{mid}.diar_embeddings.json. This clusters ALL slots across the
whole corpus into voice identities, then anchors each cluster to the known-
person gallery. Keyed on embeddings, NOT diarization labels -> immune to the
label-permutation problem and consistent across videos.

Inputs:
  catalog/asr/{mid}.diar_embeddings.json        {speaker_label: [256 floats]}
  catalog/voiceprint_v2/gallery_clean/*.json    {person_id, centroid, n_clips}
Outputs (when not --report-only):
  catalog/identity_clusters/clusters.json
  catalog/identity_map/{mid}.json
      {speaker_label: {person_id, cluster_id, source, gallery_score, n_media}}

Phase 3 on-spine (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §1.5/§4): this
was the tool that drove D-Identity-04 (invisible to write AND read — no
journal/outcomes call anywhere in the script). It now records an explicit
erf.outcomes.record() row on every exit path: one "__corpus__"-scoped CLUSTERS
row per invocation (always carries the resolution-rate histogram; the
gallery_share/no_embedding_share health predicates are additionally evaluated
only on a full write, status="ok" — --report-only's row is status="skip",
health=None, see `_record_corpus`), plus one per-media CLUSTERS row for every
identity_map/{mid}.json file actually written this run (none in
--report-only mode, since it writes no identity_map files). erf.outcomes is
a hard import (no `if outcomes is not None` escape) — an unloggable run fails
loudly rather than staying silently off-spine.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam); unset = ROOT/catalog
# Phase 3 on-spine (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §1.5/§4): this
# was THE tool that drove D-Identity-04 — "invisible to write AND read". P4:
# erf.outcomes is a HARD dependency here (no `if outcomes is not None` escape,
# unlike the pre-Phase-3 guarded call sites elsewhere) — an unloggable CLUSTERS
# run must fail loudly, not silently stay off-spine.
from erf import outcomes
from erf import stages as _stages_registry

GALLERY_THR = 0.45   # cosine sim to anchor a cluster to a gallery person
MIN_SIMUL_MS = 800   # same-person labels speaking simultaneously above this = over-merge


def _diar_turns(mid: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    p = CAT / "asr" / f"{mid}.diar.jsonl"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        lab = t.get("speaker_label")
        if lab and t.get("start_ms") is not None:
            out.setdefault(lab, []).append((int(t["start_ms"]), int(t["end_ms"])))
    return out


def _overlap_ms(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    tot = 0
    for s1, e1 in a:
        for s2, e2 in b:
            tot += max(0, min(e1, e2) - max(s1, s2))
    return tot


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def _load_slots():
    """Return (good_slots, degenerate_slots).

    good       = (media_id, label, finite non-zero vec256)
    degenerate = (media_id, label, reason) for NaN/inf/zero/shape-bad embeddings
                 (these get a no_embedding fallback in the identity_map).
    """
    good, degenerate = [], []
    for f in sorted(glob.glob(str(CAT / "asr" / "*.diar_embeddings.json"))):
        mid = os.path.basename(f).split(".")[0]
        d = json.load(open(f))
        # Repair sidecar (bin/reextract_diar_embeddings.py) — preferred when the
        # canonical embedding is degenerate; keeps asr.py the single writer of the
        # canonical file while still recovering identifiable voices.
        rep = {}
        rp = CAT / "diar_embeddings_repair" / f"{mid}.json"
        if rp.exists():
            try:
                rep = json.load(open(rp)).get("embeddings", {})
            except Exception:
                rep = {}
        for label, vec in d.items():
            v = np.asarray(vec, dtype=np.float64) if (isinstance(vec, list) and len(vec) == 256) else None
            if v is None or not np.all(np.isfinite(v)) or float(v @ v) < 1e-12:
                rv = rep.get(label)
                if isinstance(rv, list) and len(rv) == 256:
                    rvn = np.asarray(rv, dtype=np.float64)
                    if np.all(np.isfinite(rvn)) and float(rvn @ rvn) >= 1e-12:
                        good.append((mid, label, rvn)); continue
                reason = "shape" if v is None else ("nonfinite" if not np.all(np.isfinite(v)) else "zero")
                degenerate.append((mid, label, reason)); continue
            good.append((mid, label, v))
    return good, degenerate


def _load_gallery():
    gal = []  # (person_id, vec256)
    for f in sorted(glob.glob(str(CAT / "voiceprint_v2" / "gallery_clean" / "*.json"))):
        d = json.load(open(f))
        c = d.get("centroid")
        if isinstance(c, list) and len(c) == 256:
            gal.append((d.get("person_id", os.path.basename(f).split(".")[0]),
                        np.asarray(c, dtype=np.float64)))
    return gal


def _cluster(X: np.ndarray, dist_thr: float) -> np.ndarray:
    if len(X) == 1:
        return np.array([0])
    ac = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dist_thr,
        metric="cosine", linkage="average")
    return ac.fit_predict(X)


def _stable_cluster_id(members) -> str:
    key = "|".join(sorted(f"{m}:{l}" for m, l in members))
    return "vc_" + hashlib.sha1(key.encode()).hexdigest()[:8]


def _source_histogram(media_map: dict) -> dict[str, int]:
    """Count identity_map slots by `source` (gallery/cluster_unknown/
    no_embedding/demoted_overmerge) across every media in `media_map`."""
    hist: dict[str, int] = {}
    for mp in media_map.values():
        for rec in mp.values():
            hist[rec["source"]] = hist.get(rec["source"], 0) + 1
    return hist


def _corpus_shares(hist: dict[str, int]) -> tuple[int, float, float]:
    """(total_slots, gallery_share, no_embedding_share) from a source histogram."""
    total = sum(hist.values())
    if not total:
        return 0, 0.0, 0.0
    gallery_share = round(hist.get("gallery", 0) / total, 4)
    no_embedding_share = round(hist.get("no_embedding", 0) / total, 4)
    return total, gallery_share, no_embedding_share


def _record_corpus(status: str, hist: dict[str, int], *, reason: str | None,
                    n_media: int, n_clusters: int = 0) -> None:
    """CLUSTERS on-spine (Phase 3): the one corpus-level outcome row. Carries
    the resolution-rate histogram + (for status="ok") the two corpus health
    predicates (gallery_share/no_embedding_share) — evaluate_health() is the
    single home for their floor logic (src/erf/stages.py)."""
    total, gallery_share, no_embedding_share = _corpus_shares(hist)
    metrics = {
        "total_slots": total, "n_media": n_media, "n_clusters": n_clusters,
        "by_source": hist, "gallery_share": gallery_share,
        "no_embedding_share": no_embedding_share,
    }
    health = _stages_registry.evaluate_health("CLUSTERS", {
        "gallery_share": gallery_share, "no_embedding_share": no_embedding_share,
    }) if status == "ok" else None
    artifacts = ["catalog/identity_clusters/clusters.json"] if status == "ok" else []
    outcomes.record("__corpus__", "CLUSTERS", status, reason=reason,
                    metrics=metrics, health=health, artifacts=artifacts,
                    producer="build_identity_clusters.py")


def _record_media(media_map: dict) -> None:
    """CLUSTERS on-spine (Phase 3): one per-media outcome row for every media
    whose identity_map file this run wrote — metrics = slot counts by source."""
    for mid, mp in media_map.items():
        by_source: dict[str, int] = {}
        for rec in mp.values():
            by_source[rec["source"]] = by_source.get(rec["source"], 0) + 1
        outcomes.record(mid, "CLUSTERS", "ok",
                        metrics={"n_slots": len(mp), "by_source": by_source},
                        artifacts=[f"catalog/identity_map/{mid}.json"],
                        producer="build_identity_clusters.py")


def build(dist_thr: float, report_only: bool, slots=None, degenerate=None, gallery=None):
    if slots is None or degenerate is None:
        slots, degenerate = _load_slots()
    if gallery is None:
        gallery = _load_gallery()
    if not slots:
        print("no slots")
        # Every exit path records (Phase 3 P4): whole-corpus degenerate state —
        # no diar_embeddings anywhere. Vacuous histogram, no health verdict
        # (nothing to floor-compare against a corpus of zero slots).
        _record_corpus("empty", {}, reason="no_slots", n_media=0)
        return
    X = _l2(np.stack([v for _, _, v in slots]))
    labels = _cluster(X, dist_thr)

    G = _l2(np.stack([v for _, v in gallery])) if gallery else None
    gnames = [p for p, _ in gallery]

    clusters = {}
    for idx, cl in enumerate(labels):
        clusters.setdefault(int(cl), []).append(idx)

    out_clusters = []
    media_map: dict[str, dict] = {}
    for cl, idxs in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        members = [(slots[i][0], slots[i][1]) for i in idxs]
        cen = _l2(X[idxs].mean(axis=0, keepdims=True))[0]
        person_id, source, gscore = None, "cluster_unknown", 0.0
        if G is not None:
            sims = G @ cen
            j = int(np.argmax(sims))
            if float(sims[j]) >= GALLERY_THR:
                person_id, source, gscore = gnames[j], "gallery", float(sims[j])
        cid = _stable_cluster_id(members)
        if person_id is None:
            person_id = "p_unk_" + cid[3:]
        n_media = len({m for m, _ in members})
        rec = {
            "cluster_id": cid, "person_id": person_id, "source": source,
            "gallery_score": round(gscore, 4),
            "n_members": len(members), "n_media": n_media,
            "members": [{"media_id": m, "speaker_label": l} for m, l in members],
        }
        out_clusters.append(rec)
        for m, l in members:
            media_map.setdefault(m, {})[l] = {
                "person_id": person_id, "cluster_id": cid, "source": source,
                "gallery_score": round(gscore, 4), "n_media": n_media,
            }

    # Degenerate slots (NaN/zero embedding): no voice identity -> per-media fallback.
    for m, l, reason in degenerate:
        media_map.setdefault(m, {})[l] = {
            "person_id": None, "cluster_id": None, "source": "no_embedding",
            "gallery_score": 0.0, "n_media": 0, "reason": reason,
        }

    # ---- physical-exclusivity post-pass (audit 2026-07-03) ----
    # One person cannot speak twice at the same instant. If two labels in one
    # media resolve to the SAME identity and their diar turns overlap by more
    # than MIN_SIMUL_MS, that identity is over-merged: keep the strongest
    # claimant, demote the rest to a stable per-slot unknown (anonymous stable
    # colour beats a wrong name). Sequential multi-label = legitimate diar
    # over-segmentation and is left alone.
    slot_vec = {(m, l): _l2(v[None, :])[0] for m, l, v in slots}
    gal_by_pid = {pid: _l2(np.asarray(vec, dtype=np.float64)[None, :])[0]
                  for pid, vec in gallery}
    n_demoted = 0
    for mid, mp in media_map.items():
        by_pid: dict[str, list[str]] = {}
        for lab, rec in mp.items():
            pid = rec.get("person_id")
            if pid:
                by_pid.setdefault(pid, []).append(lab)
        multi = {pid: labs for pid, labs in by_pid.items() if len(labs) > 1}
        if not multi:
            continue
        turns = _diar_turns(mid)

        def _dur(lab: str) -> int:
            return sum(e - s for s, e in turns.get(lab, []))

        for pid, labs in multi.items():
            g = gal_by_pid.get(pid)

            def _rank(lab: str):
                v = slot_vec.get((mid, lab))
                sim = float(v @ g) if (v is not None and g is not None) else None
                # anchored: own similarity to the person's gallery centroid;
                # unknown clusters: most speech wins
                return (sim if sim is not None else -1.0, _dur(lab))

            labs_sorted = sorted(labs, key=_rank, reverse=True)
            keeper = labs_sorted[0]
            for lab in labs_sorted[1:]:
                ov = _overlap_ms(turns.get(keeper, []), turns.get(lab, []))
                if ov < MIN_SIMUL_MS:
                    continue
                rec = mp[lab]
                own = _rank(lab)[0]
                unk = "p_unk_" + hashlib.sha1(
                    f"{mid}:{lab}".encode()).hexdigest()[:8]
                rec.update({
                    "person_id": unk,
                    # sever the keeper's cluster/score: this slot is by
                    # definition NOT the keeper's voice (physical exclusivity)
                    "cluster_id": unk,
                    "gallery_score": 0.0,
                    "source": "demoted_overmerge",
                    "demoted_from": pid,
                    "demote_simultaneous_ms": int(ov),
                    "demote_own_gallery_sim": round(own, 4),
                    "keeper_label": keeper,
                    "keeper_own_gallery_sim": round(_rank(keeper)[0], 4),
                })
                n_demoted += 1
    print(f"exclusivity post-pass: demoted {n_demoted} over-merged slots")

    # ---- report ----
    n_singleton = sum(1 for c in out_clusters if c["n_members"] == 1)
    n_multimedia = sum(1 for c in out_clusters if c["n_media"] >= 2)
    n_anchored = sum(1 for c in out_clusters if c["source"] == "gallery")
    print(f"\n=== dist_thr={dist_thr} (cosine sim >= {round(1-dist_thr,2)} to merge) ===")
    print(f"good_slots={len(slots)} degenerate={len(degenerate)} clusters={len(out_clusters)} "
          f"singletons={n_singleton} multi_media={n_multimedia} anchored={n_anchored}")
    # anchored summary (one line per gallery person)
    by_person: dict[str, list] = {}
    for c in out_clusters:
        if c["source"] == "gallery":
            by_person.setdefault(c["person_id"], []).append(c)
    for pid, cs in sorted(by_person.items()):
        tot_media = sum(c["n_media"] for cs2 in [cs] for c in cs2)
        print(f"  [GAL] {pid:18s} clusters={len(cs)} "
              f"slots={sum(c['n_members'] for c in cs)} media={tot_media} "
              f"score~{max(c['gallery_score'] for c in cs):.3f}")
    # recurring unknowns (multi-media, not anchored) = randos worth a stable colour
    rec_unk = [c for c in out_clusters if c["source"] != "gallery" and c["n_media"] >= 2]
    print(f"  recurring-unknown clusters (>=2 media): {len(rec_unk)}")
    for c in sorted(rec_unk, key=lambda c: -c["n_media"])[:8]:
        print(f"    {c['person_id']} media={c['n_media']} slots={c['n_members']}")

    if report_only:
        # Diagnostic sweep/report — no identity_map file is written this call,
        # so no per-media rows (would misrepresent artifacts that don't exist
        # on disk); still surfaces the corpus histogram+health for visibility.
        _record_corpus("skip", _source_histogram(media_map), reason="report_only",
                       n_media=len(media_map), n_clusters=len(out_clusters))
        return

    (CAT / "identity_clusters").mkdir(parents=True, exist_ok=True)
    cp = CAT / "identity_clusters" / "clusters.json"
    tmp = cp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "dist_thr": dist_thr, "gallery_thr": GALLERY_THR,
        "n_slots": len(slots), "n_degenerate": len(degenerate),
        "n_clusters": len(out_clusters),
        "clusters": out_clusters,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, cp)

    (CAT / "identity_map").mkdir(parents=True, exist_ok=True)
    for mid, mp in media_map.items():
        p = CAT / "identity_map" / f"{mid}.json"
        t = p.with_suffix(".json.tmp")
        t.write_text(json.dumps({"media_id": mid, "speakers": mp},
                                ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(t, p)
    print(f"\nWROTE clusters.json + {len(media_map)} identity_map files")

    # ---- CLUSTERS on-spine (Phase 3, D-Log-04 §4): explicit outcomes for the
    # write path — per-media rows for every identity_map file just written,
    # plus the one corpus row carrying the resolution-rate health verdict. ----
    _record_media(media_map)
    _record_corpus("ok", _source_histogram(media_map), reason=None,
                   n_media=len(media_map), n_clusters=len(out_clusters))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="cosine DISTANCE threshold (1 - cos_sim) for merge")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="report at several thresholds in one process")
    a = ap.parse_args()
    if a.sweep:
        slots, degenerate = _load_slots()
        gallery = _load_gallery()
        for t in (0.40, 0.50, 0.55, 0.60):
            build(t, True, slots=slots, degenerate=degenerate, gallery=gallery)
        return
    build(a.threshold, a.report_only)


if __name__ == "__main__":
    main()

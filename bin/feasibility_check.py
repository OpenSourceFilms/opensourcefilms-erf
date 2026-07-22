#!/usr/bin/env python3
"""
§3 Per-beat retrieval + §5.1 Feasibility gate across all story proposals.

For each beat: embed semantic_query → cosine search over event_embeddings →
apply signal boosts → check against relevance floor (0.40).

Also checks visual_basis against visual_grounding data (V0), and surfaces
availability_status / production_delta for beats that require unavailable production.
"""
from __future__ import annotations
import sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from erf.scan import load_all_embeddings
from erf.query import embed_query

CATALOG        = Path("/workspace/erf/catalog")
STATE          = Path("/workspace/erf/state")
PROPOSALS_DIR  = CATALOG / "story_proposals"

_UNAVAILABLE_STATUSES = {"core_vfx_missing", "core_shot_missing", "reshoot_needed"}

RELEVANCE_FLOOR  = 0.40   # 0.30 too permissive — pure-noise matches sneak through
BOOST_EMOTIONAL  = 0.05
BOOST_SIGNAL     = 0.04   # applied per want_hook / want_quote when score >= 0.7
BOOST_SPEAKER    = 0.03

# Keywords that indicate a beat is asking for a production artifact, not footage.
# These beats may still have a corpus-grounded topic (score can be high) but the
# delivery format is editorial — added in post, not retrievable from raw footage.
_PRODUCTION_KEYWORDS = {
    # on-screen text / graphics
    "title card", "end card", "end screen", "lower third",
    # channel / social production elements
    "follow button", "subscribe button", "social media handles",
    "channel logo", "channel branding", "channel end card",
    # overlay / animation effects (narrow enough to avoid false hits)
    "overlay",
    "animated overlay", "animated effect", "animated shot",
    "cartoon overlay", "cartoon effect",
    # catch bare "animated" and "cartoon" only when they precede a noun
    # (works for: "animated browser tab", "animated comedy", "cartoon naked")
    "animated ", "cartoon ",
    # audio post-production cues
    "music sting", "audio sting",
    # edit-only moments with no footage backing
    "freeze frame", "smash cut",
}


def load_visual_groundings(catalog: Path) -> dict[str, dict]:
    """Return {media_id: visual_grounding} from catalog/visual_grounding/."""
    result: dict[str, dict] = {}
    vg_dir = catalog / "visual_grounding"
    if not vg_dir.exists():
        return result
    for f in vg_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            result[d.get("media_id", f.stem)] = d
        except Exception:
            pass
    return result


def load_event_meta(catalog: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for line in (catalog / "event_digest.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        meta[ev["event_id"]] = {
            "emotional_beat":  ev.get("emotional_beat", ""),
            "hook_potential":  float(ev.get("hook_potential")  or 0.0),
            "quote_potential": float(ev.get("quote_potential") or 0.0),
            "speaker_name":    ev.get("speaker_name", "") or "",
            "summary":         ev.get("summary", ""),
            "t_s":             ev.get("t_s", 0.0),
            "media_id":        ev.get("media_id", ""),
        }
    return meta


def fused(base: float, ev_meta: dict, spec: dict) -> float:
    score = base
    eb_target = (spec.get("emotional_beat_target") or "").lower()
    if eb_target and eb_target in (ev_meta.get("emotional_beat") or "").lower():
        score += BOOST_EMOTIONAL
    if spec.get("want_hook") and ev_meta.get("hook_potential", 0.0) >= 0.7:
        score += BOOST_SIGNAL
    if spec.get("want_quote") and ev_meta.get("quote_potential", 0.0) >= 0.7:
        score += BOOST_SIGNAL
    hint = (spec.get("speaker_hint") or "").lower()
    if hint and hint in (ev_meta.get("speaker_name") or "").lower():
        score += BOOST_SPEAKER
    return score


def _beat_type(beat: dict, score: float) -> str:
    """Classify beat as 'footage' or 'production_layer'.

    production_layer = the beat requests an editorial artifact (title card,
    animation, end screen, etc.) that cannot be retrieved from raw footage.
    Two triggers:
    1. Score below floor — no corpus grounding at all.
    2. Score above floor BUT narrative_intent / semantic_query contain
       production-format keywords (topic may be grounded, format is editorial).
    """
    text = (
        (beat.get("narrative_intent") or "") + " " +
        (beat.get("retrieval_spec", {}).get("semantic_query") or "")
    ).lower()
    has_prod_keyword = any(kw in text for kw in _PRODUCTION_KEYWORDS)
    if score < RELEVANCE_FLOOR:
        return "production_layer"
    if has_prod_keyword:
        return "production_layer"
    return "footage"


ANCHOR_BOOST = 0.15   # boost when narrative_intent names a verified anchor event


def retrieve_beat(
    beat: dict,
    all_vecs: dict[str, np.ndarray],
    event_meta: dict,
    anchor_event_ids: list[str] | None = None,
    visual_groundings: dict[str, dict] | None = None,
) -> dict:
    spec   = beat.get("retrieval_spec") or {}
    query  = spec.get("semantic_query") or ""

    intent = beat.get("narrative_intent") or ""
    explicit_anchors = {eid for eid in (anchor_event_ids or []) if eid in intent}

    qvec = np.array(embed_query(query, STATE), dtype=np.float32)
    n = np.linalg.norm(qvec)
    if n > 0:
        qvec /= n

    best_id, best_base, best_fused = None, 0.0, 0.0
    for ev_id, evec in all_vecs.items():
        base  = float(np.dot(qvec, evec))
        score = fused(base, event_meta.get(ev_id, {}), spec)
        if ev_id in explicit_anchors:
            score += ANCHOR_BOOST
        if score > best_fused:
            best_id, best_base, best_fused = ev_id, base, score

    btype = _beat_type(beat, best_fused)

    # Visual basis check: verify cited event_ids have visual grounding
    visual_basis = beat.get("visual_basis") or []
    vg = visual_groundings or {}
    visual_basis_ok: list[str] = []
    visual_basis_missing: list[str] = []
    for eid in visual_basis:
        mid = event_meta.get(eid, {}).get("media_id", "")
        if mid and mid in vg:
            visual_basis_ok.append(eid)
        else:
            visual_basis_missing.append(eid)

    # Availability status from proposal (V0-grounded honesty mandate)
    avail_status = beat.get("availability_status", "unclear")
    prod_delta = beat.get("production_delta")
    has_unavailable = avail_status in _UNAVAILABLE_STATUSES

    # Override beat_type if V0 explicitly flags the beat as requiring unavailable production
    if has_unavailable and btype == "footage":
        btype = "production_layer"

    return {
        "beat_index":              beat["beat_index"],
        "role":                    beat.get("role", ""),
        "narrative_intent":        beat.get("narrative_intent", ""),
        "semantic_query":          query,
        "top_event_id":            best_id,
        "top_base_score":          round(best_base,  4),
        "top_fused_score":         round(best_fused, 4),
        "top_summary":             event_meta.get(best_id or "", {}).get("summary", ""),
        "beat_type":               btype,
        "passed":                  btype == "footage",
        "availability_status":     avail_status,
        "production_delta":        prod_delta,
        "visual_basis":            visual_basis,
        "visual_basis_grounded":   visual_basis_ok,
        "visual_basis_unresolved": visual_basis_missing,
    }


def main() -> None:
    print("Loading event embeddings …")
    all_vecs   = load_all_embeddings(CATALOG)
    print(f"  {len(all_vecs)} event vectors loaded")

    print("Loading event metadata …")
    event_meta = load_event_meta(CATALOG)
    print(f"  {len(event_meta)} events loaded")

    print("Loading visual groundings …")
    vg = load_visual_groundings(CATALOG)
    print(f"  {len(vg)} media with V0 visual grounding\n")

    proposals = sorted(PROPOSALS_DIR.glob("sto_*.json"))
    print(f"{len(proposals)} proposals found\n")

    summaries: list[dict] = []

    for ppath in proposals:
        proposal = json.loads(ppath.read_text())
        pid   = proposal["proposal_id"]
        title = proposal["title"]
        beats = proposal.get("beats", [])

        print(f"{'='*72}")
        print(f"  {pid}  |  {title}")
        print(f"  Logline: {proposal.get('logline','')[:80]}")
        print(f"  {len(beats)} beats")
        print()

        anchor_ids = proposal.get("anchor_event_ids", [])
        results = [
            retrieve_beat(b, all_vecs, event_meta, anchor_ids, visual_groundings=vg)
            for b in beats
        ]

        prod_layer = sum(1 for r in results if r["beat_type"] == "production_layer")
        unavail_beats = [r for r in results if r.get("availability_status") in _UNAVAILABLE_STATUSES]

        for r in results:
            btype = r["beat_type"]
            score = r["top_fused_score"]
            q     = r["semantic_query"][:48]
            avail = r.get("availability_status", "")
            delta = r.get("production_delta") or ""
            if btype == "production_layer":
                label = "PROD "
                print(f"    Beat {r['beat_index']:02d} [{label}] {score:.3f}  {q:<48}  ← {avail or 'editorial'}")
                if delta:
                    print(f"             ↳ {delta}")
                if score < RELEVANCE_FLOOR:
                    print(f"             no corpus grounding  top: {r['top_event_id']}  base={r['top_base_score']:.3f}")
                    print(f"             top_ev: {r['top_summary'][:80]}")
                vb = r.get("visual_basis", [])
                if vb:
                    unresolved = r.get("visual_basis_unresolved", [])
                    print(f"             visual_basis={vb}  unresolved={unresolved}")
            else:
                vb = r.get("visual_basis", [])
                print(f"    Beat {r['beat_index']:02d} [FOOT ] {score:.3f}  {q:<48}  avail={avail}")
                if vb:
                    print(f"             visual_basis={vb}")

        footage_beats = [r for r in results if r["beat_type"] == "footage"]
        pct = round(100 * len(footage_beats) / len(results)) if results else 0
        print()
        print(
            f"  RESULT: {len(footage_beats)}/{len(results)} footage  |  "
            f"{prod_layer} production_layer  |  {len(unavail_beats)} unavailable"
        )
        if unavail_beats:
            print(f"  SHELVE REASON: {len(unavail_beats)} core beat(s) require unavailable production:")
            for r in unavail_beats:
                print(f"    beat {r['beat_index']:02d} [{r['availability_status']}]: {(r.get('production_delta') or '')[:80]}")
        summaries.append({
            "pid":           pid,
            "title":         title,
            "total":         len(results),
            "footage":       len(footage_beats),
            "prod_layer":    prod_layer,
            "unavailable":   len(unavail_beats),
            "pct_footage":   pct,
        })
        print()

    print(f"{'='*72}")
    print("CROSS-PROPOSAL SUMMARY  (FOOT=footage  PROD=production_layer  UNAV=unavailable)")
    print(f"{'='*72}")
    for s in summaries:
        foot_bar = "█" * s["footage"]
        prod_bar = "▒" * s["prod_layer"]
        unav_bar = "✗" * s["unavailable"]
        print(
            f"  {s['pct_footage']:3d}% footage  [{foot_bar}{prod_bar}]  "
            f"UNAV={s['unavailable']}  {s['pid']}  {s['title'][:38]}"
        )
    print()


if __name__ == "__main__":
    main()

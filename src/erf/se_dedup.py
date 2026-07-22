"""SE:C3 per-lane candidate deduplication — 5-signal rule from patch_v3 §7.

Two candidates are duplicates iff: same primary_format_lane AND ≥2 of 5 signals.
Exception: sentence_jaccard >= 0.8 alone clusters.
"""
from __future__ import annotations


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _word_set(text: str | None) -> set[str]:
    return set((text or "").lower().split())


def _beat_clip_range(beats: list[dict], role: str) -> tuple | None:
    for b in beats:
        if b.get("role", "").upper() == role:
            return (b.get("clip_id"), tuple(b.get("sentence_range", [])))
    return None


def _top2_clips(beats: list[dict]) -> frozenset[str]:
    from collections import Counter
    counts = Counter(b.get("clip_id") for b in beats if b.get("clip_id"))
    return frozenset(c for c, _ in counts.most_common(2))


def _signals(a: dict, b: dict) -> list[bool]:
    beats_a = a.get("beats") or []
    beats_b = b.get("beats") or []
    ev_a = set(a.get("evidence_ids") or [eid for b in beats_a for eid in b.get("evidence_ids", [])])
    ev_b = set(b.get("evidence_ids") or [eid for b in beats_b for eid in b.get("evidence_ids", [])])
    clips_a = {b.get("clip_id") for b in beats_a if b.get("clip_id")}
    clips_b = {b.get("clip_id") for b in beats_b if b.get("clip_id")}
    return [
        _jaccard(ev_a, ev_b) >= 0.5,
        clips_a == clips_b and a.get("archetype") == b.get("archetype"),
        _jaccard(_word_set(a.get("central_question")), _word_set(b.get("central_question"))) >= 0.4,
        _beat_clip_range(beats_a, "PAYOFF") is not None
            and _beat_clip_range(beats_a, "PAYOFF") == _beat_clip_range(beats_b, "PAYOFF")
            and _beat_clip_range(beats_a, "HOOK") == _beat_clip_range(beats_b, "HOOK"),
        _top2_clips(beats_a) == _top2_clips(beats_b) and a.get("archetype") == b.get("archetype"),
    ]


def _are_dups(a: dict, b: dict) -> bool:
    if a.get("primary_format_lane") != b.get("primary_format_lane"):
        return False
    beats_a = a.get("beats") or []
    beats_b = b.get("beats") or []
    ev_a = set(a.get("evidence_ids") or [eid for b in beats_a for eid in b.get("evidence_ids", [])])
    ev_b = set(b.get("evidence_ids") or [eid for b in beats_b for eid in b.get("evidence_ids", [])])
    if _jaccard(ev_a, ev_b) >= 0.8:
        return True
    return sum(_signals(a, b)) >= 2


def _score(cand: dict) -> float:
    beats = cand.get("beats") or []
    roles = {b.get("role", "").upper() for b in beats}
    hook = max((b.get("hook_score", 0) for b in beats), default=0)
    quote = max((b.get("quote_score", 0) for b in beats), default=0)
    av = max((b.get("av_quality", 0) for b in beats), default=0)
    has_payoff = 1 if "PAYOFF" in roles else 0
    has_hook = 1 if "HOOK" in roles else 0
    return 0.25 * has_payoff + 0.20 * has_hook + 0.25 * hook + 0.20 * quote + 0.10 * av


def cluster_candidates(candidates: list[dict], lane: str) -> list[dict]:
    """Return one representative per cluster (highest composite score). Per-lane."""
    same_lane = [c for c in candidates if c.get("primary_format_lane") == lane]
    n = len(same_lane)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) != find(j) and _are_dups(same_lane[i], same_lane[j]):
                parent[find(i)] = find(j)

    clusters: dict[int, list[dict]] = {}
    for i, c in enumerate(same_lane):
        clusters.setdefault(find(i), []).append(c)

    return [max(members, key=_score) for members in clusters.values()]

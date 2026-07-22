from __future__ import annotations


def filter_turns(turns: list[dict], *, gap_ms: int = 1000, min_ms: int = 2000) -> tuple[list[dict], list[dict]]:
    if not turns:
        return [], []

    overlap_skipped = []
    clean = []
    for t in turns:
        if t.get("overlap", False):
            overlap_skipped.append({**t, "skip_reason": "overlap"})
        else:
            clean.append(t)

    clean.sort(key=lambda x: x["start_ms"])

    merged: list[dict] = []
    i = 0
    while i < len(clean):
        curr = dict(clean[i])
        j = i + 1
        while (
            j < len(clean)
            and clean[j]["speaker_label"] == curr["speaker_label"]
            and (clean[j]["start_ms"] - curr["end_ms"]) <= gap_ms
        ):
            curr["end_ms"] = clean[j]["end_ms"]
            j += 1
        curr["overlap"] = False
        merged.append(curr)
        i = j

    kept = []
    skipped = list(overlap_skipped)
    for t in merged:
        if (t["end_ms"] - t["start_ms"]) >= min_ms:
            kept.append(t)
        else:
            skipped.append({**t, "skip_reason": "turn_too_short"})

    return kept, skipped

#!/usr/bin/env python3
"""QC: flag identity_map over-merges (D-Identity-01 follow-up, audit 2026-07-03).

Two diar labels in one media mapping to the SAME person_id is legitimate when the
diarizer over-segmented one voice — but physically impossible if the two labels
SPEAK AT THE SAME TIME. This scans identity_map, finds same-person multi-label
media, and measures simultaneous-speech overlap between the labels' baseline diar
turns. Overlap above threshold = true over-merge candidate for manual review.

Read-only: writes nothing, changes nothing. Output is the review list.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAT = ROOT / "catalog"
MIN_OVERLAP_MS = 800  # ignore trivial boundary slop


def _turns(mid: str) -> dict[str, list[tuple[int, int]]]:
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


def main() -> None:
    flagged = 0
    scanned = 0
    for p in sorted((CAT / "identity_map").glob("m_*.json")):
        mid = p.stem
        d = json.loads(p.read_text(encoding="utf-8"))
        by_person: dict[str, list[str]] = {}
        for lab, a in (d.get("speakers") or {}).items():
            pid = a.get("person_id")
            if pid:
                by_person.setdefault(pid, []).append(lab)
        multi = {pid: labs for pid, labs in by_person.items() if len(labs) > 1}
        if not multi:
            continue
        scanned += 1
        turns = _turns(mid)
        for pid, labs in multi.items():
            for i in range(len(labs)):
                for j in range(i + 1, len(labs)):
                    ov = _overlap_ms(turns.get(labs[i], []), turns.get(labs[j], []))
                    tag = "OVER-MERGE?" if ov >= MIN_OVERLAP_MS else "ok(sequential)"
                    if ov >= MIN_OVERLAP_MS:
                        flagged += 1
                    print(f"{mid}  {pid:<18} {labs[i]}+{labs[j]}  "
                          f"simultaneous={ov / 1000:.1f}s  {tag}")
    print(f"\nmedia with same-person multi-label: {scanned}; "
          f"flagged over-merge candidates: {flagged} (threshold {MIN_OVERLAP_MS}ms)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Golden-leak stratification sidecar (additive, non-destructive).

The gallery_redimnet2_b6 centroids are built from full diarization turns of
specific (media_id, speaker_label) slots (state/identity/gallery_redimnet2_b6/
manifest.json). If a golden eval window's audio falls inside one of those same
turns for the SAME person the window asserts, the gallery already "knows" that
exact audio -- resolving it correctly is not a fair test of generalisation.

This script never touches attribution_golden.jsonl or any resolver output. It
only computes, per golden_id, whether the window overlaps gallery-training
audio for one of its own `expected` person_ids, and writes the verdict to a
sidecar file that eval_attribution.py optionally folds into its report as an
ADDITIONAL clean-vs-contaminated breakdown alongside (not instead of) the
existing full-63 stratum gate.

  reads : state/eval/attribution_golden.jsonl
          state/identity/gallery_redimnet2_b6/manifest.json
          catalog/asr/{mid}.diar.jsonl
  writes: state/eval/golden_leak_stratification.json
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "state/eval/attribution_golden.jsonl"
GALLERY_MANIFEST = ROOT / "state/identity/gallery_redimnet2_b6/manifest.json"
CAT = ROOT / "catalog"
OUT = ROOT / "state/eval/golden_leak_stratification.json"


def _load_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _diar_turns(media_id: str, speaker_label: str) -> List[Tuple[float, float]]:
    p = CAT / "asr" / f"{media_id}.diar.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if t.get("speaker_label") == speaker_label:
            out.append((float(t["start_ms"]), float(t["end_ms"])))
    return out


def _overlaps(a_start: float, a_end: float, spans: List[Tuple[float, float]]) -> bool:
    return any(max(a_start, s) < min(a_end, e) for s, e in spans)


def compute_leak_status() -> Dict[str, dict]:
    """Return {golden_id: {status, contaminating_clips, ...}} for every golden window."""
    golden = _load_jsonl(GOLDEN)
    manifest = json.loads(GALLERY_MANIFEST.read_text(encoding="utf-8"))

    # gallery training turns, keyed by (media_id, person_id) -> list of clips
    # that could leak, each with its own span(s). "gold_label" clips are keyed
    # by speaker_label and expand to the slot's FULL diarization turn set;
    # "enrollment_supplement" clips already carry an exact trimmed [start_s,
    # end_s) range (golden-leak-checked at enrollment time, GAMEPLAN 2026-07-04)
    # so no turn lookup is needed for those.
    turns_by_media_person: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    turn_cache: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for clip in manifest.get("clips", []):
        mid, pid = clip["media_id"], clip["person_id"]
        spk = clip.get("speaker_label")
        if spk is not None:
            key = (mid, spk)
            if key not in turn_cache:
                turn_cache[key] = _diar_turns(mid, spk)
            spans = turn_cache[key]
        else:
            spans = [(float(clip["start_s"]) * 1000.0, float(clip["end_s"]) * 1000.0)]
        turns_by_media_person[(mid, pid)].append({
            "speaker_label": spk or clip.get("source_clip"), "source": clip.get("source"),
            "spans": spans,
        })

    result: Dict[str, dict] = {}
    for entry in golden:
        gid = entry["golden_id"]
        mid = entry["media_id"]
        s, e = float(entry["start_ms"]), float(entry["end_ms"])
        expected_pids = [a.get("person_id") for a in entry.get("expected", []) if a.get("person_id")]

        contaminating = []
        for pid in expected_pids:
            for clip in turns_by_media_person.get((mid, pid), []):
                if _overlaps(s, e, clip["spans"]):
                    contaminating.append({
                        "person_id": pid, "speaker_label": clip["speaker_label"],
                        "source": clip["source"],
                    })

        result[gid] = {
            "media_id": mid, "stratum": entry.get("stratum"),
            "expected_person_ids": expected_pids,
            "status": "contaminated" if contaminating else "clean",
            "contaminating_clips": contaminating,
        }
    return result


def main() -> None:
    status = compute_leak_status()
    n_clean = sum(1 for v in status.values() if v["status"] == "clean")
    n_contam = sum(1 for v in status.values() if v["status"] == "contaminated")

    print(f"=== golden leak stratification :: {len(status)} windows ===")
    print(f"  clean:        {n_clean}")
    print(f"  contaminated: {n_contam}")
    by_stratum: Dict[str, Dict[str, int]] = defaultdict(lambda: {"clean": 0, "contaminated": 0})
    for v in status.values():
        by_stratum[v["stratum"] or "?"][v["status"]] += 1
    for strat in sorted(by_stratum):
        c = by_stratum[strat]
        print(f"    stratum {strat}: clean={c['clean']} contaminated={c['contaminated']}")
    print("\n  contaminated windows:")
    for gid, v in sorted(status.items()):
        if v["status"] == "contaminated":
            who = ", ".join(f"{c['person_id']}({c['source']})" for c in v["contaminating_clips"])
            print(f"    {gid:14s} {v['media_id']}  expected={v['expected_person_ids']}  leaks_from={who}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "n_windows": len(status), "n_clean": n_clean, "n_contaminated": n_contam,
        "windows": status,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()

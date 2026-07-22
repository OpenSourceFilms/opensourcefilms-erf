"""Sentence re-segmentation — split giant ASR sentences into editable units.

The punctuation model that builds `sentences` occasionally fails to split long
continuous speech, leaving 30-80s "sentences" (6.6% of corpus sentences are >8s).
Those are unusable as edit atoms: a beat can't be tighter than its sentence, so the
adapter rejects giant-sentence beats and the clip's material is lost.

Word-level timing (`catalog/asr/{mid}.transcript.jsonl`, each word carries
start_ms/end_ms/sentence_index) lets us re-split deterministically — no re-ASR. The
split is greedy-largest-gap with a minimum-duration floor: it cuts a too-long unit at
its biggest internal pause that keeps BOTH sides >= min_dur_ms, recursing toward
max_ms; a run with no such pause is left intact (better one long unit than fragments).

This is the algorithm; wiring it into the T0 sentence-builder + a one-time corpus
re-segmentation (which regenerates evidence + editorial labels) is a separate decision.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import List

CAT = Path("/workspace/erf/catalog")


def split_words(words: List[dict], max_ms: int = 8000, min_dur_ms: int = 1500) -> List[List[dict]]:
    """Split one sentence's words into units <= max_ms, never creating a unit < min_dur_ms."""
    s0, e1 = int(words[0]["start_ms"]), int(words[-1]["end_ms"])
    if e1 - s0 <= max_ms or len(words) < 2:
        return [words]
    best_i, best_gap = None, -1
    for i in range(len(words) - 1):
        left_dur = int(words[i]["end_ms"]) - s0
        right_dur = e1 - int(words[i + 1]["start_ms"])
        if left_dur >= min_dur_ms and right_dur >= min_dur_ms:
            gap = int(words[i + 1]["start_ms"]) - int(words[i]["end_ms"])
            if gap > best_gap:
                best_gap, best_i = gap, i
    if best_i is None:  # cannot split without making a fragment → leave intact
        return [words]
    return (split_words(words[:best_i + 1], max_ms, min_dur_ms)
            + split_words(words[best_i + 1:], max_ms, min_dur_ms))


def resegment_clip(media_id: str, *, max_ms: int = 8000, min_dur_ms: int = 1500,
                   catalog: Path = CAT) -> List[dict]:
    """Re-segmented sentence units for a clip, from its word-level transcript.

    Returns [{start_ms, end_ms, text, speaker_label, orig_sentence_index}], time-ordered.
    Sentences already <= max_ms pass through unchanged.
    """
    tp = catalog / "asr" / f"{media_id}.transcript.jsonl"
    words = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_sent: "OrderedDict[int, List[dict]]" = OrderedDict()
    for w in words:
        by_sent.setdefault(w["sentence_index"], []).append(w)
    out: List[dict] = []
    for sidx, ws in by_sent.items():
        ws = sorted(ws, key=lambda x: int(x["start_ms"]))
        for part in split_words(ws, max_ms, min_dur_ms):
            out.append({
                "start_ms": int(part[0]["start_ms"]),
                "end_ms": int(part[-1]["end_ms"]),
                "text": " ".join(x["text"] for x in part),
                "speaker_label": part[0].get("speaker_label"),
                "orig_sentence_index": sidx,
            })
    out.sort(key=lambda s: s["start_ms"])
    return out


if __name__ == "__main__":
    import sys
    mid = sys.argv[1]
    units = resegment_clip(mid)
    for u in units:
        print(f'[{u["start_ms"]:>7}-{u["end_ms"]:<7}] ({(u["end_ms"]-u["start_ms"])/1000:.1f}s) {u["text"][:80]}')
    print(f"# {mid}: {len(units)} units, max {max((u['end_ms']-u['start_ms'])/1000 for u in units):.1f}s")

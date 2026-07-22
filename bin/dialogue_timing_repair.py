#!/usr/bin/env python3
"""D0 — dialogue timing repair (additive, pre-render).

Selected beat windows come from P0 sentence spans, i.e. the first word's start and
the last word's end. Cutting exactly there feels "snipped" — ASR word boundaries are
imprecise and there's no breathing room. D0 extends each beat's in/out into the
natural pause on either side (pre-roll / tail), bounded so it never reaches into the
adjacent word (a different sentence or speaker). Uses existing ASR word timestamps —
no re-ASR, no media_id churn. Energy/forced-alignment refinement can layer on later.
"""
from __future__ import annotations

import json
from pathlib import Path

_words_cache: dict[str, list[dict]] = {}


def _load_words(catalog: Path, media_id: str) -> list[dict]:
    if media_id not in _words_cache:
        ws: list[dict] = []
        p = catalog / "asr" / f"{media_id}.words.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    w = json.loads(line)
                except Exception:
                    continue
                if w.get("start_ms") is not None and w.get("end_ms") is not None:
                    ws.append(w)
            ws.sort(key=lambda x: x["start_ms"])
        _words_cache[media_id] = ws
    return _words_cache[media_id]


def repair(
    catalog: Path,
    media_id: str,
    in_ms: int,
    out_ms: int,
    pre_roll_ms: int = 150,
    tail_ms: int = 320,
    guard_ms: int = 60,
) -> tuple[int, int]:
    """Return (repaired_in_ms, repaired_out_ms) with handles into surrounding pauses.

    pre_roll/tail are the desired handles; they are clipped to leave guard_ms of
    silence before/after the neighbouring word so we never clip into it.
    """
    words = _load_words(catalog, media_id)
    if not words:
        return in_ms, out_ms
    inside = [w for w in words if w["start_ms"] < out_ms and w["end_ms"] > in_ms]
    if not inside:
        return in_ms, out_ms
    fw, lw = inside[0], inside[-1]

    prev_end = max((w["end_ms"] for w in words if w["end_ms"] <= fw["start_ms"]), default=0)
    next_start = min((w["start_ms"] for w in words if w["start_ms"] >= lw["end_ms"]),
                     default=lw["end_ms"] + 100000)

    new_in = max(prev_end + guard_ms, fw["start_ms"] - pre_roll_ms)
    new_in = min(new_in, fw["start_ms"])               # only extend backward
    new_out = min(next_start - guard_ms, lw["end_ms"] + tail_ms)
    new_out = max(new_out, lw["end_ms"])               # only extend forward
    if new_out <= new_in:
        return in_ms, out_ms
    return int(new_in), int(new_out)

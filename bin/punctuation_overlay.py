#!/usr/bin/env python3
"""P0 — punctuation / sentence-boundary overlay (additive facet).

Consumes the EXISTING ASR word stream (catalog/asr/<media_id>.words.jsonl) and
emits catalog/asr/<media_id>.sentences.jsonl — sentence spans aligned to the
existing word timestamps. It NEVER re-transcribes, NEVER mutates words, and
NEVER touches media_ids, events or embeddings (same "additive facet" discipline
as V0 visual grounding).

Boundaries come from a token-classification punctuation model (FullStop via
deepmultilingualpunctuation) combined with pause-aware repair on the existing
timestamps. The model's output is hard-validated to leave the word sequence
unchanged; on any mutation the media falls back to pause-only boundaries.

Run on CPU (the punctuation model is small; the pod's GPU torch build mismatches
this venv's kernels) — CUDA is disabled before torch import.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU — must precede torch import

import argparse
import glob
import hashlib
import json
import logging
import re
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("p0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    from erf import outcomes
except Exception:
    outcomes = None

ASR_DIR = Path("catalog/asr")

_TERMINAL = (".", "?", "!", "…")
_PUNCT_CHARS = ".,?!:;…-"
_FILLER = {"um", "uh", "like", "you", "know", "i", "mean", "sort", "kind", "of", "er", "erm", "ah"}

# Boundary tuning (ms / word counts)
PAUSE_MS = 650
STRONG_PAUSE_MS = 1000
MIN_SENT_WORDS = 4
SOFT_MIN_WORDS = 6
LONG_WORDS = 35
FORCE_WORDS = 45


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load_words(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            w = json.loads(line)
        except Exception:
            continue
        if w.get("start_ms") is not None and w.get("end_ms") is not None:
            out.append(w)
    return out


def _model_punct_after(words: list[dict], model) -> list[str] | None:
    """Return a per-word trailing-punctuation list from the model, or None if the
    model's output mutated the word sequence (caller falls back to pause-only)."""
    if model is None:
        return None
    text = " ".join((w.get("text") or "").strip() for w in words)
    try:
        restored = model.restore_punctuation(text)
    except Exception as e:
        log.warning("restore_punctuation failed: %s", e)
        return None

    out_tokens = restored.split()
    punct_after = [""] * len(words)
    wi = 0
    for tok in out_tokens:
        core = tok.strip(_PUNCT_CHARS)
        trail = tok[len(tok.rstrip(_PUNCT_CHARS)):]  # trailing punctuation only
        if not _norm(core):
            # pure-punctuation token: attach to the previous word
            if wi > 0 and trail:
                punct_after[wi - 1] = trail
            continue
        if wi >= len(words):
            return None  # model produced extra words → mutation
        if _norm(core) != _norm(words[wi].get("text") or ""):
            return None  # word changed → mutation, reject
        punct_after[wi] = trail
        wi += 1
    if wi != len(words):
        return None  # model dropped words → mutation
    return punct_after


def _sentence_boundaries(words: list[dict], punct_after: list[str] | None) -> list[int]:
    """Return indices i after which a sentence boundary falls (inclusive end)."""
    n = len(words)
    pause = [0] * n
    for i in range(n - 1):
        pause[i] = max(0, int(words[i + 1]["start_ms"]) - int(words[i]["end_ms"]))

    def is_filler(i: int) -> bool:
        return _norm(words[i].get("text") or "") in _FILLER

    boundaries: list[int] = []
    sent_start = 0
    for i in range(n):
        sent_len = i - sent_start + 1
        terminal = bool(punct_after and punct_after[i] and punct_after[i][-1] in _TERMINAL)
        cut = False
        if terminal and sent_len >= MIN_SENT_WORDS:
            cut = True
        elif terminal and sent_len < MIN_SENT_WORDS:
            cut = False  # too short — let it ride unless a later real boundary
        elif pause[i] >= PAUSE_MS and sent_len >= SOFT_MIN_WORDS:
            cut = True
        elif sent_len >= LONG_WORDS:
            lo, hi = max(sent_start, i - 5), min(n - 1, i + 5)
            if pause[i] > 0 and pause[i] >= max(pause[lo:hi + 1]):
                cut = True
        if sent_len >= FORCE_WORDS:
            # force a boundary: prefer the largest pause in the run, but tie-break
            # toward the END (largest index). With contiguous ASR (all pauses ~0)
            # this cuts at the current word — a full FORCE_WORDS chunk — instead of
            # shaving one word off the front and re-triggering every iteration.
            seg = range(sent_start, i + 1)
            best = max(seg, key=lambda j: (pause[j], j))
            boundaries.append(best)
            sent_start = best + 1
            continue
        # block boundaries adjacent to filler unless the pause is strong
        if cut and pause[i] < STRONG_PAUSE_MS and not terminal:
            if (i + 1 < n and is_filler(i + 1)) or is_filler(i):
                cut = False
        if cut:
            boundaries.append(i)
            sent_start = i + 1
    if not boundaries or boundaries[-1] != n - 1:
        boundaries.append(n - 1)
    # de-dup + sort
    seen: set[int] = set()
    out = []
    for b in sorted(boundaries):
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _boundary_source(words, punct_after, end_i, start_i) -> str:
    if punct_after and punct_after[end_i] and punct_after[end_i][-1] in _TERMINAL:
        return "model_terminal"
    if (end_i - start_i + 1) >= FORCE_WORDS:
        return "maxlen"
    return "pause"


def process_media(media_id: str, model) -> dict | None:
    wpath = ASR_DIR / f"{media_id}.words.jsonl"
    if not wpath.exists():
        log.warning("%s: no words file", media_id)
        return None
    # Gate on A's explicit has_speech flag (cross-stage awareness), not file presence.
    if outcomes and outcomes.flag(media_id, "has_speech") is False:
        outcomes.record(media_id, "P0", "skip", reason="no_speech")
        outcomes.set_flag(media_id, "has_sentences", False, reason="no_speech")
        return None
    words = _load_words(wpath)
    if not words:
        if outcomes:
            outcomes.record(media_id, "P0", "skip", reason="no_words")
            outcomes.set_flag(media_id, "has_sentences", False, reason="no_words")
        log.warning("%s: 0 usable words", media_id)
        return None

    punct_after = _model_punct_after(words, model)
    mutation = "passed" if punct_after is not None else ("fallback" if model is not None else "no_model")

    boundaries = _sentence_boundaries(words, punct_after)
    sentences = []
    start = 0
    for si, end in enumerate(boundaries):
        seg = words[start:end + 1]
        if not seg:
            start = end + 1
            continue
        text = " ".join((w.get("text") or "").strip() for w in seg).strip()
        sentences.append({
            "sentence_id": f"{media_id}:sent:{si:04d}",
            "media_id": media_id,
            "start_word": start,
            "end_word": end,
            "start_ms": int(seg[0]["start_ms"]),
            "end_ms": int(seg[-1]["end_ms"]),
            "text": text,
            "boundary_source": _boundary_source(words, punct_after, end, start),
            "synthetic": punct_after is None or not (punct_after[end] and punct_after[end][-1] in _TERMINAL),
        })
        start = end + 1

    norm_join = " ".join(_norm(w.get("text") or "") for w in words)
    header = {
        "schema_version": "1.0.0",
        "media_id": media_id,
        "punctuation_model": "fullstop" if model is not None else "none",
        "word_count": len(words),
        "word_sequence_sha256": hashlib.sha256(norm_join.encode()).hexdigest(),
        "mutation_check": mutation,
        "sentence_count": len(sentences),
    }

    # atomic write
    tmp = ASR_DIR / f".{media_id}.sentences.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for s in sentences:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp, ASR_DIR / f"{media_id}.sentences.jsonl")

    lens = [s["end_word"] - s["start_word"] + 1 for s in sentences]
    med = sorted(lens)[len(lens) // 2] if lens else 0
    log.info("%s: %d words -> %d sentences (median_len=%d, mutation=%s)",
             media_id, len(words), len(sentences), med, mutation)
    if outcomes:
        outcomes.record(media_id, "P0", "ok",
                        counts={"words": len(words), "sentences": len(sentences)})
        outcomes.set_flag(media_id, "has_sentences", len(sentences) > 0)
    return {"media_id": media_id, "n_words": len(words), "n_sentences": len(sentences),
            "median_len": med, "mutation": mutation}


def main() -> None:
    ap = argparse.ArgumentParser(description="P0 punctuation/sentence-boundary overlay")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--media-id")
    g.add_argument("--all", action="store_true")
    g.add_argument("--media-list")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    try:
        from deepmultilingualpunctuation import PunctuationModel
        model = PunctuationModel()
    except Exception as e:
        log.warning("punctuation model unavailable (%s) — pause-only boundaries", e)
        model = None

    if args.media_id:
        ids = [args.media_id]
    elif args.media_list:
        ids = [l.strip() for l in Path(args.media_list).read_text().splitlines() if l.strip()]
    else:
        ids = sorted(p.name[:-len(".words.jsonl")]
                     for p in ASR_DIR.glob("*.words.jsonl"))
    if args.limit:
        ids = ids[:args.limit]

    ok = mut = 0
    for mid in ids:
        try:
            # Single-writer: process_media records every terminal outcome
            # (ok/skip) via erf.outcomes itself. Wrapping in stage_run
            # double-wrote a bare journal ok over each no_speech skip
            # (same defect class as T0, fixed 2026-07-04).
            r = process_media(mid, model)
            if r:
                ok += 1
                if r["mutation"] == "fallback":
                    mut += 1
        except Exception as e:
            if outcomes:
                try:
                    outcomes.record(mid, "P0", "error", reason=str(e)[:200])
                except Exception:
                    pass
            log.exception("%s failed", mid)
    log.info("DONE: %d media processed, %d used pause-only fallback", ok, mut)


if __name__ == "__main__":
    main()

"""Stage tags — per-chunk scoring for short-form assembly.

hook_score, quote_score, proof_score: cheap deterministic heuristics.
format_fit: duration + beat + b_roll heuristic.
No LLM calls. All scores nullable if input data is missing.
"""
from __future__ import annotations

import re
import tomllib
import uuid
from pathlib import Path

from .atomicio import jsonl_append, read_jsonl
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
}

# Hook: starts with attention-grabbing opener
_HOOK_OPENERS = re.compile(
    r"(?i)^(wait|here'?s? (the|a)|you('?re| won'?t)|did you|what if|the reason|"
    r"this is|i (never|always|can'?t|won'?t)|the truth|why (does|do|is|are)|"
    r"how (do|does|can|to)|imagine|the biggest|stop|listen|watch)"
)
_QUESTION_START = re.compile(r"(?i)^(what|why|how|when|where|who|can|could|would|should|is|are|do|does)")

# Proof: quantitative or before/after framing
_PROOF_PATTERNS = re.compile(
    r"(?i)(\d+[%x]|\d+\s*(times|percent|million|billion|thousand|k|x)|"
    r"before (and after|vs)|from \d+ to \d+|increased? by|decreased? by|"
    r"went from|doubled|tripled|results?|proven?|study|research|data shows?)"
)


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
    return cfg


def _hook_score(text: str, duration_ms: int, speech_clarity: float | None) -> float | None:
    if not text:
        return None
    text = text.strip()
    score = 0.0
    if _HOOK_OPENERS.match(text):
        score += 0.5
    elif _QUESTION_START.match(text):
        score += 0.3
    # Short is better for hooks
    dur_s = duration_ms / 1000
    if dur_s <= 8:
        score += 0.3
    elif dur_s <= 15:
        score += 0.15
    # Clear speech boosts hook
    if speech_clarity is not None and speech_clarity > 0.5:
        score += 0.2
    return round(min(score, 1.0), 3)


def _quote_score(text: str, duration_ms: int, speech_clarity: float | None, n_speakers: int) -> float | None:
    if not text:
        return None
    dur_s = duration_ms / 1000
    score = 0.0
    # Single speaker complete sentence in moderate duration
    if n_speakers <= 1:
        score += 0.3
    # Target 8-25s
    if 8 <= dur_s <= 25:
        score += 0.4
    elif 5 <= dur_s < 8 or 25 < dur_s <= 35:
        score += 0.2
    # Ends with sentence-terminal punctuation → likely complete
    stripped = text.strip()
    if stripped and stripped[-1] in ".!?":
        score += 0.15
    if speech_clarity is not None and speech_clarity > 0.4:
        score += 0.15
    return round(min(score, 1.0), 3)


def _proof_score(text: str) -> float | None:
    if not text:
        return None
    matches = len(_PROOF_PATTERNS.findall(text))
    score = min(matches * 0.35, 1.0)
    return round(score, 3)


def _format_fit_for_chunk(duration_ms: int, b_roll: bool | None, emotional_beat: str | None) -> list[str]:
    dur_s = duration_ms / 1000
    if b_roll:
        return ["tiktok", "reels", "youtube_shorts", "longform"]
    high_energy = emotional_beat in ("tension", "action", "comedic", "reveal")
    if dur_s < 15 and high_energy:
        return ["tiktok", "reels", "youtube_shorts"]
    if dur_s < 15:
        return ["tiktok", "reels", "youtube_shorts"]
    if dur_s <= 60:
        return ["youtube_shorts", "reels"]
    return ["longform"]


def run(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
) -> StageManifest:
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])

    run_id = f"tags-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "tags", run_id)

    error: str | None = None
    tag_count = 0
    outputs: list[str] = []
    skip_no_chunks = False

    try:
        # Gate on B0's explicit has_chunks flag (cross-stage awareness) — avoids
        # scoring a no-speech/no-chunks media as if chunk data existed.
        from . import outcomes as _oc
        chunks_path = catalog / "chunks" / f"{media_id}.jsonl"
        _has_chunks = _oc.flag(media_id, "has_chunks")
        chunks = [] if _has_chunks is False else read_jsonl(chunks_path)

        if not chunks:
            # Nothing to tag — explicit skip outcome, no zero-byte/stale file.
            # reconcile_status.py's verify() treats skip/empty as "no artifact
            # expected" for this stage (tags is not in explicit_empty_stages),
            # so a stale tags.jsonl left over from a prior (pre-gate) run would
            # surface forever as "unrecorded" drift — remove it.
            # The skip is recorded AFTER reap_manifest below — the reap bridge
            # writes a bare "ok" and would overwrite it here.
            skip_no_chunks = True
            tags_path = catalog / "tags" / f"{media_id}.jsonl"
            if tags_path.exists():
                tags_path.unlink()
        else:
            # Build quality index: media_id → audio metrics (one per media)
            quality_path = catalog / "quality" / f"{media_id}.jsonl"
            speech_clarity: float | None = None
            if quality_path.exists():
                for qrec in read_jsonl(quality_path):
                    if qrec.get("speech_clarity") is not None:
                        speech_clarity = float(qrec["speech_clarity"])
                        break

            # Load events for emotional_beat per scene
            events_path = catalog / "events" / f"{media_id}.jsonl"
            emotional_beat_by_chunk: dict[str, str | None] = {}
            b_roll_by_chunk: dict[str, bool | None] = {}
            if events_path.exists():
                for ev in read_jsonl(events_path):
                    beat = ev.get("emotional_beat")
                    b_roll = ev.get("b_roll_flag")
                    for cid in (ev.get("chunk_ids") or []):
                        emotional_beat_by_chunk[cid] = beat
                        b_roll_by_chunk[cid] = b_roll

            tags_path = catalog / "tags" / f"{media_id}.jsonl"
            tags_path.parent.mkdir(parents=True, exist_ok=True)
            if tags_path.exists():
                tags_path.unlink()

            # Count distinct speakers per chunk for quote scoring
            speaker_set: set[str] = set()
            for c in chunks:
                lbl = c.get("speaker_label")
                if lbl:
                    speaker_set.add(lbl)

            for chunk in chunks:
                cid = chunk["chunk_id"]
                text = chunk.get("text", "")
                dur_ms = chunk.get("end_ms", 0) - chunk.get("start_ms", 0)
                b_roll = b_roll_by_chunk.get(cid)
                beat = emotional_beat_by_chunk.get(cid)

                # Speaker count heuristic: if single speaker label or no diarization
                sp = chunk.get("speaker_label")
                n_speakers = 1 if sp else len(speaker_set) or 1

                rec = {
                    "schema_version": "1.0.0",
                    "chunk_id": cid,
                    "media_id": media_id,
                    "hook_score": _hook_score(text, dur_ms, speech_clarity),
                    "quote_score": _quote_score(text, dur_ms, speech_clarity, n_speakers),
                    "proof_score": _proof_score(text),
                    "format_fit": _format_fit_for_chunk(dur_ms, b_roll, beat),
                    "emotional_beat": beat,
                    "b_roll_flag": b_roll,
                }
                jsonl_append(tags_path, rec)
                tag_count += 1

            outputs.append(str(tags_path))

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="tags",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"tags": tag_count},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    if skip_no_chunks:
        try:
            from . import outcomes as _oc2
            # Override the bridge's "ok" with the accurate skip (runs after reap).
            _oc2.record(media_id, "tags", "skip", reason="no_chunks")
        except Exception:
            pass
    return manifest

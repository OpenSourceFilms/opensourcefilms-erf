"""Stage B0 — Chunking.

Splits ASR words+segments into retrieval chunks on speaker-turn boundaries,
then merges/splits to target windows. Runs two passes per file:
  - search pass:    12–35 s windows, chunk_rule=speaker_turn_search|duration_search
  - shortform pass:  4–15 s windows, chunk_rule=speaker_turn_short|duration_short

Both sets written to catalog/chunks/{media_id}.jsonl (same file).
Both are indexed in FTS and embedded by downstream stages.
"""
from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

from .atomicio import jsonl_append, write_json
from .ids import chunk_id
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "erf_root": "/workspace/erf",
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "search_min_s": 12,
    "search_max_s": 35,
    "shortform_min_s": 4,
    "shortform_max_s": 15,
}


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        ck = raw.get("chunking", {})
        # New keys; fall back to legacy target_min/max for search pass
        legacy_min = ck.get("target_min_s", cfg["search_min_s"])
        legacy_max = ck.get("target_max_s", cfg["search_max_s"])
        cfg["search_min_s"] = ck.get("search_min_s", legacy_min)
        cfg["search_max_s"] = ck.get("search_max_s", legacy_max)
        cfg["shortform_min_s"] = ck.get("shortform_min_s", cfg["shortform_min_s"])
        cfg["shortform_max_s"] = ck.get("shortform_max_s", cfg["shortform_max_s"])
    return cfg


def _ms(s: float) -> int:
    return int(round(s * 1000))


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _assign_speaker_to_words(words: list[dict], turns: list[dict]) -> list[dict]:
    """Assign speaker_label from diar turns to each word (best-overlap wins)."""
    for w in words:
        best_label = w.get("speaker_label")
        best_overlap = 0
        for t in turns:
            ov = _overlap_ms(w["start_ms"], w["end_ms"], t["start_ms"], t["end_ms"])
            if ov > best_overlap:
                best_overlap = ov
                best_label = t["speaker_label"]
        w["speaker_label"] = best_label
    return words


def _group_words_by_speaker_turn(words: list[dict]) -> list[list[dict]]:
    """Group consecutive words with the same speaker_label into runs."""
    if not words:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = [words[0]]
    current_speaker = words[0].get("speaker_label")
    for w in words[1:]:
        sp = w.get("speaker_label")
        if sp == current_speaker:
            current.append(w)
        else:
            groups.append(current)
            current = [w]
            current_speaker = sp
    groups.append(current)
    return groups


def _split_group(words: list[dict], min_ms: int, max_ms: int) -> list[list[dict]]:
    """Split a single-speaker word group into windows of min_ms–max_ms ms."""
    if not words:
        return []
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_start = words[0]["start_ms"]

    for w in words:
        current.append(w)
        span = w["end_ms"] - current_start
        if span >= max_ms:
            chunks.append(current)
            current = []
            current_start = w["end_ms"]

    if current:
        if chunks and (current[-1]["end_ms"] - current[0]["start_ms"]) < min_ms:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


def _merge_short_groups(
    groups: list[list[dict]], min_ms: int, max_ms: int
) -> list[list[dict]]:
    """Merge adjacent same-speaker groups when the result fits within max_ms."""
    if not groups:
        return []
    result: list[list[dict]] = [groups[0]]
    for grp in groups[1:]:
        prev = result[-1]
        span_merged = grp[-1]["end_ms"] - prev[0]["start_ms"]
        same_speaker = prev[0].get("speaker_label") == grp[0].get("speaker_label")
        if same_speaker and span_merged <= max_ms:
            result[-1] = prev + grp
        else:
            result.append(grp)
    return result


def _make_chunk_record(words: list[dict], media_id: str, chunk_rule: str) -> dict:
    start_ms = words[0]["start_ms"]
    end_ms = words[-1]["end_ms"]
    text = " ".join(w["text"].strip() for w in words if w.get("text", "").strip())
    word_ids = [w["word_id"] for w in words]
    speaker_label = words[0].get("speaker_label")
    return {
        "schema_version": "1.0.0",
        "chunk_id": chunk_id(media_id, start_ms, end_ms),
        "media_id": media_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "word_ids": word_ids,
        "speaker_label": speaker_label,
        "speaker_person_id": None,
        "chunk_rule": chunk_rule,
        "emotional_beat": None,
        "b_roll_flag": None,
    }


def _build_chunks(
    words: list[dict],
    turns: list[dict],
    min_ms: int,
    max_ms: int,
    media_id: str,
    has_diar: bool,
    shortform: bool,
) -> list[dict]:
    """Produce chunk records for one pass (search or shortform)."""
    suffix = "short" if shortform else "search"

    # Re-assign speaker labels (words already have them from the search pass,
    # but we redo it here to be safe since this function is pure).
    if turns:
        words = _assign_speaker_to_words([dict(w) for w in words], turns)

    # Group by speaker turn
    speaker_groups = _group_words_by_speaker_turn(words)

    # Determine chunk_rule
    if has_diar:
        base_rule = f"speaker_turn_{suffix}"
    else:
        base_rule = f"duration_{suffix}"

    # Merge then split
    merged = _merge_short_groups(speaker_groups, min_ms, max_ms)
    final_word_groups: list[list[dict]] = []
    for grp in merged:
        span = grp[-1]["end_ms"] - grp[0]["start_ms"] if grp else 0
        if span > max_ms:
            final_word_groups.extend(_split_group(grp, min_ms, max_ms))
        else:
            final_word_groups.append(grp)

    # Enforce 2s floor on tail only: a sub-2s tail group is an artifact (e.g. a single
    # leftover word after merging), not a natural speaker boundary. Absorb it regardless.
    _FLOOR_MS = 2000
    if len(final_word_groups) >= 2:
        last = final_word_groups[-1]
        if last and (last[-1]["end_ms"] - last[0]["start_ms"]) < _FLOOR_MS:
            final_word_groups[-2].extend(final_word_groups.pop())

    records: list[dict] = []
    for grp in final_word_groups:
        if not grp:
            continue
        records.append(_make_chunk_record(grp, media_id, base_rule))
    return records


def run(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
    source_note: str | None = None,
) -> StageManifest:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])

    run_id = f"B0-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "B0", run_id)

    error: str | None = None
    search_count = 0
    shortform_count = 0
    outputs: list[str] = []
    skip_no_speech = False

    try:
        search_min_ms = int(cfg["search_min_s"] * 1000)
        search_max_ms = int(cfg["search_max_s"] * 1000)
        short_min_ms = int(cfg["shortform_min_s"] * 1000)
        short_max_ms = int(cfg["shortform_max_s"] * 1000)

        words_path = catalog / "asr" / f"{media_id}.words.jsonl"
        if not words_path.exists():
            raise FileNotFoundError(f"Words file not found: {words_path}")

        from .atomicio import read_jsonl
        from . import outcomes
        words = [w for w in read_jsonl(words_path) if w.get("start_ms") != w.get("end_ms")]

        chunks_path = catalog / "chunks" / f"{media_id}.jsonl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        if chunks_path.exists():
            chunks_path.unlink()

        # Gate explicitly on A's declared has_speech flag (cross-stage awareness),
        # not by re-deriving emptiness from the words file. `words` is still read above
        # because it is the DATA B0 chunks — that read is legitimate I/O, not a gate.
        if outcomes.flag(media_id, "has_speech") is False or not words:
            # Explicit no-speech outcome: no chunks. has_chunks is B0's own flag
            # (has_speech is owned/declared by stage A). No zero-byte file is written.
            # The skip outcome itself is recorded AFTER reap_manifest below —
            # the reap bridge writes a bare "ok" and would overwrite it here.
            skip_no_speech = True
            outcomes.set_flag(media_id, "has_chunks", False, reason="no_speech")
        else:
            diar_path = catalog / "asr" / f"{media_id}.diar.jsonl"
            turns = read_jsonl(diar_path) if diar_path.exists() else []
            has_diar = bool(turns)

            # Search pass (12–35 s)
            search_chunks = _build_chunks(
                words, turns, search_min_ms, search_max_ms,
                media_id, has_diar, shortform=False,
            )
            for rec in search_chunks:
                jsonl_append(chunks_path, rec)
            search_count = len(search_chunks)

            # Shortform pass (4–15 s)
            short_chunks = _build_chunks(
                words, turns, short_min_ms, short_max_ms,
                media_id, has_diar, shortform=True,
            )
            for rec in short_chunks:
                jsonl_append(chunks_path, rec)
            shortform_count = len(short_chunks)

            outputs.append(str(chunks_path))

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="B0",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={
            "chunks": search_count + shortform_count,
            "search_chunks": search_count,
            "shortform_chunks": shortform_count,
        },
        error=error,
        outputs=outputs,
    )
    status_dir = catalog / "status"
    reap_manifest(journal_db, manifest, status_dir)
    if skip_no_speech:
        try:
            from . import outcomes as _oc
            # Override the bridge's "ok" with the accurate skip (runs after reap).
            _oc.record(media_id, "B0", "skip", reason="no_speech",
                       counts={"chunks": 0, "words": 0}, producer=run_id, db=journal_db)
        except Exception:
            pass
    if (search_count + shortform_count) > 0:
        try:
            from . import outcomes as _oc
            _oc.set_flag(media_id, "has_chunks", True)
        except Exception:
            pass
    return manifest

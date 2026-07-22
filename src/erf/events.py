"""Stage E0 — Events.

Phase 2 skeleton: one scene_event per scene with text rollup and format_fit.
Phase 3a extension: LLM beat-tagging pass adds emotional_beat, edit_role, summary.
Phase 3b extension: when diarization turns exist, generate one dialogue_event per
  merged speaker-turn group rather than one event per scene (scene fallback retained).
"""
from __future__ import annotations

import json
import sqlite3
import tomllib
import uuid
import warnings
from pathlib import Path

from .atomicio import jsonl_append, read_jsonl, write_tmp_rename
from .ids import event_id
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
}

_EMOTIONAL_BEAT_ENUM = {
    "setup","tension","release","comedic","emotional",
    "action","reveal","exposition","transition",
}
_EDIT_ROLE_ENUM = {
    "hook","body","b_roll","proof","quote",
    "transition","call_to_action","exposition",
}

_LLM_SYSTEM = """\
You are a documentary and short-form video editor classifying transcript excerpts for editorial assembly.

Respond ONLY with a JSON object matching this exact schema — no prose, no markdown, no extra keys:
{
  "emotional_beat": string | null,
  "edit_role": [string, ...],
  "b_roll_flag": boolean | null,
  "summary": string
}

Field rules:
- "emotional_beat": one of ["setup","tension","release","comedic","emotional","action","reveal","exposition","transition"] or null if no clear beat or no speech
- "edit_role": non-empty array; each element must be one of ["hook","body","b_roll","proof","quote","transition","call_to_action","exposition"]. If the clip has dialogue and no stronger signal, use ["body"]. Never return an empty array.
- "b_roll_flag": true if clip is visual-only / no sync dialogue (camera movement, scenery, cutaway); false if it carries essential spoken content; null if unclear
- "summary": one factual sentence of max 20 words stating what happens in the clip

Example output for "We increased revenue by 40% after switching suppliers":
{"emotional_beat":"reveal","edit_role":["proof","body"],"b_roll_flag":false,"summary":"Speaker reveals 40% revenue increase from supplier change."}

When uncertain, prefer null over guessing. Never return fields outside the schema.\
"""


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
    return cfg


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _format_fit(duration_ms: int, emotional_beat: str | None = None, b_roll: bool | None = None) -> list[str]:
    dur_s = duration_ms / 1000
    if b_roll:
        return ["youtube_shorts", "reels", "tiktok", "longform"]
    if dur_s < 15:
        return ["tiktok", "reels", "youtube_shorts"]
    if dur_s < 60:
        return ["youtube_shorts", "reels"]
    return ["longform"]


def _tag_with_llm(text: str, duration_ms: int) -> dict | None:
    """Call LLM to classify one event. Returns cleaned dict or None on failure."""
    if not text or len(text.strip()) < 10:
        return None
    try:
        from .llm import complete
        user_msg = f"Scene duration: {duration_ms // 1000}s\nTranscript: {text[:800]}"
        raw = complete(
            [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            json_only=True,
        )
        parsed = json.loads(raw)

        emotional_beat = parsed.get("emotional_beat")
        if emotional_beat not in _EMOTIONAL_BEAT_ENUM:
            emotional_beat = None

        edit_role_raw = parsed.get("edit_role")
        if isinstance(edit_role_raw, list):
            valid = [r for r in edit_role_raw if r in _EDIT_ROLE_ENUM]
            edit_role = valid if valid else ["body"]
        else:
            edit_role = None

        b_roll_flag = parsed.get("b_roll_flag")
        if not isinstance(b_roll_flag, bool):
            b_roll_flag = None

        summary = parsed.get("summary")
        if summary and not isinstance(summary, str):
            summary = None
        if summary:
            summary = summary[:200]

        return {
            "emotional_beat": emotional_beat,
            "edit_role": edit_role,
            "b_roll_flag": b_roll_flag,
            "summary": summary,
        }
    except Exception as exc:
        warnings.warn(f"LLM tagging failed: {exc}")
        return None


def _emit_tagging_stats(journal_db: Path, media_id: str, run_id: str, tagged: int, total: int) -> None:
    conn = sqlite3.connect(str(journal_db))
    try:
        conn.execute(
            # Phase 1 (D-Log-04): sub-event trace, not E0's own disposition
            # (E0's real outcome is recorded via reap_manifest) -> events_trace.
            "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
            "VALUES (?, ?, ?, 'E0', 'ok', ?, ?)",
            (
                StageManifest.now(), f"{media_id}:E0:llm", media_id, run_id,
                json.dumps({"tagged": tagged, "total": total, "null_rate": round(1 - tagged / max(total, 1), 2)}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _merge_turns(turns: list[dict], gap_ms: int = 2000) -> list[dict]:
    """Merge consecutive same-speaker turns within gap_ms into span groups.

    Returns list of {speaker_label, start_ms, end_ms} dicts.
    """
    if not turns:
        return []
    sorted_turns = sorted(turns, key=lambda t: t["start_ms"])
    groups: list[dict] = []
    cur = {
        "speaker_label": sorted_turns[0]["speaker_label"],
        "start_ms": sorted_turns[0]["start_ms"],
        "end_ms": sorted_turns[0]["end_ms"],
    }
    for t in sorted_turns[1:]:
        gap = t["start_ms"] - cur["end_ms"]
        if t["speaker_label"] == cur["speaker_label"] and gap <= gap_ms:
            cur["end_ms"] = max(cur["end_ms"], t["end_ms"])
        else:
            groups.append(cur)
            cur = {
                "speaker_label": t["speaker_label"],
                "start_ms": t["start_ms"],
                "end_ms": t["end_ms"],
            }
    groups.append(cur)
    return groups


def run(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
    llm_tag: bool = True,
) -> StageManifest:
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])

    run_id = f"E0-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "E0", run_id)

    error: str | None = None
    event_count = 0
    tagged_count = 0
    outputs: list[str] = []
    skip_no_speech = False

    try:
        scenes = read_jsonl(catalog / "scenes" / f"{media_id}.jsonl")
        chunks = read_jsonl(catalog / "chunks" / f"{media_id}.jsonl")

        # Only use search chunks for event text rollup (12-35 s semantic windows)
        search_chunks = [
            c for c in chunks
            if c.get("chunk_rule", "").endswith("_search") or "chunk_rule" not in c
        ]

        # Build quality index: scene_id → b_roll_heuristic
        quality_path = catalog / "quality" / f"{media_id}.jsonl"
        b_roll_by_scene: dict[str, bool | None] = {}
        if quality_path.exists():
            kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
            kf_by_id = {kf["keyframe_id"]: kf for kf in read_jsonl(kf_path)} if kf_path.exists() else {}
            for qrec in read_jsonl(quality_path):
                kf_id = qrec.get("keyframe_id")
                b_roll = qrec.get("b_roll_heuristic")
                if kf_id and kf_id in kf_by_id:
                    sc_id = kf_by_id[kf_id].get("scene_id")
                    if sc_id and b_roll is not None:
                        b_roll_by_scene[sc_id] = b_roll_by_scene.get(sc_id) or b_roll

        events_path = catalog / "events" / f"{media_id}.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        if events_path.exists():
            events_path.unlink()

        # No-speech guard: words file exists and is empty → skip all event generation.
        # Silent clips have no transcript content to segment; producing scene_events
        # from scene detection would be misleading (no speech means no editorial value).
        # Cross-stage awareness (Outcome Contract): read A's explicit has_speech flag
        # rather than re-deriving "no speech" by re-reading the words file.
        from . import outcomes as _oc
        _no_speech = (_oc.flag(media_id, "has_speech") is False)
        if _no_speech:
            # The skip is recorded AFTER reap_manifest below — the reap bridge
            # writes a bare "ok" and would overwrite it here.
            skip_no_speech = True
            _oc.set_flag(media_id, "has_events", False, reason="no_speech")
            # events_path already unlinked above; leave absent (0 events is correct)

        # Determine event segmentation mode (skipped for no-speech clips)
        diar_path = catalog / "asr" / f"{media_id}.diar.jsonl"
        raw_turns = read_jsonl(diar_path) if diar_path.exists() else []

        if _no_speech:
            pass  # no events to write
        elif raw_turns:
            # Diarization-based mode: dialogue_event per merged speaker-turn group
            merged_groups = _merge_turns(raw_turns, gap_ms=2000)

            for idx, grp in enumerate(merged_groups):
                grp_start = grp["start_ms"]
                grp_end = grp["end_ms"]
                duration_ms = grp_end - grp_start

                overlapping_chunks = [
                    c for c in search_chunks
                    if _overlap_ms(grp_start, grp_end, c["start_ms"], c["end_ms"]) > 0
                ]
                text_rollup = " ".join(
                    c["text"].strip() for c in overlapping_chunks if c.get("text")
                )
                chunk_ids = [c["chunk_id"] for c in overlapping_chunks]

                # LLM tagging
                emotional_beat: str | None = None
                edit_role: list | None = None
                b_roll_flag: bool | None = None
                summary: str | None = text_rollup[:500] if text_rollup else None

                if llm_tag and text_rollup:
                    llm_result = _tag_with_llm(text_rollup, duration_ms)
                    if llm_result:
                        emotional_beat = llm_result["emotional_beat"]
                        edit_role = llm_result["edit_role"]
                        if llm_result["b_roll_flag"] is not None:
                            b_roll_flag = llm_result["b_roll_flag"]
                        if llm_result["summary"]:
                            summary = llm_result["summary"]
                        tagged_count += 1

                ff = _format_fit(duration_ms, emotional_beat, b_roll_flag)

                rec = {
                    "schema_version": "1.1.0",
                    "event_id": event_id(media_id, idx),
                    "media_id": media_id,
                    "start_ms": grp_start,
                    "end_ms": grp_end,
                    "kind": "dialogue_event",
                    "speaker_label": grp["speaker_label"],
                    "chunk_ids": chunk_ids,
                    "scene_ids": [],
                    "summary": summary,
                    "emotional_beat": emotional_beat,
                    "edit_role": edit_role,
                    "format_fit": ff,
                    "publicability": "public",
                    "person_ids": None,
                    "b_roll_flag": b_roll_flag,
                    "visual_quality_score": None,
                }
                jsonl_append(events_path, rec)
                event_count += 1

        else:
            # Fallback: one scene_event per scene (original behaviour)
            for idx, sc in enumerate(scenes):
                sc_id = sc["scene_id"]
                sc_start = sc["start_ms"]
                sc_end = sc["end_ms"]
                duration_ms = sc_end - sc_start

                overlapping_chunks = [
                    c for c in search_chunks
                    if _overlap_ms(sc_start, sc_end, c["start_ms"], c["end_ms"]) > 0
                ]
                text_rollup = " ".join(
                    c["text"].strip() for c in overlapping_chunks if c.get("text")
                )
                chunk_ids = [c["chunk_id"] for c in overlapping_chunks]

                b_roll_from_quality = b_roll_by_scene.get(sc_id)

                emotional_beat = None
                edit_role = None
                b_roll_flag: bool | None = b_roll_from_quality
                summary: str | None = text_rollup[:500] if text_rollup else None

                if llm_tag and text_rollup:
                    llm_result = _tag_with_llm(text_rollup, duration_ms)
                    if llm_result:
                        emotional_beat = llm_result["emotional_beat"]
                        edit_role = llm_result["edit_role"]
                        if llm_result["b_roll_flag"] is not None:
                            b_roll_flag = llm_result["b_roll_flag"]
                        if llm_result["summary"]:
                            summary = llm_result["summary"]
                        tagged_count += 1

                ff = _format_fit(duration_ms, emotional_beat, b_roll_flag)

                rec = {
                    "schema_version": "1.1.0",
                    "event_id": event_id(media_id, idx),
                    "media_id": media_id,
                    "start_ms": sc_start,
                    "end_ms": sc_end,
                    "kind": "scene_event",
                    "chunk_ids": chunk_ids,
                    "scene_ids": [sc_id],
                    "summary": summary,
                    "emotional_beat": emotional_beat,
                    "edit_role": edit_role,
                    "format_fit": ff,
                    "publicability": "public",
                    "person_ids": None,
                    "b_roll_flag": b_roll_flag,
                    "visual_quality_score": None,
                }
                jsonl_append(events_path, rec)
                event_count += 1

        if not _no_speech:
            outputs.append(str(events_path))
        _emit_tagging_stats(journal_db, media_id, run_id, tagged_count, event_count)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="E0",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"events": event_count, "llm_tagged": tagged_count},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    if skip_no_speech:
        try:
            from . import outcomes as _oc2
            # Override the bridge's "ok" with the accurate skip (runs after reap).
            _oc2.record(media_id, "E0", "skip", reason="no_speech", db=journal_db)
        except Exception:
            pass
    if error is None:  # explicit flag for E1/CARD/EV to read instead of probing files
        try:
            from . import outcomes as _oc2
            _oc2.set_flag(media_id, "has_events", event_count > 0,
                          reason=None if event_count else "no_events")
        except Exception:
            pass
    return manifest

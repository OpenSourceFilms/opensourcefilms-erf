"""Stage §3 Pass 1 — narrative design via DeepSeek Pro."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomicio import read_jsonl, write_tmp_rename
from .contracts import validate
from .llm import complete

logger = logging.getLogger(__name__)

_STAGE = "P1"

_SYSTEM_PROMPT = """\
You are a documentary short-form storyteller. You receive a narrative thread — a cluster of related moments from BTS footage.

Design the story this thread tells. Decide the emotional journey, the hook (first 2 seconds), the punchline or reveal. Lay out an ordered beat structure. For EACH beat:
- State its emotional function
- State what it must accomplish (intent)
- Decide whether it wants b-roll relief: reason about b-roll as storytelling, not decoration. B-roll should earn its place — does this beat need visual breathing room, texture, or contrast to land? Set wants_broll: true when yes, false when the talking head carries it.
- If wants_broll is true, describe the specific visual texture in broll_intent (e.g. 'hands adjusting camera in soft light', 'empty set from above', 'faces watching playback')

Suggest which event_id(s) from the thread would serve each beat in candidate_event_ids. A single event can serve multiple beats.

Do NOT select chunk_ids. Do NOT write timecodes. This is pure story design.

Return ONLY strict JSON matching the narrative_brief schema. No markdown, no prose outside the JSON.\
"""

_SCHEMA_EXAMPLE = {
    "schema_version": "1.0.0",
    "thread_id": "thr_xxxxxxxxxxxx",
    "title": "Short descriptive title",
    "logline": "One sentence: the story this video tells.",
    "emotional_journey": "Describe the emotional arc from start to end.",
    "hook": {
        "intent": "What grabs the viewer in the first 2 seconds.",
        "seed_event_id": "m_xxx:ev:0000"
    },
    "punchline_or_reveal": "The payoff line or visual reveal, or null.",
    "beats": [
        {
            "beat_index": 0,
            "role": "hook",
            "emotional_function": "Immediate grab — surprise or tension.",
            "intent": "What this beat must make the viewer feel or do.",
            "candidate_event_ids": ["m_xxx:ev:0000"],
            "wants_broll": False,
            "broll_intent": None
        },
        {
            "beat_index": 1,
            "role": "setup",
            "emotional_function": "Ground the viewer — who, where, stakes.",
            "intent": "Establish context before tension builds.",
            "candidate_event_ids": ["m_xxx:ev:0001"],
            "wants_broll": True,
            "broll_intent": "Wide shot of the set from above, crew visible."
        }
    ],
    "target_duration_s": 60.0
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _journal(journal_db: Path, thread_id: str, run_id: str, event_type: str, payload: dict | None = None) -> None:
    try:
        with sqlite3.connect(str(journal_db)) as conn:
            conn.execute(
                # Phase 1 (D-Log-04): thread-scoped operational trace -> events_trace,
                # not the outcome-only `events` table (§3.1).
                "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (_now(), f"{thread_id}:{_STAGE}", thread_id, _STAGE, event_type, run_id,
                 json.dumps(payload) if payload is not None else None),
            )
    except Exception as exc:
        logger.warning("journal write failed for %s: %s", thread_id, exc)


def _media_id_from_event(event_id: str) -> str:
    # event_id format: m_<hash>:ev:<NNNN>
    return event_id.split(":")[0]


def _load_events_and_signals(media_ids: set[str], catalog: Path) -> dict[str, dict]:
    """Load events merged with E1 signals for a set of media_ids."""
    events_by_id: dict[str, dict] = {}
    for media_id in media_ids:
        events_path = catalog / "events" / f"{media_id}.jsonl"
        if not events_path.exists():
            continue
        signals: dict[str, dict] = {}
        sig_path = catalog / "event_signals" / f"{media_id}.jsonl"
        if sig_path.exists():
            for sig in read_jsonl(sig_path):
                signals[sig["event_id"]] = sig
        for evt in read_jsonl(events_path):
            eid = evt["event_id"]
            merged = dict(evt)
            sig = signals.get(eid, {})
            merged["hook_potential"] = sig.get("hook_potential")
            merged["quote_potential"] = sig.get("quote_potential")
            merged["narrative_role_suggestion"] = sig.get("narrative_role_suggestion")
            events_by_id[eid] = merged
    return events_by_id


def _resolve_speaker_names(member_event_ids: list[str], events_by_id: dict[str, dict], catalog: Path) -> dict[str, str]:
    """Return {event_id: display_name_or_label} for all member events."""
    media_ids = {_media_id_from_event(eid) for eid in member_event_ids}

    # chunk_id → person_id from identity_overlay
    chunk_to_person: dict[str, str] = {}
    for media_id in media_ids:
        overlay_path = catalog / "identity_overlay" / f"{media_id}.json"
        if not overlay_path.exists():
            continue
        try:
            data = json.loads(overlay_path.read_text(encoding="utf-8"))
            for entry in data.get("chunk_speaker", []):
                cid, pid = entry.get("chunk_id"), entry.get("person_id")
                if cid and pid:
                    chunk_to_person[cid] = pid
        except Exception:
            pass

    # person_id → display_name
    person_names: dict[str, str] = {}
    persons_dir = catalog / "persons"
    for pid in set(chunk_to_person.values()):
        pf = persons_dir / f"{pid}.json"
        if pf.exists():
            try:
                person_names[pid] = json.loads(pf.read_text(encoding="utf-8")).get("display_name", pid)
            except Exception:
                person_names[pid] = pid

    # event_id → speaker display name (dominant person by chunk count, else speaker_label)
    speaker_names: dict[str, str] = {}
    for eid in member_event_ids:
        evt = events_by_id.get(eid, {})
        chunk_ids = evt.get("chunk_ids") or []
        counts: dict[str, int] = {}
        for cid in chunk_ids:
            pid = chunk_to_person.get(cid)
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
        if counts:
            dom = max(counts, key=counts.get)
            speaker_names[eid] = person_names.get(dom, dom)
        else:
            speaker_names[eid] = evt.get("speaker_label") or "unknown"
    return speaker_names


_VALID_ROLES = frozenset(["hook", "setup", "body", "tension", "reveal", "payoff", "quote", "cta", "transition"])

# Pro invents synonym roles; map them to the nearest valid value before validation.
_ROLE_NORM: dict[str, str] = {
    "development": "body", "elaboration": "body", "build": "body",
    "comedic": "body", "comedic_beat": "body", "comedic twist": "payoff",
    "rising_tension": "tension", "tension_build": "tension", "escalation": "tension",
    "complication": "tension", "turn": "reveal",
    "exposition": "setup", "context": "setup",
    "climax": "reveal", "resolution": "payoff", "conclusion": "payoff",
    "call_to_action": "cta",
}


def _normalize_beat_roles(brief: dict) -> dict:
    for beat in brief.get("beats", []):
        role = beat.get("role", "body")
        if role not in _VALID_ROLES:
            normalized = _ROLE_NORM.get(role.lower().replace(" ", "_"), "body")
            logger.debug("Normalizing beat role %r → %r", role, normalized)
            beat["role"] = normalized
    return brief


def pass1_brief(
    thread: dict[str, Any],
    events_by_id: dict[str, dict],
    speaker_names: dict[str, str],
    guidance_text: str,
) -> dict[str, Any]:
    thread_id = thread["thread_id"]
    member_event_ids: list[str] = thread.get("member_event_ids", [])

    event_lines: list[str] = []
    for eid in member_event_ids:
        evt = events_by_id.get(eid)
        if not evt:
            logger.warning("Missing event %s for thread %s", eid, thread_id)
            continue
        hp = evt.get("hook_potential")
        qp = evt.get("quote_potential")
        event_lines.append(
            f"Event ID: {eid}\n"
            f"  Speaker: {speaker_names.get(eid, 'unknown')}\n"
            f"  Emotional beat: {evt.get('emotional_beat') or 'unknown'}\n"
            f"  B-roll available: {bool(evt.get('b_roll_flag'))}\n"
            f"  Hook potential: {'null' if hp is None else f'{hp:.2f}'} | "
            f"Quote potential: {'null' if qp is None else f'{qp:.2f}'}\n"
            f"  Summary: {evt.get('summary') or '(no summary)'}"
        )

    user_prompt = (
        f"Thread ID: {thread_id}\n"
        f"Seed event: {thread.get('seed_event_id')}\n"
        f"Arc shape: {thread.get('arc_shape', [])}\n"
        f"Dominant person: {thread.get('dominant_person_id') or 'unknown'}\n"
        f"Estimated duration: {thread.get('est_duration_s', 60):.1f}s\n"
        f"Composite score: {thread.get('scores', {}).get('composite', 0):.3f}\n"
        "\n"
        "Events in arc order:\n"
        + "\n\n".join(event_lines)
        + "\n\nProject guidance:\n"
        + (guidance_text or "(none)")
        + "\n\nOutput schema (follow exactly):\n"
        + json.dumps(_SCHEMA_EXAMPLE, indent=2)
        + "\n\nReturn ONLY the JSON object. No other text."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = complete(messages, json_only=True)
    try:
        brief = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pass 1 JSON parse failed: {raw[:200]}") from exc

    # Normalize and fill defaults before validation
    brief.setdefault("schema_version", "1.0.0")
    brief.setdefault("thread_id", thread_id)
    brief.setdefault("punchline_or_reveal", None)
    for beat in brief.get("beats", []):
        beat.setdefault("candidate_event_ids", [])
        beat.setdefault("broll_intent", None)
    _normalize_beat_roles(brief)

    try:
        validate(brief, "narrative_brief")
    except Exception as exc:
        raise ValueError(f"Pass 1 schema validation failed: {exc}") from exc

    return brief


def run_pass1(thread_id: str, *, catalog: Path, state: Path, journal_db: Path) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    _journal(journal_db, thread_id, run_id, "start")

    thread_path = catalog / "narrative_threads" / f"{thread_id}.json"
    thread = json.loads(thread_path.read_text(encoding="utf-8"))

    guidance_path = state / "guidance.md"
    guidance_text = guidance_path.read_text(encoding="utf-8") if guidance_path.exists() else ""

    member_event_ids: list[str] = thread.get("member_event_ids", [])
    media_ids = {_media_id_from_event(eid) for eid in member_event_ids}
    events_by_id = _load_events_and_signals(media_ids, catalog)
    speaker_names = _resolve_speaker_names(member_event_ids, events_by_id, catalog)

    try:
        brief = pass1_brief(thread, events_by_id, speaker_names, guidance_text)
    except Exception as exc:
        _journal(journal_db, thread_id, run_id, "error", {"error": str(exc)})
        raise

    out_dir = catalog / "narrative_briefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tmp_rename(
        out_dir / f"{thread_id}.json",
        json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
    )

    _journal(journal_db, thread_id, run_id, "ok", {"beats": len(brief.get("beats", []))})
    return brief


def run_pass1_batch(
    *,
    catalog: Path,
    state: Path,
    journal_db: Path,
    limit: int | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    threads_dir = catalog / "narrative_threads"
    thread_files = sorted(threads_dir.glob("thr_*.json"))
    if not thread_files:
        logger.warning("No thread files found in %s", threads_dir)
        return []

    # Sort by composite score descending
    threads: list[dict] = []
    for tf in thread_files:
        try:
            t = json.loads(tf.read_text(encoding="utf-8"))
            threads.append(t)
        except Exception as exc:
            logger.error("Skipping corrupt thread file %s: %s", tf, exc)

    threads.sort(key=lambda t: t.get("scores", {}).get("composite", 0.0), reverse=True)
    if limit is not None:
        threads = threads[:limit]

    results: list[dict] = []
    briefs_dir = catalog / "narrative_briefs"
    for t in threads:
        thread_id = t.get("thread_id")
        if not thread_id:
            continue
        brief_path = briefs_dir / f"{thread_id}.json"
        if not force and brief_path.exists():
            try:
                results.append(json.loads(brief_path.read_text(encoding="utf-8")))
            except Exception:
                pass
            continue
        try:
            brief = run_pass1(thread_id, catalog=catalog, state=state, journal_db=journal_db)
            results.append(brief)
            logger.info("Pass 1 complete for %s — %d beats", thread_id, len(brief.get("beats", [])))
        except Exception as exc:
            logger.error("Pass 1 failed for %s: %s", thread_id, exc)

    return results


def check_broll_refs(beat_plans: list[dict], catalog: Path) -> dict[str, Any]:
    """Post-run referential check: verify every broll.keyframe_id exists in catalog/keyframes/.

    Returns a summary dict: {total_enabled, ok, missing, missing_ids}.
    """
    # Build a set of all known keyframe_ids per media_id (lazy-loaded)
    kf_index: dict[str, set[str]] = {}

    def _get_kf_ids(media_id: str) -> set[str]:
        if media_id not in kf_index:
            ids: set[str] = set()
            path = catalog / "keyframes" / f"{media_id}.jsonl"
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            ids.add(json.loads(line)["keyframe_id"])
                        except Exception:
                            pass
            kf_index[media_id] = ids
        return kf_index[media_id]

    total_enabled = 0
    ok = 0
    missing_ids: list[str] = []
    for bp in beat_plans:
        for beat in bp.get("beats", []):
            broll = beat.get("broll")
            if broll and broll.get("enabled") and broll.get("keyframe_id"):
                kid = broll["keyframe_id"]
                total_enabled += 1
                media_id = kid.split(":")[0]
                if kid in _get_kf_ids(media_id):
                    ok += 1
                else:
                    missing_ids.append(kid)
                    logger.warning("broll keyframe_id not found in catalog: %s", kid)

    return {
        "total_enabled": total_enabled,
        "ok": ok,
        "missing": len(missing_ids),
        "missing_ids": missing_ids,
    }


# ---------------------------------------------------------------------------
# Pass 2 — atom selection / execution
# ---------------------------------------------------------------------------

_STAGE_P2 = "P2"

_SYSTEM_PROMPT_P2 = """\
You are a clip selector for a documentary short-form video. You have a narrative brief describing the story structure and a pool of available clips (atoms), each with its event summary, emotional beat, and scores.

Each atom in the pool represents ONE event with its own in/out timestamps. Multiple atoms may share the same chunk_id (they are different moments within the same source take). Use the atom's in_ms/out_ms — not the chunk boundaries — as your selection; the renderer will trim to those exact timestamps.

For each beat in the narrative brief:
1. Select the atom (by chunk_id) from the pool that best executes the beat's intent and emotional_function. Prefer atoms whose emotional_beat aligns with the beat's emotional_function, and whose summary matches the beat's intent.
2. Set in_ms/out_ms to the selected atom's values exactly.
3. Set edit_role based on the beat's role.
4. Write a brief note explaining your selection.
5. If wants_broll is true: select a keyframe_id from the keyframe pool matching the broll_intent.

Constraints:
- Every beat must map to exactly ONE atom from the pool
- Do NOT reuse the same atom for multiple beats unless the pool has fewer atoms than beats
- Use the exact chunk_id, in_ms, out_ms, keyframe_id as provided — do not invent IDs or timestamps
- Honour beat order from the narrative brief
- Keep total duration close to target_duration_s
- Return ONLY strict JSON matching the beat_plan schema (1.3.0). No markdown, no prose outside the JSON.\
"""

# Defensive mapping: Pro may return narrative brief roles or synonyms in edit_role.
# Allowed top-level keys in the beat_plan schema (anything else is stripped)
_ALLOWED_BP_KEYS = frozenset([
    "schema_version", "bp_id", "created_at_utc", "revision", "supersedes_revision",
    "topic", "format_fit", "duration_target_s", "beats",
    "total_duration_s", "assembler_note", "template_id",
])

# Allowed per-beat keys
_ALLOWED_BEAT_KEYS = frozenset([
    "beat_index", "chunk_id", "media_id", "in_ms", "out_ms", "edit_role",
    "note", "hook_score", "quote_score", "proof_score", "suggested_padding_ms",
    "duration_s", "b_roll_candidates", "b_roll_selected", "broll",
])

_EDIT_ROLE_NORM: dict[str, str] = {
    "hook": "hook",
    "setup": "body",
    "body": "body",
    "tension": "body",
    "reveal": "body",
    "payoff": "body",
    "quote": "quote",
    "cta": "call_to_action",
    "call_to_action": "call_to_action",
    "transition": "transition",
    "development": "body",
    "exposition": "exposition",
    "comedic": "body",
    "b-roll": "b_roll",
    "broll": "b_roll",
    "b_roll": "b_roll",
    "action": "body",
    "proof": "proof",
}


def _normalize_edit_role(role: str | None) -> str:
    if not isinstance(role, str):
        return "body"
    return _EDIT_ROLE_NORM.get(role.strip().lower(), "body")


def build_atom_pool(
    member_event_ids: list[str],
    events_by_id: dict[str, dict],
    catalog: Path,
    min_atom_ms: int = 3000,
    max_atom_ms: int = 12000,
) -> list[dict]:
    """Return selectable clip atoms for Pass 2, subdivided to ~sentence granularity.

    Chunks are speaker-turn sized (often 10-40s). One atom per turn starves Pass 2:
    a brief with N beats drawing from a thread with fewer distinct turns is forced to
    reuse the same window across beats (degenerate beat plans — see DECISIONS
    D-Author-01). We therefore subdivide each member chunk into segment-aligned atoms
    (~min_atom_ms..max_atom_ms each) using catalog/asr/<media>.segments.jsonl.
    Segment-aligned windows also make cuts land on sentence boundaries rather than
    mid-utterance.

    Falls back to the event/chunk window when a media's segments are unavailable. Each
    source chunk is subdivided once; identical windows are collapsed.

    min_atom_ms: floor for an atom's duration (short trailing segments merge upward).
    max_atom_ms: ceiling — a running segment group is emitted before exceeding this.
    """
    event_media: dict[str, str] = {}
    for ev_id in member_event_ids:
        evt = events_by_id.get(ev_id)
        if not evt:
            continue
        event_media[ev_id] = evt.get("media_id", "")

    media_ids = set(event_media.values())

    # Load ALL chunks per media (time-sorted) and tags keyed by chunk_id. Event
    # chunk_id references can be stale after a re-chunk (the chunks file is
    # regenerated with different boundaries), so atom metadata is resolved by TIME
    # overlap against the current chunks rather than by (possibly dead) chunk_id.
    chunks_by_media: dict[str, list[dict]] = {}
    tags_by_cid: dict[str, dict] = {}
    for media_id in media_ids:
        chunk_list: list[dict] = []
        chunk_path = catalog / "chunks" / f"{media_id}.jsonl"
        if chunk_path.exists():
            for line in chunk_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    c = json.loads(line)
                except Exception:
                    continue
                if c.get("start_ms") is not None and c.get("end_ms") is not None:
                    chunk_list.append(c)
            chunk_list.sort(key=lambda c: c["start_ms"])
        chunks_by_media[media_id] = chunk_list
        tag_path = catalog / "tags" / f"{media_id}.jsonl"
        if tag_path.exists():
            for line in tag_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue
                cid = t.get("chunk_id", "")
                if cid:
                    tags_by_cid[cid] = t

    def _chunk_at(media_id: str, t_ms: int) -> "dict | None":
        """Current chunk whose window contains t_ms (None if none)."""
        for c in chunks_by_media.get(media_id, []):
            if c["start_ms"] <= t_ms < c["end_ms"]:
                return c
        return None

    # Sentence-level segment spans per media, for subdividing coarse turn-chunks.
    segments_by_media: dict[str, list[dict]] = {}

    def _segments(media_id: str) -> list[dict]:
        if media_id not in segments_by_media:
            segs: list[dict] = []
            sp = catalog / "asr" / f"{media_id}.segments.jsonl"
            if sp.exists():
                for line in sp.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        s = json.loads(line)
                    except Exception:
                        continue
                    if s.get("start_ms") is not None and s.get("end_ms") is not None:
                        segs.append(s)
                segs.sort(key=lambda x: x["start_ms"])
            segments_by_media[media_id] = segs
        return segments_by_media[media_id]

    # Sentence-level units from punctuated word ASR. When the transcript carries
    # sentence punctuation we group/cut on sentence boundaries (no mid-sentence
    # cuts); otherwise we fall back to the coarser ASR segments.
    units_by_media: dict[str, list[dict]] = {}

    def _units(media_id: str) -> list[dict]:
        if media_id not in units_by_media:
            # Prefer the P0 sentence-boundary facet (catalog/asr/<m>.sentences.jsonl)
            # when present — it's pause-aware and works even on unpunctuated ASR.
            sp = catalog / "asr" / f"{media_id}.sentences.jsonl"
            if sp.exists():
                facet: list[dict] = []
                for line in sp.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    if o.get("start_ms") is not None and o.get("end_ms") is not None \
                            and "sentence_id" in o:
                        facet.append({"start_ms": o["start_ms"], "end_ms": o["end_ms"],
                                      "text": o.get("text", "")})
                if facet:
                    units_by_media[media_id] = facet
                    return facet
            wp = catalog / "asr" / f"{media_id}.words.jsonl"
            words: list[dict] = []
            if wp.exists():
                for line in wp.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        w = json.loads(line)
                    except Exception:
                        continue
                    if w.get("start_ms") is not None and w.get("end_ms") is not None:
                        words.append(w)
            sentences: list[list[dict]] = []
            cur: list[dict] = []
            for w in words:
                cur.append(w)
                if (w.get("text") or "").rstrip().endswith((".", "?", "!", "…")):
                    sentences.append(cur)
                    cur = []
            if cur:
                sentences.append(cur)
            # Only treat as sentence units when punctuation actually split the
            # transcript (>1 sentence); else fall back to ASR segments.
            if len(sentences) > 1:
                units = [
                    {"start_ms": s[0]["start_ms"], "end_ms": s[-1]["end_ms"],
                     "text": " ".join((w.get("text") or "").strip() for w in s).strip()}
                    for s in sentences
                ]
            else:
                units = _segments(media_id)
            units_by_media[media_id] = units
        return units_by_media[media_id]

    def _group_segments(segs: list[dict]) -> list[tuple[int, int, str]]:
        """Greedily group consecutive segments into (start_ms, end_ms, text) windows
        of at least min_atom_ms, emitting before a group would exceed max_atom_ms. A
        trailing group shorter than min_atom_ms is merged into the previous one."""
        groups: list[list[dict]] = []
        cur: list[dict] = []
        for s in segs:
            if cur:
                cur_span = cur[-1]["end_ms"] - cur[0]["start_ms"]
                next_span = s["end_ms"] - cur[0]["start_ms"]
                if cur_span >= min_atom_ms or next_span > max_atom_ms:
                    groups.append(cur)
                    cur = []
            cur.append(s)
        if cur:
            groups.append(cur)
        if len(groups) >= 2:
            tail = groups[-1]
            if tail[-1]["end_ms"] - tail[0]["start_ms"] < min_atom_ms:
                groups[-2].extend(groups.pop())
        out: list[tuple[int, int, str]] = []
        for g in groups:
            text = " ".join((s.get("text") or "").strip() for s in g).strip()
            out.append((g[0]["start_ms"], g[-1]["end_ms"], text[:200]))
        return out

    seen_windows: set[tuple] = set()  # (media_id, in_ms, out_ms) — dedupe video clips
    seen_events: set[str] = set()
    atoms: list[dict] = []
    for ev_id in member_event_ids:
        if ev_id in seen_events:
            continue
        seen_events.add(ev_id)
        evt = events_by_id.get(ev_id)
        if not evt:
            continue
        media_id = evt.get("media_id", "") or event_media.get(ev_id, "")

        # Region this event covers, by TIME (robust to stale chunk_id references).
        ev_start = evt.get("start_ms")
        ev_end = evt.get("end_ms")
        if ev_start is None or ev_end is None or ev_end <= ev_start:
            continue

        # Subdivide the event's span into segment-aligned windows.
        all_segs = _units(media_id)
        ev_segs = [
            s for s in all_segs
            if s["end_ms"] > ev_start and s["start_ms"] < ev_end
        ]
        if ev_segs:
            windows = _group_segments(ev_segs)
        elif not all_segs:
            # Media has no segment ASR at all — best-effort whole-event window.
            windows = [(ev_start, ev_end, (evt.get("summary") or "")[:200])]
        else:
            # Media HAS segments but none overlap this event's span: the event is
            # stale relative to the current ASR (media re-transcribed/re-chunked
            # after events were written — see D-Data-01). Emitting the event span
            # here would create a word-less beat that renders with no captions and
            # no real speech (the "stub" failure). Skip it.
            continue

        for w_start, w_end, w_text in windows:
            w_start = int(w_start)
            w_end = int(w_end)
            if w_end - w_start < min_atom_ms:
                continue
            window_key = (media_id, w_start, w_end)
            if window_key in seen_windows:
                continue
            # Overlap dedup: adjacent events can grab segment-groups sharing a
            # boundary, producing near-duplicate windows that show the same footage.
            # Skip a window overlapping an existing same-media atom by >50% of the
            # shorter window so Pass 2 never sees two interchangeable clips.
            _overlaps = False
            for _a in atoms:
                if _a["media_id"] != media_id:
                    continue
                _ov = min(w_end, _a["out_ms"]) - max(w_start, _a["in_ms"])
                if _ov > 0 and _ov > 0.5 * min(w_end - w_start, _a["out_ms"] - _a["in_ms"]):
                    _overlaps = True
                    break
            if _overlaps:
                continue
            seen_windows.add(window_key)
            # Resolve current chunk by time for metadata (chunk_id, speaker, tags).
            mid_ms = (w_start + w_end) // 2
            cur_chunk = _chunk_at(media_id, mid_ms)
            cid = cur_chunk["chunk_id"] if cur_chunk else f"{media_id}:c:{w_start}-{w_end}"
            tag = tags_by_cid.get(cid, {})
            spk = (cur_chunk.get("speaker_label") if cur_chunk else None) or ""
            atoms.append({
                "chunk_id": cid,
                "media_id": media_id,
                "in_ms": w_start,
                "out_ms": w_end,
                "text": w_text or (evt.get("summary") or "")[:200],
                "speaker_label": spk,
                "hook_score": tag.get("hook_score") or 0.0,
                "quote_score": tag.get("quote_score") or 0.0,
                "proof_score": tag.get("proof_score") or 0.0,
                "event_id": ev_id,
                "emotional_beat": evt.get("emotional_beat") or "",
                "duration_s": (w_end - w_start) / 1000.0,
            })

    atoms.sort(key=lambda a: (a["media_id"], a["in_ms"]))
    return atoms


def build_keyframe_pool(media_ids: set[str], catalog: Path) -> list[dict]:
    """Return all keyframes for the given media, sorted by time_ms."""
    pool: list[dict] = []
    for media_id in media_ids:
        kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
        if not kf_path.exists():
            continue
        for line in kf_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                kf = json.loads(line)
                pool.append({
                    "keyframe_id": kf["keyframe_id"],
                    "media_id": kf.get("media_id", media_id),
                    "time_ms": kf["time_ms"],
                    "visual_quality_score": kf.get("visual_quality_score"),
                    "b_roll_flag": kf.get("b_roll_flag"),
                })
            except Exception:
                pass
    pool.sort(key=lambda kf: kf["time_ms"])
    return pool


def pass2_beat_plan(
    narrative_brief: dict[str, Any],
    atom_pool: list[dict],
    keyframe_pool: list[dict],
    thread: dict[str, Any],
) -> dict[str, Any]:
    beats_text: list[str] = []
    for beat in narrative_brief.get("beats", []):
        beats_text.append(json.dumps({
            "beat_index": beat.get("beat_index"),
            "role": beat.get("role"),
            "intent": beat.get("intent", ""),
            "emotional_function": beat.get("emotional_function", ""),
            "wants_broll": beat.get("wants_broll", False),
            "broll_intent": beat.get("broll_intent"),
            "candidate_event_ids": beat.get("candidate_event_ids", []),
        }))

    brief_summary = (
        f"Title: {narrative_brief.get('title', '')}\n"
        f"Logline: {narrative_brief.get('logline', '')}\n"
        f"Beats:\n" + "\n".join(beats_text)
    )

    atom_lines: list[str] = []
    for idx, atom in enumerate(atom_pool):
        scores_str = f"hook:{atom['hook_score']:.2f} quote:{atom['quote_score']:.2f}"
        eb = atom.get("emotional_beat") or ""
        atom_lines.append(
            f"{idx+1:04d} | {atom['chunk_id']} | event:{atom['event_id']} | "
            f"{atom['speaker_label']} | {atom['in_ms']}-{atom['out_ms']}ms ({atom['duration_s']:.1f}s) | "
            f"{scores_str} | {eb} | {atom['text']}"
        )

    has_broll = any(b.get("wants_broll") for b in narrative_brief.get("beats", []))
    kf_section = ""
    if has_broll:
        if keyframe_pool:
            kf_lines: list[str] = []
            for idx, kf in enumerate(keyframe_pool):
                vqs = kf.get("visual_quality_score")
                vqs_str = f"{vqs:.2f}" if vqs is not None else "null"
                kf_lines.append(
                    f"{idx+1:04d} | {kf['keyframe_id']} | {kf['time_ms']}ms | vqs:{vqs_str}"
                )
            kf_section = "\nKEYFRAME POOL:\n" + "\n".join(kf_lines) + "\n"
        else:
            kf_section = "\nKEYFRAME POOL: empty — set broll.enabled=false for all broll beats\n"

    brief_beat_count = len(narrative_brief.get("beats", []))
    pool_size = len(atom_pool)
    pool_cap_note = ""
    if pool_size < brief_beat_count:
        pool_cap_note = (
            f"\nCONSTRAINT: The atom pool has only {pool_size} unique video windows "
            f"but the brief requests {brief_beat_count} beats. "
            f"Plan exactly {pool_size} beats — do not exceed the pool size. "
            "Each atom may be used at most once.\n"
        )

    user_prompt = (
        f"NARRATIVE BRIEF:\n{brief_summary}\n\n"
        f"ATOM POOL ({pool_size} clips):\n" + "\n".join(atom_lines) + "\n"
        + kf_section
        + pool_cap_note
        + f"\nTARGET DURATION (seconds): {narrative_brief.get('target_duration_s', 60)}\n\n"
        "Return ONLY the JSON beat_plan object. No other text."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_P2},
        {"role": "user", "content": user_prompt},
    ]
    raw = complete(messages, json_only=True)
    try:
        beat_plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pass 2 JSON parse failed: {raw[:200]}") from exc

    # Unwrap if Pro returned {"beat_plan": {...}} instead of the object directly.
    if "beat_plan" in beat_plan and "beats" not in beat_plan:
        inner = beat_plan["beat_plan"]
        if isinstance(inner, dict):
            # Preserve any top-level fields Pro included outside the wrapper
            outer = {k: v for k, v in beat_plan.items() if k != "beat_plan"}
            beat_plan = {**inner, **outer}
        elif isinstance(inner, list):
            # Some models return {"beat_plan": [...beats...]}
            outer = {k: v for k, v in beat_plan.items() if k != "beat_plan"}
            beat_plan = {"beats": inner, **outer}

    # Handle "version" alias for "schema_version"
    if "version" in beat_plan and "schema_version" not in beat_plan:
        beat_plan["schema_version"] = beat_plan.pop("version")

    # Fill required top-level defaults
    beat_plan.setdefault("schema_version", "1.3.0")
    beat_plan.setdefault("topic", thread.get("theme_label") or thread.get("thread_id", ""))
    beat_plan.setdefault("format_fit", "youtube_shorts")
    target = int(narrative_brief.get("target_duration_s") or 60)
    beat_plan.setdefault("duration_target_s", target)
    beat_plan.setdefault("total_duration_s", None)
    beat_plan.setdefault("assembler_note", None)
    beat_plan.setdefault("template_id", None)
    beat_plan.setdefault("revision", 0)
    beat_plan.setdefault("supersedes_revision", None)

    # Strip unknown top-level keys (Pro invents e.g. total_duration_estimate_s)
    for k in list(beat_plan.keys()):
        if k not in _ALLOWED_BP_KEYS:
            beat_plan.pop(k)

    # Build index for resolving chunk_id references.
    # Multiple atoms may share a chunk_id (different sub-clips); keep ordered list per chunk_id.
    from collections import defaultdict as _defaultdict
    atoms_by_cid: dict[str, list[dict]] = _defaultdict(list)
    for a in atom_pool:
        atoms_by_cid[a["chunk_id"]].append(a)
    # Track used video windows (chunk_id, in_ms, out_ms) so the same physical clip
    # is never assigned to two beats, even if Pro picks different event_ids for each.
    _used_event_ids: set[str] = set()
    _used_windows: set[tuple] = set()

    for beat in beat_plan.get("beats", []):
        # Unwrap "selected" or "clip" wrapper if Pro nested chunk/timing data inside them
        for wrapper_key in ("selected", "clip"):
            wrapper = beat.pop(wrapper_key, None)
            if isinstance(wrapper, dict):
                # Merge fields from wrapper into beat (beat-level wins for existing fields)
                for field in ("chunk_id", "edit_role", "note", "broll"):
                    if field in wrapper and field not in beat:
                        beat[field] = wrapper[field]
                # clip uses start_ms/end_ms instead of in_ms/out_ms
                if "start_ms" in wrapper and "in_ms" not in beat:
                    beat["in_ms"] = wrapper["start_ms"]
                if "end_ms" in wrapper and "out_ms" not in beat:
                    beat["out_ms"] = wrapper["end_ms"]
                # media_id may be in the clip wrapper
                if "media_id" in wrapper and "media_id" not in beat:
                    beat["media_id"] = wrapper["media_id"]

        # Alias selection_note → note
        if "selection_note" in beat and "note" not in beat:
            beat["note"] = beat.pop("selection_note")
        else:
            beat.pop("selection_note", None)

        cid = str(beat.get("chunk_id") or "")
        # If model returned an index number instead of a real chunk_id, resolve it
        stripped = cid.lstrip("0")
        if not stripped:
            stripped = "0"
        if stripped.isdigit() and len(cid) <= 4:
            idx = int(stripped) - 1  # 1-indexed; use stripped not cid — cid may be "" or "0001"
            if 0 <= idx < len(atom_pool):
                atom = atom_pool[idx]
                beat["chunk_id"] = atom["chunk_id"]
                cid = atom["chunk_id"]

        # If Pro omitted or nulled chunk_id, fall back to the next unused atom in pool order.
        if not cid and atom_pool:
            fallback = next(
                (a for a in atom_pool
                 if (a["chunk_id"], a["in_ms"], a["out_ms"]) not in _used_windows),
                atom_pool[0],
            )
            beat["chunk_id"] = fallback["chunk_id"]
            cid = fallback["chunk_id"]

        # Resolve each beat to a DISTINCT real atom and snap its window to it.
        # The LLM sometimes reuses the same window across beats (and the old code
        # trusted its in/out when they were ints), producing degenerate beat plans
        # that the renderer then collapses to one clip. Selection order:
        #   1. the atom matching the LLM's exact window, IF still unused;
        #   2. else the next unused atom sharing the LLM's chunk_id;
        #   3. else the next unused atom anywhere in the pool (LLM reused a chunk);
        #   4. else any candidate (pool genuinely exhausted — render dedup handles it).
        candidates = atoms_by_cid.get(cid, [])
        _llm_in = beat.get("in_ms") if isinstance(beat.get("in_ms"), int) else None
        _llm_out = beat.get("out_ms") if isinstance(beat.get("out_ms"), int) else None

        def _unused(a: dict) -> bool:
            return (
                (a["chunk_id"], a["in_ms"], a["out_ms"]) not in _used_windows
                and a["event_id"] not in _used_event_ids
            )

        atom = None
        if _llm_in is not None and _llm_out is not None:
            atom = next(
                (a for a in candidates
                 if a["in_ms"] == _llm_in and a["out_ms"] == _llm_out and _unused(a)),
                None,
            )
        if atom is None:
            atom = next((a for a in candidates if _unused(a)), None)
        if atom is None:
            atom = next((a for a in atom_pool if _unused(a)), None)
        if atom is None:
            atom = candidates[0] if candidates else (atom_pool[0] if atom_pool else None)

        if atom:
            _used_event_ids.add(atom["event_id"])
            _used_windows.add((atom["chunk_id"], atom["in_ms"], atom["out_ms"]))
            # Snap to the selected atom (overrides any LLM window reuse).
            beat["chunk_id"] = atom["chunk_id"]
            beat["media_id"] = atom["media_id"]
            beat["in_ms"] = atom["in_ms"]
            beat["out_ms"] = atom["out_ms"]
            cid = atom["chunk_id"]

        beat["edit_role"] = _normalize_edit_role(beat.get("edit_role"))
        beat.setdefault("note", None)
        beat.setdefault("hook_score", None)
        beat.setdefault("quote_score", None)
        beat.setdefault("proof_score", None)
        beat.setdefault("suggested_padding_ms", None)
        beat.setdefault("duration_s", None)
        beat.setdefault("b_roll_candidates", None)
        beat.setdefault("b_roll_selected", None)
        if "broll" not in beat:
            beat["broll"] = None

        # Resolve index-based keyframe_id references in broll
        broll = beat.get("broll")
        if isinstance(broll, dict):
            kid = broll.get("keyframe_id")
            if isinstance(kid, str):
                stripped_kid = kid.lstrip("0") or "0"
                if stripped_kid.isdigit() and len(kid) <= 4:
                    idx = int(stripped_kid) - 1  # 1-indexed; use stripped not kid
                    if 0 <= idx < len(keyframe_pool):
                        broll["keyframe_id"] = keyframe_pool[idx]["keyframe_id"]
                    else:
                        broll["enabled"] = False
                        broll["keyframe_id"] = None
                        broll["reason"] = f"Index {kid} out of range for keyframe_pool"

        # Normalise broll dict to the schema (required 'enabled'; no extra keys).
        # The LLM sometimes returns {keyframe_id: ...} without 'enabled', which
        # fails validation and aborts the whole Pass 2 run.
        broll = beat.get("broll")
        if isinstance(broll, dict):
            if "enabled" not in broll:
                broll["enabled"] = bool(broll.get("keyframe_id"))
            for _bk in list(broll.keys()):
                if _bk not in ("enabled", "keyframe_id", "reason"):
                    broll.pop(_bk)
            beat["broll"] = broll

        # Strip any remaining unknown beat keys
        for k in list(beat.keys()):
            if k not in _ALLOWED_BEAT_KEYS:
                beat.pop(k)

    # Validation is deferred to run_pass2 after bp_id/created_at_utc are set.
    return beat_plan


def run_pass2(thread_id: str, *, catalog: Path, state: Path, journal_db: Path) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]

    brief_path = catalog / "narrative_briefs" / f"{thread_id}.json"
    if not brief_path.exists():
        raise FileNotFoundError(f"narrative_brief missing: {brief_path}")
    narrative_brief = json.loads(brief_path.read_text(encoding="utf-8"))

    thread_path = catalog / "narrative_threads" / f"{thread_id}.json"
    thread = json.loads(thread_path.read_text(encoding="utf-8"))

    member_event_ids: list[str] = thread.get("member_event_ids", [])
    if not member_event_ids:
        raise ValueError(f"Thread {thread_id} has no member_event_ids")

    media_ids = {_media_id_from_event(eid) for eid in member_event_ids}
    events_by_id = _load_events_and_signals(media_ids, catalog)

    atom_pool = build_atom_pool(member_event_ids, events_by_id, catalog)
    if not atom_pool:
        raise ValueError(f"Thread {thread_id}: atom_pool is empty — no chunks resolved")

    keyframe_pool = build_keyframe_pool(media_ids, catalog)

    bp_id = f"bp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

    try:
        beat_plan = pass2_beat_plan(narrative_brief, atom_pool, keyframe_pool, thread)
    except Exception as exc:
        try:
            with sqlite3.connect(str(journal_db)) as conn:
                conn.execute(
                    # Phase 1 (D-Log-04): trace-only -> events_trace (§3.1).
                    "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (_now(), f"{thread_id}:{_STAGE_P2}", thread_id, _STAGE_P2, "error", run_id,
                     json.dumps({"error": str(exc)})),
                )
        except Exception:
            pass
        raise

    beat_plan["bp_id"] = bp_id
    beat_plan["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    # Validate after bp_id and created_at_utc are populated (both are required by schema)
    try:
        validate(beat_plan, "beat_plan")
    except Exception as exc:
        raise ValueError(f"Pass 2 schema validation failed for {thread_id}: {exc}") from exc

    out_dir = catalog / "beat_plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tmp_rename(
        out_dir / f"{bp_id}.jsonl",
        json.dumps(beat_plan, ensure_ascii=False).encode("utf-8") + b"\n",
    )

    # Update thread→bp sidecar index atomically
    index_path = out_dir / "_thread_index.json"
    try:
        index: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    except Exception:
        index = {}
    index[thread_id] = bp_id
    write_tmp_rename(index_path, json.dumps(index, indent=2).encode("utf-8"))

    try:
        with sqlite3.connect(str(journal_db)) as conn:
            conn.execute(
                # Phase 1 (D-Log-04): trace-only -> events_trace (§3.1).
                "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (_now(), f"{thread_id}:{_STAGE_P2}", thread_id, _STAGE_P2, "ok", run_id,
                 json.dumps({"bp_id": bp_id, "beats": len(beat_plan.get("beats", []))})),
            )
    except Exception as exc:
        logger.warning("journal write failed for %s P2: %s", thread_id, exc)

    logger.info("Pass 2 complete: %s → %s (%d beats)", thread_id, bp_id, len(beat_plan.get("beats", [])))
    return beat_plan


def run_pass2_batch(
    *,
    catalog: Path,
    state: Path,
    journal_db: Path,
    limit: int | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    threads_dir = catalog / "narrative_threads"
    thread_files = sorted(threads_dir.glob("thr_*.json"))
    if not thread_files:
        logger.warning("No thread files in %s", threads_dir)
        return []

    threads: list[dict] = []
    for tf in thread_files:
        try:
            threads.append(json.loads(tf.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.error("Skipping corrupt thread %s: %s", tf, exc)

    threads.sort(key=lambda t: t.get("scores", {}).get("composite", 0.0), reverse=True)
    if limit is not None:
        threads = threads[:limit]

    # Load existing thread→bp mapping
    index_path = catalog / "beat_plans" / "_thread_index.json"
    try:
        index: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    except Exception:
        index = {}

    results: list[dict] = []
    for t in threads:
        thread_id = t.get("thread_id")
        if not thread_id:
            continue
        if not (catalog / "narrative_briefs" / f"{thread_id}.json").exists():
            logger.warning("No narrative_brief for %s — skipping Pass 2", thread_id)
            continue
        if not force and thread_id in index:
            logger.info("Beat plan already exists for %s (force=False)", thread_id)
            continue
        try:
            bp = run_pass2(thread_id, catalog=catalog, state=state, journal_db=journal_db)
            results.append(bp)
            # refresh index from disk after each write
            try:
                index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
            except Exception:
                pass
        except Exception as exc:
            logger.error("Pass 2 failed for %s: %s", thread_id, exc)

    return results

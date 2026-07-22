"""Stage E1 — event narrative-signal re-tag (PHASE_NARRATIVE_CODE_BRIEF.md §1).

Runs DeepSeek Flash over every event summary, assigns hook_potential,
quote_potential, and narrative_role_suggestion per event.

Overlay-not-rewrite: writes to catalog/event_signals/{media_id}.jsonl.
Never touches catalog/events/.

Manifest: letter="E1", tier="cpu", consumes_hard=["event@*"],
          consumes_soft=["identity_overlay@*"], produces=["event_signals"],
          idempotent=True, stage_version="1.0.0".
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .atomicio import read_jsonl, write_tmp_rename
from .contracts import validate

logger = logging.getLogger(__name__)

_STAGE = "E1"
_STAGE_VERSION = "1.0.0"
_PRODUCED_BY = "e1_signals.py"
_FLASH_MODEL = "deepseek-v4-flash"
_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
_FLASH_TIMEOUT = 60

_SYSTEM_PROMPT = """\
Score a documentary dialogue moment for short-form social media potential.

Return ONLY this exact JSON — no other keys, no markdown, no prose:
{"hook_potential": 0.0, "quote_potential": 0.0, "narrative_role_suggestion": "body", "rationale": "reason"}

Field rules (ALL are required):
- hook_potential: float 0.0–1.0. Measures first-2-second grab (surprise, question, bold claim, visual/comedic punch). Most moments score below 0.3.
- quote_potential: float 0.0–1.0. Measures whether the line is self-contained, memorable, shareable. Most moments score below 0.3.
- narrative_role_suggestion: EXACTLY one of: "hook", "body", "quote", "b_roll" — no other values.
- rationale: ≤120 character string explaining the scores.

Be discriminating — if everything scores 0.7+ the signal is useless. Score low by default.\
"""

_NARRATIVE_ROLES = {"hook", "body", "quote", "b_roll"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_edit_role(raw) -> str:
    if raw is None:
        return "body"
    if isinstance(raw, list):
        return raw[0].lower() if raw else "body"
    return str(raw).lower()


def _flash_call(system: str, user: str) -> str:
    import urllib.request
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    payload = {
        "model": _FLASH_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        _DEEPSEEK_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_FLASH_TIMEOUT) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def _parse_flash_response(text: str) -> dict | None:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        parsed = json.loads(text)
        hp = float(parsed["hook_potential"])
        qp = float(parsed["quote_potential"])
        nrs = parsed["narrative_role_suggestion"]
        if not (0.0 <= hp <= 1.0 and 0.0 <= qp <= 1.0):
            return None
        if nrs not in _NARRATIVE_ROLES:
            return None
        rationale = parsed.get("rationale")
        if rationale and len(rationale) > 120:
            rationale = rationale[:120]
        return {
            "hook_potential": hp,
            "quote_potential": qp,
            "narrative_role_suggestion": nrs,
            "rationale": rationale,
        }
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _journal_event(
    journal_db: Path,
    media_id: str,
    event_type: str,
    run_id: str,
    payload: dict | None,
) -> None:
    """Phase 1 (D-Log-04): funnels into the single outcome writer, `erf.outcomes`.

    Only "ok"/"error" are terminal dispositions for stage E1 here (matching the
    PRE-existing special-case in this function: status_projection was only ever
    updated for ok/error, never for "start" or the per-event "skip" — those are
    per-item diagnostics emitted mid-run, not the stage's final disposition, so
    they route to trace() exactly as before, just via the new events_trace table
    instead of the raw events table). "ok"/"error" -> outcomes.record() (P4: no
    silent swallow — a DB write failure now raises, unlike the try/except this
    function used to wrap everything in)."""
    from . import outcomes
    if event_type in ("ok", "error"):
        reason = None
        if event_type == "error":
            reason = (payload or {}).get("error") or "unknown_error"
        metrics = payload if isinstance(payload, dict) else None
        outcomes.record(media_id, _STAGE, event_type, reason=reason, metrics=metrics,
                         run_id=run_id, db=journal_db)
    else:
        outcomes.trace(media_id, _STAGE, event_type, payload, run_id=run_id, db=journal_db)


def _load_overlay(overlay_path: Path) -> dict | None:
    if not overlay_path.exists():
        return None
    try:
        return json.loads(overlay_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read identity_overlay %s: %s", overlay_path, exc)
        return None


def _person_display_name(person_id: str, persons_dir: Path) -> str:
    person_path = persons_dir / f"{person_id}.json"
    if not person_path.exists():
        return person_id
    try:
        data = json.loads(person_path.read_text(encoding="utf-8"))
        return data.get("display_name", person_id)
    except (json.JSONDecodeError, OSError):
        return person_id


def _speaker_name(
    event: dict,
    overlay: dict | None,
    persons_dir: Path,
    person_cache: dict[str, str],
) -> str:
    if overlay:
        chunk_ids = event.get("chunk_ids")
        first_chunk = chunk_ids[0] if isinstance(chunk_ids, list) and chunk_ids else None
        if first_chunk:
            for entry in overlay.get("chunk_speaker", []):
                if entry.get("chunk_id") == first_chunk:
                    person_id = entry.get("person_id")
                    if person_id:
                        if person_id not in person_cache:
                            person_cache[person_id] = _person_display_name(
                                person_id, persons_dir
                            )
                        return person_cache[person_id]
    label = event.get("speaker_label")
    return label if label else "unknown"


def _null_signal_record(event_id: str, media_id: str) -> dict:
    return {
        "schema_version": _STAGE_VERSION,
        "event_id": event_id,
        "media_id": media_id,
        "hook_potential": None,
        "quote_potential": None,
        "narrative_role_suggestion": "body",
        "rationale": None,
        "produced_by": _PRODUCED_BY,
        "stage_version": _STAGE_VERSION,
    }


# ── per-event scoring ─────────────────────────────────────────────────────────

def _score_event(
    event: dict,
    speaker_name: str,
    media_id: str,
    journal_db: Path,
    run_id: str,
) -> dict:
    ev_id = event.get("event_id", "unknown")
    summary = event.get("summary") or ""
    emotional_beat = event.get("emotional_beat") or ""
    edit_role = _norm_edit_role(event.get("edit_role"))

    if not summary.strip():
        # No content to score — return null-signal without calling Flash
        return _null_signal_record(ev_id, media_id)

    user_prompt = (
        f"Summary: {summary}\n"
        f"Emotional beat: {emotional_beat}\n"
        f"Speaker: {speaker_name}\n"
        f"Edit role: {edit_role}\n\n"
        f"Return JSON: {{\"hook_potential\": <float>, \"quote_potential\": <float>, "
        f"\"narrative_role_suggestion\": <hook|body|quote|b_roll>, \"rationale\": <string>}}"
    )

    try:
        raw = _flash_call(_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.warning("WARN E1 flash_error event=%s: %s", ev_id, exc)
        _journal_event(
            journal_db,
            media_id,
            "skip",
            run_id,
            {"reason": "flash_error", "event_id": ev_id, "error": str(exc)},
        )
        return _null_signal_record(ev_id, media_id)

    parsed = _parse_flash_response(raw)
    if parsed is None:
        logger.warning("WARN E1 parse_error event=%s raw=%s", ev_id, raw[:120])
        _journal_event(
            journal_db,
            media_id,
            "skip",
            run_id,
            {"reason": "parse_error", "event_id": ev_id},
        )
        return _null_signal_record(ev_id, media_id)

    record = {
        "schema_version": _STAGE_VERSION,
        "event_id": ev_id,
        "media_id": media_id,
        "hook_potential": parsed["hook_potential"],
        "quote_potential": parsed["quote_potential"],
        "narrative_role_suggestion": parsed["narrative_role_suggestion"],
        "rationale": parsed["rationale"],
        "produced_by": _PRODUCED_BY,
        "stage_version": _STAGE_VERSION,
    }

    # Validate against schema; on failure fall back to null-signal
    try:
        validate(record, "event_signals")
    except Exception as exc:
        logger.warning("WARN E1 schema_invalid event=%s: %s", ev_id, exc)
        return _null_signal_record(ev_id, media_id)

    return record


# ── main entry point ──────────────────────────────────────────────────────────

def run(
    media_id: str,
    *,
    catalog: Path,
    journal_db: Path,
    run_id: str | None = None,
    max_workers: int = 8,
) -> dict:
    """Score all events for media_id through Flash; write catalog/event_signals/{media_id}.jsonl.

    Returns: {"ok": bool, "records_written": int, "parse_errors": int}
    Idempotent: re-running overwrites the existing signals file.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    _journal_event(journal_db, media_id, "start", run_id, None)

    events_path = catalog / "events" / f"{media_id}.jsonl"
    signals_path = catalog / "event_signals" / f"{media_id}.jsonl"
    overlay_path = catalog / "identity_overlay" / f"{media_id}.json"
    persons_dir = catalog / "persons"

    events = read_jsonl(events_path)
    if not events:
        write_tmp_rename(signals_path, b"\n")
        _journal_event(journal_db, media_id, "ok", run_id, {"records_written": 0, "parse_errors": 0})
        return {"ok": True, "records_written": 0, "parse_errors": 0}

    overlay = _load_overlay(overlay_path)
    person_cache: dict[str, str] = {}
    speaker_names = [
        _speaker_name(ev, overlay, persons_dir, person_cache) for ev in events
    ]

    results: list[dict | None] = [None] * len(events)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _score_event, events[i], speaker_names[i], media_id, journal_db, run_id
            ): i
            for i in range(len(events))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                ev_id = events[idx].get("event_id", "unknown")
                logger.error("ERROR E1 unexpected event=%s: %s", ev_id, exc)
                results[idx] = _null_signal_record(ev_id, media_id)

    # Fill any gaps (defensive)
    for i, r in enumerate(results):
        if r is None:
            ev_id = events[i].get("event_id", "unknown")
            results[i] = _null_signal_record(ev_id, media_id)

    parse_errors = sum(
        1 for r in results if r.get("hook_potential") is None
    )

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in results).encode("utf-8") + b"\n"
    write_tmp_rename(signals_path, content)

    _journal_event(
        journal_db,
        media_id,
        "ok",
        run_id,
        {"records_written": len(results), "parse_errors": parse_errors},
    )
    return {"ok": True, "records_written": len(results), "parse_errors": parse_errors}

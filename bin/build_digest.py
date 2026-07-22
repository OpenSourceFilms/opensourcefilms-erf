#!/usr/bin/env python3
"""Build corpus-level digest artifacts for the ERF narrative pipeline.

Produces:
  - catalog/event_digest.jsonl      (one line per event, all media)
  - catalog/media_cards/{mid}.json  (one Flash-generated card per media)

Run:
  python bin/build_digest.py [--catalog PATH] [--skip-cards] [--only-cards] [--force]

Idempotent by default: digest is skipped when newer than all event files;
cards are skipped when their JSON already exists.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# ── constants ─────────────────────────────────────────────────────────────────

_FLASH_URL = "https://api.deepseek.com/v1/chat/completions"
_FLASH_MODEL = "deepseek-v4-flash"
_FLASH_TIMEOUT = 45
_SCHEMA_VERSION = "1.0.0"
_PRODUCED_BY = "build_digest.py"

# ── atomic I/O ────────────────────────────────────────────────────────────────

def _atomic_write_json(target: Path, data: object) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".tmp{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _atomic_write_jsonl(target: Path, rows: list[dict]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".tmp{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ── data loading ──────────────────────────────────────────────────────────────

def _load_all_events(catalog: Path) -> dict[str, list[dict]]:
    """Return {media_id: [event, ...]} from catalog/events/."""
    d = catalog / "events"
    if not d.exists():
        return {}
    result: dict[str, list[dict]] = {}
    for f in sorted(d.glob("*.jsonl")):
        evts = _read_jsonl(f)
        if evts:
            result[f.stem] = evts
    return result


def _load_signals(catalog: Path) -> dict[str, dict[str, dict]]:
    """Return {media_id: {event_id: {hook, quote, role}}} from event_signals/."""
    d = catalog / "event_signals"
    if not d.exists():
        return {}
    result: dict[str, dict[str, dict]] = {}
    for f in sorted(d.glob("*.jsonl")):
        mid = f.stem
        sig_map: dict[str, dict] = {}
        for row in _read_jsonl(f):
            eid = row.get("event_id")
            if eid:
                sig_map[eid] = {
                    "hook_potential": row.get("hook_potential"),
                    "quote_potential": row.get("quote_potential"),
                    "narrative_role_suggestion": row.get("narrative_role_suggestion"),
                }
        if sig_map:
            result[mid] = sig_map
    return result


def _load_overlay(catalog: Path, media_id: str) -> dict[str, str]:
    """Return {chunk_id: person_id} from identity_overlay/{media_id}.json.
    Entries where person_id is None are excluded.
    """
    path = catalog / "identity_overlay" / f"{media_id}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result: dict[str, str] = {}
    for entry in data.get("chunk_speaker", []):
        cid = entry.get("chunk_id")
        pid = entry.get("person_id")
        if cid and pid:
            result[cid] = pid
    return result


_person_cache: dict[str, str] = {}


def _person_display_name(catalog: Path, person_id: str) -> str:
    if person_id in _person_cache:
        return _person_cache[person_id]
    path = catalog / "persons" / f"{person_id}.json"
    name = person_id
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("display_name") or person_id
        except Exception:
            pass
    _person_cache[person_id] = name
    return name


def _resolve_speaker(
    event: dict,
    chunk_to_person: dict[str, str],
    catalog: Path,
) -> str:
    """Resolve event's speaker to a display_name via identity_overlay.

    Uses event.chunk_ids[0] -> chunk_to_person -> display_name.
    Falls back to event.speaker_label, then 'unknown'.
    """
    chunk_ids = event.get("chunk_ids") or []
    if chunk_ids:
        first_cid = chunk_ids[0]
        pid = chunk_to_person.get(first_cid)
        if pid:
            return _person_display_name(catalog, pid)
    label = event.get("speaker_label")
    return label.strip() if label and label.strip() else "unknown"


# ── Flash API ─────────────────────────────────────────────────────────────────

def _flash_call(system: str, user: str) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    payload = {
        "model": _FLASH_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _FLASH_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_FLASH_TIMEOUT) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"].strip()


# ── event_digest ──────────────────────────────────────────────────────────────

def _load_shoot_contexts(catalog: Path) -> dict[str, str]:
    """Return {media_id: source_note} from media_manifest/ for shoot_context field."""
    result: dict[str, str] = {}
    manifest_dir = catalog / "media_manifest"
    if not manifest_dir.exists():
        return result
    for f in manifest_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            note = (d.get("source_note") or "").strip()
            if note:
                result[d["media_id"]] = note
        except Exception:
            pass
    return result


def _load_visual_grounding(catalog: Path) -> dict[str, dict]:
    """Return {media_id: visual_grounding_record} from catalog/visual_grounding/."""
    result: dict[str, dict] = {}
    vg_dir = catalog / "visual_grounding"
    if not vg_dir.exists():
        return result
    for f in vg_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            mid = d.get("media_id", f.stem)
            result[mid] = d
        except Exception:
            pass
    return result


def _vis_tag(vg: dict) -> str:
    """Compact visual tag for event_digest lines: [vis: footage_type | shot_type | subjects]"""
    ft = vg.get("footage_type", "")
    st = vg.get("shot_type") or ""
    subjects = vg.get("on_screen_subjects", [])
    top_subj = ", ".join(subjects[:3]) if subjects else ""
    parts = [ft]
    if st:
        parts.append(st)
    if top_subj:
        parts.append(top_subj)
    return "[vis: " + " | ".join(parts) + "]"


def build_event_digest(catalog: Path, force: bool = False) -> int:
    """Write catalog/event_digest.jsonl. Returns events written (0 if skipped)."""
    out_path = catalog / "event_digest.jsonl"

    if not force and out_path.exists():
        events_dir = catalog / "events"
        if events_dir.exists():
            newest = max((f.stat().st_mtime for f in events_dir.glob("*.jsonl")), default=0.0)
            if out_path.stat().st_mtime >= newest:
                existing = _read_jsonl(out_path)
                print(f"[INFO] event_digest.jsonl up-to-date — {len(existing)} events, skipping.")
                return len(existing)

    events_by_media = _load_all_events(catalog)
    signals_by_media = _load_signals(catalog)
    shoot_contexts = _load_shoot_contexts(catalog)
    visual_groundings = _load_visual_grounding(catalog)

    rows: list[dict] = []
    for media_id, events in events_by_media.items():
        chunk_to_person = _load_overlay(catalog, media_id)
        media_sigs = signals_by_media.get(media_id, {})
        vg = visual_groundings.get(media_id)
        for ev in events:
            eid = ev.get("event_id")
            if not eid:
                continue
            summary = ev.get("summary") or ""
            sig = media_sigs.get(eid, {})
            hook = sig.get("hook_potential")
            quote = sig.get("quote_potential")
            role = sig.get("narrative_role_suggestion")

            # Skip truly empty records (no text, no signal)
            if not summary.strip() and hook is None:
                continue

            t_s = (ev.get("start_ms") or 0) / 1000.0
            end_ms = ev.get("end_ms") or ev.get("start_ms") or 0
            duration_s = (end_ms - (ev.get("start_ms") or 0)) / 1000.0
            speaker_name = _resolve_speaker(ev, chunk_to_person, catalog)

            row: dict = {
                "event_id": eid,
                "media_id": media_id,
                "t_s": t_s,
                "duration_s": duration_s,
                "emotional_beat": ev.get("emotional_beat"),
                "speaker_name": speaker_name,
                "hook_potential": hook,
                "quote_potential": quote,
                "narrative_role_suggestion": role,
                "summary": summary,
                "shoot_context": shoot_contexts.get(media_id, ""),
            }
            if vg:
                row["vis_tag"] = _vis_tag(vg)
                row["vis_footage_type"] = vg.get("footage_type")
                row["vis_on_screen_subjects"] = vg.get("on_screen_subjects", [])
            rows.append(row)

    rows.sort(key=lambda r: (r["media_id"], r["t_s"]))
    _atomic_write_jsonl(out_path, rows)
    print(f"[INFO] event_digest.jsonl written — {len(rows)} events.")
    return len(rows)


# ── media_cards ───────────────────────────────────────────────────────────────

_CARD_SYSTEM = (
    "You are a documentary footage analyst. You will receive a list of events "
    "from a single video recording. Your task: write ONE specific sentence that "
    "captures what makes THIS recording distinctive — not what is generally true "
    "about documentary footage, but what is concrete and specific to THIS recording. "
    "Name the real subject, name the real action, name what is memorable or surprising. "
    "Then write ONE more sentence on the dominant emotional register. 2 sentences total. "
    "If the recording has no memorable moments, say so explicitly. "
    "Do NOT use generic phrases like 'person discusses topics' or 'discusses various subjects'. "
    "Be specific about what actually happens."
)


def _emotional_tenor(events: list[dict]) -> str:
    _tenor_map = {
        "comedic": "comedic",
        "tension": "tense",
        "suspense": "tense",
        "dramatic": "dramatic",
        "action": "dramatic",
        "informational": "informational",
        "exposition": "informational",
        "reveal": "dramatic",
    }
    counts: dict[str, int] = {}
    for ev in events:
        beat = (ev.get("emotional_beat") or "").lower().strip()
        if beat:
            tenor = _tenor_map.get(beat, "mixed")
            counts[tenor] = counts.get(tenor, 0) + 1
    if not counts:
        return "informational"
    top = max(counts.values())
    winners = [t for t, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else "mixed"


def _build_one_card(
    media_id: str,
    events: list[dict],
    speaker_names: list[str],
    catalog: Path,
    visual_grounding: Optional[dict] = None,
) -> dict:
    """Build a media_card dict for one media. Flash call may raise."""
    # beat breakdown
    beat_counts: dict[str, int] = {}
    for ev in events:
        b = (ev.get("emotional_beat") or "").strip()
        if b:
            beat_counts[b] = beat_counts.get(b, 0) + 1
    beat_str = ", ".join(
        f"{k}: {v}" for k, v in sorted(beat_counts.items(), key=lambda x: -x[1])
    ) or "none"

    # dominant speakers (by freq, skip unknown, max 3)
    speaker_freq: dict[str, int] = {}
    for name in speaker_names:
        if name and name != "unknown":
            speaker_freq[name] = speaker_freq.get(name, 0) + 1
    dominant = [n for n, _ in sorted(speaker_freq.items(), key=lambda x: -x[1])][:3]

    # standout events (hook_potential > 0.2, top 3)
    hooks = [(ev["event_id"], ev.get("hook_potential") or 0.0) for ev in events]
    hooks.sort(key=lambda x: -x[1])
    standout = [eid for eid, h in hooks if h > 0.2][:3]

    # flash prompt events (up to 20 with summaries)
    prompt_events = [ev for ev in events if (ev.get("summary") or "").strip()][:20]
    event_lines = "\n".join(
        f"[{ev['t_s']:.0f}s] {ev.get('emotional_beat','?')}: {(ev.get('summary') or '')[:100]}"
        for ev in prompt_events
    ) or "No events with summaries."

    unique_speakers = list(dict.fromkeys(s for s in speaker_names if s != "unknown"))
    speakers_joined = ", ".join(unique_speakers) or "unknown"

    user_prompt = (
        f"Recording: {media_id}\n"
        f"Speakers: {speakers_joined}\n"
        f"Event count: {len(events)}\n"
        f"Emotional beats breakdown: {beat_str}\n\n"
        f"Events (in order):\n{event_lines}\n\n"
        "Write your 2-sentence analysis."
    )

    try:
        synopsis = _flash_call(_CARD_SYSTEM, user_prompt)
    except Exception as exc:
        print(f"[WARN] Flash error for {media_id}: {exc}")
        synopsis = "[flash error]"

    card: dict = {
        "schema_version": _SCHEMA_VERSION,
        "media_id": media_id,
        "synopsis": synopsis,
        "dominant_speakers": dominant,
        "emotional_tenor": _emotional_tenor(events),
        "event_count": len(events),
        "standout_event_ids": standout,
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": _PRODUCED_BY,
    }
    if visual_grounding:
        card["visual_grounding"] = {
            "footage_type": visual_grounding.get("footage_type"),
            "setting": visual_grounding.get("setting"),
            "on_screen_subjects": visual_grounding.get("on_screen_subjects", []),
            "shot_type": visual_grounding.get("shot_type"),
            "visual_summary": visual_grounding.get("visual_summary", ""),
        }
    return card


def build_all_media_cards(
    catalog: Path, max_workers: int = 8, force: bool = False
) -> dict[str, int]:
    """Build catalog/media_cards/{media_id}.json for all media with ≥1 event.

    Returns: {'built': N, 'skipped': N, 'error': N}
    """
    events_by_media = _load_all_events(catalog)
    signals_by_media = _load_signals(catalog)
    visual_groundings = _load_visual_grounding(catalog)
    cards_dir = catalog / "media_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from erf.runlog import log_event as _log_event
        from erf import outcomes as _outcomes
    except Exception:
        _log_event = None
        _outcomes = None

    to_build: list[tuple[str, list[dict], list[str], Optional[dict]]] = []
    skipped = 0

    for media_id, raw_events in events_by_media.items():
        card_path = cards_dir / f"{media_id}.json"
        if not force and card_path.exists():
            skipped += 1
            continue

        chunk_to_person = _load_overlay(catalog, media_id)
        media_sigs = signals_by_media.get(media_id, {})

        filtered: list[dict] = []
        names: list[str] = []
        for ev in raw_events:
            eid = ev.get("event_id")
            if not eid:
                continue
            summary = ev.get("summary") or ""
            sig = media_sigs.get(eid, {})
            hook = sig.get("hook_potential")
            if not summary.strip() and hook is None:
                continue
            t_s = (ev.get("start_ms") or 0) / 1000.0
            filtered.append({
                "event_id": eid,
                "t_s": t_s,
                "emotional_beat": ev.get("emotional_beat"),
                "hook_potential": hook,
                "summary": summary,
            })
            names.append(_resolve_speaker(ev, chunk_to_person, catalog))

        if not filtered:
            if _log_event:  # in-scope candidate but no card-worthy events — journal the skip
                _log_event("CARD", media_id, "skip", {"reason": "no card-worthy events"})
            if _outcomes:
                _outcomes.record(media_id, "CARD", "skip", reason="no_card_worthy_events")
                _outcomes.set_flag(media_id, "has_card", False, reason="no_card_worthy_events")
            continue
        to_build.append((media_id, filtered, names, visual_groundings.get(media_id)))

    print(f"[INFO] Media cards: {len(to_build)} to build, {skipped} already exist.")

    stats = {"built": 0, "skipped": skipped, "error": 0}

    def _task(args):
        mid, evts, sp_names, vg = args
        card = _build_one_card(mid, evts, sp_names, catalog, visual_grounding=vg)
        _atomic_write_json(cards_dir / f"{mid}.json", card)
        return mid

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from erf.runlog import log_event as _log_event
        from erf import outcomes as _outcomes
    except Exception:
        _log_event = None
        _outcomes = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, args): args[0] for args in to_build}
        for fut in concurrent.futures.as_completed(futures):
            mid = futures[fut]
            try:
                fut.result()
                stats["built"] += 1
                print(f"  ok  {mid}")
                if _log_event:
                    _log_event("CARD", mid, "ok")
                if _outcomes:
                    _outcomes.record(mid, "CARD", "ok")
                    _outcomes.set_flag(mid, "has_card", True)
            except Exception as exc:
                print(f"  ERR {mid}: {exc}")
                stats["error"] += 1
                if _log_event:
                    _log_event("CARD", mid, "error", {"error": str(exc)[:300]})
                if _outcomes:
                    _outcomes.record(mid, "CARD", "error", reason=str(exc)[:200])

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build ERF corpus digest artifacts.")
    ap.add_argument("--catalog", default="/workspace/erf/catalog")
    ap.add_argument("--skip-cards", action="store_true", help="Build digest only.")
    ap.add_argument("--only-cards", action="store_true", help="Build cards only.")
    ap.add_argument("--force", action="store_true", help="Rebuild even if files exist.")
    args = ap.parse_args()

    catalog = Path(args.catalog)

    if not args.only_cards:
        print("[1/2] Building event_digest.jsonl ...")
        n = build_event_digest(catalog, force=args.force)
        print(f"      {n} events in digest.")

    if not args.skip_cards:
        print("[2/2] Building media_cards/ ...")
        stats = build_all_media_cards(catalog, force=args.force)
        print(f"      built={stats['built']} skipped={stats['skipped']} error={stats['error']}")

    print("Done.")


if __name__ == "__main__":
    main()

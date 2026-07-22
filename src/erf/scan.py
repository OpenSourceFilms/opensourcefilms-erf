"""Stage E2 — Autonomous corpus scan. Hook-seeded, semantically-grown, arc-scored thread discovery."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import warnings

import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from sklearn.cluster import HDBSCAN

from .atomicio import read_jsonl, write_tmp_rename

logger = logging.getLogger(__name__)

EMOTIONAL_WEIGHT: dict[str, float] = {
    "reveal": 1.0, "tension": 0.8, "comedic": 0.6,
    "action": 0.5, "exposition": 0.2, "setup": 0.3,
    "payoff": 0.7, "release": 0.6,
}
_DEFAULT_WEIGHT = 0.3
IDENTIFIED_SPEAKER_BONUS = 0.2
TARGET_DURATION_S = 60.0
_STAGE = "E2"

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "it", "its",
    "they", "he", "she", "we", "i", "you", "about", "from", "as", "into",
    "up", "out", "so", "if", "not", "no", "also", "then", "than", "more",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _journal(journal_db: Path, run_id: str, event_type: str, payload: dict | None = None) -> None:
    try:
        with sqlite3.connect(str(journal_db)) as conn:
            conn.execute(
                # Phase 1 (D-Log-04): corpus-scan trace -> events_trace (§3.1).
                "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (_now(), f"scan:{run_id}", "corpus", _STAGE, event_type, run_id,
                 json.dumps(payload) if payload is not None else None),
            )
    except Exception as exc:
        logger.warning("journal write failed: %s", exc)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _arc_completeness(beats: list[str | None]) -> float:
    clean = [b for b in beats if b]
    if not clean:
        return 0.0
    unique = set(clean)
    has_setup = bool(unique & {"setup", "exposition"})
    has_peak = bool(unique & {"tension", "action", "reveal"})
    has_release = bool(unique & {"payoff", "release", "comedic"})
    if has_setup and has_peak and has_release:
        return 1.0
    if (has_setup and has_peak) or (has_peak and has_release):
        return 0.8
    if has_setup and has_release:
        return 0.7
    if has_peak:
        return 0.5
    if has_setup or has_release:
        return 0.3
    return 0.1


def build_event_embeddings(
    media_id: str,
    catalog: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Mean-pool chunk text embeddings → per-event vector. Reads from catalog/embeddings/{media_id}.text.jsonl."""
    cache_path = catalog / "event_embeddings" / f"{media_id}.jsonl"
    if not force and cache_path.exists():
        return read_jsonl(cache_path)

    events_path = catalog / "events" / f"{media_id}.jsonl"
    chunk_emb_path = catalog / "embeddings" / f"{media_id}.text.jsonl"
    if not events_path.exists() or not chunk_emb_path.exists():
        return []

    chunk_vecs: dict[str, np.ndarray] = {}
    for row in read_jsonl(chunk_emb_path):
        cid = row.get("source_id")
        vec = row.get("vector")
        if cid and vec:
            chunk_vecs[cid] = np.array(vec, dtype=np.float32)

    results: list[dict[str, Any]] = []
    for evt in read_jsonl(events_path):
        chunk_ids = evt.get("chunk_ids") or []
        if not chunk_ids or not (evt.get("summary") or "").strip():
            continue
        vecs = [chunk_vecs[cid] for cid in chunk_ids if cid in chunk_vecs]
        if not vecs:
            continue
        mean_vec: np.ndarray = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        results.append({
            "event_id": evt["event_id"],
            "media_id": media_id,
            "modality": "text",
            "model_tag": "qwen3-8b",
            "sources": list(dict.fromkeys(chunk_ids)),  # deduplicated, order-preserving
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "vector": mean_vec.tolist(),
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
    write_tmp_rename(cache_path, (lines + "\n").encode("utf-8") if lines else b"\n")
    return results


def load_all_embeddings(
    catalog: Path,
    force: bool = False,
) -> dict[str, np.ndarray]:
    emb_by_id: dict[str, np.ndarray] = {}
    for fpath in sorted((catalog / "events").glob("*.jsonl")):
        media_id = fpath.stem
        for rec in build_event_embeddings(media_id, catalog, force=force):
            emb_by_id[rec["event_id"]] = np.array(rec["vector"], dtype=np.float32)
    return emb_by_id


def _load_identity_overlay(catalog: Path, media_id: str) -> dict[str, str]:
    """Return {chunk_id: person_id} for this media."""
    overlay_path = catalog / "identity_overlay" / f"{media_id}.json"
    if not overlay_path.exists():
        return {}
    try:
        data = json.loads(overlay_path.read_text(encoding="utf-8"))
        return {entry["chunk_id"]: entry["person_id"]
                for entry in data.get("chunk_speaker", [])
                if entry.get("chunk_id") and entry.get("person_id")}
    except Exception:
        return {}


def load_all_events_and_signals(
    catalog: Path,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Return (events_by_id merged with signals, speaker_names by event_id)."""
    events_by_id: dict[str, dict] = {}

    for fpath in sorted((catalog / "events").glob("*.jsonl")):
        media_id = fpath.stem
        signals: dict[str, dict] = {}
        sig_path = catalog / "event_signals" / f"{media_id}.jsonl"
        if sig_path.exists():
            for sig in read_jsonl(sig_path):
                signals[sig["event_id"]] = sig

        chunk_to_person = _load_identity_overlay(catalog, media_id)

        for evt in read_jsonl(fpath):
            eid = evt["event_id"]
            merged = dict(evt)
            sig = signals.get(eid, {})
            merged["hook_potential"] = sig.get("hook_potential")
            merged["quote_potential"] = sig.get("quote_potential")
            merged["narrative_role_suggestion"] = sig.get("narrative_role_suggestion")

            # Dominant speaker from identity_overlay
            chunk_ids = evt.get("chunk_ids") or []
            person_counts: dict[str, int] = {}
            for cid in chunk_ids:
                pid = chunk_to_person.get(cid)
                if pid:
                    person_counts[pid] = person_counts.get(pid, 0) + 1
            merged["_dominant_person"] = max(person_counts, key=person_counts.get) if person_counts else None

            events_by_id[eid] = merged

    speaker_names = {eid: (ev.get("_dominant_person") or "") for eid, ev in events_by_id.items()}
    return events_by_id, speaker_names


def _theme_label(events_by_id: dict, thread_ids: list[str], seed_eid: str) -> str | None:
    """Short phrase from seed event summary, stop-word filtered."""
    summary = events_by_id[seed_eid].get("summary") or ""
    words = [w.strip(".,;:!?\"'()") for w in summary.split()]
    meaningful = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2]
    return " ".join(meaningful[:5]) if meaningful else None


def run(
    *,
    catalog: Path,
    lance_path: Path,  # kept for interface compatibility; embeddings read from catalog/embeddings
    journal_db: Path,
    limit: int = 20,
    emotional_filter: str | None = None,
    platform: str = "youtube_shorts",
    force_rebuild_embeddings: bool = False,
) -> list[dict[str, Any]]:
    run_id = uuid.uuid4().hex[:12]
    _journal(journal_db, run_id, "start")

    try:
        embeddings = load_all_embeddings(catalog, force=force_rebuild_embeddings)
        events_by_id, speaker_names = load_all_events_and_signals(catalog)

        valid_eids = [
            eid for eid in events_by_id
            if eid in embeddings
            and events_by_id[eid].get("hook_potential") is not None
            and (events_by_id[eid].get("summary") or "").strip()
        ]

        if len(valid_eids) < 3:
            _journal(journal_db, run_id, "ok", {"threads": 0, "valid_events": len(valid_eids)})
            return []

        X = np.stack([embeddings[eid] for eid in valid_eids])
        labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean").fit_predict(X)
        cluster_map = {eid: int(lbl) for eid, lbl in zip(valid_eids, labels)}

        # Seed ranking
        seed_candidates = []
        for eid in valid_eids:
            evt = events_by_id[eid]
            hook = float(evt.get("hook_potential") or 0.0)
            beat = evt.get("emotional_beat") or ""
            w = EMOTIONAL_WEIGHT.get(beat, _DEFAULT_WEIGHT)
            speaker_mult = 1.0 + IDENTIFIED_SPEAKER_BONUS if speaker_names.get(eid) else 1.0
            seed_candidates.append((hook * w * speaker_mult, eid))

        seed_candidates.sort(key=lambda x: x[0], reverse=True)
        if emotional_filter:
            seed_candidates = [
                (s, eid) for s, eid in seed_candidates
                if events_by_id[eid].get("emotional_beat") == emotional_filter
            ]

        valid_set = set(valid_eids)
        raw_threads: list[dict[str, Any]] = []

        for _, seed_eid in seed_candidates[: 3 * limit]:
            thread_ids: list[str] = [seed_eid]
            thread_vecs: list[np.ndarray] = [embeddings[seed_eid]]
            seed_media = events_by_id[seed_eid]["media_id"]
            seed_speaker = speaker_names.get(seed_eid, "")

            while len(thread_ids) < 8:
                centroid = np.mean(thread_vecs, axis=0)
                best_eid: str | None = None
                best_score = -1.0

                for eid in valid_eids:
                    if eid in thread_ids:
                        continue
                    sim = _cosine(embeddings[eid], centroid)
                    bonus = 0.0
                    if events_by_id[eid]["media_id"] == seed_media:
                        bonus += 0.15
                    if seed_speaker and speaker_names.get(eid) == seed_speaker:
                        bonus += 0.10
                    # reward beat variety
                    current_beats = {events_by_id[t].get("emotional_beat") for t in thread_ids}
                    cand_beat = events_by_id[eid].get("emotional_beat")
                    if cand_beat and cand_beat not in current_beats:
                        bonus += 0.10

                    score = sim + bonus
                    if score > best_score:
                        best_score = score
                        best_eid = eid

                if best_eid is None or best_score <= 0.30:
                    break
                thread_ids.append(best_eid)
                thread_vecs.append(embeddings[best_eid])

                # Arc-complete: setup + peak + resolution with ≥5 events
                if len(thread_ids) >= 5:
                    beats = [events_by_id[t].get("emotional_beat") for t in thread_ids]
                    if _arc_completeness(beats) >= 1.0:
                        break

            if len(thread_ids) < 3:
                continue

            # Score
            member_vecs = [embeddings[eid] for eid in thread_ids]
            n = len(member_vecs)
            pw_sims = [_cosine(member_vecs[i], member_vecs[j]) for i in range(n) for j in range(i + 1, n)]
            coherence = max(0.0, min(1.0, float(np.mean(pw_sims)) if pw_sims else 0.0))
            beats = [events_by_id[eid].get("emotional_beat") for eid in thread_ids]
            arc = _arc_completeness(beats)
            hook_s = max(float(events_by_id[eid].get("hook_potential") or 0.0) for eid in thread_ids)
            quote_s = max(float(events_by_id[eid].get("quote_potential") or 0.0) for eid in thread_ids)
            broll_s = sum(1 for eid in thread_ids if events_by_id[eid].get("b_roll_flag")) / n
            est_dur = sum(
                int(events_by_id[eid].get("end_ms", 0)) - int(events_by_id[eid].get("start_ms", 0))
                for eid in thread_ids
            ) / 1000.0
            dur_fit = 1.0 - min(1.0, abs(est_dur - TARGET_DURATION_S) / TARGET_DURATION_S)
            composite = (0.25 * hook_s + 0.25 * arc + 0.20 * coherence +
                         0.15 * quote_s + 0.10 * broll_s + 0.05 * dur_fit)

            persons = [speaker_names[eid] for eid in thread_ids if speaker_names.get(eid)]
            dom_person = max(set(persons), key=persons.count) if persons else None

            raw_threads.append({
                "schema_version": "1.0.0",
                "thread_id": "thr_" + uuid.uuid4().hex[:12],
                "seed_event_id": seed_eid,
                "member_event_ids": thread_ids,
                "arc_shape": [b or "unknown" for b in beats],
                "dominant_person_id": dom_person,
                "theme_label": _theme_label(events_by_id, thread_ids, seed_eid),
                "cluster_id": cluster_map.get(seed_eid, -1),
                "scores": {
                    "composite": round(composite, 4),
                    "hook": round(hook_s, 4),
                    "arc": round(arc, 4),
                    "coherence": round(coherence, 4),
                    "quote": round(quote_s, 4),
                    "broll": round(broll_s, 4),
                    "duration_fit": round(dur_fit, 4),
                },
                "est_duration_s": round(est_dur, 2),
                "created_at_utc": _now(),
            })

        # Dedupe by Jaccard > 0.5, keep highest composite
        raw_threads.sort(key=lambda t: t["scores"]["composite"], reverse=True)
        kept: list[dict[str, Any]] = []
        for t in raw_threads:
            members_a = set(t["member_event_ids"])
            if any(_jaccard(members_a, set(k["member_event_ids"])) > 0.5 for k in kept):
                continue
            kept.append(t)
            if len(kept) == limit:
                break

        out_dir = catalog / "narrative_threads"
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in kept:
            fpath = out_dir / f"{t['thread_id']}.json"
            write_tmp_rename(fpath, json.dumps(t, indent=2, ensure_ascii=False).encode("utf-8"))

        _journal(journal_db, run_id, "ok", {"threads": len(kept), "valid_events": len(valid_eids)})
        return kept

    except Exception as exc:
        logger.exception("E2 scan failed")
        _journal(journal_db, run_id, "error", {"error": str(exc)})
        raise

"""Stage F1 — identity resolution + overlay (control, no GPU).

Reads face_tracks + diar turns + chunks + keyframes + events,
matches tracks against persons_v1 gallery, runs bootstrap speaker assignment,
writes catalog/identity_overlay/{media_id}.json (atomic write).
"""
from __future__ import annotations

import json
import os
import tempfile
import tomllib
import uuid
from pathlib import Path

from .atomicio import read_jsonl
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULT_FLOOR = 0.38
_DEFAULT_MARGIN = 0.10
_BOOTSTRAP_DOMINANCE = 0.60
_BOOTSTRAP_MARGIN = 0.25


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "rb") as fh:
        return tomllib.load(fh)


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    a_arr, b_arr = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _face_assignments(tracks: list[dict], floor: float, margin: float,
                      manual_face_links: dict[str, str]) -> list[dict]:
    """Match each face_track against persons_v1 gallery. Returns face_assignments list."""
    try:
        import lancedb
        db = lancedb.connect("index/lance")
        if "persons_v1" not in db.table_names():
            # No gallery — all null
            return [
                {"face_track_id": t["face_track_id"], "person_id": None,
                 "match_score": None, "method": "none"}
                for t in tracks
            ]
        tbl = db.open_table("persons_v1")
        gallery_rows = tbl.search().limit(10000).to_list()
    except Exception:
        return [
            {"face_track_id": t["face_track_id"], "person_id": None,
             "match_score": None, "method": "none"}
            for t in tracks
        ]

    # Group gallery by person_id
    by_person: dict[str, list[list[float]]] = {}
    display_names: dict[str, str] = {}
    for row in gallery_rows:
        pid = row["person_id"]
        by_person.setdefault(pid, []).append(row["vector"])
        display_names[pid] = row.get("display_name", pid)

    results = []
    for tr in tracks:
        ft_id = tr["face_track_id"]

        # Manual override
        if ft_id in manual_face_links:
            results.append({
                "face_track_id": ft_id,
                "person_id": manual_face_links[ft_id],
                "match_score": None,
                "method": "manual",
            })
            continue

        emb = tr.get("embedding")
        if not emb or not by_person:
            results.append({"face_track_id": ft_id, "person_id": None,
                            "match_score": None, "method": "none"})
            continue

        # Max-cosine per person
        scores: list[tuple[float, str]] = []
        for pid, templates in by_person.items():
            max_cos = max(_cosine(emb, tmpl) for tmpl in templates)
            scores.append((max_cos, pid))
        scores.sort(reverse=True)

        top1_score, top1_pid = scores[0]
        top2_score = scores[1][0] if len(scores) > 1 else 0.0

        if top1_score >= floor and (top1_score - top2_score) >= margin:
            results.append({
                "face_track_id": ft_id,
                "person_id": top1_pid,
                "match_score": round(top1_score, 4),
                "method": "gallery",
            })
        else:
            results.append({
                "face_track_id": ft_id,
                "person_id": None,
                "match_score": round(top1_score, 4),
                "method": "none",
            })

    return results


def _speaker_assignments(diar_turns: list[dict], face_assigns: list[dict],
                          tracks: list[dict], manual_links: list[dict],
                          media_id: str) -> list[dict]:
    """Bootstrap speaker→person assignment per D4. Returns speaker_assignments list."""
    # Build track span → person_id lookup
    track_by_id = {t["face_track_id"]: t for t in tracks}
    ft_person = {fa["face_track_id"]: fa["person_id"] for fa in face_assigns if fa["person_id"]}

    distinct_labels = sorted({t["speaker_label"] for t in diar_turns})
    results = []

    for label in distinct_labels:
        label_turns = [t for t in diar_turns if t["speaker_label"] == label and not t.get("overlap", False)]
        visible: dict[str, float] = {}

        for turn in label_turns:
            t_start, t_end = turn["start_ms"], turn["end_ms"]
            for ft_id, pid in ft_person.items():
                tr = track_by_id.get(ft_id)
                if not tr:
                    continue
                ov = _overlap_ms(t_start, t_end, tr["start_ms"], tr["end_ms"])
                if ov > 0:
                    visible[pid] = visible.get(pid, 0.0) + ov / 1000.0

        total = sum(visible.values())
        assigned_pid = None
        confidence = None
        method = "none"
        evidence = []
        evidence_seconds = 0.0  # absolute face-overlap seconds for the assigned person

        if total > 0:
            ranked = sorted(visible.items(), key=lambda x: x[1], reverse=True)
            top_pid, top_s = ranked[0]
            next_s = ranked[1][1] if len(ranked) > 1 else 0.0
            if top_s >= _BOOTSTRAP_DOMINANCE * total and next_s <= _BOOTSTRAP_MARGIN * total:
                assigned_pid = top_pid
                confidence = round(top_s / total, 3)
                evidence_seconds = round(top_s, 1)
                method = "bootstrap"
                evidence = [ft_id for ft_id, pid in ft_person.items() if pid == top_pid]

        results.append({
            "speaker_label": label,
            "person_id": assigned_pid,
            "method": method,
            "confidence": confidence,
            "evidence_seconds": evidence_seconds,
            "evidence": evidence,
        })

    # De-dup ambiguous face over-claim (D-Identity-01): a person can be the FACE-bootstrapped
    # speaker for at most ONE label. When the camera stays on one person while another speaks
    # off-camera, face-overlap wrongly assigns every label to the visible face (e.g. both
    # SPEAKER_00 and SPEAKER_01 -> the same person in a single clip). Keep the label whose turns
    # co-occur MOST with that person's face (evidence_seconds); demote the rest to unresolved
    # so an off-camera speaker reads as honest "unknown" (manually linkable) rather than being
    # mis-attributed to the on-camera face. Manual links are applied after and always win.
    by_person: dict[str, list[dict]] = {}
    for sa in results:
        if sa["method"] == "bootstrap" and sa["person_id"]:
            by_person.setdefault(sa["person_id"], []).append(sa)
    for pid, group in by_person.items():
        if len(group) <= 1:
            continue
        keeper = max(group, key=lambda s: (s.get("evidence_seconds") or 0.0,
                                            s.get("confidence") or 0.0))
        for sa in group:
            if sa is not keeper:
                sa["person_id"] = None
                sa["method"] = "dedup_ambiguous"
                sa["confidence"] = None
                sa["evidence"] = []

    # Apply manual links (win over bootstrap and de-dup)
    manual_for_media = {ml["speaker_label"]: ml["person_id"]
                        for ml in manual_links if ml.get("media_id") == media_id}
    for sa in results:
        if sa["speaker_label"] in manual_for_media:
            sa["person_id"] = manual_for_media[sa["speaker_label"]]
            sa["method"] = "manual"
            sa["confidence"] = None
            sa["evidence"] = []

    return results


def run(media_id: str, *, config_path: str | Path = "state/config.toml") -> StageManifest:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    paths = cfg.get("paths", {})
    catalog = Path(paths.get("catalog", "catalog"))
    state = Path(paths.get("state", "state"))

    identity_cfg = cfg.get("identity", {})
    floor = float(identity_cfg.get("face_match_floor", _DEFAULT_FLOOR))
    margin = float(identity_cfg.get("face_match_margin", _DEFAULT_MARGIN))

    run_id = f"F1-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"
    emit_start(journal_db, media_id, "F1", run_id)

    error: str | None = None
    counts: dict = {}

    try:
        # Load inputs
        tracks = read_jsonl(catalog / "face_tracks" / f"{media_id}.jsonl")
        diar_turns = read_jsonl(catalog / "asr" / f"{media_id}.diar.jsonl")
        chunks = read_jsonl(catalog / "chunks" / f"{media_id}.jsonl")
        keyframes = read_jsonl(catalog / "keyframes" / f"{media_id}.jsonl")
        events = read_jsonl(catalog / "events" / f"{media_id}.jsonl")

        # Manual links
        links_path = catalog / "persons" / "_manual_links.jsonl"
        manual_links = read_jsonl(links_path) if links_path.exists() else []

        # Manual face links (rare, stored separately)
        face_links_path = catalog / "persons" / "_manual_face_links.jsonl"
        manual_face_links = {}
        if face_links_path.exists():
            for row in read_jsonl(face_links_path):
                if row.get("media_id") == media_id:
                    manual_face_links[row["face_track_id"]] = row["person_id"]

        # D5 step 1: face assignments
        fa = _face_assignments(tracks, floor, margin, manual_face_links)
        assigned_faces = sum(1 for x in fa if x["person_id"])

        # D5 step 2: speaker assignments
        sa = _speaker_assignments(diar_turns, fa, tracks, manual_links, media_id)
        assigned_speakers = sum(1 for x in sa if x["person_id"])

        # D5 step 3: chunk_speaker
        spk_map = {x["speaker_label"]: x["person_id"] for x in sa}
        chunk_speaker = []
        for ch in chunks:
            lbl = ch.get("speaker_label")
            chunk_speaker.append({
                "chunk_id": ch["chunk_id"],
                "person_id": spk_map.get(lbl) if lbl else None,
            })

        # D5 step 4: keyframe_persons
        ft_person = {x["face_track_id"]: x["person_id"] for x in fa if x["person_id"]}
        track_spans = {t["face_track_id"]: (t["start_ms"], t["end_ms"]) for t in tracks}
        keyframe_persons = []
        for kf in keyframes:
            kf_ms = kf["time_ms"]
            pids = list({
                pid for ft_id, pid in ft_person.items()
                if track_spans.get(ft_id, (0, 0))[0] <= kf_ms <= track_spans.get(ft_id, (0, 0))[1]
            })
            keyframe_persons.append({"keyframe_id": kf["keyframe_id"], "person_ids": sorted(pids)})

        # D5 step 5: event_persons
        chunk_pid = {cs["chunk_id"]: cs["person_id"] for cs in chunk_speaker if cs["person_id"]}
        kf_pid = {kp["keyframe_id"]: kp["person_ids"] for kp in keyframe_persons}
        event_persons = []
        for ev in events:
            ev_start, ev_end = ev.get("start_ms", 0), ev.get("end_ms", 0)
            pids: set[str] = set()
            for ch in chunks:
                if _overlap_ms(ev_start, ev_end, ch["start_ms"], ch["end_ms"]) > 0:
                    pid = chunk_pid.get(ch["chunk_id"])
                    if pid:
                        pids.add(pid)
            for kf in keyframes:
                if ev_start <= kf["time_ms"] <= ev_end:
                    pids.update(kf_pid.get(kf["keyframe_id"], []))
            event_persons.append({"event_id": ev["event_id"], "person_ids": sorted(pids)})

        # Write overlay (atomic)
        overlay_dir = catalog / "identity_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay = {
            "schema_version": "1.0.0",
            "media_id": media_id,
            "produced_by": run_id,
            "stage_version": "1.0.0",
            "face_assignments": fa,
            "speaker_assignments": sa,
            "chunk_speaker": chunk_speaker,
            "keyframe_persons": keyframe_persons,
            "event_persons": event_persons,
        }
        out_path = overlay_dir / f"{media_id}.json"
        with tempfile.NamedTemporaryFile("w", dir=overlay_dir, delete=False, suffix=".tmp") as tf:
            json.dump(overlay, tf, indent=2)
            tmp_name = tf.name
        os.replace(tmp_name, out_path)

        counts = {
            "face_tracks": len(tracks),
            "assigned_faces": assigned_faces,
            "speaker_labels": len(sa),
            "assigned_speakers": assigned_speakers,
        }

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="F1",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts=counts,
        error=error,
        outputs=[str(catalog / "identity_overlay" / f"{media_id}.json")],
    )
    status_dir = catalog / "status"
    reap_manifest(journal_db, manifest, status_dir)
    if ok:  # explicit flag for EV/render/voice_enroll to read instead of probing files
        try:
            from . import outcomes as _oc
            _oc.set_flag(media_id, "has_identity",
                         (assigned_speakers > 0 or assigned_faces > 0))
        except Exception:
            pass
    return manifest

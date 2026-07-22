"""ASD join-time resolver — reads active_speaker + F1 face_assignments.

Answers: "who is speaking during this diar turn, and how confident?"
Does NOT modify F0 or F1 artifacts. Read-only on both.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

_log = logging.getLogger("erf.asd_resolver")

ASD_ATTRIB_THRESHOLD = 0.5  # default; override via resolve_speaker_turn(threshold=...)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_identity_overlay(catalog: Path, media_id: str) -> Optional[dict]:
    return _load_json(catalog / "identity_overlay" / f"{media_id}.json")


def _load_active_speaker(catalog: Path, media_id: str) -> Optional[dict]:
    return _load_json(catalog / "active_speaker" / f"{media_id}.json")


def resolve_speaker_turn(
    media_id: str,
    turn_index: int,
    speaker_label: str,
    catalog: Path,
    *,
    threshold: float = ASD_ATTRIB_THRESHOLD,
) -> dict:
    """Return the best person attribution for a diarization turn.

    Result schema:
    {
        "person_id":     str | None,
        "method":        "asd" | "f1_bootstrap" | "f1_gallery" | "f1_manual" | "none",
        "confidence":    float,          # 0–1
        "face_track_id": str | None,     # ASD-grounded track (if method=="asd")
        "note":          str | None      # "asd_correlated_unnamed" if ASD wins but person is null
    }
    """
    asd_data = _load_active_speaker(catalog, media_id)
    overlay = _load_identity_overlay(catalog, media_id)

    asd_turn_entry = None
    if asd_data is not None:
        for t in asd_data.get("turn_active_speaker", []):
            if t["turn_index"] == turn_index:
                asd_turn_entry = t
                break

    # Try ASD-grounded path
    if (
        asd_turn_entry is not None
        and asd_turn_entry.get("face_track_id") is not None
        and asd_turn_entry.get("confidence", 0.0) >= threshold
    ):
        ft_id = asd_turn_entry["face_track_id"]
        conf = float(asd_turn_entry["confidence"])

        person_id = None
        if overlay is not None:
            face_assignments = overlay.get("face_assignments", {})
            assignment = face_assignments.get(ft_id)
            if assignment:
                person_id = assignment.get("person_id")

        if person_id is not None:
            return {
                "person_id": person_id,
                "method": "asd",
                "confidence": conf,
                "face_track_id": ft_id,
                "note": None,
            }
        # ASD grounded but person not yet identified
        return {
            "person_id": None,
            "method": "asd",
            "confidence": conf,
            "face_track_id": ft_id,
            "note": "asd_correlated_unnamed",
        }

    # Fallback: F1 speaker_assignments for this speaker_label
    if overlay is not None:
        speaker_assignments = overlay.get("speaker_assignments", {})
        sa = speaker_assignments.get(speaker_label)
        if sa:
            return {
                "person_id": sa.get("person_id"),
                "method": sa.get("method", "f1_bootstrap"),
                "confidence": float(sa.get("confidence", 0.0)),
                "face_track_id": None,
                "note": None,
            }

    return {
        "person_id": None,
        "method": "none",
        "confidence": 0.0,
        "face_track_id": None,
        "note": None,
    }

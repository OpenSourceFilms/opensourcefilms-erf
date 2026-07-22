"""Seam 2 — versioned, immutable writer + reader resolver for editorial labels.

Replaces the flat, overwrite-on-regen `catalog/editorial/{media_id}.json` (which
violated Invariant 2, immutability — D-Story-01) with an immutable versioned record
`catalog/editorial/{media_id}/{editorial_job_id}.json` plus a `manifest.json`
latest-version resolver. The path is keyed by the editorial job's OWN id, never a
story-batch run_id (patch_v3 §8.3). Consumers read labels ONLY through `latest()`
(via the manifest) — never by globbing the directory.

The label shape + vocabulary (FUNCTIONS / TENSIONS / ARCHETYPES) are owned by
`erf.editorial`; this module freezes them into the `editorial_signals/1.0.0` schema
and sanitizes Flash drift (out-of-vocab labels) against the frozen vocabulary before
writing, so the schema can stay strict.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import contracts, runlog
from .atomicio import write_json
from .editorial import ARCHETYPES, FUNCTIONS, TENSIONS

CAT = Path("/workspace/erf/catalog")
EDIR = CAT / "editorial"

_REMAP = {"REVELATION": "REVEAL"}  # near-synonym Flash drift → canonical FUNCTION
_FUN = set(FUNCTIONS)
_TEN = set(TENSIONS)
_ARCH = set(ARCHETYPES)


def source_evidence_hash(evidence_pkg: dict) -> str:
    """sha256 of the evidence package fed to the labeller — the staleness key."""
    blob = json.dumps(evidence_pkg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sanitize(label_doc: dict) -> tuple[dict, list[str]]:
    """Coerce raw LABEL output to the frozen vocabulary. Returns (clean, coercions).

    Drops out-of-vocab tension_tags / archetypes / central_tension, remaps known
    near-synonym functions (REVELATION→REVEAL), drops unknown functions. Structural
    objects (chain_affordances, quote_completeness) pass through unchanged.
    """
    coercions: list[str] = []
    cl_in = dict(label_doc.get("clip_level") or {})

    ct = cl_in.get("central_tension")
    if ct is not None and ct not in _TEN:
        coercions.append(f"clip: dropped central_tension {ct!r}")
        ct = None

    arch_in = cl_in.get("archetype_candidates") or []
    arch = [a for a in arch_in if a in _ARCH]
    if len(arch) != len(arch_in):
        coercions.append("clip: dropped out-of-vocab archetype(s)")

    clip_level = {
        "story_forces": cl_in.get("story_forces") or {},
        "central_tension": ct,
        "archetype_candidates": arch,
    }

    sentences = []
    for s in label_doc.get("sentences") or []:
        idx = s.get("sentence_index")
        fns_in = s.get("functions") or []
        fns: list[str] = []
        for x in fns_in:
            x = _REMAP.get(x, x)
            if x in _FUN and x not in fns:
                fns.append(x)
        if len(fns) != len(fns_in):
            coercions.append(f"s{idx}: coerced functions")

        tts_in = s.get("tension_tags") or []
        tts = [t for t in tts_in if t in _TEN]
        if len(tts) != len(tts_in):
            coercions.append(f"s{idx}: dropped tension_tag(s)")

        sentences.append({
            "sentence_index": idx,
            "functions": fns,
            "tension_tags": tts,
            "chain_affordances": s.get("chain_affordances"),
            "quote_completeness": s.get("quote_completeness"),
        })

    return {"clip_level": clip_level, "sentences": sentences}, coercions


def build_record(
    media_id: str,
    label_doc: dict,
    *,
    source_evidence_hash: str,
    editorial_job_id: str,
    model: str,
    stage_version: str,
    mode: str = "non-thinking",
    produced_by: str = "LABEL",
) -> dict:
    """Assemble a record conforming to editorial_signals/1.0.0. Does NOT write.

    Coercions from sanitize() are dropped here; callers that want them log
    sanitize() directly.
    """
    clean, _ = sanitize(label_doc)
    return {
        "schema_version": "1.0.0",
        "media_id": media_id,
        "editorial_job_id": editorial_job_id,
        "source_evidence_hash": source_evidence_hash,
        "produced_by": produced_by,
        "model": model,
        "mode": mode,
        "stage_version": stage_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parse_error": None,
        "clip_level": clean["clip_level"],
        "sentences": clean["sentences"],
    }


def write(media_id: str, record: dict, *, journal: bool = True) -> Path:
    """Persist a validated record immutably + update the manifest. Returns its path.

    Validates against schema; if the record file already exists it is a no-op
    (immutable, never overwrite). Atomic writes throughout.
    """
    contracts.validate(record, "editorial_signals")

    out_dir = EDIR / media_id
    out_file = out_dir / f"{record['editorial_job_id']}.json"
    if out_file.exists():
        return out_file  # immutable: never overwrite an existing record

    write_json(out_file, record, indent=1)
    write_json(out_dir / "manifest.json", {
        "media_id": media_id,
        "latest_job_id": record["editorial_job_id"],
        "created_at_utc": record["created_at_utc"],
        "source_evidence_hash": record["source_evidence_hash"],
        "schema_version": "1.0.0",
    }, indent=1)

    if journal:
        runlog.log_event("LABEL", media_id, "ok", {
            "editorial_job_id": record["editorial_job_id"],
            "path": str(out_file),
            "n_sentences": len(record["sentences"]),
        })
    return out_file


def latest(media_id: str) -> Optional[dict]:
    """Resolve the latest record via the manifest. Never scans the directory."""
    manifest_path = EDIR / media_id / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        mf = json.loads(manifest_path.read_text())
        return json.loads((EDIR / media_id / f"{mf['latest_job_id']}.json").read_text())
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def is_stale(media_id: str, current_evidence_hash: str) -> bool:
    """True if the latest record was built from a different evidence hash (or none)."""
    manifest_path = EDIR / media_id / "manifest.json"
    if not manifest_path.exists():
        return True
    try:
        mf = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return True
    return mf.get("source_evidence_hash") != current_evidence_hash


def backfill_flat(*, dry_run: bool = False) -> dict:
    """Migrate legacy flat EDIR/<media_id>.json files to versioned records.

    Each flat file becomes an immutable record under EDIR/<media_id>/ and the
    original is moved to EDIR/_legacy_flat/. One backfill job id for the batch.
    """
    stats: dict = {"n": 0, "written": 0, "coerced": 0, "errors": []}
    legacy_dir = EDIR / "_legacy_flat"
    candidates = [
        p for p in EDIR.iterdir()
        if p.is_file() and p.suffix == ".json"
        and p.stem.startswith("m_") and not p.name.startswith("_")
    ]
    stats["n"] = len(candidates)
    ejid = f"backfill_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

    for flat_file in candidates:
        media_id = flat_file.stem
        try:
            doc = json.loads(flat_file.read_text())
        except Exception as exc:
            stats["errors"].append(f"{media_id}: load error: {exc}")
            continue

        ev_path = CAT / "evidence" / f"{media_id}.json"
        seh = source_evidence_hash(json.loads(ev_path.read_text())) if ev_path.exists() else "0" * 64
        _, coerced = sanitize(doc)
        record = build_record(
            media_id, doc, source_evidence_hash=seh, editorial_job_id=ejid,
            model="deepseek-v4-flash", stage_version="LABEL/backfill",
        )

        if not dry_run:
            try:
                write(media_id, record)
                legacy_dir.mkdir(parents=True, exist_ok=True)
                flat_file.rename(legacy_dir / flat_file.name)
                stats["written"] += 1
            except Exception as exc:
                stats["errors"].append(f"{media_id}: write/move error: {exc}")
                continue
        if coerced:
            stats["coerced"] += 1

    return stats

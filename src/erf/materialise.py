"""Stage C — Vector materialise.

Streams catalog/embeddings/{media_id}.text.jsonl → LanceDB text_v1 upsert.
Streams catalog/embeddings/{media_id}.visual.jsonl → LanceDB visual_v1 upsert.
Dim for visual_v1 is read from state/model_registry.json (never hardcoded).
Control-plane only. Never runs on a GPU worker.
"""
from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

from .atomicio import read_jsonl
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "index": "/workspace/erf/index/lance",
    "text_dim": 4096,
    "text_metric": "cosine",
}


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["index"] = raw.get("paths", {}).get("index", cfg["index"])
        cfg["text_dim"] = raw.get("collections", {}).get("text_dim", cfg["text_dim"])
        cfg["text_metric"] = raw.get("collections", {}).get("text_metric", cfg["text_metric"])
    return cfg


def _lancedb_row(emb: dict) -> dict:
    """Convert an embedding record to a text_v1 LanceDB row."""
    return {
        "id": emb["source_id"],          # chunk_id
        "media_id": emb.get("media_id", ""),
        "vector": emb["vector"],
        "text": "",                        # filled in below from chunk
        "start_ms": 0,
        "end_ms": 0,
        "speaker_label": "",
        "speaker_person_id": None,
        "emotional_beat": None,
        "b_roll_flag": None,
    }


def run(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
) -> StageManifest:
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    index_path = Path(cfg["index"])
    text_dim = int(cfg["text_dim"])

    run_id = f"C-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "C", run_id)

    error: str | None = None
    upsert_count = 0

    try:
        import lancedb
        import lancedb.pydantic
        import pyarrow as pa

        # Load embeddings
        emb_path = catalog / "embeddings" / f"{media_id}.text.jsonl"
        if not emb_path.exists():
            # No embeddings file. Distinguish the two reasons (the StageManifest path
            # is binary ok|error and cannot express skip, so without this gate a
            # no-speech clip records a FALSE C error). Consult the upstream explicit
            # outcome per the spine contract rather than assuming.
            from . import outcomes as _oc
            if _oc.flag(media_id, "has_speech") is False:
                # No speech → no transcript → no text to embed. A legitimate skip.
                manifest = StageManifest(
                    media_id=media_id, stage="C", run_id=run_id, ok=True,
                    at_utc=at_utc, tool_version=None,
                    counts={"upserted": 0}, error=None, outputs=[],
                )
                reap_manifest(journal_db, manifest, catalog / "status")
                # Override the bridge's "ok" with the accurate skip (runs after reap).
                _oc.record(media_id, "C", "skip", reason="no_speech")
                return manifest
            # Speech present but embeddings absent = a real failure upstream.
            raise FileNotFoundError(f"No text embeddings found: {emb_path}")

        embeddings = read_jsonl(emb_path)
        if not embeddings:
            # Nothing to upsert — succeed silently
            manifest = StageManifest(
                media_id=media_id, stage="C", run_id=run_id, ok=True,
                at_utc=at_utc, tool_version=None,
                counts={"upserted": 0}, error=None, outputs=[],
            )
            reap_manifest(journal_db, manifest, catalog / "status")
            return manifest

        # Load chunk metadata for text/timecode fields
        chunks_path = catalog / "chunks" / f"{media_id}.jsonl"
        chunks_by_id: dict[str, dict] = {}
        for c in read_jsonl(chunks_path):
            chunks_by_id[c["chunk_id"]] = c

        # Read kind from media_manifest (defaults to "raw_footage" if absent)
        import json as _json
        mm_path = catalog / "media_manifest" / f"{media_id}.json"
        media_kind = "raw_footage"
        if mm_path.exists():
            try:
                media_kind = _json.loads(mm_path.read_text()).get("kind") or "raw_footage"
            except Exception:
                pass

        # Build rows
        rows = []
        for emb in embeddings:
            chunk_id = emb["source_id"]
            chunk = chunks_by_id.get(chunk_id, {})
            row = {
                "id": chunk_id,
                "media_id": emb.get("media_id", media_id),
                "vector": emb["vector"],
                "text": chunk.get("text", ""),
                "start_ms": chunk.get("start_ms", 0),
                "end_ms": chunk.get("end_ms", 0),
                "speaker_label": chunk.get("speaker_label") or "",
                "speaker_person_id": chunk.get("speaker_person_id"),
                "emotional_beat": chunk.get("emotional_beat"),
                "b_roll_flag": chunk.get("b_roll_flag"),
                "kind": media_kind,
            }
            rows.append(row)

        # Connect to LanceDB (creates if absent)
        index_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(index_path))

        # Define schema for new table creation
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("media_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), text_dim)),
            pa.field("text", pa.string()),
            pa.field("start_ms", pa.int64()),
            pa.field("end_ms", pa.int64()),
            pa.field("speaker_label", pa.string()),
            pa.field("speaker_person_id", pa.string()),
            pa.field("emotional_beat", pa.string()),
            pa.field("b_roll_flag", pa.bool_()),
            pa.field("kind", pa.string()),
        ])

        # list_tables() returns ListTablesResponse in LanceDB 0.33 (not a plain list).
        # Use .tables to get the string list; table_names() is deprecated.
        existing_tables = db.list_tables().tables
        if "text_v1" not in existing_tables:
            tbl = db.create_table("text_v1", schema=schema)
        else:
            tbl = db.open_table("text_v1")
            # Add kind column if missing (additive nullable — D-sess2-00)
            if "kind" not in [f.name for f in tbl.schema]:
                tbl.add_columns(pa.field("kind", pa.string()))

        # Delete existing rows for this media (idempotent)
        try:
            tbl.delete(f"media_id = '{media_id}'")
        except Exception:
            pass

        # Add rows
        tbl.add(rows)
        upsert_count = len(rows)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="C",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"upserted": upsert_count},
        error=error,
        outputs=[],
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest


def _get_visual_dim(state: Path) -> int:
    """Read visual dim from model_registry.json. Never hardcoded."""
    reg = state / "model_registry.json"
    if reg.exists():
        import json
        data = json.loads(reg.read_text())
        for entry in data.values():
            if entry.get("modality") == "visual" and entry.get("gate_passed"):
                return int(entry["dim"])
    raise RuntimeError(
        "No visual model in model_registry.json. Run bin/gate_visual_embed.py first."
    )


def run_visual(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
) -> StageManifest:
    """Materialise visual embeddings → visual_v1 LanceDB table."""
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    index_path = Path(cfg["index"])

    run_id = f"C_vis-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "C_vis", run_id)

    error: str | None = None
    upsert_count = 0

    try:
        import lancedb
        import pyarrow as pa

        visual_dim = _get_visual_dim(state)

        emb_path = catalog / "embeddings" / f"{media_id}.visual.jsonl"
        if not emb_path.exists():
            raise FileNotFoundError(f"No visual embeddings found: {emb_path}")

        embeddings = read_jsonl(emb_path)
        if not embeddings:
            manifest = StageManifest(
                media_id=media_id, stage="C_vis", run_id=run_id, ok=True,
                at_utc=at_utc, tool_version=None, counts={"upserted": 0},
                error=None, outputs=[],
            )
            reap_manifest(journal_db, manifest, catalog / "status")
            return manifest

        # Load keyframe metadata
        kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
        kf_by_id: dict[str, dict] = {}
        for kf in read_jsonl(kf_path):
            kf_by_id[kf["keyframe_id"]] = kf

        # Read kind from media_manifest
        import json as _json
        mm_path = catalog / "media_manifest" / f"{media_id}.json"
        media_kind = "raw_footage"
        if mm_path.exists():
            try:
                media_kind = _json.loads(mm_path.read_text()).get("kind") or "raw_footage"
            except Exception:
                pass

        rows = []
        for emb in embeddings:
            kf = kf_by_id.get(emb["source_id"], {})
            rows.append({
                "id": emb["source_id"],
                "media_id": emb.get("media_id", media_id),
                "vector": emb["vector"],
                "scene_id": kf.get("scene_id") or "",
                "time_ms": kf.get("time_ms", 0),
                "image_relpath": kf.get("image_relpath") or "",
                "visual_quality_score": None,
                "person_ids": None,
                "b_roll_flag": None,
                "kind": media_kind,
            })

        index_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(index_path))

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("media_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), visual_dim)),
            pa.field("scene_id", pa.string()),
            pa.field("time_ms", pa.int64()),
            pa.field("image_relpath", pa.string()),
            pa.field("visual_quality_score", pa.float32()),
            pa.field("person_ids", pa.list_(pa.string())),
            pa.field("b_roll_flag", pa.bool_()),
            pa.field("kind", pa.string()),
        ])

        existing_tables = db.list_tables().tables
        if "visual_v1" not in existing_tables:
            tbl = db.create_table("visual_v1", schema=schema)
        else:
            tbl = db.open_table("visual_v1")
            if "kind" not in [f.name for f in tbl.schema]:
                tbl.add_columns(pa.field("kind", pa.string()))

        try:
            tbl.delete(f"media_id = '{media_id}'")
        except Exception:
            pass

        tbl.add(rows)
        upsert_count = len(rows)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="C_vis",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"upserted": upsert_count},
        error=error,
        outputs=[],
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest

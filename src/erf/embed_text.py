"""Stage B1 — Text embedding.

Loads Qwen3-Embedding-8B using the confirmed invocation from
state/embed_text_invocation.json (written by G-TEXT-EMBED gate).
Embeds all chunks for a media file, writes catalog/embeddings/{media_id}.text.jsonl.
"""
from __future__ import annotations

import json
import tomllib
import uuid
from pathlib import Path

from .atomicio import jsonl_append, read_jsonl
from .ids import embedding_id
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "erf_root": "/workspace/erf",
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "text_embed_path": "/workspace/erf/weights/Qwen3-Embedding-8B",
    "text_dim": 4096,
    "text_model_tag": "qwen3-8b",
}
# Module-level model cache — avoids reloading 15GB model on every run() call.
# device_map="auto" with Accelerate does not release VRAM cleanly on model deletion;
# keeping a single reference prevents repeated load/OOM across batched runs.
_MODEL_CACHE: dict = {}


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["text_embed_path"] = raw.get("models", {}).get("text_embed_path", cfg["text_embed_path"])
        cfg["text_dim"] = raw.get("collections", {}).get("text_dim", cfg["text_dim"])
        cfg["text_model_tag"] = raw.get("collections", {}).get("text_model_tag", cfg["text_model_tag"])
    return cfg


def _load_invocation(state: Path) -> dict:
    inv_path = state / "embed_text_invocation.json"
    if not inv_path.exists():
        raise FileNotFoundError(
            f"G-TEXT-EMBED gate not passed: {inv_path} not found. "
            "Run bin/gate_text_embed.py first."
        )
    return json.loads(inv_path.read_text())


def run(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
    batch_size: int = 64,
) -> StageManifest:
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    model_tag = cfg["text_model_tag"]
    expected_dim = int(cfg["text_dim"])

    run_id = f"B1-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "B1", run_id)

    error: str | None = None
    embed_count = 0
    outputs: list[str] = []
    tool_version: str | None = None
    skip_no_chunks = False

    try:
        # Load confirmed invocation
        inv = _load_invocation(state)
        model_path = inv.get("model_path", cfg["text_embed_path"])

        # Gate on B0's explicit has_chunks flag (cross-stage awareness) — avoids the
        # 15 GB model load on silent clips without re-deriving emptiness from the file.
        from . import outcomes as _oc
        _has_chunks = _oc.flag(media_id, "has_chunks")
        chunks_path = catalog / "chunks" / f"{media_id}.jsonl"
        chunks = [] if _has_chunks is False else read_jsonl(chunks_path)
        emb_path = catalog / "embeddings" / f"{media_id}.text.jsonl"
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        if not chunks:
            # Nothing to embed — explicit skip outcome, no zero-byte file.
            # The skip is recorded AFTER reap_manifest below — the reap bridge
            # writes a bare "ok" and would overwrite it here.
            skip_no_chunks = True
            _oc.set_flag(media_id, "has_text_vec", False, reason="no_chunks")
        else:
            from sentence_transformers import SentenceTransformer
            import sentence_transformers as _st
            tool_version = _st.__version__

            if model_path not in _MODEL_CACHE:
                _MODEL_CACHE[model_path] = SentenceTransformer(
                    model_path,
                    model_kwargs=inv.get("model_kwargs", {"device_map": "auto"}),
                    tokenizer_kwargs=inv.get("tokenizer_kwargs", {"padding_side": "left"}),
                )
            model = _MODEL_CACHE[model_path]

            query_prompt_name = inv.get("query_prompt_name", "query")
            normalize = inv.get("encode_kwargs", {}).get("normalize_embeddings", True)

            if emb_path.exists():
                emb_path.unlink()

            # Batch encode — documents use no prompt per the gate
            texts = [c.get("text", "") for c in chunks]
            all_vectors = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vecs = model.encode(
                    batch,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
                all_vectors.extend(vecs.tolist())

            for chunk, vector in zip(chunks, all_vectors):
                if len(vector) != expected_dim:
                    raise ValueError(
                        f"Embedding dim mismatch: expected {expected_dim}, got {len(vector)}"
                    )
                emb_id = embedding_id(chunk["chunk_id"], model_tag)
                rec = {
                    "schema_version": "1.0.0",
                    "embedding_id": emb_id,
                    "source_id": chunk["chunk_id"],
                    "source_type": "chunk",
                    "media_id": media_id,
                    "modality": "text",
                    "model_tag": model_tag,
                    "dim": expected_dim,
                    "vector": vector,
                    "normalized": normalize,
                }
                jsonl_append(emb_path, rec)
                embed_count += 1

            outputs.append(str(emb_path))

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="B1",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=tool_version,
        counts={"embeddings": embed_count},
        error=error,
        outputs=outputs,
    )
    status_dir = catalog / "status"
    reap_manifest(journal_db, manifest, status_dir)
    if skip_no_chunks:
        try:
            from . import outcomes as _oc2
            # Override the bridge's "ok" with the accurate skip (runs after reap).
            _oc2.record(media_id, "B1", "skip", reason="no_chunks")
        except Exception:
            pass
    if error is None and embed_count > 0:
        try:
            from . import outcomes as _oc2
            _oc2.set_flag(media_id, "has_text_vec", True)
        except Exception:
            pass
    return manifest

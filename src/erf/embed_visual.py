"""Stage B4 — Visual embedding.

Loads Qwen3-VL-Embedding-8B using confirmed invocation from G-VISUAL-EMBED gate.
Encodes keyframe jpg files → embedding JSONL (modality=visual).
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
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "erf_root": "/workspace/erf",
}
_MODEL_CACHE: dict = {}


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
    return cfg


def _load_model(state: Path):
    inv_path = state / "embed_visual_invocation.json"
    if not inv_path.exists():
        raise FileNotFoundError(
            f"G-VISUAL-EMBED gate not passed: {inv_path} missing. Run bin/gate_visual_embed.py."
        )
    inv = json.loads(inv_path.read_text())
    model_path = inv["model_path"]

    if model_path not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            model_path,
            model_kwargs=inv.get("model_kwargs", {"device_map": "auto"}),
            tokenizer_kwargs=inv.get("tokenizer_kwargs", {"padding_side": "left"}),
        )
        _MODEL_CACHE[model_path] = (model, inv)
    return _MODEL_CACHE[model_path]


def run(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
    batch_size: int = 16,
) -> StageManifest:
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    erf_root = Path(cfg["erf_root"])

    run_id = f"B4-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "B4", run_id)

    error: str | None = None
    embed_count = 0
    outputs: list[str] = []
    tool_version: str | None = None

    try:
        model, inv = _load_model(state)
        import sentence_transformers as _st
        tool_version = _st.__version__

        model_tag = inv["model_tag"]
        expected_dim = int(inv["dim"])
        normalize = inv.get("encode_kwargs", {}).get("normalize_embeddings", True)

        # Load keyframes
        kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
        if not kf_path.exists():
            raise FileNotFoundError(f"Keyframes not found: {kf_path}")
        keyframes = read_jsonl(kf_path)

        emb_path = catalog / "embeddings" / f"{media_id}.visual.jsonl"
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        if emb_path.exists():
            emb_path.unlink()

        if not keyframes:
            emb_path.touch()
            outputs.append(str(emb_path))
        else:
            # Build image path list
            image_paths = [str(erf_root / kf["image_relpath"]) for kf in keyframes]

            # Batch encode
            all_vectors = []
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]
                vecs = model.encode(
                    batch,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
                all_vectors.extend(vecs.tolist())

            for kf, vector in zip(keyframes, all_vectors):
                if len(vector) != expected_dim:
                    raise ValueError(f"Dim mismatch: expected {expected_dim}, got {len(vector)}")
                emb_id = embedding_id(kf["keyframe_id"], model_tag)
                rec = {
                    "schema_version": "1.0.0",
                    "embedding_id": emb_id,
                    "source_id": kf["keyframe_id"],
                    "source_type": "keyframe",
                    "media_id": media_id,
                    "modality": "visual",
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
        stage="B4",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=tool_version,
        counts={"embeddings": embed_count},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest

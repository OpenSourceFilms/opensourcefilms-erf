"""Hybrid query pipeline — FTS5 + LanceDB semantic + RRF fusion.

Steps: parse → embed → lance search → FTS search → RRF → assemble clips.
"""
from __future__ import annotations

import json
import threading
import tomllib
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_TIMERS: dict[str, threading.Timer] = {}
_EMBED_IDLE_SECONDS: int = int(__import__("os").environ.get("EMBED_IDLE_SECONDS", "300"))

def _release_model(key: str) -> None:
    _MODEL_CACHE.pop(key, None)
    _MODEL_TIMERS.pop(key, None)
    try:
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _schedule_model_release(key: str, idle_s: int = _EMBED_IDLE_SECONDS) -> None:
    existing = _MODEL_TIMERS.pop(key, None)
    if existing:
        existing.cancel()
    t = threading.Timer(idle_s, _release_model, args=(key,))
    t.daemon = True
    t.start()
    _MODEL_TIMERS[key] = t


def get_model_status() -> dict:
    try:
        import torch
        vram = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    except Exception:
        vram = 0.0
    keys = list(_MODEL_CACHE.keys())
    return {
        "text_model_loaded": any("visual" not in k for k in keys),
        "visual_model_loaded": any("visual" in k for k in keys),
        "cached_keys": keys,
        "vram_allocated_mb": round(vram, 1),
    }


_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "index": "/workspace/erf/index/lance",
    "erf_fts_db": "/workspace/erf/state/erf_fts.db",
    "top_k": 50,
    "hyde_enabled": True,
    "hyde_word_threshold": 6,
}

_QUERYSPEC_SYSTEM = (
    "You are a media search query parser. Convert the user's natural-language query into a "
    "QuerySpec JSON object. Output ONLY valid JSON. Schema: "
    '{"modality": ["text"], "semantic_text": "...", "filters": {"media_id": null, '
    '"speaker_person_id": null, "time_range_ms": null, "emotional_beat": null, '
    '"b_roll_flag": null}, "top_k": 50}. '
    "Default modality to [\"text\"]. Default top_k to 50. Never add prose."
)

_HYDE_SYSTEM = (
    "You are a video corpus search assistant. Given a search query, write one concrete "
    "sentence describing what an ideal matching video clip would look like — visual, "
    "specific, descriptive. Output only the description, no preamble or explanation."
)


def _hyde_expand(query_str: str, state: Path) -> list[float]:
    """HyDE: generate a hypothetical clip description, embed it, fuse with the original query.

    Returns an L2-normalised average of the original and hypothetical embeddings.
    Falls back to standard embed_query on any error.
    """
    try:
        from .llm import complete as _llm_complete
        import numpy as np

        hypothesis = _llm_complete(
            [
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query_str},
            ]
        ).strip()
        if not hypothesis:
            return embed_query(query_str, state)

        orig_vec = embed_query(query_str, state)
        hypo_vec = embed_query(hypothesis, state)

        fused = np.add(orig_vec, hypo_vec, dtype=np.float32)
        norm = float(np.linalg.norm(fused))
        if norm > 1e-8:
            fused = fused / norm
        return fused.tolist()

    except Exception:
        return embed_query(query_str, state)


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["index"] = raw.get("paths", {}).get("index", cfg["index"])
        cfg["erf_fts_db"] = raw.get("paths", {}).get("erf_fts_db", cfg["erf_fts_db"])
        cfg["top_k"] = raw.get("query", {}).get("top_k", cfg["top_k"])
        hyde_sec = raw.get("query", {}).get("hyde", {})
        cfg["hyde_enabled"] = hyde_sec.get("enabled", cfg["hyde_enabled"])
        cfg["hyde_word_threshold"] = int(hyde_sec.get("word_count_threshold", cfg["hyde_word_threshold"]))
    return cfg


# ── Query parsing ─────────────────────────────────────────────────────────────

def parse_query(query_str: str) -> dict:
    """Call LLM to parse plain-English query into QuerySpec. Falls back to a default spec."""
    from .llm import complete
    try:
        messages = [
            {"role": "system", "content": _QUERYSPEC_SYSTEM},
            {"role": "user", "content": query_str},
        ]
        raw = complete(messages, json_only=True)
        spec = json.loads(raw)
        spec.setdefault("modality", ["text"])
        spec.setdefault("semantic_text", query_str)
        spec.setdefault("filters", {})
        spec.setdefault("top_k", 50)
        return spec
    except Exception:
        return {
            "modality": ["text"],
            "semantic_text": query_str,
            "filters": {},
            "top_k": 50,
        }


# ── Embedding ─────────────────────────────────────────────────────────────────

def _load_embed_model(state: Path):
    inv_path = state / "embed_text_invocation.json"
    if not inv_path.exists():
        raise FileNotFoundError(f"G-TEXT-EMBED gate not passed: {inv_path}")
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
    _schedule_model_release(model_path)
    return _MODEL_CACHE[model_path]


def embed_query(query_text: str, state: Path) -> list[float]:
    model, inv = _load_embed_model(state)
    query_prompt_name = inv.get("query_prompt_name", "query")
    vec = model.encode(
        [query_text],
        prompt_name=query_prompt_name,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec[0].tolist()


# ── LanceDB search ────────────────────────────────────────────────────────────

def search_lance(
    query_vector: list[float],
    lance_path: Path,
    *,
    top_k: int = 50,
    filters: dict | None = None,
) -> list[dict]:
    try:
        import lancedb
        db = lancedb.connect(str(lance_path))
        # list_tables() returns ListTablesResponse in LanceDB 0.33 (not a plain list).
        # Use .tables to get the string list; table_names() is deprecated.
        if "text_v1" not in db.list_tables().tables:
            return []
        tbl = db.open_table("text_v1")

        q = tbl.search(query_vector).limit(top_k)

        # Apply filters as SQL WHERE clause
        if filters:
            parts = []
            if filters.get("media_id"):
                parts.append(f"media_id = '{filters['media_id']}'")
            if filters.get("speaker_person_id"):
                parts.append(f"speaker_person_id = '{filters['speaker_person_id']}'")
            if filters.get("emotional_beat"):
                parts.append(f"emotional_beat = '{filters['emotional_beat']}'")
            if filters.get("b_roll_flag") is not None:
                val = "true" if filters["b_roll_flag"] else "false"
                parts.append(f"b_roll_flag = {val}")
            if filters.get("kind"):
                parts.append(f"kind = '{filters['kind']}'")
            if parts:
                q = q.where(" AND ".join(parts))

        rows = q.to_list()
        results = []
        for row in rows:
            results.append({
                "chunk_id": row.get("id", ""),
                "media_id": row.get("media_id", ""),
                "start_ms": row.get("start_ms", 0),
                "end_ms": row.get("end_ms", 0),
                "text": row.get("text", ""),
                "speaker_label": row.get("speaker_label", ""),
                "score": float(row.get("_distance", 0.0)),
                "source": "vector",
            })
        return results
    except Exception:
        return []


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fuse(
    fts_results: list[dict],
    lance_results: list[dict],
    *,
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}

    for rank, r in enumerate(fts_results):
        cid = r.get("chunk_id", r.get("id", ""))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in merged:
            merged[cid] = {**r, "chunk_id": cid, "sources": ["fts"]}
        else:
            merged[cid]["sources"].append("fts")

    for rank, r in enumerate(lance_results):
        cid = r.get("chunk_id", r.get("id", ""))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in merged:
            merged[cid] = {**r, "chunk_id": cid, "sources": ["vector"]}
        else:
            if "vector" not in merged[cid]["sources"]:
                merged[cid]["sources"].append("vector")

    fused = sorted(merged.values(), key=lambda x: scores[x["chunk_id"]], reverse=True)
    for r in fused:
        r["score"] = scores[r["chunk_id"]]
    return fused


# ── Timecode ──────────────────────────────────────────────────────────────────

def ms_to_tc(ms: int, fps: float, fps_rational: str | None = None) -> str:
    drop = fps_rational in ("30000/1001", "60000/1001")
    total_frames = round(ms / 1000 * fps)
    fps_int = round(fps)
    ff = total_frames % fps_int
    total_s = total_frames // fps_int
    hh, rem = divmod(total_s, 3600)
    mm, ss = divmod(rem, 60)
    sep = ";" if drop else ":"
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"


# ── Manifest loader ───────────────────────────────────────────────────────────
# The media registry is FILE-BASED: catalog/media_manifest/{media_id}.json.
# erf.db contains a `media` table but it has 0 rows and is not written to by
# any pipeline code. Do NOT query erf.db for media records — use this function.

def load_media_manifest(media_id: str, catalog: Path) -> dict | None:
    """Return the media manifest dict for media_id, or None if not ingested.

    The authoritative media registry is catalog/media_manifest/{media_id}.json.
    erf.db.media exists but is permanently empty — never query it for media records.
    """
    p = catalog / "media_manifest" / f"{media_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    query_str: str,
    *,
    config_path: str | Path = "state/config.toml",
    kind: str | None = None,
    rerank_mode: str = "off",
) -> dict:
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    lance_path = Path(cfg["index"])
    fts_db = Path(cfg["erf_fts_db"])
    default_top_k = int(cfg["top_k"])

    # 1. Parse
    spec = parse_query(query_str)
    top_k = int(spec.get("top_k", default_top_k))
    semantic_text = spec.get("semantic_text", query_str)
    filters = spec.get("filters") or {}

    # Merge caller-supplied kind override into filters
    if kind:
        filters["kind"] = kind

    # Capture warmness BEFORE model load — opportunistic rerank skips if cold at request start
    _was_warm = _is_warm()

    # 2. Embed (with optional HyDE expansion for short queries)
    word_count = len(semantic_text.strip().split())
    if cfg.get("hyde_enabled", True) and word_count <= cfg.get("hyde_word_threshold", 6):
        query_vector = _hyde_expand(semantic_text, state)
    else:
        query_vector = embed_query(semantic_text, state)

    # 3. Vector search
    lance_results = search_lance(query_vector, lance_path, top_k=top_k, filters=filters)

    # 4. FTS search
    from .fts import search_chunks as fts_search_chunks
    fts_raw = fts_search_chunks(fts_db, semantic_text, limit=top_k)
    fts_results = [
        {**r, "chunk_id": r.get("chunk_id", r.get("id", "")), "source": "fts"}
        for r in fts_raw
    ]

    # 5. RRF fuse
    fused = rrf_fuse(fts_results, lance_results)

    # 6. Assemble result objects
    results = []
    manifests: dict[str, dict | None] = {}
    for r in fused[:top_k]:
        mid = r.get("media_id", "")
        if mid not in manifests:
            manifests[mid] = load_media_manifest(mid, catalog)
        manifest = manifests[mid] or {}

        video = manifest.get("video") or {}
        fps = float(video.get("fps") or 25.0)
        fps_rational = video.get("fps_rational")
        relpath = manifest.get("relpath", mid)

        in_ms = int(r.get("start_ms", 0))
        out_ms = int(r.get("end_ms", 0))
        in_tc = ms_to_tc(in_ms, fps, fps_rational)
        out_tc = ms_to_tc(out_ms, fps, fps_rational)

        result = {
            "result_id": r["chunk_id"],
            "media_id": mid,
            "media_relpath": relpath,
            "in_ms": in_ms,
            "out_ms": out_ms,
            "in_tc": in_tc,
            "out_tc": out_tc,
            "fps": fps,
            "thumb_url": f"/media/thumbs/{mid}/{mid}_mid.webp",
            "snippet": r.get("text", "")[:300],
            "speaker": r.get("speaker_label") or "",
            "modality": "text",
            "score": round(r.get("score", 0.0), 6),
            "rerank_score": None,
        }
        results.append(result)

    # Join tag scores (hook/quote/proof) by chunk_id from catalog/tags/
    from .atomicio import read_jsonl as _read_jsonl
    _tags_by_chunk: dict[str, dict] = {}
    _tags_loaded: set[str] = set()
    for r in results:
        mid = r["media_id"]
        if mid not in _tags_loaded:
            _tags_loaded.add(mid)
            tp = catalog / "tags" / f"{mid}.jsonl"
            if tp.exists():
                for t in _read_jsonl(tp):
                    _tags_by_chunk[t["chunk_id"]] = t
    for r in results:
        tag = _tags_by_chunk.get(r["result_id"], {})
        r["hook_score"] = tag.get("hook_score")
        r["quote_score"] = tag.get("quote_score")
        r["proof_score"] = tag.get("proof_score")

    # Left-join identity_overlay: fill speaker_person_id per chunk (Invariant 6 — null if no overlay)
    _overlay_cache: dict[str, dict] = {}
    for r in results:
        mid = r["media_id"]
        if mid not in _overlay_cache:
            ov_path = catalog / "identity_overlay" / f"{mid}.json"
            if ov_path.exists():
                import json as _json
                ov = _json.loads(ov_path.read_text())
                _overlay_cache[mid] = {cs["chunk_id"]: cs["person_id"]
                                       for cs in ov.get("chunk_speaker", [])}
            else:
                _overlay_cache[mid] = {}
        r["speaker_person_id"] = _overlay_cache[r["media_id"]].get(r["result_id"])

    # Freshness: derive "unused" | "used_in_published" from published_clip_refs.chunk_id
    import sqlite3 as _sqlite3
    fts_db = Path(cfg["erf_fts_db"])
    _used_chunks: set[str] = set()
    chunk_ids = [r["result_id"] for r in results if r.get("result_id")]
    if chunk_ids and fts_db.exists():
        try:
            _fts_conn = _sqlite3.connect(str(fts_db))
            placeholders = ",".join("?" * len(chunk_ids))
            _used_chunks = {
                row[0] for row in _fts_conn.execute(
                    f"SELECT chunk_id FROM published_clip_refs WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                if row[0]
            }
            _fts_conn.close()
        except Exception:
            pass
    for r in results:
        r["freshness"] = "used_in_published" if r.get("result_id") in _used_chunks else "unused"

    if rerank_mode == "opportunistic":
        results = rerank(semantic_text, results, state=state, opportunistic=True, was_warm=_was_warm)
    elif rerank_mode == "force":
        results = rerank(semantic_text, results, state=state, opportunistic=False)

    return {"query": query_str, "results": results}


# ── Visual search ─────────────────────────────────────────────────────────────

def _load_visual_model(state: Path):
    inv_path = state / "embed_visual_invocation.json"
    if not inv_path.exists():
        raise FileNotFoundError("G-VISUAL-EMBED gate not passed")
    inv = json.loads(inv_path.read_text())
    model_path = inv["model_path"]

    cache_key = f"visual:{model_path}"
    if cache_key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            model_path,
            model_kwargs=inv.get("model_kwargs", {"device_map": "auto"}),
            tokenizer_kwargs=inv.get("tokenizer_kwargs", {"padding_side": "left"}),
        )
        _MODEL_CACHE[cache_key] = (model, inv)
    _schedule_model_release(cache_key)
    return _MODEL_CACHE[cache_key]


def embed_visual_query(query_text: str, state: Path) -> list[float]:
    """Embed a text query in the visual embedding space (cross-modal)."""
    model, inv = _load_visual_model(state)
    normalize = inv.get("encode_kwargs", {}).get("normalize_embeddings", True)
    vec = model.encode([query_text], normalize_embeddings=normalize, show_progress_bar=False)
    return vec[0].tolist()


def search_visual_lance(
    query_vector: list[float],
    lance_path: Path,
    *,
    top_k: int = 50,
    media_id: str | None = None,
) -> list[dict]:
    """Search visual_v1 LanceDB table."""
    try:
        import lancedb
        db = lancedb.connect(str(lance_path))
        if "visual_v1" not in db.list_tables().tables:
            return []
        tbl = db.open_table("visual_v1")
        q = tbl.search(query_vector).limit(top_k)
        if media_id:
            q = q.where(f"media_id = '{media_id}'")
        rows = q.to_list()
        return [
            {
                "keyframe_id": r.get("id", ""),
                "media_id": r.get("media_id", ""),
                "time_ms": r.get("time_ms", 0),
                "image_relpath": r.get("image_relpath", ""),
                "scene_id": r.get("scene_id", ""),
                "score": float(r.get("_distance", 0.0)),
                "source": "visual_vector",
            }
            for r in rows
        ]
    except Exception:
        return []


def run_visual(
    query_str: str,
    *,
    config_path: str | Path = "state/config.toml",
    media_id: str | None = None,
) -> dict:
    """Cross-modal text→frame search via visual_v1."""
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    state = Path(cfg["state"])
    lance_path = Path(cfg["index"])
    catalog = Path(cfg["catalog"])
    default_top_k = int(cfg["top_k"])

    query_vector = embed_visual_query(query_str, state)
    results_raw = search_visual_lance(query_vector, lance_path, top_k=default_top_k, media_id=media_id)

    results = []
    manifests: dict[str, dict | None] = {}
    for r in results_raw:
        mid = r.get("media_id", "")
        if mid not in manifests:
            manifests[mid] = load_media_manifest(mid, catalog)
        mm = manifests[mid] or {}
        video = mm.get("video") or {}
        fps = float(video.get("fps") or 25.0)
        fps_rational = video.get("fps_rational")
        time_ms = int(r.get("time_ms", 0))

        results.append({
            "result_id": r["keyframe_id"],
            "media_id": mid,
            "media_relpath": mm.get("relpath", mid),
            "in_ms": max(0, time_ms - 2000),
            "out_ms": time_ms + 2000,
            "in_tc": ms_to_tc(max(0, time_ms - 2000), fps, fps_rational),
            "out_tc": ms_to_tc(time_ms + 2000, fps, fps_rational),
            "fps": fps,
            "thumb_url": f"/media/thumbs/{mid}/{r['image_relpath'].replace('media/keyframes/' + mid + '/', '').replace('.jpg', '.webp')}",
            "image_url": f"/{r['image_relpath']}",
            "snippet": "",
            "speaker": "",
            "modality": "visual",
            "score": round(r.get("score", 0.0), 6),
        })
    return {"query": query_str, "results": results}


# ── OCR search ────────────────────────────────────────────────────────────────

def run_ocr(query_str: str, *, config_path: str | Path = "state/config.toml") -> dict:
    """Lexical search over OCR text."""
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    fts_db = Path(cfg["erf_fts_db"])
    from .fts import search_ocr
    hits = search_ocr(fts_db, query_str)
    return {"query": query_str, "results": hits}


# ── Similar-by-clip ───────────────────────────────────────────────────────────

def run_similar(
    source_id: str,
    *,
    config_path: str | Path = "state/config.toml",
    modality: str = "text",
) -> dict:
    """More-like-this: fetch stored embedding for source_id, search same table."""
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    catalog = Path(cfg["catalog"])
    lance_path = Path(cfg["index"])
    state = Path(cfg["state"])

    try:
        import lancedb
        db = lancedb.connect(str(lance_path))
        table_name = "text_v1" if modality == "text" else "visual_v1"
        if table_name not in db.list_tables().tables:
            return {"source_id": source_id, "results": []}
        tbl = db.open_table(table_name)
        rows = tbl.search().where(f"id = '{source_id}'").limit(1).to_list()
        if not rows:
            return {"source_id": source_id, "results": []}
        stored_vec = rows[0]["vector"]
        results_raw = tbl.search(stored_vec).limit(20).to_list()
        # Exclude self
        results_raw = [r for r in results_raw if r.get("id") != source_id]
        return {"source_id": source_id, "results": results_raw}
    except Exception as e:
        return {"source_id": source_id, "results": [], "error": str(e)}


# ── Reranker ──────────────────────────────────────────────────────────────────

_RERANKER_CACHE: dict = {}


def _is_warm() -> bool:
    """Return True if the text embedding model is already loaded in cache."""
    return any("visual" not in k for k in _MODEL_CACHE)


def rerank(
    query: str,
    results: list[dict],
    *,
    state: Path,
    top_n: int = 30,
    opportunistic: bool = True,
    was_warm: bool | None = None,
) -> list[dict]:
    """Rerank top-N results with Qwen3-VL-Reranker-2B CrossEncoder.

    DEC-3 policy:
    - opportunistic=True: skip if worker was not warm at request start and ERF_RERANK!=1
    - opportunistic=False: always rerank unless ERF_RERANK=0 (shorts batch path)
    - ERF_RERANK=1: force-enable regardless of warm/cold
    - ERF_RERANK=0: disable regardless
    - was_warm: warmness captured BEFORE this request's model load (avoids false-positive)
    """
    import os
    erf_rerank = os.environ.get("ERF_RERANK", "")
    if erf_rerank == "0":
        return results

    force = (erf_rerank == "1")
    # Use was_warm if supplied (captures state before this request loaded the model)
    warm_check = was_warm if was_warm is not None else _is_warm()
    if opportunistic and not force and not warm_check:
        return results  # cold worker at request start, not forced — skip

    inv_path = state / "reranker_invocation.json"
    if not inv_path.exists():
        return results

    import json as _json
    inv = _json.loads(inv_path.read_text())
    reranker_path = inv.get("model_path")
    prompt = inv.get("prompt", "Retrieve text relevant to the user query.")

    if not reranker_path or not Path(reranker_path).exists():
        return results

    if reranker_path not in _RERANKER_CACHE:
        try:
            from sentence_transformers import CrossEncoder
            reranker_model = CrossEncoder(reranker_path, model_kwargs={"device_map": "auto"})
            _RERANKER_CACHE[reranker_path] = reranker_model
        except Exception:
            return results

    reranker_model = _RERANKER_CACHE[reranker_path]
    candidates = results[:top_n]
    rest = results[top_n:]
    pairs = [(query, r.get("snippet") or r.get("text") or "") for r in candidates]

    try:
        scores = reranker_model.predict(pairs, prompt=prompt)
        for r, score in zip(candidates, scores):
            r["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception:
        pass

    return candidates + rest

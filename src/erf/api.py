"""FastAPI control-plane API + minimal static UI.

Endpoints:
  GET  /                         — minimal search UI (static HTML)
  POST /query                    — hybrid semantic + FTS search
  POST /query/exact              — exact-phrase FTS search
  GET  /collections              — list indexed media with stage status
  POST /export                   — write FCPXML/EDL/OTIO from clip selection
  GET  /media/thumbs/{media_id}/{filename} — serve thumbnail files
  GET  /media/proxies/{media_id}.mp4      — serve proxy video

Database note:
  erf.db is used for: review_queue, short_templates, persons, audit, jobs, workers.
  erf.db.media EXISTS but has 0 rows and is not maintained by any pipeline code.
  For media record lookups always use catalog/media_manifest/{media_id}.json
  (via query.load_media_manifest). Never query erf.db for media records.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_CONFIG_PATH = Path("state/config.toml")
_CFG: dict[str, Any] = {}
_ideate_jobs: dict[str, dict] = {}   # job_id → {status, proposals_developed, proposal_ids, ...}


def _cfg() -> dict:
    global _CFG
    if not _CFG:
        defaults = {
            "catalog": "/workspace/erf/catalog",
            "state": "/workspace/erf/state",
            "erf_root": "/workspace/erf",
            "erf_fts_db": "/workspace/erf/state/erf_fts.db",
        }
        _CFG = dict(defaults)
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "rb") as fh:
                raw = tomllib.load(fh)
            _CFG["catalog"] = raw.get("paths", {}).get("catalog", defaults["catalog"])
            _CFG["state"] = raw.get("paths", {}).get("state", defaults["state"])
            _CFG["erf_root"] = raw.get("paths", {}).get("erf_root", defaults["erf_root"])
            _CFG["erf_fts_db"] = raw.get("paths", {}).get("erf_fts_db", defaults["erf_fts_db"])
    return _CFG


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 50
    kind: str | None = None       # filter: "raw_footage" | "published_output" | "audience_text"
    freshness: str | None = None  # post-filter: "unused" | "used_in_published"


class ExactQueryRequest(BaseModel):
    phrase: str
    limit: int = 50


class ClipSpec(BaseModel):
    media_id: str
    in_ms: int
    out_ms: int
    label: str | None = None
    source_relpath: str | None = None


class ExportRequest(BaseModel):
    clips: list[ClipSpec]
    format: str = "fcpxml"
    query: str | None = None


class SimilarRequest(BaseModel):
    source_id: str
    modality: str = "text"


class ShortsScoreRequest(BaseModel):
    query: str
    filters: dict = {}
    top_k: int = 20


class ShortsAssembleRequest(BaseModel):
    beat_plan: dict
    format: str = "fcpxml"
    render: bool = False
    reframe_strategy: str = "face_centroid"
    music_bed_path: str | None = None


class ShortsIdeasRequest(BaseModel):
    topic: str
    format_fit: str = "youtube_shorts"
    duration_target_s: int = 60
    filters: dict = {}
    template_id: str | None = None


class ShortsScanRequest(BaseModel):
    limit: int = 20
    emotional_filter: str | None = None
    platform: str = "youtube_shorts"
    force_rebuild_embeddings: bool = False


class ShortsIdeateRequest(BaseModel):
    max_proposals: int = 5
    target_duration_s: int = 60
    force_rebuild: bool = False


class ShortsAuthorRequest(BaseModel):
    pass1: bool = True
    pass2: bool = True
    limit: int | None = None
    force: bool = False


class ShortsGateRequest(BaseModel):
    composite_min: float = 0.50
    coherence_min: float = 0.60
    dry_run: bool = False


class NarratorRunRequest(BaseModel):
    concurrency: int = 3
    limit: int | None = None
    force: bool = False
    composite_min: float = 0.50
    coherence_min: float = 0.60


class TemplateCreateRequest(BaseModel):
    name: str
    platform: str
    beat_structure: list
    duration_target_s: float | None = None
    caption_style: str = "centre"
    tone_guidance: str | None = None
    exemplar_beat_plan_ids: list = []
    created_by: str = "api"


class TemplateUpdateRequest(BaseModel):
    name: str | None = None
    platform: str | None = None
    beat_structure: list | None = None
    duration_target_s: float | None = None
    caption_style: str | None = None
    tone_guidance: str | None = None
    exemplar_beat_plan_ids: list | None = None
    created_by: str = "api"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ERF", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
async def index():
    return _UI_HTML


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    from .query import run as run_query
    try:
        result = run_query(
            req.query,
            config_path=_CONFIG_PATH,
            kind=req.kind or None,
            rerank_mode="opportunistic",
        )
        # Post-filter by freshness (derived field, not in LanceDB)
        if req.freshness:
            result["results"] = [
                r for r in result["results"]
                if r.get("freshness") == req.freshness
            ]
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/visual")
async def query_visual(req: QueryRequest):
    from .query import run_visual
    try:
        result = run_visual(req.query, config_path=_CONFIG_PATH)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/ocr")
async def query_ocr(req: QueryRequest):
    from .query import run_ocr
    try:
        result = run_ocr(req.query, config_path=_CONFIG_PATH)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/similar")
async def query_similar(req: SimilarRequest):
    from .query import run_similar
    try:
        result = run_similar(req.source_id, modality=req.modality, config_path=_CONFIG_PATH)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/exact")
async def query_exact(req: ExactQueryRequest):
    from .fts import search_exact
    cfg = _cfg()
    fts_db = Path(cfg["erf_fts_db"])
    try:
        hits = search_exact(fts_db, req.phrase, limit=req.limit)
        return JSONResponse(content={"phrase": req.phrase, "results": hits})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model/status")
async def model_status():
    from .query import get_model_status
    return JSONResponse(content=get_model_status())


@app.get("/collections")
async def collections():
    cfg = _cfg()
    catalog = Path(cfg["catalog"])
    status_dir = catalog / "status"
    items = []
    if status_dir.exists():
        for f in sorted(status_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                items.append(data)
            except Exception:
                pass
    return JSONResponse(content={"collections": items, "count": len(items)})


@app.post("/export")
async def export_endpoint(req: ExportRequest):
    from .export import export
    cfg = _cfg()
    clips = [c.model_dump() for c in req.clips]
    try:
        manifest = export(
            clips,
            fmt=req.format,
            query=req.query,
            config_path=_CONFIG_PATH,
        )
        return JSONResponse(content=manifest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/media/thumbs/{media_id}/{filename}")
async def serve_thumb(media_id: str, filename: str):
    cfg = _cfg()
    erf_root = Path(cfg["erf_root"])
    thumb_path = erf_root / "media" / "thumbs" / media_id / filename
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(thumb_path), media_type="image/webp")


@app.get("/media/proxies/{media_id}.mp4")
async def serve_proxy(media_id: str):
    cfg = _cfg()
    erf_root = Path(cfg["erf_root"])
    proxy_path = erf_root / "media" / "proxies" / f"{media_id}.mp4"
    if not proxy_path.exists():
        raise HTTPException(status_code=404, detail="Proxy not found")
    return FileResponse(str(proxy_path), media_type="video/mp4")


# ── Template endpoints ───────────────────────────────────────────────────────


@app.post("/templates")
async def create_template_endpoint(req: TemplateCreateRequest):
    from .templates import create_template
    try:
        template_id = create_template(
            req.name, req.platform, req.beat_structure,
            duration_target_s=req.duration_target_s,
            caption_style=req.caption_style,
            tone_guidance=req.tone_guidance,
            exemplar_beat_plan_ids=req.exemplar_beat_plan_ids,
            created_by=req.created_by,
            config_path=_CONFIG_PATH,
        )
        from .templates import get_template
        record = get_template(template_id, config_path=_CONFIG_PATH)
        return JSONResponse(content=record, status_code=201)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/templates")
async def list_templates_endpoint(platform: str | None = None, active_only: bool = True):
    from .templates import list_templates
    try:
        templates = list_templates(platform=platform, active_only=active_only, config_path=_CONFIG_PATH)
        return JSONResponse(content={"templates": templates, "count": len(templates)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/templates/{template_id}")
async def get_template_endpoint(template_id: str):
    from .templates import get_template
    try:
        record = get_template(template_id, config_path=_CONFIG_PATH)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        return JSONResponse(content=record)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/templates/{template_id}")
async def update_template_endpoint(template_id: str, req: TemplateUpdateRequest):
    """Deactivate old version, create new version with updated fields. Returns new template_id."""
    from .templates import get_template, deactivate_template, create_template
    try:
        old = get_template(template_id, config_path=_CONFIG_PATH)
        if old is None:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

        deactivate_template(template_id, config_path=_CONFIG_PATH)

        new_fields = {
            "name": req.name if req.name is not None else old["name"],
            "platform": req.platform if req.platform is not None else old["platform"],
            "beat_structure": req.beat_structure if req.beat_structure is not None else old["beat_structure"],
            "duration_target_s": req.duration_target_s if req.duration_target_s is not None else old.get("duration_target_s"),
            "caption_style": req.caption_style if req.caption_style is not None else old.get("caption_style", "centre"),
            "tone_guidance": req.tone_guidance if req.tone_guidance is not None else old.get("tone_guidance"),
            "exemplar_beat_plan_ids": req.exemplar_beat_plan_ids if req.exemplar_beat_plan_ids is not None else old.get("exemplar_beat_plan_ids", []),
            "created_by": req.created_by,
        }

        new_id = create_template(
            new_fields["name"], new_fields["platform"], new_fields["beat_structure"],
            duration_target_s=new_fields["duration_target_s"],
            caption_style=new_fields["caption_style"],
            tone_guidance=new_fields["tone_guidance"],
            exemplar_beat_plan_ids=new_fields["exemplar_beat_plan_ids"],
            created_by=new_fields["created_by"],
            config_path=_CONFIG_PATH,
        )
        new_record = get_template(new_id, config_path=_CONFIG_PATH)
        return JSONResponse(content={"old_template_id": template_id, "new_template_id": new_id, "template": new_record})
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Review endpoints ─────────────────────────────────────────────────────────


class ReviewApproveRequest(BaseModel):
    reviewed_by: str


class ReviewRejectRequest(BaseModel):
    reviewed_by: str
    notes: str = ""


class ReviewEditRequest(BaseModel):
    beats: list
    reviewed_by: str
    diff_summary: str | None = None


@app.get("/review")
async def list_reviews_endpoint(state: str | None = None, platform: str | None = None):
    from .review import list_reviews, get_review
    from .beat_plans import latest_beat_plan
    try:
        reviews = list_reviews(state=state, platform=platform, config_path=_CONFIG_PATH)
        return JSONResponse(content={"reviews": reviews, "count": len(reviews)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/review/{review_id}")
async def get_review_endpoint(review_id: str):
    from .review import get_review
    from .beat_plans import latest_beat_plan
    from pathlib import Path as _Path
    try:
        rec = get_review(review_id, config_path=_CONFIG_PATH)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
        # Attach current beat_plan
        cfg = _cfg()
        catalog = _Path(cfg["catalog"])
        bp = latest_beat_plan(rec["beat_plan_id"], catalog)
        return JSONResponse(content={"review": rec, "beat_plan": bp})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/review/{review_id}/approve")
async def approve_review_endpoint(review_id: str, req: ReviewApproveRequest):
    from .review import approve_review
    try:
        rec = approve_review(review_id, req.reviewed_by, config_path=_CONFIG_PATH)
        return JSONResponse(content=rec)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/review/{review_id}/reject")
async def reject_review_endpoint(review_id: str, req: ReviewRejectRequest):
    from .review import reject_review
    try:
        rec = reject_review(review_id, req.reviewed_by, req.notes, config_path=_CONFIG_PATH)
        return JSONResponse(content=rec)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/review/{review_id}/edit")
async def edit_review_endpoint(review_id: str, req: ReviewEditRequest):
    from .review import edit_review
    try:
        result = edit_review(
            review_id, req.beats, req.reviewed_by,
            diff_summary=req.diff_summary,
            config_path=_CONFIG_PATH,
        )
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/review/{review_id}/assemble")
async def reassemble_review_endpoint(review_id: str):
    from .review import reassemble_review
    try:
        result = reassemble_review(review_id, config_path=_CONFIG_PATH)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Shorts endpoints ─────────────────────────────────────────────────────────


@app.post("/shorts/score")
async def shorts_score(req: ShortsScoreRequest):
    from .query import run as run_query
    from .atomicio import read_jsonl
    from .usage import is_used
    try:
        cfg = _cfg()
        catalog = Path(cfg["catalog"])

        raw = run_query(req.query, config_path=_CONFIG_PATH, rerank_mode="force")
        results = raw.get("results", [])[:req.top_k]

        # Cache tags and events per media_id to avoid repeated IO
        tags_cache: dict[str, dict] = {}  # chunk_id → tag_rec
        events_cache: dict[str, list] = {}

        for r in results:
            mid = r.get("media_id", "")
            cid = r.get("result_id", "")

            if mid not in tags_cache:
                tags_path = catalog / "tags" / f"{mid}.jsonl"
                for t in (read_jsonl(tags_path) if tags_path.exists() else []):
                    tags_cache[t["chunk_id"]] = t

            if mid not in events_cache:
                ev_path = catalog / "events" / f"{mid}.jsonl"
                events_cache[mid] = read_jsonl(ev_path) if ev_path.exists() else []

            tag = tags_cache.get(cid, {})
            in_ms, out_ms = int(r.get("in_ms", 0)), int(r.get("out_ms", 0))

            # Find overlapping event
            edit_role = None
            for ev in events_cache.get(mid, []):
                if max(ev["start_ms"], in_ms) < min(ev["end_ms"], out_ms):
                    edit_role = ev.get("edit_role")
                    break

            r["hook_score"] = tag.get("hook_score")
            r["quote_score"] = tag.get("quote_score")
            r["proof_score"] = tag.get("proof_score")
            r["edit_role"] = edit_role
            r["format_fit"] = tag.get("format_fit")
            r["used_before"] = is_used(mid, in_ms, out_ms, config_path=_CONFIG_PATH)
            r["suggested_padding_ms"] = None

        return JSONResponse(content={"query": req.query, "results": results})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/shorts/assemble")
async def shorts_assemble(req: ShortsAssembleRequest):
    import hashlib
    from datetime import datetime, timezone
    from .export import export
    from .usage import record_export
    from .beat_plans import append_beat_plan
    from .atomicio import read_jsonl
    import jsonschema
    try:
        cfg = _cfg()
        catalog = Path(cfg["catalog"])
        bp = dict(req.beat_plan)

        if "beats" not in bp or "topic" not in bp:
            raise HTTPException(status_code=422, detail="beat_plan must have 'beats' and 'topic'")

        # Validate against schema
        schema_path = Path("schemas/beat_plan.schema.json")
        if schema_path.exists():
            import json as _json
            bp_schema = _json.loads(schema_path.read_text())
            try:
                jsonschema.validate(bp, bp_schema)
            except jsonschema.ValidationError as ve:
                raise HTTPException(status_code=422, detail=f"beat_plan schema error: {ve.message}")

        # Assign bp_id if missing
        if "bp_id" not in bp:
            now = datetime.now(timezone.utc)
            suffix = hashlib.blake2b(bp["topic"].encode(), digest_size=3).hexdigest()
            bp["bp_id"] = f"bp_{now.strftime('%Y%m%dT%H%M%SZ')}_{suffix}"
        if "created_at_utc" not in bp:
            bp["created_at_utc"] = datetime.now(timezone.utc).isoformat()

        beats = bp["beats"]

        # Prefer shortform chunks for hook/call_to_action beats: if a shortform
        # chunk covers the beat's span, use its tighter in_ms/out_ms.
        _SHORTFORM_ROLES = {"hook", "call_to_action"}
        _shortform_by_media: dict[str, list[dict]] = {}

        def _get_shortform_chunks(mid: str) -> list[dict]:
            if mid not in _shortform_by_media:
                cp = catalog / "chunks" / f"{mid}.jsonl"
                all_c = read_jsonl(cp) if cp.exists() else []
                _shortform_by_media[mid] = [
                    c for c in all_c
                    if c.get("chunk_rule", "").endswith("_short")
                ]
            return _shortform_by_media[mid]

        def _resolve_beat_boundaries(b: dict) -> tuple[int, int]:
            roles = set(b.get("edit_role") or [])
            if not roles.intersection(_SHORTFORM_ROLES):
                return b["in_ms"], b["out_ms"]
            mid = b["media_id"]
            beat_in, beat_out = b["in_ms"], b["out_ms"]
            best_overlap = 0
            best_in, best_out = beat_in, beat_out
            for sc in _get_shortform_chunks(mid):
                ov = max(0, min(beat_out, sc["end_ms"]) - max(beat_in, sc["start_ms"]))
                if ov > best_overlap:
                    best_overlap = ov
                    best_in, best_out = sc["start_ms"], sc["end_ms"]
            return best_in, best_out

        # Crosstalk out-point trim: if a foreign speaker turn starts within the
        # final 2s of a beat, pull out_ms back to the start of that turn.
        _diar_cache: dict[str, list[dict]] = {}
        import json as _json_diar

        def _get_diar_turns(mid: str) -> list[dict]:
            if mid not in _diar_cache:
                dp = catalog / "asr" / f"{mid}.diar.jsonl"
                _diar_cache[mid] = (
                    [_json_diar.loads(l) for l in dp.read_text().splitlines() if l.strip()]
                    if dp.exists() else []
                )
            return _diar_cache[mid]

        _words_cache: dict[str, list[dict]] = {}

        def _get_words(mid: str) -> list[dict]:
            if mid not in _words_cache:
                wp = catalog / "asr" / f"{mid}.transcript.jsonl"
                _words_cache[mid] = (
                    [_json_diar.loads(l) for l in wp.read_text().splitlines() if l.strip()]
                    if wp.exists() else []
                )
            return _words_cache[mid]

        def _trim_crosstalk_out(mid: str, in_ms: int, out_ms: int, lookahead_ms: int = 2000) -> tuple[int, bool]:
            # `<=` (not `<`) on the out_ms comparisons makes re-detection idempotent:
            # when re-rendering a stored beat plan whose out_ms was ALREADY pulled back
            # to a foreign-speaker onset by a prior run, the foreign turn now starts
            # exactly at out_ms. Strict `<` would miss it, leave _crosstalk_trimmed False,
            # and let the render OUT-pad re-expand straight back into the crosstalk.
            turns = [t for t in _get_diar_turns(mid) if t["start_ms"] <= out_ms and t["end_ms"] > in_ms]
            if not turns:
                return out_ms, False
            spk_ms: dict[str, int] = {}
            for t in turns:
                spk_ms[t["speaker_label"]] = spk_ms.get(t["speaker_label"], 0) + (
                    min(t["end_ms"], out_ms) - max(t["start_ms"], in_ms)
                )
            primary = max(spk_ms, key=spk_ms.__getitem__)
            words = _get_words(mid)
            for t in sorted(turns, key=lambda x: x["start_ms"]):
                if out_ms - lookahead_ms <= t["start_ms"] <= out_ms and t["speaker_label"] != primary:
                    # Skip phantom diar turns (foreign label, no transcribed words) — a
                    # breath/noise the diarizer mislabeled. Trimming on them chops the
                    # primary speaker's own continuing sentence. See D-Render-09.
                    if words and not any(
                        w.get("speaker_label") == t["speaker_label"]
                        and t["start_ms"] <= int(w["start_ms"]) < t["end_ms"]
                        for w in words
                    ):
                        continue
                    return t["start_ms"], True
            return out_ms, False

        clips = []
        for b in beats:
            in_ms, out_ms = _resolve_beat_boundaries(b)
            lookahead = 4000 if b.get("edit_role") == "quote" else 2000
            out_ms, _ct = _trim_crosstalk_out(b["media_id"], in_ms, out_ms, lookahead_ms=lookahead)
            b["out_ms"] = out_ms  # keep beat in sync for total_duration_s
            clips.append({
                "media_id": b["media_id"],
                "in_ms": in_ms,
                "out_ms": out_ms,
                "label": f"{b.get('edit_role','')} {(b.get('note') or '')[:40]}".strip(),
                "source_relpath": b.get("source_relpath"),
                "_is_broll": b.get("edit_role") == "b_roll",
                "_crosstalk_trimmed": _ct,
            })

        export_manifest = export(clips, fmt=req.format, query=bp["topic"], config_path=_CONFIG_PATH)
        record_export(export_manifest["export_id"], clips, config_path=_CONFIG_PATH)
        append_beat_plan(bp, catalog)

        # Auto-create review_queue row (pending)
        from .review import _make_review_id, _load_config as _review_cfg, _connect as _review_connect, _emit_journal as _review_journal
        from datetime import datetime as _dt, timezone as _tz
        import sqlite3 as _sqlite3, pathlib as _pl
        _rv_id = _make_review_id()
        _rv_cfg = _review_cfg(_CONFIG_PATH)
        _rv_state = _pl.Path(_rv_cfg["state"])
        _rv_now = _dt.now(_tz.utc).isoformat()
        _rv_platform = bp.get("format_fit", "any")
        _rv_revision = bp.get("revision", 0)
        _rv_tmpl_id = bp.get("template_id")
        _rv_conn = _review_connect(_rv_state / "erf.db")
        _rv_conn.execute(
            "INSERT INTO review_queue (review_id, export_id, beat_plan_id, beat_plan_revision, template_id, platform, state, created_at_utc, output_otio_relpath) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (_rv_id, export_manifest["export_id"], bp["bp_id"], _rv_revision, _rv_tmpl_id, _rv_platform, _rv_now, export_manifest["output_relpath"]),
        )
        _rv_conn.commit()
        _rv_conn.close()
        _review_journal(_rv_state, _rv_id, "create", {"export_id": export_manifest["export_id"], "beat_plan_id": bp["bp_id"]})

        total_s = sum((b["out_ms"] - b["in_ms"]) / 1000 for b in beats)
        resp: dict = {
            "export_id": export_manifest["export_id"],
            "review_id": _rv_id,
            "output_relpath": export_manifest["output_relpath"],
            "beat_plan_id": bp["bp_id"],
            "total_duration_s": round(total_s, 2),
        }

        if req.render:
            from .render import render as do_render
            from datetime import datetime as _dt2, timezone as _tz2
            import sqlite3 as _sq2, json as _j2
            _catalog = Path(cfg["catalog"])
            _proxy_dir = Path("media/proxies")
            _exports_dir = _catalog / "exports"
            _exp_id = export_manifest["export_id"]
            _jdb = Path(cfg["state"]) / "journal.db"

            def _render_journal(event_type: str, extra: dict | None = None) -> None:
                try:
                    _c = _sq2.connect(str(_jdb))
                    _c.execute("PRAGMA journal_mode = TRUNCATE")
                    _c.execute(
                        # Phase 1 (D-Log-04): trace-only, not an outcome disposition
                        # -> events_trace, not the outcome-only `events` table (§3.1).
                        "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) VALUES (?, ?, 'erf', 'render', ?, ?, ?)",
                        (_dt2.now(_tz2.utc).isoformat(), f"erf:render:{_exp_id}", event_type, _exp_id, _j2.dumps({"export_id": _exp_id, **(extra or {})})),
                    )
                    _c.commit(); _c.close()
                except Exception:
                    pass  # best-effort

            # B-roll expansion: inject b-roll clips mid-beat when candidates qualify
            _render_clips = list(clips)
            _render_beats = list(beats)
            try:
                _ab_toml: dict = {}
                if _CONFIG_PATH.exists():
                    with open(_CONFIG_PATH, "rb") as _ab_fh:
                        _ab_toml = tomllib.load(_ab_fh)
                _ab_cfg = _ab_toml.get("assembly", {}).get("broll", {})
                if _ab_cfg.get("enabled", True):
                    _ab_min = float(_ab_cfg.get("min_combined_score", 0.3))
                    _render_clips = []
                    _render_beats = []
                    for _b, _c in zip(beats, clips):
                        _cands = _b.get("b_roll_candidates") or []
                        _valid = [x for x in _cands if x.get("combined_score", 0) >= _ab_min]
                        _c_in = int(_c["in_ms"])
                        _c_out = int(_c["out_ms"])
                        _c_dur = _c_out - _c_in
                        if _valid and _c_dur >= 6000:
                            _top = _valid[0]
                            _br_mid = _top["media_id"]
                            _br_t = int(_top["time_ms"])
                            _br_in = max(0, _br_t - 2000)
                            _br_out = _br_t + 2000
                            _br_kf = _top["keyframe_id"]
                            _br_proxy = _proxy_dir / f"{_br_mid}.mp4"
                            if _br_proxy.exists():
                                _mid_ms = _c_in + _c_dur // 2
                                # first half of dialogue
                                _render_clips.append({**_c, "out_ms": _mid_ms})
                                _render_beats.append({**_b, "out_ms": _mid_ms})
                                # b-roll overlay (muted audio)
                                _render_clips.append({
                                    "media_id": _br_mid, "in_ms": _br_in, "out_ms": _br_out,
                                    "label": f"broll {_br_kf}", "_is_broll": True,
                                })
                                _render_beats.append({
                                    "beat_index": -1, "chunk_id": f"broll:{_br_kf}",
                                    "media_id": _br_mid, "in_ms": _br_in, "out_ms": _br_out,
                                    "edit_role": "b_roll",
                                })
                                # second half of dialogue
                                _render_clips.append({**_c, "in_ms": _mid_ms})
                                _render_beats.append({**_b, "in_ms": _mid_ms})
                                _b["b_roll_selected"] = _br_kf
                                continue
                        _render_clips.append(_c)
                        _render_beats.append(_b)
            except Exception:
                _render_clips = list(clips)
                _render_beats = list(beats)

            _render_bp = {**bp, "beats": _render_beats}
            _render_journal("start", {"strategy": req.reframe_strategy})
            try:
                render_result = do_render(
                    beat_plan=_render_bp,
                    clips=_render_clips,
                    export_id=_exp_id,
                    catalog=_catalog,
                    proxy_dir=_proxy_dir,
                    output_dir=_exports_dir,
                    reframe_strategy=req.reframe_strategy,
                    music_bed_path=req.music_bed_path,
                    journal_db=_jdb,
                    config_path=_CONFIG_PATH,
                )
            except Exception as _re:
                _render_journal("error", {"error": str(_re)[:200]})
                raise
            resp["output_mp4_relpath"] = str(
                Path(render_result["output_mp4_path"]).relative_to(Path(".").resolve())
            )
            resp["render_duration_s"] = render_result["duration_s"]
            _render_journal("ok", {"output_mp4_relpath": resp["output_mp4_relpath"], "duration_s": render_result["duration_s"]})
            # Patch export_manifest with mp4 path
            from .atomicio import write_json as _write_json
            import json as _json_patch
            _manifest_path = Path(cfg["catalog"]) / "exports" / f"{_exp_id}.json"
            if _manifest_path.exists():
                _mdata = _json_patch.loads(_manifest_path.read_text(encoding="utf-8"))
                _mdata["output_mp4_relpath"] = resp["output_mp4_relpath"]
                _write_json(_manifest_path, _mdata, indent=2)
            # Update review_queue with mp4 path
            _rv_conn2 = _review_connect(_rv_state / "erf.db")
            _rv_conn2.execute(
                "UPDATE review_queue SET output_mp4_relpath=? WHERE review_id=?",
                (resp["output_mp4_relpath"], _rv_id),
            )
            _rv_conn2.commit()
            _rv_conn2.close()

        return JSONResponse(content=resp)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/shorts/ideas")
async def shorts_ideas(req: ShortsIdeasRequest):
    import hashlib, json as _json
    from datetime import datetime, timezone
    from .query import run as run_query
    from .atomicio import read_jsonl
    from .llm import complete
    from .templates import get_template
    import jsonschema
    try:
        cfg = _cfg()
        catalog = Path(cfg["catalog"])

        # Resolve template if provided
        tmpl: dict | None = None
        if req.template_id:
            tmpl = get_template(req.template_id, config_path=_CONFIG_PATH)
            if tmpl is None or not tmpl.get("active", True):
                raise HTTPException(status_code=404, detail=f"Template {req.template_id} not found or inactive")

        # 1. Score results
        raw = run_query(req.topic, config_path=_CONFIG_PATH)
        results = raw.get("results", [])[:30]

        tags_cache: dict[str, dict] = {}
        events_cache: dict[str, list] = {}
        chunks_cache: dict[str, dict] = {}   # chunk_id -> chunk record (for speaker_label fallback)
        identity_cache: dict[str, dict] = {} # media_id -> {chunk_id: person_id} from identity_overlay
        manifest_cache: dict[str, dict] = {} # media_id -> manifest
        persons_cache: dict[str, str] = {}   # person_id -> display_name
        atoms = []

        for r in results:
            mid = r.get("media_id", "")
            cid = r.get("result_id", "")
            if mid not in tags_cache:
                tp = catalog / "tags" / f"{mid}.jsonl"
                for t in (read_jsonl(tp) if tp.exists() else []):
                    tags_cache[t["chunk_id"]] = t
            if mid not in events_cache:
                ep = catalog / "events" / f"{mid}.jsonl"
                events_cache[mid] = read_jsonl(ep) if ep.exists() else []
            if mid not in manifest_cache:
                mp = catalog / "media_manifest" / f"{mid}.json"
                manifest_cache[mid] = _json.loads(mp.read_text()) if mp.exists() else {}
            if mid not in identity_cache:
                ip = catalog / "identity_overlay" / f"{mid}.json"
                if ip.exists():
                    _io = _json.loads(ip.read_text())
                    identity_cache[mid] = {
                        cs["chunk_id"]: cs["person_id"]
                        for cs in _io.get("chunk_speaker", [])
                    }
                else:
                    identity_cache[mid] = {}
            # Index chunks by chunk_id for speaker_label fallback (lazy, per media_id)
            _chunk_sentinel = f"__loaded_{mid}"
            if _chunk_sentinel not in chunks_cache:
                cp = catalog / "chunks" / f"{mid}.jsonl"
                for c in (read_jsonl(cp) if cp.exists() else []):
                    chunks_cache[c["chunk_id"]] = c
                chunks_cache[_chunk_sentinel] = {}

            tag = tags_cache.get(cid, {})
            in_ms, out_ms = int(r.get("in_ms", 0)), int(r.get("out_ms", 0))
            edit_role = None
            for ev in events_cache.get(mid, []):
                if max(ev["start_ms"], in_ms) < min(ev["end_ms"], out_ms):
                    edit_role = ev.get("edit_role")
                    break

            # Null-coerce scores so DeepSeek sees numbers, not nulls
            hook_score = tag.get("hook_score") if tag.get("hook_score") is not None else 0.0
            quote_score = tag.get("quote_score") if tag.get("quote_score") is not None else 0.0
            proof_score = tag.get("proof_score") if tag.get("proof_score") is not None else 0.0
            score = float(r.get("score", 0))
            composite_score = round(0.4 * score + 0.3 * hook_score + 0.2 * quote_score + 0.1 * proof_score, 4)

            # Speaker name: identity_overlay chunk_speaker → display_name; fallback to speaker_label
            # identity_overlay.chunk_speaker is authoritative — chunks.speaker_person_id is stale, do not use.
            person_id = identity_cache.get(mid, {}).get(cid)
            speaker_name: str | None = None
            if person_id:
                if person_id not in persons_cache:
                    pp = catalog / "persons" / f"{person_id}.json"
                    if pp.exists():
                        pd = _json.loads(pp.read_text())
                        persons_cache[person_id] = pd.get("display_name") or person_id
                    else:
                        persons_cache[person_id] = person_id
                speaker_name = persons_cache[person_id]
            else:
                chunk_rec = chunks_cache.get(cid, {})
                if chunk_rec.get("speaker_label"):
                    speaker_name = chunk_rec["speaker_label"]

            # Source context: prefer source_note, fallback to filename
            mfest = manifest_cache.get(mid, {})
            _ctx_raw = mfest.get("source_note") or mfest.get("relpath", "")
            source_context: str | None = _ctx_raw[:60] or None

            atom: dict = {
                "chunk_id": cid,
                "media_id": mid,
                "in_ms": in_ms,
                "out_ms": out_ms,
                "duration_s": round((out_ms - in_ms) / 1000, 1),
                "score": score,
                "composite_score": composite_score,
                "snippet": r.get("snippet", "")[:400],
                "hook_score": hook_score,
                "quote_score": quote_score,
                "proof_score": proof_score,
                "edit_role": edit_role,
                "format_fit": tag.get("format_fit"),
            }
            if speaker_name:
                atom["speaker_name"] = speaker_name
            if source_context:
                atom["source_context"] = source_context
            atoms.append(atom)

        # Sort by composite score so best atoms surface first before 20-item truncation
        atoms.sort(key=lambda a: a["composite_score"], reverse=True)

        # 2. LLM beat selection
        format_fit = tmpl["platform"] if tmpl else req.format_fit
        if tmpl:
            duration_s = tmpl.get("target_duration_s") or sum(
                b.get("duration_target_s", 0) for b in tmpl.get("beat_structure", [])
            ) or req.duration_target_s
        else:
            duration_s = req.duration_target_s

        guidance_prefix = ""
        _guidance_path = Path("state/guidance.md")
        if _guidance_path.exists():
            _guidance_text = _guidance_path.read_text(encoding="utf-8").strip()
            if _guidance_text:
                guidance_prefix = f"Editorial guidance (follow this):\n{_guidance_text}\n---\n\n"

        system_prompt = guidance_prefix + (
            f"You are a short-form video producer. Given a list of video atoms with scores, "
            f"select and sequence atoms into a beat_plan for a {format_fit} video of "
            f"approximately {duration_s}s about: {req.topic}. "
            "Respond ONLY with a valid JSON beat_plan object. "
            "For each beat, set 'note' to your selection rationale in max 40 words. "
            "beat_plan schema: {bp_id: string, created_at_utc: iso8601, revision: 0, "
            "supersedes_revision: null, topic: string, format_fit: string, duration_target_s: int, "
            "total_duration_s: number, assembler_note: string|null, template_id: string|null, "
            "beats: [{beat_index, chunk_id, media_id, in_ms, out_ms, edit_role, note, "
            "hook_score, quote_score, proof_score, suggested_padding_ms}]}. "
            "edit_role values: hook|body|b_roll|proof|quote|transition|call_to_action|exposition. "
            "The LLM reasons over atoms — do NOT hallucinate timecodes, use only those provided."
        )

        if tmpl:
            beats_desc = "\n".join(
                f"  {i+1}. {b['role']} (~{b['duration_target_s']}s)"
                + (f": {b['guidance']}" if b.get('guidance') else "")
                for i, b in enumerate(tmpl["beat_structure"])
            )
            system_prompt += (
                f"\n\nUse this template structure:\n"
                f"Platform: {tmpl['platform']}\n"
                f"Target duration: {duration_s}s\n"
                f"Tone: {tmpl.get('tone_guidance') or 'not specified'}\n"
                f"Beat structure:\n{beats_desc}\n"
                f"Select atoms that fit each beat role and target duration from the template."
            )

        # Wire exemplar beat plans into prompt (up to 2) when template provides them
        if tmpl and tmpl.get("exemplar_beat_plan_ids"):
            from .beat_plans import latest_beat_plan
            for ex_bp_id in tmpl["exemplar_beat_plan_ids"][:2]:
                ex_bp = latest_beat_plan(ex_bp_id, catalog)
                if ex_bp:
                    system_prompt += (
                        f"\n\nExample beat_plan (for structural reference only):\n"
                        f"{_json.dumps(ex_bp, indent=2)}"
                    )

        now = datetime.now(timezone.utc)
        suffix = hashlib.blake2b(req.topic.encode(), digest_size=3).hexdigest()
        bp_id = f"bp_{now.strftime('%Y%m%dT%H%M%SZ')}_{suffix}"

        llm_raw = complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Atoms:\n{_json.dumps(atoms[:20], indent=2)}"},
            ],
            json_only=True,
        )

        try:
            bp = _json.loads(llm_raw)
        except Exception:
            raise HTTPException(status_code=422, detail=f"LLM returned invalid JSON: {llm_raw[:200]}")

        # Unwrap if LLM returned {"beat_plan": {...}} instead of the object directly
        if isinstance(bp, dict) and "beats" not in bp:
            inner = bp.get("beat_plan")
            if isinstance(inner, dict) and "beats" in inner:
                bp = inner

        # Fill required fields
        bp.setdefault("schema_version", "1.2.0")
        bp["schema_version"] = "1.2.0"
        bp.setdefault("bp_id", bp_id)
        bp.setdefault("created_at_utc", now.isoformat())
        bp.setdefault("revision", 0)
        bp.setdefault("supersedes_revision", None)
        bp.setdefault("topic", req.topic)
        bp.setdefault("format_fit", format_fit)
        bp.setdefault("duration_target_s", int(duration_s))
        bp["template_id"] = req.template_id  # null if no template

        # Validate
        schema_path = Path("schemas/beat_plan.schema.json")
        if schema_path.exists():
            bp_schema = _json.loads(schema_path.read_text())
            try:
                jsonschema.validate(bp, bp_schema)
            except jsonschema.ValidationError as ve:
                return JSONResponse(
                    status_code=422,
                    content={"error": ve.message, "raw_llm_output": llm_raw[:500], "bp_draft": bp},
                )

        # Duration guard: warn when a selected atom is < 50% of the template's per-beat target duration.
        # Does not reject — appends a visible warning to the beat's note so reviewers catch it.
        if tmpl:
            import logging as _logging
            _dur_log = _logging.getLogger(__name__)
            _role_targets: dict[str, float] = {
                b["role"]: float(b["duration_target_s"])
                for b in tmpl.get("beat_structure", [])
                if b.get("role") and b.get("duration_target_s")
            }
            for _beat in bp.get("beats", []):
                _role = _beat.get("edit_role", "")
                _target = _role_targets.get(_role)
                if _target is None:
                    continue
                _actual = (_beat.get("out_ms", 0) - _beat.get("in_ms", 0)) / 1000.0
                if _actual < 0.5 * _target:
                    _dur_log.warning(
                        "beat_plan duration shortfall: beat_index=%s edit_role=%s actual=%.1fs target=%.0fs",
                        _beat.get("beat_index"), _role, _actual, _target,
                    )
                    _warn = f" [DURATION WARNING: {_actual:.1f}s selected, template target {_target:.0f}s — replace before rendering]"
                    _beat["note"] = (_beat.get("note") or "") + _warn

        # B-roll enrichment: cross-modal visual query per beat (best-effort)
        try:
            from .query import embed_visual_query as _embed_vq, search_visual_lance as _svl
            _broll_toml: dict = {}
            if _CONFIG_PATH.exists():
                with open(_CONFIG_PATH, "rb") as _fh:
                    _broll_toml = tomllib.load(_fh)
            _broll_cfg = _broll_toml.get("assembly", {}).get("broll", {})
            _broll_on = _broll_cfg.get("enabled", True)
            _min_comb = float(_broll_cfg.get("min_combined_score", 0.3))
            _max_cands = int(_broll_cfg.get("max_broll_per_beat", 3))
            _state_p = Path(cfg["state"])
            _lance_p = Path(_broll_toml.get("paths", {}).get("index", cfg.get("erf_root", "/workspace/erf") + "/index/lance"))

            if _broll_on:
                _aes_cache: dict[str, dict] = {}

                def _get_aes(mid: str, kid: str) -> "float | None":
                    if mid not in _aes_cache:
                        qp = catalog / "quality" / f"{mid}.jsonl"
                        _aes_cache[mid] = {
                            r["keyframe_id"]: r.get("aesthetic_score")
                            for r in (read_jsonl(qp) if qp.exists() else [])
                            if r.get("keyframe_id")
                        }
                    return _aes_cache[mid].get(kid)

                for _beat in bp.get("beats", []):
                    _snippet = (_beat.get("note") or "")[:120].strip()
                    _beat_mid = _beat.get("media_id", "")
                    _beat["b_roll_selected"] = None
                    if not _snippet:
                        _beat["b_roll_candidates"] = None
                        continue
                    try:
                        _qvec = _embed_vq(_snippet, _state_p)
                        _hits = _svl(_qvec, _lance_p, top_k=10)
                        # Rank-normalize similarity: visual_v1 embeddings are not L2-normalized
                        # so _distance > 1.0 is normal; use within-result rank to get 0-1 sim.
                        _dists = [float(_h.get("score", 1.0)) for _h in _hits]
                        _min_d = min(_dists) if _dists else 0.0
                        _max_d = max(_dists) if _dists else 1.0
                        _d_span = _max_d - _min_d
                        _cands = []
                        for _h, _dist in zip(_hits, _dists):
                            if _h["media_id"] == _beat_mid:
                                continue
                            _sim = round((_max_d - _dist) / _d_span if _d_span > 1e-6 else 0.5, 4)
                            _aes = _get_aes(_h["media_id"], _h["keyframe_id"])
                            _aes_v = _aes if _aes is not None else 0.5
                            _comb = round(_sim * _aes_v, 4)
                            if _comb < _min_comb:
                                continue
                            _cands.append({
                                "keyframe_id": _h["keyframe_id"],
                                "media_id": _h["media_id"],
                                "time_ms": int(_h["time_ms"]),
                                "similarity_score": _sim,
                                "aesthetic_score": _aes,
                                "combined_score": _comb,
                                "thumb_url": f"/{_h.get('image_relpath', '')}",
                            })
                        _cands.sort(key=lambda c: c["combined_score"], reverse=True)
                        _beat["b_roll_candidates"] = _cands[:_max_cands] if _cands else None
                    except Exception:
                        _beat["b_roll_candidates"] = None
        except Exception:
            pass  # b-roll enrichment is best-effort; never fail the response

        return JSONResponse(content=bp)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


_GUIDANCE_PATH = Path("state/guidance.md")


@app.get("/guidance")
async def get_guidance():
    if _GUIDANCE_PATH.exists():
        content = _GUIDANCE_PATH.read_text(encoding="utf-8")
    else:
        content = ""
    return JSONResponse(content={"content": content})


class GuidanceWriteRequest(BaseModel):
    content: str


@app.put("/guidance")
async def put_guidance(req: GuidanceWriteRequest):
    from .atomicio import write_tmp_rename
    data = req.content.encode("utf-8")
    write_tmp_rename(_GUIDANCE_PATH, data)
    return JSONResponse(content={"ok": True, "bytes": len(data)})


@app.get("/shorts/gaps")
async def shorts_gaps(media_id: str | None = None):
    from .atomicio import read_jsonl
    from pathlib import Path as _Path
    try:
        cfg = _cfg()
        catalog = _Path(cfg["catalog"])
        counts: dict[str, int] = {r: 0 for r in
            ["hook","body","b_roll","proof","quote","transition","call_to_action","exposition"]}

        if media_id:
            paths = [catalog / "events" / f"{media_id}.jsonl"]
        else:
            events_dir = catalog / "events"
            paths = list(events_dir.glob("*.jsonl")) if events_dir.exists() else []

        for p in paths:
            for ev in read_jsonl(p):
                for role in (ev.get("edit_role") or []):
                    if role in counts:
                        counts[role] += 1

        gaps = [role for role, cnt in counts.items() if cnt < 2]
        return JSONResponse(content={"gaps": gaps, "counts": counts})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/shorts/scan")
async def shorts_scan(req: ShortsScanRequest):
    try:
        cfg = _cfg()
        catalog = Path(cfg["catalog"])
        lance_path = Path(cfg.get("index", "/workspace/erf/index/lance"))
        state = Path(cfg["state"])
        journal_db = state / "journal.db"
        from .scan import run as scan_run
        threads = scan_run(
            catalog=catalog,
            lance_path=lance_path,
            journal_db=journal_db,
            limit=req.limit,
            emotional_filter=req.emotional_filter,
            platform=req.platform,
            force_rebuild_embeddings=req.force_rebuild_embeddings,
        )
        return JSONResponse(content={"threads": threads, "count": len(threads)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _ideate_sync(job_id: str, params: dict, cfg: dict) -> dict:
    """Blocking survey+develop pipeline. Called from a daemon thread by shorts_ideate.

    Updates _ideate_jobs[job_id] with per-proposal progress so the poll endpoint
    can report partial results while the job is still running.
    """
    import hashlib
    import os as _os
    import uuid as _uuid
    from math import ceil
    from datetime import datetime, timezone
    from .llm import complete

    max_proposals = params["max_proposals"]
    target_duration_s = params["target_duration_s"]

    catalog = Path(cfg["catalog"])
    digest_path = catalog / "event_digest.jsonl"

    # Load event digest
    raw_digest = [json.loads(l) for l in digest_path.read_text().splitlines() if l.strip()]
    digest_lookup = {ev["event_id"]: ev for ev in raw_digest}
    all_event_ids = set(digest_lookup.keys())

    # Format digest as CSV
    csv_rows = []
    for ev in raw_digest:
        eid = ev["event_id"]
        t_s = ev.get("t_s", 0)
        dur_s = ev.get("duration_s", 0)
        beat = ev.get("emotional_beat") or ""
        speaker = ev.get("speaker_name") or ""
        hook = ev.get("hook_potential")
        quote = ev.get("quote_potential")
        role = ev.get("narrative_role_suggestion") or ""
        summary = (ev.get("summary") or "")[:60].replace(",", " ")
        hook_str = f"{hook:.2f}" if hook is not None else "null"
        quote_str = f"{quote:.2f}" if quote is not None else "null"
        csv_rows.append(
            f"{eid},{t_s:.1f},{dur_s:.1f},{beat},{speaker},{hook_str},{quote_str},{role},{summary}"
        )
    digest_csv = "event_id,t_s,dur_s,beat,speaker,hook,quote,role,summary\n" + "\n".join(csv_rows)

    # Load media cards (with visual_grounding block if present)
    cards_dir = catalog / "media_cards"
    cards_text = ""
    if cards_dir.exists():
        for cf in sorted(cards_dir.glob("*.json")):
            try:
                card = json.loads(cf.read_text())
                mid = card.get("media_id", cf.stem)
                standout = ", ".join(card.get("standout_event_ids") or []) or "none"
                vg = card.get("visual_grounding") or {}
                vis_line = ""
                if vg:
                    subjects = ", ".join(vg.get("on_screen_subjects", [])[:4]) or "none"
                    vis_summary = (vg.get("visual_summary") or "")[:120]
                    vis_line = (
                        f"Visual: {vg.get('footage_type','')} | {vg.get('shot_type','')} | "
                        f"on-screen: {subjects}\n"
                        f"  {vis_summary}\n"
                    )
                cards_text += (
                    f"## {mid}\n"
                    f"{card.get('synopsis','')}\n"
                    f"Speakers: {card.get('dominant_speakers', [])}\n"
                    f"Tenor: {card.get('emotional_tenor','')}\n"
                    f"Events: {card.get('event_count','')}\n"
                    f"Standout: {standout}\n"
                    f"{vis_line}\n"
                )
            except Exception as exc:
                logger.warning("media card load error %s: %s", cf.name, exc)
    if not cards_text:
        logger.warning("/shorts/ideate: no media cards found; survey runs without card summaries")

    # ── Pass 1: Survey ─────────────────────────────────────────────────────────
    survey_system = (
        "You are a documentary story analyst. Survey this entire corpus of video events.\n"
        "Identify the genuine short-form video stories (30–90 seconds) that ACTUALLY EXIST in this material"
        " — stories with a real arc, a hook, and a payoff or reveal.\n"
        "CRITICAL: Each media card includes a 'Visual:' line showing what is ACTUALLY ON SCREEN. "
        "The subjects people discuss may be off-screen. A clip tagged 'talking_head' with no insect "
        "in on-screen subjects cannot show a fly, regardless of what is discussed.\n"
        "You are explicitly allowed to find few — even zero. Only propose stories the footage can actually support.\n"
        "For each story you identify, give:\n"
        "  - title: short punchy title\n"
        "  - logline: what is this story in one sentence (concrete, not abstract)\n"
        "  - emotional_arc: what changes emotionally from first beat to last\n"
        "  - anchor_event_ids: 2–5 event_ids that form the backbone of this story (must exist in the digest)\n"
        "  - viability_note: honest assessment — how strongly does the corpus support this story? "
        "If a core story element requires footage not yet produced (VFX, reshoots, off-screen subjects), "
        "say so explicitly.\n"
        "If you cannot find a story that truly works with available footage, say so explicitly.\n"
        "Do NOT invent moments not in the digest. Return ONLY a JSON array (no markdown). Max 10 seeds.\n"
        'Each seed: {"title": str, "logline": str, "emotional_arc": str, "anchor_event_ids": [str, ...], "viability_note": str}'
    )
    survey_user = (
        f"MEDIA CARDS:\n{cards_text}\n"
        f"EVENT DIGEST (CSV):\n{digest_csv}\n\n"
        "Identify the genuine stories. Return JSON array only."
    )

    raw_survey = complete(
        [{"role": "system", "content": survey_system},
         {"role": "user", "content": survey_user}],
        json_only=True,
    )

    # Parse seeds
    seeds: list[dict] = []
    try:
        parsed = json.loads(raw_survey)
        if isinstance(parsed, list):
            seeds = parsed
        elif isinstance(parsed, dict) and "seeds" in parsed:
            seeds = parsed["seeds"]
    except Exception:
        start = raw_survey.find("[")
        end = raw_survey.rfind("]")
        if start != -1 and end != -1:
            try:
                seeds = json.loads(raw_survey[start : end + 1])
            except Exception:
                pass
        if not seeds:
            logger.error("/shorts/ideate survey: could not parse JSON from response")

    # Validate and filter seeds
    valid_seeds: list[dict] = []
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        anchor_ids = seed.get("anchor_event_ids")
        if not isinstance(anchor_ids, list):
            continue
        # Keep only ids that actually exist in the digest
        valid_ids = [aid for aid in anchor_ids if aid in all_event_ids]
        if len(valid_ids) < 2:
            logger.warning("seed '%s' dropped: only %d valid anchor_event_ids", seed.get("title"), len(valid_ids))
            continue
        seed["anchor_event_ids"] = valid_ids
        valid_seeds.append(seed)

    if not valid_seeds:
        return {"survey_seeds_found": 0, "proposals_developed": 0, "proposal_ids": []}

    _ideate_jobs[job_id]["survey_seeds_found"] = len(valid_seeds)

    # ── Pass 2: Develop each seed ──────────────────────────────────────────────
    proposals_dir = catalog / "story_proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)

    beat_cap = ceil(target_duration_s / 3)
    beat_approx = max(1, ceil(target_duration_s / 4))

    develop_system = (
        "You are a short-form video producer. Develop the story seed into a full beat-by-beat story proposal.\n"
        f"Beat count discipline: ~1 beat per 4 seconds ({target_duration_s}s ≈ {beat_approx} beats; "
        f"NEVER exceed {beat_cap} beats total).\n"
        "VISUAL HONESTY MANDATE: Each clip's media card includes a 'Visual:' line showing footage_type and "
        "what is ACTUALLY ON SCREEN. Do not direct any shot the footage does not contain. "
        "A talking_head clip with no insect in on_screen_subjects cannot show a fly. "
        "For every beat, cite in visual_basis the event_id(s) whose visual content supports the intended shot. "
        "If a beat's shot cannot be grounded in available footage, set availability_status accordingly — "
        "never present an unfilmable shot as buildable.\n"
        "For each beat:\n"
        "  - beat_index (int, 0-based)\n"
        "  - role: exactly one of hook|setup|body|tension|reveal|payoff|quote|cta|transition|b_roll\n"
        "  - narrative_intent: plain language — what must this beat DO for the story\n"
        "  - min_duration_s: watchability floor (2.5 default; hook 1.5 ok; b_roll 1.5–3.0)\n"
        "  - target_duration_s: per-beat target seconds\n"
        "  - retrieval_spec: {semantic_query: str, emotional_beat_target: str|null, speaker_hint: str|null, "
        "want_hook: bool, want_quote: bool, want_broll: bool}\n"
        "  - visual_basis: list of event_ids whose visual content supports this beat's intended shot "
        "(cite the real event_ids from anchor_event_ids or the digest)\n"
        "  - availability_status: one of 'available' | 'minor_graphics_needed' | 'core_vfx_missing' | "
        "'core_shot_missing' | 'reshoot_needed' | 'unclear'\n"
        "  - production_delta: null if available; otherwise a one-sentence description of what production "
        "work is needed before this beat can be built (e.g. 'Needs fly VFX plate not yet produced')\n"
        "Return ONLY a JSON story_proposal object (no markdown). "
        "Every beat MUST include visual_basis and availability_status — these are required, not optional.\n"
        '{"title": str, "logline": str, "emotional_arc": str, "anchor_event_ids": [str], '
        '"viability_note": str, "target_duration_s": int, "beats": [\n'
        '  {"beat_index": 0, "role": "hook", "narrative_intent": "...", '
        '"min_duration_s": 1.5, "target_duration_s": 2.0, '
        '"retrieval_spec": {"semantic_query": "...", "emotional_beat_target": "...", '
        '"speaker_hint": "SPEAKER_01", "want_hook": true, "want_quote": true, "want_broll": false}, '
        '"visual_basis": ["m_XXXXXXXX:ev:0001"], '
        '"availability_status": "available", '
        '"production_delta": null},\n'
        '  ... more beats following the same schema ...\n'
        ']}'
    )

    developed: list[dict] = []
    proposal_ids: list[str] = []

    for seed in valid_seeds[:max_proposals]:
        # Item 13: pre-develop duration sanity check
        anchor_speech_s = sum(
            digest_lookup.get(eid, {}).get("duration_s", 0.0)
            for eid in seed["anchor_event_ids"]
        )
        if anchor_speech_s < target_duration_s * 0.5:
            logger.warning(
                "Seed '%s': anchor events sum to %.1fs speech (target %.1fs) — may be short",
                seed.get("title"), anchor_speech_s, target_duration_s,
            )

        anchor_lines = []
        for eid in seed["anchor_event_ids"]:
            ev = digest_lookup.get(eid)
            if ev is None:
                continue
            t_s = ev.get("t_s", 0)
            beat = ev.get("emotional_beat") or ""
            speaker = ev.get("speaker_name") or ""
            hook = ev.get("hook_potential")
            quote = ev.get("quote_potential")
            summary = ev.get("summary") or ""
            hook_str = f"{hook:.2f}" if hook is not None else "null"
            quote_str = f"{quote:.2f}" if quote is not None else "null"
            anchor_lines.append(
                f"[{t_s:.0f}s] {eid} | {beat} | {speaker} | hook={hook_str} quote={quote_str} | {summary}"
            )

        develop_user = (
            f"STORY SEED:\n"
            f"Title: {seed.get('title', '')}\n"
            f"Logline: {seed.get('logline', '')}\n"
            f"Emotional arc: {seed.get('emotional_arc', '')}\n"
            f"Anchor events:\n" + "\n".join(anchor_lines) + "\n\n"
            f"Target duration: {target_duration_s}s\n\n"
            "Develop into a full story_proposal. Return JSON only."
        )

        try:
            raw_dev = complete(
                [{"role": "system", "content": develop_system},
                 {"role": "user", "content": develop_user}],
                json_only=True,
            )
            proposal = json.loads(raw_dev)
        except Exception as exc:
            logger.error("develop failed for seed '%s': %s", seed.get("title"), exc)
            continue

        if not isinstance(proposal, dict) or "beats" not in proposal:
            logger.error("develop returned invalid proposal for seed '%s'", seed.get("title"))
            continue

        # Item 12: strip hallucinated event_ids from visual_basis before further checks
        for _b in proposal.get("beats", []):
            _vb = _b.get("visual_basis")
            if isinstance(_vb, list):
                _invalid = [eid for eid in _vb if eid not in digest_lookup]
                if _invalid:
                    logger.warning(
                        "Proposal '%s' beat %s: hallucinated visual_basis ids stripped: %s",
                        seed.get("title"), _b.get("beat_index", "?"), _invalid,
                    )
                _b["visual_basis"] = [eid for eid in _vb if eid in digest_lookup]

        # NEW-01: warn on missing visual grounding fields (non-blocking — logs for audit)
        missing_vg = [
            b.get("beat_index", i)
            for i, b in enumerate(proposal.get("beats", []))
            if not b.get("visual_basis")
        ]
        if missing_vg:
            logger.warning(
                "Proposal '%s': beats missing visual_basis: %s — visual grounding non-compliant",
                seed.get("title"), missing_vg,
            )

        # NEW-03: cross-reference visual_basis event_ids against media_card on_screen_subjects
        for _i, _b in enumerate(proposal.get("beats", [])):
            for _eid in (_b.get("visual_basis") or []):
                _mid = digest_lookup.get(_eid, {}).get("media_id", "")
                if not _mid:
                    continue
                _card_path = catalog / "media_cards" / f"{_mid}.json"
                if not _card_path.exists():
                    continue
                try:
                    _card = json.loads(_card_path.read_text())
                    _vg = _card.get("visual_grounding") or {}
                    _subjects = _vg.get("on_screen_subjects") or []
                    _ftype = _vg.get("footage_type", "")
                    if _ftype == "talking_head" and len(_subjects) <= 1:
                        logger.warning(
                            "Proposal '%s' beat %d: visual_basis %s → talking_head sparse"
                            " subjects=%s; intent: '%s'",
                            seed.get("title"), _b.get("beat_index", _i), _eid, _subjects,
                            (_b.get("narrative_intent") or "")[:60],
                        )
                except Exception:
                    pass

        proposal_id = "sto_" + hashlib.blake2b(
            seed.get("logline", seed.get("title", "")).encode(), digest_size=6
        ).hexdigest()
        now_utc = datetime.now(timezone.utc).isoformat()
        proposal.setdefault("schema_version", "1.0.0")
        proposal["proposal_id"] = proposal_id
        proposal["created_at_utc"] = now_utc
        proposal["anchor_event_ids"] = seed["anchor_event_ids"]
        proposal.setdefault("target_duration_s", target_duration_s)

        final_path = proposals_dir / f"{proposal_id}.json"
        tmp_path = proposals_dir / f"{proposal_id}.tmp{_uuid.uuid4().hex[:8]}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(proposal, fh, indent=2)
            _os.replace(tmp_path, final_path)
        except Exception as exc:
            logger.error("write proposal %s failed: %s", proposal_id, exc)
            tmp_path.unlink(missing_ok=True)
            continue

        developed.append(proposal)
        proposal_ids.append(proposal_id)
        # Live progress update so the poll endpoint can report mid-run state
        _ideate_jobs[job_id]["proposals_developed"] = len(developed)
        _ideate_jobs[job_id]["proposal_ids"] = list(proposal_ids)

    return {
        "survey_seeds_found": len(valid_seeds),
        "proposals_developed": len(developed),
        "proposal_ids": proposal_ids,
    }


@app.post("/shorts/ideate")
async def shorts_ideate(req: ShortsIdeateRequest):
    """Survey the full corpus digest and develop story_proposals via DeepSeek Pro.

    Returns immediately with a job_id. Poll GET /shorts/ideate/{job_id} for progress.
    The work (1 survey LLM call + N develop calls) runs in a daemon thread — no client timeout.
    """
    import uuid as _uuid, threading

    cfg = _cfg()
    catalog = Path(cfg["catalog"])
    digest_path = catalog / "event_digest.jsonl"
    if not digest_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Digest not built — run build_digest.py first",
        )

    job_id = "ideate_" + _uuid.uuid4().hex[:8]
    params = {
        "max_proposals": req.max_proposals,
        "target_duration_s": req.target_duration_s,
        "force_rebuild": req.force_rebuild,
    }
    _ideate_jobs[job_id] = {
        "status": "running",
        "proposals_developed": 0,
        "survey_seeds_found": 0,
        "proposal_ids": [],
    }

    def _bg() -> None:
        try:
            result = _ideate_sync(job_id, params, dict(cfg))
            _ideate_jobs[job_id].update({"status": "done", **result})
        except Exception as exc:
            _ideate_jobs[job_id].update({"status": "error", "error": str(exc)[:500]})

    threading.Thread(target=_bg, daemon=True, name=f"ideate-{job_id}").start()
    return JSONResponse(content={
        "job_id": job_id,
        "status": "running",
        "message": f"Ideate started — poll GET /shorts/ideate/{job_id} for progress",
    })


@app.get("/shorts/ideate/{job_id}")
async def shorts_ideate_status(job_id: str):
    """Poll the status of a running or completed ideate job."""
    job = _ideate_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id {job_id!r} not found (server may have restarted)")
    return JSONResponse(content=job)


@app.post("/shorts/gate")
async def shorts_gate(req: ShortsGateRequest):
    """Run §4 narrative gate: score all threads and emit shortlist."""
    try:
        cfg = _cfg()
        catalog = Path(cfg["catalog"])
        import importlib.util, sys as _sys
        gate_module_path = Path(__file__).parent.parent.parent / "bin" / "gate_narrative.py"
        spec = importlib.util.spec_from_file_location("gate_narrative", gate_module_path)
        gate_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate_mod)
        gate_record = gate_mod.run_gate(
            catalog=catalog,
            composite_min=req.composite_min,
            coherence_min=req.coherence_min,
        )
        if not req.dry_run:
            gate_mod._write_gate(gate_record, catalog)
        return JSONResponse(content={
            "gate_id": gate_record["gate_id"],
            "summary": gate_record["summary"],
            "shortlist": gate_record["shortlist"],
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/shorts/author")
async def shorts_author(req: ShortsAuthorRequest):
    """Run Pass 1 (narrative design) and/or Pass 2 (atom selection) on top threads."""
    try:
        cfg = _cfg()
        catalog = Path(cfg["catalog"])
        state = Path(cfg["state"])
        journal_db = state / "journal.db"
        from .author import run_pass1_batch, run_pass2_batch
        result: dict = {"pass1": None, "pass2": None}
        if req.pass1:
            briefs = run_pass1_batch(catalog=catalog, state=state, journal_db=journal_db,
                                     limit=req.limit, force=req.force)
            result["pass1"] = {"count": len(briefs)}
        if req.pass2:
            plans = run_pass2_batch(catalog=catalog, state=state, journal_db=journal_db,
                                    limit=req.limit, force=req.force)
            result["pass2"] = {"count": len(plans)}
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/shorts/narrative/run")
async def shorts_narrative_run(req: NarratorRunRequest):
    """§5 orchestrator: run the full narrative pipeline queue.

    Processes all pending threads through Pass 1, Pass 2, §4 gate, and regen.
    State is persisted to state/narrative_pipeline.db for restart safety.
    Long-running — returns summary after completion.
    """
    try:
        cfg = _cfg()
        catalog = Path(cfg["catalog"])
        state = Path(cfg["state"])
        journal_db = state / "journal.db"
        from .narrator import run_orchestrator
        result = run_orchestrator(
            catalog=catalog,
            state=state,
            journal_db=journal_db,
            concurrency=req.concurrency,
            limit=req.limit,
            force=req.force,
            composite_min=req.composite_min,
            coherence_min=req.coherence_min,
        )
        return JSONResponse(content={
            "ok": True,
            "total": result["total"],
            "work_items": result["work_items"],
            "elapsed_s": result.get("elapsed_s"),
            "passed": result["passed"],
            "regen_passed": result["regen_passed"],
            "failed": result["failed"],
            "summary": result["summary"],
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/shorts/narrative/status")
async def shorts_narrative_status():
    """Return current narrative pipeline state counts and shortlist."""
    try:
        cfg = _cfg()
        state = Path(cfg["state"])
        from . import narrative_state as ns
        pipeline_db = state / "narrative_pipeline.db"
        if not pipeline_db.exists():
            return JSONResponse(content={"initialised": False, "summary": {}})
        summary = ns.get_summary(pipeline_db)
        shortlist = [r["thread_id"] for r in ns.get_by_state(pipeline_db, "gate_passed")]
        discarded = [
            {"thread_id": r["thread_id"], "reasons": r.get("failure_reasons")}
            for r in ns.get_by_state(pipeline_db, "discarded")
        ]
        return JSONResponse(content={
            "initialised": True,
            "summary": summary,
            "shortlist": shortlist,
            "discarded": discarded,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/used-clips")
async def used_clips():
    from .usage import list_published
    try:
        published = list_published(config_path=_CONFIG_PATH)
        return JSONResponse(content={"published": published})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Minimal static UI ─────────────────────────────────────────────────────────

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ERF — Editorial Retrieval Fabric</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; min-height: 100vh; }
  header { padding: 1rem 2rem; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.05em; color: #aaa; }
  .search-bar { flex: 1; display: flex; gap: 0.5rem; }
  .search-bar input { flex: 1; background: #1e1e1e; border: 1px solid #444; border-radius: 6px;
    padding: 0.5rem 1rem; color: #eee; font-size: 1rem; }
  .search-bar input:focus { outline: none; border-color: #888; }
  .search-bar button { background: #333; border: 1px solid #555; border-radius: 6px;
    padding: 0.5rem 1.2rem; color: #eee; cursor: pointer; font-size: 0.9rem; }
  .search-bar button:hover { background: #444; }
  #results { padding: 1rem 2rem; display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; overflow: hidden; cursor: pointer;
    transition: border-color 0.15s; }
  .card:hover { border-color: #555; }
  .card.selected { border-color: #4a9eff; }
  .card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; background: #222; display: block; }
  .card .meta { padding: 0.75rem; }
  .card .tc { font-family: monospace; font-size: 0.8rem; color: #888; }
  .card .snippet { font-size: 0.85rem; margin-top: 0.4rem; color: #ccc; line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .card .score { font-size: 0.75rem; color: #666; margin-top: 0.3rem; }
  .card .speaker { font-size: 0.75rem; color: #888; margin-top: 0.2rem; }
  #status { padding: 1rem 2rem; color: #666; font-size: 0.9rem; }
  #selbar { position: fixed; bottom: 0; left: 0; right: 0; background: #1a1a1a;
    border-top: 1px solid #333; padding: 0.75rem 2rem; display: none;
    align-items: center; gap: 1rem; }
  #selbar span { color: #aaa; font-size: 0.9rem; }
  #selbar select { background: #222; border: 1px solid #444; border-radius: 4px;
    color: #eee; padding: 0.3rem 0.6rem; }
  #selbar button { background: #4a9eff; border: none; border-radius: 6px;
    padding: 0.4rem 1rem; color: #fff; cursor: pointer; font-size: 0.9rem; }
  #selbar button:hover { background: #5aafff; }
  #selbar .clear { background: #333; }
</style>
</head>
<body>
<header>
  <h1>ERF</h1>
  <div class="search-bar">
    <input id="q" type="text" placeholder="Search transcripts… e.g. warehouse argument at night" />
    <button onclick="doSearch()">Search</button>
    <button onclick="doExact()" style="background:#222;">Exact</button>
  </div>
</header>
<div id="status"></div>
<div id="results"></div>
<div id="selbar">
  <span id="selcount">0 selected</span>
  <select id="fmt"><option value="fcpxml">FCPXML</option><option value="edl">EDL</option><option value="otio">OTIO</option><option value="srt">SRT</option></select>
  <button onclick="doExport()">Export</button>
  <button class="clear" onclick="clearSel()">Clear</button>
</div>
<script>
let selected = {};

function status(msg) { document.getElementById('status').textContent = msg; }

function renderResults(data) {
  const el = document.getElementById('results');
  if (!data.results || !data.results.length) { el.innerHTML = ''; status('No results.'); return; }
  status(`${data.results.length} results`);
  el.innerHTML = data.results.map(r => {
    const sel = selected[r.result_id] ? ' selected' : '';
    return `<div class="card${sel}" data-id="${r.result_id}" onclick="toggleSel(this, ${JSON.stringify(r).replace(/"/g,'&quot;')})">
      <img src="${r.thumb_url}" onerror="this.style.display='none'" loading="lazy">
      <div class="meta">
        <div class="tc">${r.media_relpath} &nbsp; ${r.in_tc} → ${r.out_tc}</div>
        <div class="snippet">${r.snippet}</div>
        <div class="speaker">${r.speaker}</div>
        <div class="score">score ${r.score.toFixed(4)} &middot; ${r.modality}</div>
      </div>
    </div>`;
  }).join('');
}

async function doSearch() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  status('Searching…');
  const r = await fetch('/query', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query: q})});
  const data = await r.json();
  renderResults(data);
}

async function doExact() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  status('Exact search…');
  const r = await fetch('/query/exact', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({phrase: q})});
  const data = await r.json();
  const results = (data.results || []).map(h => ({
    result_id: h.chunk_id, media_id: h.media_id, media_relpath: h.media_id,
    in_ms: h.start_ms, out_ms: h.end_ms, in_tc: '', out_tc: '',
    fps: 25, thumb_url: `/media/thumbs/${h.media_id}/${h.media_id}_mid.webp`,
    snippet: h.text, speaker: h.speaker_label || '', modality: 'fts', score: Math.abs(h.score)
  }));
  renderResults({results});
}

function toggleSel(el, r) {
  if (selected[r.result_id]) { delete selected[r.result_id]; el.classList.remove('selected'); }
  else { selected[r.result_id] = r; el.classList.add('selected'); }
  const n = Object.keys(selected).length;
  document.getElementById('selcount').textContent = n + ' selected';
  document.getElementById('selbar').style.display = n > 0 ? 'flex' : 'none';
}

function clearSel() {
  selected = {};
  document.querySelectorAll('.card.selected').forEach(e => e.classList.remove('selected'));
  document.getElementById('selbar').style.display = 'none';
}

async function doExport() {
  const fmt = document.getElementById('fmt').value;
  const clips = Object.values(selected).map(r => ({
    media_id: r.media_id, in_ms: r.in_ms, out_ms: r.out_ms, label: r.snippet?.slice(0,40)
  }));
  const r = await fetch('/export', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({clips, format: fmt, query: document.getElementById('q').value})});
  const data = await r.json();
  alert('Export written: ' + data.output_relpath);
}

document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
</script>
</body>
</html>"""


def start(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

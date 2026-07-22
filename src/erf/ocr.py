"""Stage B5 — Selective OCR.

Only runs on frames where media_class is 'screen_rec' or 'tutorial'
(per media_manifest.media_class / source_note).

PaddleOCR → ocr JSONL per schema. Privacy risk flagging via regex.
If privacy_risk is non-null: raises publicability to 'needs_review' and
emits a warning journal event.
"""
from __future__ import annotations

import json
import re
import sqlite3
import tomllib
import uuid
from pathlib import Path

from .atomicio import jsonl_append, read_jsonl
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "erf_root": "/workspace/erf",
}

# Privacy risk patterns
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(sk-[A-Za-z0-9]{20,}|"       # OpenAI-style keys
    r"ghp_[A-Za-z0-9]{36}|"        # GitHub personal access tokens
    r"AKIA[A-Z0-9]{16}|"           # AWS access key
    r"Bearer [A-Za-z0-9\-._~+/]{20,}|"  # Bearer tokens
    r"(password|passwd|pwd|secret|api.?key)\s*[:=]\s*\S{6,})"  # generic creds
)
_KEY_OR_TOKEN_PATTERN = re.compile(
    r"([A-Za-z0-9+/]{40,}={0,2}|"   # base64-ish long strings (API keys)
    r"eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+)"  # JWTs
)
_PII_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"  # email
)


def _privacy_risk(text: str) -> str | None:
    if _CREDENTIAL_PATTERN.search(text):
        return "credential"
    if _KEY_OR_TOKEN_PATTERN.search(text):
        return "key_or_token"
    if _PII_PATTERN.search(text):
        return "pii"
    return None


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
    return cfg


def _should_run_ocr(media_manifest: dict) -> bool:
    source_note = (media_manifest.get("source_note") or "").lower()
    return any(kw in source_note for kw in ("screen_rec", "screen rec", "tutorial"))


def _emit_privacy_warning(journal_db: Path, media_id: str, run_id: str, kf_id: str, risk: str) -> None:
    conn = sqlite3.connect(str(journal_db))
    try:
        conn.execute(
            # Phase 1 (D-Log-04): sub-event trace, not B5's own disposition
            # -> events_trace, not the outcome-only `events` table (§3.1).
            "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
            "VALUES (?, ?, ?, 'B5', 'skip', ?, ?)",
            (
                StageManifest.now(), f"{media_id}:B5:privacy", media_id, run_id,
                json.dumps({"reason": "privacy_risk", "keyframe_id": kf_id, "risk": risk}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def run(
    media_id: str,
    *,
    config_path: str | Path = "state/config.toml",
    force: bool = False,
) -> StageManifest:
    """force=True bypasses the media_class gate (for testing)."""
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    erf_root = Path(cfg["erf_root"])

    run_id = f"B5-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    emit_start(journal_db, media_id, "B5", run_id)

    error: str | None = None
    ocr_count = 0
    outputs: list[str] = []

    try:
        # Gate: only screen recordings / tutorials
        mm_path = catalog / "media_manifest" / f"{media_id}.json"
        if mm_path.exists():
            mm = json.loads(mm_path.read_text())
        else:
            mm = {}

        if not force and not _should_run_ocr(mm):
            # Emit skip — not a screen recording
            conn = sqlite3.connect(str(journal_db))
            try:
                conn.execute(
                    # Phase 1 (D-Log-04): sub-event trace, not B5's own disposition
                    # -> events_trace, not the outcome-only `events` table (§3.1).
                    "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                    "VALUES (?, ?, ?, 'B5', 'skip', ?, ?)",
                    (
                        StageManifest.now(), f"{media_id}:B5:gate", media_id, run_id,
                        json.dumps({"reason": "not_screen_rec"}),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            manifest = StageManifest(
                media_id=media_id, stage="B5", run_id=run_id, ok=True, at_utc=at_utc,
                tool_version=None, counts={"ocr_records": 0, "skipped": "not_screen_rec"},
                error=None, outputs=[],
            )
            reap_manifest(journal_db, manifest, catalog / "status")
            return manifest

        import warnings
        warnings.filterwarnings("ignore")
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_doc_orientation_predictor=False, use_angle_cls=False,
                        lang="en", show_log=False)

        kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
        keyframes = read_jsonl(kf_path) if kf_path.exists() else []

        ocr_path = catalog / "ocr" if hasattr(catalog, "ocr") else catalog.parent / "catalog" / "ocr"
        ocr_path = Path(str(catalog)) / "ocr"
        ocr_path.mkdir(parents=True, exist_ok=True)
        out_path = ocr_path / f"{media_id}.jsonl"
        if out_path.exists():
            out_path.unlink()

        for kf in keyframes:
            img_path = erf_root / kf["image_relpath"]
            if not img_path.exists():
                continue

            try:
                result = ocr.predict(str(img_path))
            except Exception:
                continue

            # PaddleOCR 3.x returns list of dicts per page
            texts = []
            if result and isinstance(result, list):
                for page in result:
                    if isinstance(page, dict):
                        for item in (page.get("rec_res") or []):
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                texts.append(item)
                    elif isinstance(page, list):
                        texts.extend(page)

            img_w = kf.get("width", 1) or 1
            img_h = kf.get("height", 1) or 1

            for t_idx, item in enumerate(texts):
                # item: [[pts], (text, conf)] or similar
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        pts = item[0]
                        text_conf = item[1]
                        if isinstance(text_conf, (list, tuple)):
                            text, conf = str(text_conf[0]), float(text_conf[1])
                        else:
                            continue
                        # Compute bbox from polygon points [[x,y], ...]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x = min(xs) / img_w
                        y = min(ys) / img_h
                        w = (max(xs) - min(xs)) / img_w
                        h = (max(ys) - min(ys)) / img_h
                    else:
                        continue
                except Exception:
                    continue

                risk = _privacy_risk(text)
                if risk:
                    _emit_privacy_warning(journal_db, media_id, run_id, kf["keyframe_id"], risk)

                rec = {
                    "schema_version": "1.0.0",
                    "ocr_id": f"{kf['keyframe_id']}:ocr:{t_idx:03d}",
                    "media_id": media_id,
                    "keyframe_id": kf["keyframe_id"],
                    "text": text,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
                    "privacy_risk": risk,
                }
                jsonl_append(out_path, rec)
                ocr_count += 1

        outputs.append(str(out_path))

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ok = error is None
    manifest = StageManifest(
        media_id=media_id,
        stage="B5",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=None,
        counts={"ocr_records": ocr_count},
        error=error,
        outputs=outputs,
    )
    reap_manifest(journal_db, manifest, catalog / "status")
    return manifest

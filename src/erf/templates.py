"""Template library — short_template CRUD backed by SQLite + catalog JSON."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .atomicio import write_json
from .contracts import validate

_DEFAULTS = {
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
}


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
    return cfg


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = TRUNCATE")
    return conn


def _emit_journal(journal_db: Path, template_id: str, catalog_relpath: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps({
        "template_id": template_id,
        "outputs_produced": [catalog_relpath],
    })
    try:
        conn = _connect(journal_db)
        conn.execute(
            # Phase 1 (D-Log-04): template-registration trace -> events_trace (§3.1).
            "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
            "VALUES (?, ?, 'erf', 'template', 'ok', ?, ?)",
            (now, f"erf:template:{template_id}", template_id, payload),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # journal is best-effort


def create_template(
    name: str,
    platform: str,
    beat_structure: list,
    *,
    duration_target_s: float | None = None,
    caption_style: str = "centre",
    tone_guidance: str | None = None,
    exemplar_beat_plan_ids: list | None = None,
    created_by: str = "api",
    config_path: str | Path = "state/config.toml",
) -> str:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])

    created_at_utc = datetime.now(timezone.utc).isoformat()
    id_src = (name + platform + created_at_utc).encode()
    template_id = "tmpl_" + hashlib.blake2b(id_src, digest_size=6).hexdigest()

    record = {
        "schema_version": "1.0.0",
        "template_id": template_id,
        "name": name,
        "platform": platform,
        "beat_structure": beat_structure,
        "created_at_utc": created_at_utc,
        "created_by": created_by,
        "active": True,
    }
    if duration_target_s is not None:
        record["duration_target_s"] = duration_target_s
    if caption_style != "centre":
        record["caption_style"] = caption_style
    else:
        record["caption_style"] = "centre"
    if tone_guidance is not None:
        record["tone_guidance"] = tone_guidance
    record["exemplar_beat_plan_ids"] = exemplar_beat_plan_ids or []

    validate(record, "short_template")

    catalog_dir = catalog / "templates"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_relpath = f"catalog/templates/{template_id}.json"
    write_json(catalog_dir / f"{template_id}.json", record)

    erf_db = state / "erf.db"
    conn = _connect(erf_db)
    conn.execute(
        """INSERT INTO short_templates
           (template_id, name, platform, duration_target_s, caption_style, tone_guidance,
            beat_structure, exemplar_beat_plan_ids, created_at_utc, created_by, active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (
            template_id, name, platform,
            duration_target_s, caption_style,
            tone_guidance,
            json.dumps(beat_structure),
            json.dumps(exemplar_beat_plan_ids or []),
            created_at_utc, created_by,
        ),
    )
    conn.commit()
    conn.close()

    _emit_journal(state / "journal.db", template_id, catalog_relpath)
    return template_id


def get_template(
    template_id: str,
    *,
    config_path: str | Path = "state/config.toml",
) -> dict | None:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    catalog = Path(cfg["catalog"])
    p = catalog / "templates" / f"{template_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_templates(
    *,
    platform: str | None = None,
    active_only: bool = True,
    config_path: str | Path = "state/config.toml",
) -> list[dict]:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    state = Path(cfg["state"])

    erf_db = state / "erf.db"
    conn = _connect(erf_db)
    conn.row_factory = sqlite3.Row

    sql = "SELECT * FROM short_templates WHERE 1=1"
    params: list = []
    if platform is not None:
        sql += " AND platform = ?"
        params.append(platform)
    if active_only:
        sql += " AND active = 1"
    sql += " ORDER BY created_at_utc DESC"

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        d["beat_structure"] = json.loads(d["beat_structure"])
        d["exemplar_beat_plan_ids"] = json.loads(d["exemplar_beat_plan_ids"])
        d["active"] = bool(d["active"])
        result.append(d)
    return result


def deactivate_template(
    template_id: str,
    *,
    config_path: str | Path = "state/config.toml",
) -> bool:
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    state = Path(cfg["state"])

    erf_db = state / "erf.db"
    conn = _connect(erf_db)
    cur = conn.execute(
        "UPDATE short_templates SET active = 0 WHERE template_id = ?",
        (template_id,),
    )
    conn.commit()
    conn.close()
    return cur.rowcount > 0

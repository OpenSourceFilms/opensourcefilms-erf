"""U0 — Usage registry.

Tracks exports and published outputs. Answers is_used() for clip overlap checks.
Tables live in state/erf_fts.db (already has exports + export_clips from Phase 1).

CLI: erf-publish log <url> --platform <p> --title "..." --used <media_id[:chunk_id],…>
"""
from __future__ import annotations

import hashlib
import sqlite3
import tomllib
from datetime import datetime, timezone
from pathlib import Path

_DEFAULTS = {
    "erf_fts_db": "/workspace/erf/state/erf_fts.db",
    "state": "/workspace/erf/state",
}


def _load_fts_db_path(config_path: Path) -> Path:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["erf_fts_db"] = raw.get("paths", {}).get("erf_fts_db", cfg["erf_fts_db"])
    return Path(cfg["erf_fts_db"])


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = TRUNCATE")
    conn.execute("PRAGMA synchronous = FULL")
    return conn


def _ensure_published_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS published_outputs (
            output_id   TEXT PRIMARY KEY,
            export_id   TEXT,
            platform    TEXT NOT NULL,
            url         TEXT UNIQUE,
            title       TEXT,
            published_at_utc TEXT NOT NULL,
            note        TEXT
        );
        CREATE TABLE IF NOT EXISTS published_clip_refs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            output_id   TEXT NOT NULL REFERENCES published_outputs(output_id),
            media_id    TEXT NOT NULL,
            in_ms       INTEGER NOT NULL,
            out_ms      INTEGER NOT NULL,
            label       TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_pcr_media ON published_clip_refs(media_id, in_ms, out_ms);
    """)
    # Additive column migrations — run after table creation to handle pre-existing tables
    for ddl in [
        "ALTER TABLE published_outputs ADD COLUMN title TEXT",
        "ALTER TABLE published_clip_refs ADD COLUMN chunk_id TEXT",
    ]:
        try:
            conn.execute(ddl)
        except Exception:
            pass  # column already exists
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS ix_pub_clip_chunk ON published_clip_refs(chunk_id)")
    except Exception:
        pass
    conn.commit()


# ── Write ─────────────────────────────────────────────────────────────────────

def record_export(
    export_id: str,
    clips: list[dict],
    *,
    config_path: str | Path = "state/config.toml",
) -> None:
    """Record an export in the exports + export_clips tables. Idempotent."""
    db = _load_fts_db_path(Path(config_path))
    conn = _connect(db)
    try:
        # exports table may already have this row from export.py
        conn.execute(
            "INSERT OR IGNORE INTO exports (export_id, created_at_utc, format, output_relpath) "
            "VALUES (?, ?, 'unknown', '')",
            (export_id, datetime.now(timezone.utc).isoformat()),
        )
        for c in clips:
            conn.execute(
                "INSERT OR IGNORE INTO export_clips (export_id, media_id, in_frame, out_frame, label) "
                "VALUES (?, ?, ?, ?, ?)",
                (export_id, c["media_id"], c.get("in_frame", 0), c.get("out_frame", 0), c.get("label")),
            )
        conn.commit()
    finally:
        conn.close()


def record_published(
    output_id: str,
    platform: str,
    clips: list[dict],
    *,
    export_id: str | None = None,
    url: str | None = None,
    note: str | None = None,
    config_path: str | Path = "state/config.toml",
) -> None:
    """Record a published output and its clip references."""
    db = _load_fts_db_path(Path(config_path))
    conn = _connect(db)
    try:
        _ensure_published_tables(conn)
        conn.execute(
            "INSERT OR REPLACE INTO published_outputs "
            "(output_id, export_id, platform, url, published_at_utc, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (output_id, export_id, platform, url,
             datetime.now(timezone.utc).isoformat(), note),
        )
        for c in clips:
            conn.execute(
                "INSERT INTO published_clip_refs (output_id, media_id, in_ms, out_ms, label) "
                "VALUES (?, ?, ?, ?, ?)",
                (output_id, c["media_id"], c.get("in_ms", 0), c.get("out_ms", 0), c.get("label")),
            )
        conn.commit()
    finally:
        conn.close()


# ── Query ─────────────────────────────────────────────────────────────────────

def is_used(
    media_id: str,
    in_ms: int,
    out_ms: int,
    *,
    config_path: str | Path = "state/config.toml",
) -> bool:
    """Return True if this time range overlaps any published_clip_refs."""
    db = _load_fts_db_path(Path(config_path))
    conn = _connect(db)
    try:
        _ensure_published_tables(conn)
        row = conn.execute(
            "SELECT 1 FROM published_clip_refs "
            "WHERE media_id = ? AND in_ms < ? AND out_ms > ? LIMIT 1",
            (media_id, out_ms, in_ms),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def list_published(
    *,
    config_path: str | Path = "state/config.toml",
) -> list[dict]:
    """List all published outputs with their clip refs."""
    db = _load_fts_db_path(Path(config_path))
    conn = _connect(db)
    try:
        _ensure_published_tables(conn)
        outputs = conn.execute(
            "SELECT * FROM published_outputs ORDER BY published_at_utc DESC"
        ).fetchall()
        result = []
        for out in outputs:
            out_dict = dict(out)
            clips = conn.execute(
                "SELECT * FROM published_clip_refs WHERE output_id = ?",
                (out_dict["output_id"],),
            ).fetchall()
            out_dict["clips"] = [dict(c) for c in clips]
            result.append(out_dict)
        return result
    finally:
        conn.close()


def log_publish(
    url: str,
    platform: str,
    title: str,
    used_specs: list[str],
    *,
    config_path: str | Path = "state/config.toml",
) -> dict:
    """Record a published output and its source atoms. Idempotent on same URL."""
    db = _load_fts_db_path(Path(config_path))
    conn = _connect(db)
    try:
        _ensure_published_tables(conn)

        existing = conn.execute(
            "SELECT output_id FROM published_outputs WHERE url = ?", (url,)
        ).fetchone()
        if existing:
            return {"output_id": existing[0], "status": "already_exists"}

        output_id = f"out_{hashlib.blake2b(url.encode(), digest_size=6).hexdigest()}"
        conn.execute(
            "INSERT INTO published_outputs "
            "(output_id, export_id, platform, url, title, published_at_utc, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (output_id, None, platform, url, title,
             datetime.now(timezone.utc).isoformat(), None),
        )

        import re as _re
        _MEDIA_ID_ONLY = _re.compile(r"^m_[0-9a-f]{12}$")
        for spec in used_specs:
            # "m_75219c4b67e1" → media_id only; "m_75219c4b67e1:c:0-13200" → full chunk_id
            if _MEDIA_ID_ONLY.match(spec):
                media_id_part, chunk_id_part = spec, None
            else:
                m = _re.match(r"^(m_[0-9a-f]{12})", spec)
                media_id_part = m.group(1) if m else spec
                chunk_id_part = spec
            conn.execute(
                "INSERT INTO published_clip_refs "
                "(output_id, media_id, chunk_id, in_ms, out_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                (output_id, media_id_part, chunk_id_part, 0, 0),
            )

        conn.commit()
        return {"output_id": output_id, "status": "created"}
    finally:
        conn.close()


def list_exports(
    *,
    config_path: str | Path = "state/config.toml",
) -> list[dict]:
    """List all export records with their clips."""
    db = _load_fts_db_path(Path(config_path))
    conn = _connect(db)
    try:
        rows = conn.execute(
            "SELECT * FROM exports ORDER BY created_at_utc DESC"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            clips = conn.execute(
                "SELECT * FROM export_clips WHERE export_id = ?", (d["export_id"],)
            ).fetchall()
            d["clips"] = [dict(c) for c in clips]
            result.append(d)
        return result
    finally:
        conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="erf-publish")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_log = sub.add_parser("log", help="Record a published output")
    p_log.add_argument("url", help="URL of the published output")
    p_log.add_argument(
        "--platform", required=True,
        choices=["youtube", "shorts", "reels", "tiktok", "patreon", "internal"],
    )
    p_log.add_argument("--title", required=True)
    p_log.add_argument(
        "--used", required=True,
        help="Comma-separated media_id[:chunk_id] references",
    )
    p_log.add_argument("--config", default="state/config.toml")

    args = parser.parse_args()

    if args.cmd == "log":
        used_specs = [s.strip() for s in args.used.split(",") if s.strip()]
        result = log_publish(
            args.url, args.platform, args.title, used_specs,
            config_path=args.config,
        )
        print(f"status={result['status']} output_id={result['output_id']}")
        sys.exit(0)


if __name__ == "__main__":
    main()

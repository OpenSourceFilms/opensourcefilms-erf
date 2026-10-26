"""Stage X0 / FTS helpers — lexical search over erf_fts.db.

Populates chunks_fts and words_fts from catalog JSONL.
Provides exact-phrase, keyword, and path/command query support.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .atomicio import read_jsonl


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = TRUNCATE")
    conn.execute("PRAGMA synchronous = FULL")
    return conn


# ── Population ────────────────────────────────────────────────────────────────

# Canonical IDs are m_<12hex>. Older/internal labels have also existed.
# Accept an ASCII basename without assuming a specific identifier version.
_SAFE_MEDIA_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def _index_media_id(media_id: str) -> str:
    """Do not allow a media key to address a different catalog directory."""
    if type(media_id) is not str or not _SAFE_MEDIA_ID.fullmatch(media_id):
        raise ValueError("media_id must be an ASCII basename (letters, digits, _ or -)")
    return media_id


def _verify_scope(records: list[dict], media_id: str, kind: str) -> None:
    """Reject mixed-media catalogs *before* the index's DELETE/INSERT phase."""
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("media_id") != media_id:
            raise ValueError(
                f"{kind} record {index} does not belong to requested media_id"
            )


def index_media(
    media_id: str,
    catalog: Path,
    db_path: Path,
) -> dict[str, int]:
    """Index one media file's chunks and words into erf_fts.db. Idempotent."""
    media_id = _index_media_id(media_id)
    chunks_path = catalog / "chunks" / f"{media_id}.jsonl"
    words_path = catalog / "asr" / f"{media_id}.words.jsonl"

    chunks = read_jsonl(chunks_path) if chunks_path.exists() else []
    words = read_jsonl(words_path) if words_path.exists() else []
    _verify_scope(chunks, media_id, "chunk")
    _verify_scope(words, media_id, "word")

    conn = _connect(db_path)
    try:
        # Delete existing rows for this media (idempotent re-index)
        conn.execute("DELETE FROM chunks_fts WHERE media_id = ?", (media_id,))
        conn.execute("DELETE FROM words_fts WHERE media_id = ?", (media_id,))

        for c in chunks:
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, media_id, start_ms, end_ms, text, speaker_label) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (c["chunk_id"], c["media_id"], c["start_ms"], c["end_ms"],
                 c.get("text", ""), c.get("speaker_label") or ""),
            )
        for w in words:
            conn.execute(
                "INSERT INTO words_fts (word_id, media_id, start_ms, end_ms, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (w["word_id"], w["media_id"], w["start_ms"], w["end_ms"],
                 w.get("text", "")),
            )
        conn.commit()
    finally:
        conn.close()

    return {"chunks": len(chunks), "words": len(words)}


def index_ocr(media_id: str, catalog: Path, db_path: Path) -> dict[str, int]:
    """Index OCR records into ocr_fts table. Idempotent."""
    media_id = _index_media_id(media_id)
    ocr_path = catalog / "ocr" / f"{media_id}.jsonl"
    records = read_jsonl(ocr_path) if ocr_path.exists() else []
    _verify_scope(records, media_id, "OCR")

    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM ocr_fts WHERE media_id = ?", (media_id,))
        for r in records:
            conn.execute(
                "INSERT INTO ocr_fts (ocr_id, media_id, keyframe_id, text) VALUES (?, ?, ?, ?)",
                (r.get("ocr_id", ""), r["media_id"], r.get("keyframe_id", ""), r.get("text", "")),
            )
        conn.commit()
    finally:
        conn.close()
    return {"ocr": len(records)}


# ── Search ────────────────────────────────────────────────────────────────────

def _fts5_escape(query: str) -> str:
    """Wrap each whitespace-separated token in double quotes so FTS5 treats them
    as literal terms rather than column names or operators.
    Skips escaping if the query already contains double quotes (e.g. phrase queries)."""
    if '"' in query:
        return query
    import re
    tokens = re.split(r"\s+", query.strip())
    return " ".join(f'"{t}"' for t in tokens if t)


def search_chunks(
    db_path: Path,
    query: str,
    *,
    media_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """FTS5 chunk search. Returns list of {chunk_id, media_id, start_ms, end_ms, text, score}."""
    query = _fts5_escape(query)
    conn = _connect(db_path)
    try:
        if media_id:
            rows = conn.execute(
                "SELECT chunk_id, media_id, start_ms, end_ms, text, speaker_label, "
                "bm25(chunks_fts) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? AND media_id = ? "
                "ORDER BY score LIMIT ?",
                (query, media_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT chunk_id, media_id, start_ms, end_ms, text, speaker_label, "
                "bm25(chunks_fts) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_words(
    db_path: Path,
    query: str,
    *,
    media_id: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """FTS5 word search for exact-phrase and command/path lookups."""
    query = _fts5_escape(query)
    conn = _connect(db_path)
    try:
        if media_id:
            rows = conn.execute(
                "SELECT word_id, media_id, start_ms, end_ms, text, "
                "bm25(words_fts) AS score "
                "FROM words_fts WHERE words_fts MATCH ? AND media_id = ? "
                "ORDER BY score LIMIT ?",
                (query, media_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT word_id, media_id, start_ms, end_ms, text, "
                "bm25(words_fts) AS score "
                "FROM words_fts WHERE words_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_exact(
    db_path: Path,
    phrase: str,
    *,
    limit: int = 50,
) -> list[dict]:
    """Exact-phrase search using FTS5 phrase syntax."""
    fts_query = f'"{phrase}"'
    return search_chunks(db_path, fts_query, limit=limit)


def search_ocr(
    db_path: Path,
    query: str,
    *,
    media_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """FTS5 search over OCR text (unicode61 tokenizer)."""
    conn = _connect(db_path)
    try:
        if media_id:
            rows = conn.execute(
                "SELECT ocr_id, media_id, keyframe_id, text, bm25(ocr_fts) AS score "
                "FROM ocr_fts WHERE ocr_fts MATCH ? AND media_id = ? ORDER BY score LIMIT ?",
                (query, media_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ocr_id, media_id, keyframe_id, text, bm25(ocr_fts) AS score "
                "FROM ocr_fts WHERE ocr_fts MATCH ? ORDER BY score LIMIT ?",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

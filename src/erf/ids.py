"""Deterministic ID generators for ERF catalog records (§6).

All IDs are deterministic from content or coordinates — no random UUIDs.
This is what makes stages idempotent (Invariant 4).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Union


# ──────────────────────────────────────────────────────────────────────────────
# Media
# ──────────────────────────────────────────────────────────────────────────────

def media_id(relpath: str, filesize_bytes: int, duration_ms: int) -> str:
    """m_<blake2b-12hex> of 'relpath|filesize_bytes|duration_ms'."""
    payload = f"{relpath}|{filesize_bytes}|{duration_ms}".encode()
    digest = hashlib.blake2b(payload, digest_size=6).hexdigest()
    return f"m_{digest}"


# ──────────────────────────────────────────────────────────────────────────────
# ASR
# ──────────────────────────────────────────────────────────────────────────────

def word_id(mid: str, index: int) -> str:
    """{media_id}:w:{index:06d}"""
    return f"{mid}:w:{index:06d}"


def segment_id(mid: str, index: int) -> str:
    """{media_id}:s:{index:05d}"""
    return f"{mid}:s:{index:05d}"


# ──────────────────────────────────────────────────────────────────────────────
# Chunks, scenes, keyframes
# ──────────────────────────────────────────────────────────────────────────────

def chunk_id(mid: str, start_ms: int, end_ms: int) -> str:
    """{media_id}:c:{start_ms}-{end_ms}"""
    return f"{mid}:c:{start_ms}-{end_ms}"


def scene_id(mid: str, index: int) -> str:
    """{media_id}:sc:{index:04d}"""
    return f"{mid}:sc:{index:04d}"


def keyframe_id(mid: str, time_ms: int) -> str:
    """{media_id}:kf:{time_ms}"""
    return f"{mid}:kf:{time_ms}"


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────

def embedding_id(source_id: str, model_tag: str) -> str:
    """emb:{source_id}:{model_tag}"""
    return f"emb:{source_id}:{model_tag}"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3
# ──────────────────────────────────────────────────────────────────────────────

def detection_id(kf_id: str, index: int) -> str:
    """{keyframe_id}:det:{index:03d}"""
    return f"{kf_id}:det:{index:03d}"


def face_track_id(mid: str, index: int) -> str:
    """{media_id}:ft:{index:04d}"""
    return f"{mid}:ft:{index:04d}"


def person_id(slug: str) -> str:
    """p_<slug> — human-assigned at enrollment; slug must be lowercase alphanumeric + underscore."""
    slug = slug.lower()
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise ValueError(f"person slug must be [a-z0-9_]+, got: {slug!r}")
    return f"p_{slug}"


# ──────────────────────────────────────────────────────────────────────────────
# Events, entity nodes
# ──────────────────────────────────────────────────────────────────────────────

def event_id(mid: str, index: int) -> str:
    """{media_id}:ev:{index:04d}"""
    return f"{mid}:ev:{index:04d}"


def entity_node_id(entity_type: str, slug: str) -> str:
    """n_<type>_<slug>"""
    return f"n_{entity_type}_{slug}"


# ──────────────────────────────────────────────────────────────────────────────
# Exports
# ──────────────────────────────────────────────────────────────────────────────

def export_id(content_hash_hex: str, at: Union[datetime, None] = None) -> str:
    """exp_<YYYYMMDDTHHMMSSZ>_<6hex>

    content_hash_hex: any hex string; first 6 chars used.
    at: UTC datetime; defaults to now.
    """
    if at is None:
        at = datetime.now(timezone.utc)
    ts = at.strftime("%Y%m%dT%H%M%SZ")
    suffix = content_hash_hex[:6].lower()
    return f"exp_{ts}_{suffix}"


# ──────────────────────────────────────────────────────────────────────────────
# Pattern validators (for tests and defensive checks)
# ──────────────────────────────────────────────────────────────────────────────

PATTERNS = {
    "media_id":      re.compile(r"^m_[0-9a-f]{12}$"),
    "word_id":       re.compile(r"^m_[0-9a-f]{12}:w:[0-9]{6}$"),
    "segment_id":    re.compile(r"^m_[0-9a-f]{12}:s:[0-9]{5}$"),
    "chunk_id":      re.compile(r"^m_[0-9a-f]{12}:c:[0-9]+-[0-9]+$"),
    "scene_id":      re.compile(r"^m_[0-9a-f]{12}:sc:[0-9]{4}$"),
    "keyframe_id":   re.compile(r"^m_[0-9a-f]{12}:kf:[0-9]+$"),
    "embedding_id":  re.compile(r"^emb:.+:.+$"),
    "detection_id":  re.compile(r"^m_[0-9a-f]{12}:kf:[0-9]+:det:[0-9]{3}$"),
    "face_track_id": re.compile(r"^m_[0-9a-f]{12}:ft:[0-9]{4}$"),
    "person_id":     re.compile(r"^p_[a-z0-9_]+$"),
    "event_id":      re.compile(r"^m_[0-9a-f]{12}:ev:[0-9]{4}$"),
    "export_id":     re.compile(r"^exp_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{6}$"),
}


def matches(id_str: str, id_type: str) -> bool:
    """Return True if id_str matches the expected pattern for id_type."""
    pat = PATTERNS.get(id_type)
    if pat is None:
        raise ValueError(f"Unknown id_type: {id_type}")
    return bool(pat.fullmatch(id_str))

#!/usr/bin/env python3
"""Disposable, genuinely executed ERF FTS5 footage-search demonstration.

Runs the repository's real erf.fts.index_media/search_chunks functions over
three invented transcript chunks in a temporary catalog. The inline SVG
posters are illustrations, NOT thumbnails extracted from real footage.
No model, GPU, network, identity data or production catalog is accessed.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from erf.fts import index_media, search_chunks  # noqa: E402
from erf.query import ms_to_tc  # noqa: E402

FPS = 24
ROWS = (
    {
        "schema_version": "1.0.0",
        "chunk_id": "m_111111111111:c:14000-19500",
        "media_id": "m_111111111111",
        "start_ms": 14000,
        "end_ms": 19500,
        "text": "We release steam through the valve and check the pressure.",
        "speaker_label": "Operator",
        "word_ids": [],
    },
    {
        "schema_version": "1.0.0",
        "chunk_id": "m_222222222222:c:51000-56000",
        "media_id": "m_222222222222",
        "start_ms": 51000,
        "end_ms": 56000,
        "text": "Steam drifts above the test chamber while the gauge stabilises.",
        "speaker_label": "Observer",
        "word_ids": [],
    },
    {
        "schema_version": "1.0.0",
        "chunk_id": "m_333333333333:c:32000-38000",
        "media_id": "m_333333333333",
        "start_ms": 32000,
        "end_ms": 38000,
        "text": "We refill the pump and record the water level.",
        "speaker_label": "Technician",
        "word_ids": [],
    },
)

# UI labels are intentionally invented and are not paths to actual recordings.
SYNTHETIC_TITLES = {
    "m_111111111111": "Synthetic workshop A",
    "m_222222222222": "Synthetic workshop B",
    "m_333333333333": "Synthetic workshop C",
}


def validate_demo_query(query: str) -> str:
    """Bound this example to simple lexical terms; it is not a general parser."""
    query = query.strip()
    if not re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){0,4}", query):
        raise ValueError("This demonstration accepts 1-5 simple alphabetic words only")
    return query


def run_demo(query: str = "steam") -> dict:
    """Execute real ERF lexical indexing/search with disposable local input."""
    query = validate_demo_query(query)
    with tempfile.TemporaryDirectory(prefix="erf_public_fts_demo_") as tmp:
        home = Path(tmp)
        catalog = home / "catalog"
        chunks_dir = catalog / "chunks"
        chunks_dir.mkdir(parents=True)
        db = home / "erf_fts.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "chunk_id UNINDEXED, media_id UNINDEXED, start_ms UNINDEXED, "
                "end_ms UNINDEXED, text, speaker_label UNINDEXED);"
                "CREATE VIRTUAL TABLE words_fts USING fts5("
                "word_id UNINDEXED, media_id UNINDEXED, start_ms UNINDEXED, "
                "end_ms UNINDEXED, text);"
            )
        for row in ROWS:
            path = chunks_dir / (row["media_id"] + ".jsonl")
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            index_media(row["media_id"], catalog, db)
        hits = search_chunks(db, query, limit=10)

    results = [
        {
            "result_id": hit["chunk_id"],
            "media_id": hit["media_id"],
            "media_label": SYNTHETIC_TITLES[hit["media_id"]],
            "in_ms": int(hit["start_ms"]),
            "out_ms": int(hit["end_ms"]),
            "in_tc": ms_to_tc(int(hit["start_ms"]), FPS),
            "out_tc": ms_to_tc(int(hit["end_ms"]), FPS),
            "speaker": hit["speaker_label"],
            "snippet": hit["text"],
            "thumbnail": "inline synthetic SVG illustration, NOT a video frame",
            "match_type": "FTS5 lexical",
        }
        for hit in hits
    ]
    return {
        "demo_only": True,
        "index": "real ERF FTS5 search over invented transcript records",
        "query": query,
        "fps": FPS,
        "results": results,
    }


def _svg_poster(label: str, timecode: str) -> str:
    label = escape(label)
    timecode = escape(timecode)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 200"
      role="img" aria-label="Synthetic illustration for {label}">
      <rect width="360" height="200" fill="#172c38"/>
      <path d="M0 148H360M0 169H360" stroke="#47616d" stroke-width="2"/>
      <rect x="54" y="75" width="110" height="68" rx="8" fill="#466c77"/>
      <rect x="75" y="90" width="67" height="20" rx="3" fill="#c9dbda"/>
      <circle cx="250" cy="99" r="38" fill="#a4bec3" stroke="#dce9e8" stroke-width="5"/>
      <path d="M250 99L273 76" stroke="#203842" stroke-width="5"/>
      <path d="M165 100h48" stroke="#c18e62" stroke-width="9"/>
      <rect x="15" y="13" width="153" height="25" rx="5" fill="#e3b58a"/>
      <text x="24" y="30" font-family="sans-serif" font-size="12"
        font-weight="bold" fill="#172c38">SYNTHETIC ILLUSTRATION</text>
      <text x="15" y="185" font-family="monospace" font-size="13"
        fill="#ecf1ee">{label} | {timecode}</text>
    </svg>"""


def render_html(payload: dict) -> str:
    """Generate a self-contained, offline preview with no linked real media."""
    cards = []
    for result in payload["results"]:
        cards.append(
            '<article class="hit">'
            + _svg_poster(result["media_label"], result["in_tc"])
            + "<div><h2>" + escape(result["media_label"]) + "</h2>"
            + '<p class="tc">In ' + escape(result["in_tc"])
            + " | Out " + escape(result["out_tc"]) + "</p>"
            + "<p>" + escape(result["snippet"]) + "</p>"
            + "<p>Speaker label: " + escape(result["speaker"])
            + " | FTS5 lexical evidence</p></div></article>"
        )
    listing = "\n".join(cards) if cards else "<p>No lexical matches in the synthetic catalog.</p>"
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ERF synthetic lexical search demonstration</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:960px;
margin:2rem auto;padding:0 1rem;background:#f6f7f7;color:#182632}
.hit{display:grid;grid-template-columns:minmax(200px,360px) 1fr;gap:18px;
background:white;border:1px solid #d5dddf;padding:16px;margin:18px 0}
.hit svg{width:100%;height:auto}.tc{font-family:monospace}
@media(max-width:650px){.hit{grid-template-columns:1fr}}</style>
</head><body><h1>ERF lexical footage search: synthetic demo</h1>
<p><strong>Invented transcript entries and illustrated posters.</strong>
This is a real execution of ERF's FTS5 index/search code, not GPU semantic
retrieval, not extracted footage, not a live ERF UI or an operational video player.</p>
<p>Query: <code>""" + escape(payload["query"]) + "</code> | " + str(len(payload["results"])) + """ result(s) | 24 fps</p>
""" + listing + "\n</body></html>\n"


def write_preview(destination: Path, payload: dict) -> None:
    """Explicit opt-in HTML export, never overwrite or write inside checkout."""
    dest = destination.expanduser().resolve(strict=False)
    if dest.is_relative_to(ROOT) or not dest.parent.is_dir():
        raise ValueError("Choose an existing output directory outside the source repository")
    with dest.open("x", encoding="utf-8") as stream:
        stream.write(render_html(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="steam", help="1-5 lexical words (default: steam)")
    parser.add_argument("--html", type=Path, help="Optional new HTML path outside the checkout")
    args = parser.parse_args(argv)
    try:
        payload = run_demo(args.query)
        if args.html is not None:
            write_preview(args.html, payload)
            print("Created local synthetic preview: " + str(args.html), file=sys.stderr)
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print("DEMO ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

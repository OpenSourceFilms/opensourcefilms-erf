#!/usr/bin/env python3
"""ERF Phase 1 pipeline runner: A → B0 → X0 → B1 → C → P.

Usage:
    python bin/run_pipeline.py <media_path> [--config state/config.toml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erf.asr import run as run_a
from erf.chunk import run as run_b0
from erf.embed_text import run as run_b1
from erf.fts import index_media
from erf.materialise import run as run_c
from erf.proxy import run as run_p


def main() -> None:
    parser = argparse.ArgumentParser(description="ERF Phase 1 pipeline")
    parser.add_argument("media_path", help="Path to media file")
    parser.add_argument("--config", default="state/config.toml", help="Config path")
    args = parser.parse_args()

    media_path = Path(args.media_path)
    config_path = Path(args.config)

    print(f"\n=== Stage A: Ingest + ASR + Diarization ===")
    m_a = run_a(media_path, config_path=config_path)
    print(f"  ok={m_a.ok}  media_id={m_a.media_id}  counts={m_a.counts}")
    if not m_a.ok:
        print(f"  ERROR: {m_a.error}")
        sys.exit(1)

    media_id = m_a.media_id

    print(f"\n=== Stage B0: Chunking ===")
    m_b0 = run_b0(media_id, config_path=config_path)
    print(f"  ok={m_b0.ok}  counts={m_b0.counts}")
    if not m_b0.ok:
        print(f"  ERROR: {m_b0.error}")
        sys.exit(1)

    print(f"\n=== Stage X0: FTS indexing ===")
    import tomllib
    cfg = {}
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["catalog"] = raw.get("paths", {}).get("catalog", "catalog")
        cfg["erf_fts_db"] = raw.get("paths", {}).get("erf_fts_db", "state/erf_fts.db")
    else:
        cfg["catalog"] = "catalog"
        cfg["erf_fts_db"] = "state/erf_fts.db"
    fts_counts = index_media(media_id, Path(cfg["catalog"]), Path(cfg["erf_fts_db"]))
    print(f"  FTS indexed: {fts_counts}")

    print(f"\n=== Stage B1: Text embedding ===")
    m_b1 = run_b1(media_id, config_path=config_path)
    print(f"  ok={m_b1.ok}  counts={m_b1.counts}")
    if not m_b1.ok:
        print(f"  ERROR: {m_b1.error}")
        sys.exit(1)

    print(f"\n=== Stage C: Materialise to LanceDB ===")
    m_c = run_c(media_id, config_path=config_path)
    print(f"  ok={m_c.ok}  counts={m_c.counts}")
    if not m_c.ok:
        print(f"  ERROR: {m_c.error}")
        sys.exit(1)

    print(f"\n=== Stage P: Proxy + thumbnails ===")
    # Load duration from manifest
    import json
    mm_path = Path(cfg["catalog"]) / "media_manifest" / f"{media_id}.json"
    duration_ms = 0
    if mm_path.exists():
        duration_ms = json.loads(mm_path.read_text()).get("duration_ms", 0)

    m_p = run_p(media_id, media_path, config_path=config_path, duration_ms=duration_ms)
    print(f"  ok={m_p.ok}  counts={m_p.counts}")
    if not m_p.ok:
        print(f"  ERROR: {m_p.error}")

    print(f"\n=== Pipeline complete for {media_id} ===")
    print(f"Media ID: {media_id}")
    print(f"Ready for /query")


if __name__ == "__main__":
    main()

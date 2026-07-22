#!/usr/bin/env python3
"""F2 batch driver — run active-speaker detection over all media.

erf.f2_asd.run_f2 handles the per-media venvs/asd subprocess, facet validation,
and journaling internally; this just iterates. Media with no face_tracks are
skipped cleanly by run_f2 (returns ok with skip_reason).

Usage: PYTHONPATH=src python bin/run_f2_batch.py [--media-ids M ...] [--limit N] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erf.f2_asd import run_f2
from erf import outcomes

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_f2_batch")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--media-ids", nargs="+", default=None, help="subset; default = all manifests")
    ap.add_argument("--limit", type=int, default=0, help="process at most N (0 = all)")
    ap.add_argument("--force", action="store_true", help="re-run even if F2 already complete")
    args = ap.parse_args()

    ids = args.media_ids or sorted(p.stem for p in (CAT / "media_manifest").glob("m_*.json"))
    if args.limit:
        ids = ids[: args.limit]

    log.info("F2 over %d media (force=%s)", len(ids), args.force)
    ok = skip = err = 0
    for i, mid in enumerate(ids, 1):
        if not args.force and outcomes.is_complete(mid, "F2"):
            skip += 1
            log.debug("[%d/%d] %s already complete — skip", i, len(ids), mid)
            continue
        try:
            res = run_f2(mid)
            if res.get("skip_reason"):
                skip += 1
                outcomes.record(mid, "F2", "skip", reason=res["skip_reason"])
                outcomes.set_flag(mid, "has_asd", False, reason=res["skip_reason"])
                log.info("[%d/%d] %s SKIP (%s)", i, len(ids), mid, res["skip_reason"])
            elif res.get("ok"):
                ok += 1
                outcomes.record(mid, "F2", "ok", artifacts=[res.get("output_path")] if res.get("output_path") else [])
                outcomes.set_flag(mid, "has_asd", True)
                log.info("[%d/%d] %s ok -> %s", i, len(ids), mid, res.get("output_path"))
            else:
                err += 1
                outcomes.record(mid, "F2", "error", reason=res.get("error") or "unknown")
                log.warning("[%d/%d] %s ERROR: %s", i, len(ids), mid, res.get("error"))
        except Exception as e:
            err += 1
            log.error("[%d/%d] %s raised: %s", i, len(ids), mid, e)
    log.info("F2 done: ok=%d skip=%d err=%d", ok, skip, err)


if __name__ == "__main__":
    main()

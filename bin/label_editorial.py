#!/usr/bin/env python3
"""LABEL — editorial-function labelling pass over evidence packages.

Writes versioned, immutable editorial_signals records via erf.editorial_signals
(catalog/editorial/{media_id}/{editorial_job_id}.json + manifest.json): per-sentence
narrative functions, production tensions, chain affordances, and quote-completeness,
plus clip-level story_forces / central_tension / archetypes. Flash-first; sanitized to
the frozen vocabulary; staleness-gated on the evidence hash; journaled (stage LABEL).

Usage: PYTHONPATH=src DEEPSEEK_API_KEY=.. venvs/embed/bin/python bin/label_editorial.py --all [--force]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from erf import editorial_signals as es
from erf import outcomes
from erf.editorial import label_clip
from erf.llm import _FLASH_MODEL
try:
    from erf.runlog import log_event, heartbeat
except Exception:
    log_event = heartbeat = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("label_editorial")
CAT = Path("/workspace/erf/catalog")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--media-ids", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.all:
        ids = sorted(Path(f).stem for f in glob.glob(str(CAT / "evidence" / "*.json")))
    else:
        ids = args.media_ids
    log.info("LABEL: %d clips", len(ids))

    job_id = f"label_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    # Parallelise across clips: label_clip is one Flash call (network-bound, stateless),
    # es.write is per-media + atomic, outcomes/runlog are concurrency-safe (busy timeout).
    # Idempotency = the staleness gate below. Counters/heartbeat tallied in the main thread.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os as _os
    workers = max(1, int(_os.environ.get("ERF_LABEL_WORKERS", "8")))

    def _one(mid: str):
        ev = CAT / "evidence" / f"{mid}.json"
        # Consult EV's explicit outcome (cross-stage awareness), not file presence.
        if outcomes.flag(mid, "has_evidence") is False or not ev.exists():
            outcomes.record(mid, "LABEL", "skip", reason="no_evidence")
            outcomes.set_flag(mid, "has_editorial", False, reason="no_evidence")
            return ("skip", mid, None)
        pkg = json.loads(ev.read_text(encoding="utf-8"))
        seh = es.source_evidence_hash(pkg)
        if not args.force and not es.is_stale(mid, seh):
            if log_event:
                log_event("LABEL", mid, "skip", {"reason": "fresh editorial record exists"})
            return ("skip", mid, None)
        doc = label_clip(pkg)
        _, coercions = es.sanitize(doc)
        rec = es.build_record(mid, doc, source_evidence_hash=seh, editorial_job_id=job_id,
                              model=_FLASH_MODEL, stage_version="LABEL/1")
        es.write(mid, rec)
        outcomes.record(mid, "LABEL", "ok",
                        counts={"sentences": len(rec.get("sentences", [])),
                                "coercions": len(coercions)})
        outcomes.set_flag(mid, "has_editorial", True)
        if coercions and log_event:
            log_event("LABEL", mid, "warn", {"coercions": coercions[:20]})
        return ("ok", mid, {"sentences": len(rec.get("sentences", [])),
                            "tension": (rec.get("clip_level") or {}).get("central_tension"),
                            "coercions": len(coercions)})

    ok = skip = err = done = 0
    log.info("LABEL: parallel workers=%d", workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, m): m for m in ids}
        for fut in as_completed(futs):
            done += 1
            mid = futs[fut]
            try:
                status, _mid, info = fut.result()
            except Exception as e:
                err += 1
                log.error("[%s] FAILED: %s", mid, e)
                try:
                    outcomes.record(mid, "LABEL", "error", reason=str(e)[:200])
                except Exception:
                    pass
                continue
            if status == "ok":
                ok += 1
                log.info("[%s] %d/%d labelled %d sentences | tension=%s%s", mid, done, len(ids),
                         info["sentences"], info["tension"],
                         f" | coerced={info['coercions']}" if info["coercions"] else "")
            else:
                skip += 1
            if heartbeat and done % 5 == 0:
                heartbeat("label_editorial", {"i": done, "n": len(ids), "ok": ok})
    log.info("Done. job=%s ok=%d skip=%d err=%d", job_id, ok, skip, err)


if __name__ == "__main__":
    main()

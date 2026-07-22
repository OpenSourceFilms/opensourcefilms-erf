#!/usr/bin/env python3
"""Retroactively flip the known FALSE 'error' outcomes to the correct 'skip', by
RE-RUNNING the (already-fixed) canonical stage code — never by hand-mutating the
outcomes surface (single-writer discipline preserved).

Targets the false-errors found in the 2026-06-24 pipeline audit:
  - C  (text materialise): no-speech media that erred on absent embeddings.
  - D0 (preprocess): no-audio-stream media that erred on ffmpeg extraction.

Run ONLY when no other process is writing catalog/outcomes (i.e. after a rebuild
finishes) to avoid a read-modify-write race on the per-media outcome files.

Usage: PYTHONPATH=src venvs/embed/bin/python bin/cleanup_false_outcomes.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from erf import outcomes
from erf.materialise import run as run_c

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="List what would be re-run; no writes")
    args = ap.parse_args()
    dry = args.dry_run

    flipped_c = flipped_d0 = 0
    still_c: list[str] = []
    still_d0: list[str] = []

    # ── C: no-speech media falsely marked error ────────────────────────────────
    c_error = outcomes.media_by_status("C", "error")
    log.info("C: %d media in 'error'", len(c_error))
    for mid in c_error:
        if outcomes.flag(mid, "has_speech") is not False:
            continue  # speech present (or unknown) → a genuine error, leave it
        if dry:
            log.info("  would re-run C for %s (has_speech=False)", mid)
            continue
        before = outcomes.status_of(mid, "C")
        try:
            run_c(mid, config_path="state/config.toml")
        except Exception as e:
            log.error("  C re-run raised for %s: %s", mid, e)
        after = outcomes.status_of(mid, "C")
        log.info("  %s: C %s -> %s", mid, before, after)
        if after == "skip":
            flipped_c += 1
        else:
            still_c.append(mid)

    # ── D0: no-audio-stream media falsely marked error ─────────────────────────
    d0_error = outcomes.media_by_status("D0", "error")
    log.info("D0: %d media in 'error'", len(d0_error))
    for mid in d0_error:
        if dry:
            log.info("  would re-run D0 for %s", mid)
            continue
        before = outcomes.status_of(mid, "D0")
        try:
            subprocess.run(
                [str(ROOT / "venvs/anyenhance/bin/python"), "bin/d0_preprocess.py", mid],
                cwd=str(ROOT), env={**os.environ, "PYTHONPATH": "src"},
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            log.error("  D0 subprocess failed for %s: %s", mid, (e.stderr or "").strip()[:200])
        after = outcomes.status_of(mid, "D0")
        log.info("  %s: D0 %s -> %s", mid, before, after)
        if after == "skip":
            flipped_d0 += 1
        else:
            still_d0.append(mid)

    log.info("Summary: C flipped=%d, D0 flipped=%d", flipped_c, flipped_d0)
    if still_c:
        log.warning("C still in error (GENUINE — investigate): %s", still_c)
    if still_d0:
        log.warning("D0 still in error (GENUINE — investigate): %s", still_d0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("cleanup_false_outcomes")
    main()

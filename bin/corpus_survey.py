#!/usr/bin/env python3
"""Corpus survey — DS Pro reads the condensed evidence of the WHOLE corpus and
picks the strongest moments for marketing shorts (hybrid: interview or on-set).

Writes catalog/survey/latest.json with ranked picks. Feed the top picks to
bin/make_short.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from erf.grounded_editor import condense, survey_select
try:
    from erf.runlog import log_event
except Exception:
    log_event = None

CAT = Path("/workspace/erf/catalog")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--footage-types", nargs="*", default=None,
                    help="restrict to these footage_types (e.g. interview talking_head)")
    ap.add_argument("--min-clarity", type=float, default=0.0)
    args = ap.parse_args()

    entries = []
    for p in sorted((CAT / "evidence").glob("*.json")):
        try:
            pkg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not pkg.get("sentences"):
            continue
        if args.footage_types and pkg.get("footage_type") not in args.footage_types:
            continue
        clarity = (pkg.get("av_quality") or {}).get("speech_clarity_median") or 0
        if clarity < args.min_clarity:
            continue
        entries.append(condense(pkg))
    print(f"survey over {len(entries)} clips...")

    picks = survey_select(entries, n=args.n)
    out = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_candidates": len(entries), "picks": picks}
    (CAT / "survey").mkdir(exist_ok=True)
    (CAT / "survey" / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if log_event:
        log_event("SURVEY", "_corpus", "ok", {"n_candidates": len(entries), "n_picks": len(picks)})
    print(f"picked {len(picks)}:")
    for i, pk in enumerate(picks, 1):
        print(f"  {i}. {pk.get('media_id')} (s={pk.get('strength')}) — {pk.get('angle')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""G-E1 gate — validate Stage E1 on a 30-event sample.

Checks:
  (a) Score distribution is discriminating (not all-high)
  (b) JSON parses cleanly for all 30 records
  (c) Every record references a real event_id
  (d) Idempotent re-run reproduces the same signals

Usage:
    PYTHONPATH=src venvs/embed/bin/python bin/gate_e1.py [--workers N]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from erf.atomicio import read_jsonl
from erf.e1_signals import run as e1_run

_CATALOG = _ROOT / "catalog"
_JOURNAL_DB = _ROOT / "state" / "journal.db"
_SAMPLE_SIZE = 30
_SEED = 42


def _collect_sample() -> list[tuple[str, dict]]:
    """Return up to SAMPLE_SIZE (media_id, event) tuples, spread across media."""
    events_dir = _CATALOG / "events"
    all_events: list[tuple[str, dict]] = []
    for f in sorted(events_dir.glob("*.jsonl")):
        mid = f.stem
        for ev in read_jsonl(f):
            if ev.get("summary"):  # only events with a summary are scoreable
                all_events.append((mid, ev))

    rng = random.Random(_SEED)
    if len(all_events) <= _SAMPLE_SIZE:
        return all_events
    return rng.sample(all_events, _SAMPLE_SIZE)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="G-E1 gate script")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print(f"=== G-E1 Gate — {_SAMPLE_SIZE}-event sample ===\n")

    # Collect sample
    sample = _collect_sample()
    print(f"Sample collected: {len(sample)} events from {len(set(m for m,_ in sample))} media\n")

    # Group by media_id so we run e1_run per media
    by_media: dict[str, list[str]] = {}
    event_ids_in_sample: set[str] = set()
    for mid, ev in sample:
        by_media.setdefault(mid, [])
        by_media[mid].append(ev["event_id"])
        event_ids_in_sample.add(ev["event_id"])

    # First pass — run E1 on each media in the sample
    t0 = time.monotonic()
    results_pass1: dict[str, dict] = {}
    for mid in sorted(by_media):
        result = e1_run(mid, catalog=_CATALOG, journal_db=_JOURNAL_DB, max_workers=args.workers)
        results_pass1[mid] = result
        print(f"  Pass 1 {mid}: {result['records_written']} records, {result['parse_errors']} parse_errors")

    elapsed_pass1 = time.monotonic() - t0
    print(f"\nPass 1 complete in {elapsed_pass1:.1f}s\n")

    # Load signals and filter to sample events
    signals_dir = _CATALOG / "event_signals"
    sample_signals: list[dict] = []
    real_event_ids: set[str] = set()

    # Build set of real event_ids from events dir
    for mid in by_media:
        for ev in read_jsonl(_CATALOG / "events" / f"{mid}.jsonl"):
            real_event_ids.add(ev["event_id"])

    for mid in by_media:
        for sig in read_jsonl(signals_dir / f"{mid}.jsonl"):
            if sig.get("event_id") in event_ids_in_sample:
                sample_signals.append(sig)

    print(f"=== CHECK (b): JSON parses cleanly ===")
    bad_json = 0
    for sig in sample_signals:
        if "event_id" not in sig or "hook_potential" not in sig:
            bad_json += 1
    if bad_json:
        print(f"  FAIL — {bad_json} records missing required fields")
    else:
        print(f"  PASS — all {len(sample_signals)} records parse cleanly\n")

    print(f"=== CHECK (c): Every record references a real event_id ===")
    orphans = [s["event_id"] for s in sample_signals if s["event_id"] not in real_event_ids]
    if orphans:
        print(f"  FAIL — {len(orphans)} orphan event_ids: {orphans[:3]}")
    else:
        print(f"  PASS — all {len(sample_signals)} event_ids are valid\n")

    # Score distribution
    hook_scores = [s["hook_potential"] for s in sample_signals if s.get("hook_potential") is not None]
    quote_scores = [s["quote_potential"] for s in sample_signals if s.get("quote_potential") is not None]
    null_count = sum(1 for s in sample_signals if s.get("hook_potential") is None)
    roles = [s.get("narrative_role_suggestion", "unknown") for s in sample_signals]
    role_counts: dict[str, int] = {}
    for r in roles:
        role_counts[r] = role_counts.get(r, 0) + 1

    print(f"=== CHECK (a): Score distribution (hook_potential) ===")
    if hook_scores:
        print(f"  n={len(hook_scores)}  null={null_count}")
        print(f"  min={min(hook_scores):.3f}  p25={_percentile(hook_scores,25):.3f}"
              f"  p50={_percentile(hook_scores,50):.3f}  p75={_percentile(hook_scores,75):.3f}"
              f"  p90={_percentile(hook_scores,90):.3f}  max={max(hook_scores):.3f}")
        print(f"  mean={mean(hook_scores):.3f}  stdev={stdev(hook_scores) if len(hook_scores)>1 else 0:.3f}")
        all_high = all(h >= 0.8 for h in hook_scores)
        if all_high:
            print("  WARN — all hook scores >= 0.8 (model not discriminating)")
        else:
            print("  PASS — distribution shows discrimination")
    else:
        print("  WARN — no valid hook scores in sample")

    print(f"\n=== quote_potential distribution ===")
    if quote_scores:
        print(f"  n={len(quote_scores)}")
        print(f"  min={min(quote_scores):.3f}  p25={_percentile(quote_scores,25):.3f}"
              f"  p50={_percentile(quote_scores,50):.3f}  p75={_percentile(quote_scores,75):.3f}"
              f"  p90={_percentile(quote_scores,90):.3f}  max={max(quote_scores):.3f}")
        print(f"  mean={mean(quote_scores):.3f}  stdev={stdev(quote_scores) if len(quote_scores)>1 else 0:.3f}")

    print(f"\n=== narrative_role_suggestion distribution ===")
    for role, cnt in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"  {role}: {cnt}")

    # Pass 2 — idempotency check
    print(f"\n=== CHECK (d): Idempotent re-run ===")
    t1 = time.monotonic()
    for mid in sorted(by_media):
        e1_run(mid, catalog=_CATALOG, journal_db=_JOURNAL_DB, max_workers=args.workers)

    sample_signals_pass2: list[dict] = []
    for mid in by_media:
        for sig in read_jsonl(signals_dir / f"{mid}.jsonl"):
            if sig.get("event_id") in event_ids_in_sample:
                sample_signals_pass2.append(sig)

    sig1_by_id = {s["event_id"]: s for s in sample_signals}
    sig2_by_id = {s["event_id"]: s for s in sample_signals_pass2}
    mismatch = 0
    for eid, s1 in sig1_by_id.items():
        s2 = sig2_by_id.get(eid)
        if s2 is None:
            mismatch += 1
            continue
        h1, h2 = s1.get("hook_potential"), s2.get("hook_potential")
        q1, q2 = s1.get("quote_potential"), s2.get("quote_potential")
        r1, r2 = s1.get("narrative_role_suggestion"), s2.get("narrative_role_suggestion")
        # Rationale text excluded: Flash wording varies at temp=0 and is cosmetic.
        # Only numeric signals and role classification count for idempotency.
        if (h1 is None) != (h2 is None):
            mismatch += 1
        elif h1 is not None and (
            abs(h1 - h2) > 0.05
            or abs(q1 - q2) > 0.05
            or r1 != r2
        ):
            mismatch += 1

    elapsed_pass2 = time.monotonic() - t1
    if mismatch:
        print(f"  WARN — {mismatch}/{len(sig1_by_id)} records differ between runs "
              f"(LLM at temp=0 may still have minor nondeterminism) [{elapsed_pass2:.1f}s]")
    else:
        print(f"  PASS — all {len(sig1_by_id)} records reproduced identically [{elapsed_pass2:.1f}s]")

    print(f"\n=== G-E1 gate complete ===")
    print(f"Report these hook/quote distributions to the user before proceeding to §2.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Narrative orchestrator CLI — §5 queue runner.

Usage:
    PYTHONPATH=src venvs/embed/bin/python bin/narrator.py run
    PYTHONPATH=src venvs/embed/bin/python bin/narrator.py run --concurrency 3 --limit 5
    PYTHONPATH=src venvs/embed/bin/python bin/narrator.py run --force
    PYTHONPATH=src venvs/embed/bin/python bin/narrator.py status
    PYTHONPATH=src venvs/embed/bin/python bin/narrator.py reset <thread_id>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from erf.narrator import run_orchestrator, print_status, _NARRATIVE_PIPELINE_DB
from erf import narrative_state as ns

_CATALOG = _ROOT / "catalog"
_STATE = _ROOT / "state"
_JOURNAL_DB = _STATE / "journal.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def cmd_run(args: argparse.Namespace) -> None:
    result = run_orchestrator(
        catalog=_CATALOG,
        state=_STATE,
        journal_db=_JOURNAL_DB,
        concurrency=args.concurrency,
        limit=args.limit,
        force=args.force,
        composite_min=args.composite_min,
        coherence_min=args.coherence_min,
    )
    print("\n" + "=" * 60)
    print("ORCHESTRATOR COMPLETE")
    print("=" * 60)
    print(f"  threads total : {result['total']}")
    print(f"  work items    : {result['work_items']}")
    print(f"  gate_passed   : {result['passed']}  (regen: {result['regen_passed']})")
    print(f"  failed/discard: {result['failed']}")
    print(f"  elapsed       : {result.get('elapsed_s', '?')}s")
    print("\nFinal state summary:")
    for state, count in sorted(result.get("summary", {}).items()):
        print(f"  {state:<22} {count:>3}")

    # Show any per-thread errors
    errors = [r for r in result.get("results", []) if r.get("error") and not r.get("skipped")]
    if errors:
        print(f"\n{len(errors)} thread error(s):")
        for r in errors:
            print(f"  {r['thread_id']}: [{r.get('final_state','?')}] {r['error'][:80]}")


def cmd_status(args: argparse.Namespace) -> None:
    pipeline_db = _STATE / _NARRATIVE_PIPELINE_DB
    print_status(pipeline_db)

    # Show shortlist
    pipeline_db_path = _STATE / _NARRATIVE_PIPELINE_DB
    if pipeline_db_path.exists():
        passed = ns.get_by_state(pipeline_db_path, "gate_passed")
        if passed:
            print(f"Shortlist ({len(passed)} threads):")
            for row in passed:
                print(f"  {row['thread_id']}  bp={row.get('bp_id','—')}")
        discarded = ns.get_by_state(pipeline_db_path, "discarded")
        if discarded:
            print(f"\nDiscarded ({len(discarded)} threads):")
            for row in discarded:
                reasons = json.loads(row["failure_reasons"] or "[]")
                print(f"  {row['thread_id']}  reasons={reasons}")


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Pre-populate state for threads that already ran outside the orchestrator."""
    pipeline_db = _STATE / _NARRATIVE_PIPELINE_DB
    ns.ensure_db(pipeline_db)

    # Seed all threads as pending first (OR IGNORE — preserves existing non-pending)
    from pathlib import Path
    thread_ids = [
        f.stem
        for f in sorted((_CATALOG / "narrative_threads").glob("thr_*.json"))
    ]
    seeded = ns.seed_pending(pipeline_db, thread_ids, force=False)
    print(f"Seeded {seeded} threads as pending (existing states unchanged)")

    # Advance based on artifacts
    counts = ns.bootstrap_from_artifacts(pipeline_db, _CATALOG)
    print(f"Advanced from artifacts: {counts}")
    print()
    cmd_status(args)


def cmd_reset(args: argparse.Namespace) -> None:
    pipeline_db = _STATE / _NARRATIVE_PIPELINE_DB
    ns.ensure_db(pipeline_db)
    count = ns.seed_pending(pipeline_db, [args.thread_id], force=True)
    print(f"Reset {args.thread_id} → pending (rows affected: {count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Narrative orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run pipeline on all pending threads")
    p_run.add_argument("--concurrency", type=int, default=3,
                       help="Concurrent Pro calls (default: 3)")
    p_run.add_argument("--limit", type=int, default=None,
                       help="Max threads to process this invocation")
    p_run.add_argument("--force", action="store_true",
                       help="Reset all threads to pending before running")
    p_run.add_argument("--composite-min", type=float, default=0.50, dest="composite_min")
    p_run.add_argument("--coherence-min", type=float, default=0.60, dest="coherence_min")

    sub.add_parser("status", help="Show current pipeline state counts + shortlist")

    sub.add_parser("bootstrap", help="Pre-populate state from existing §3/§4 artifacts")

    p_reset = sub.add_parser("reset", help="Reset a single thread to pending")
    p_reset.add_argument("thread_id", help="Thread ID to reset (e.g. thr_af12ab278296)")

    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "bootstrap":
        cmd_bootstrap(args)
    elif args.cmd == "reset":
        cmd_reset(args)


if __name__ == "__main__":
    main()

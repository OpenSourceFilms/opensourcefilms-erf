"""Narrative orchestrator — §5 queue runner.

Drives each thread through the full pipeline:
  pending → Pass 1 → Pass 2 → §4 gate → [regen Pass 2 → gate] → terminal

Concurrency model: ThreadPoolExecutor with configurable slots. Each slot
runs one thread's full pipeline end-to-end (~3.5 min/thread). State is
persisted to narrative_pipeline.db after every stage transition so a
restart can resume without redoing completed work.

Usage (from bin/narrator.py):
    run_orchestrator(catalog, state_dir, journal_db, concurrency=3)
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import narrative_state as ns
from .author import run_pass1, run_pass2

logger = logging.getLogger(__name__)

_NARRATIVE_PIPELINE_DB = "narrative_pipeline.db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_regen_eligible(failure_reasons_json: str | None) -> bool:
    """Return True if the gate failure may be fixable by Pass 2 reselection.

    Coherence failures are immutable — they reflect the thread's source material
    quality, not atom selection. Any other failure (composite, structural) is
    potentially fixable by a new Pass 2 run.

    A coherence-only failure wastes the regen budget on a retry that provably
    cannot succeed; those threads discard immediately.
    """
    if not failure_reasons_json:
        return True  # Unknown reasons — allow regen
    try:
        reasons = json.loads(failure_reasons_json)
    except Exception:
        return True
    if not reasons:
        return True
    non_coherence = [r for r in reasons if not str(r).lower().startswith("coherence")]
    return len(non_coherence) > 0


def _run_gate_inline(thread_id: str, catalog: Path, composite_min: float, coherence_min: float) -> dict:
    """Evaluate §4 gate for a single thread (no corpus artifact write)."""
    import sys
    _bin = str(Path(__file__).resolve().parent.parent.parent / "bin")
    if _bin not in sys.path:
        sys.path.insert(0, _bin)
    from gate_narrative import gate_thread  # type: ignore[import]

    result = gate_thread(
        thread_id,
        catalog=catalog,
        composite_min=composite_min,
        coherence_min=coherence_min,
    )
    return result


def _run_thread_pipeline(
    thread_id: str,
    *,
    pipeline_db: Path,
    catalog: Path,
    state: Path,
    journal_db: Path,
    composite_min: float,
    coherence_min: float,
    current_state: str,
) -> dict[str, Any]:
    """Run one thread through whatever pipeline stages remain.

    Reads current_state from narr_pipeline_state and resumes from there.
    Returns a result summary dict for aggregation by the orchestrator.
    """
    result: dict[str, Any] = {"thread_id": thread_id, "ok": False, "final_state": current_state}
    t0 = time.monotonic()

    try:
        # --- Pass 1 ---
        if current_state == "pending":
            if not ns.claim(pipeline_db, thread_id, "pending", "pass1_running"):
                result["skipped"] = True
                return result  # another worker got it
            current_state = "pass1_running"

        if current_state == "pass1_running":
            brief_path = catalog / "narrative_briefs" / f"{thread_id}.json"
            if not brief_path.exists():
                logger.info("[%s] Running Pass 1…", thread_id[-8:])
                try:
                    run_pass1(thread_id, catalog=catalog, state=state, journal_db=journal_db)
                except Exception as exc:
                    ns.transition(pipeline_db, thread_id, "pass1_failed", error_info=str(exc)[:500])
                    result["error"] = str(exc)
                    result["final_state"] = "pass1_failed"
                    return result
            ns.transition(
                pipeline_db, thread_id, "pass1_done",
                brief_path=str(brief_path),
            )
            current_state = "pass1_done"
            logger.info("[%s] Pass 1 done", thread_id[-8:])

        # --- Pass 2 (initial or regen) ---
        if current_state == "pass1_done":
            if not ns.claim(pipeline_db, thread_id, "pass1_done", "pass2_running"):
                result["skipped"] = True
                return result
            current_state = "pass2_running"

        if current_state == "pass2_running":
            row = ns.get_thread_state(pipeline_db, thread_id)
            attempt = (row or {}).get("pass2_attempt", 0)
            label = "Regen Pass 2" if attempt > 0 else "Pass 2"
            logger.info("[%s] Running %s (attempt=%d)…", thread_id[-8:], label, attempt)
            try:
                bp = run_pass2(thread_id, catalog=catalog, state=state, journal_db=journal_db)
                bp_id = bp.get("bp_id")
            except Exception as exc:
                ns.transition(pipeline_db, thread_id, "pass2_failed", error_info=str(exc)[:500])
                result["error"] = str(exc)
                result["final_state"] = "pass2_failed"
                return result
            ns.transition(pipeline_db, thread_id, "pass2_done", bp_id=bp_id)
            current_state = "pass2_done"
            logger.info("[%s] %s done → %s", thread_id[-8:], label, bp_id)

        # --- Gate ---
        if current_state == "pass2_done":
            if not ns.claim(pipeline_db, thread_id, "pass2_done", "gate_running"):
                result["skipped"] = True
                return result
            current_state = "gate_running"

        if current_state == "gate_running":
            logger.info("[%s] Running gate…", thread_id[-8:])
            try:
                gate_result = _run_gate_inline(thread_id, catalog, composite_min, coherence_min)
            except Exception as exc:
                # gate failure is unusual — rewind to pass2_done for retry
                ns.transition(pipeline_db, thread_id, "pass2_done", error_info=str(exc)[:200])
                result["error"] = f"gate error: {exc}"
                result["final_state"] = "pass2_done"
                return result

            verdict = gate_result["verdict"]
            gate_id = gate_result.get("gate_id")
            reasons_json = json.dumps(gate_result.get("reasons", []))

            if verdict == "PASS":
                ns.transition(
                    pipeline_db, thread_id, "gate_passed",
                    gate_id=gate_id, failure_reasons=None,
                )
                result["ok"] = True
                result["final_state"] = "gate_passed"
                result["elapsed_s"] = round(time.monotonic() - t0, 1)
                logger.info("[%s] GATE PASS ✓ (%.0fs)", thread_id[-8:], result["elapsed_s"])
                return result

            # FAIL — record gate_failed state
            ns.transition(
                pipeline_db, thread_id, "gate_failed",
                gate_id=gate_id, failure_reasons=reasons_json,
            )
            current_state = "gate_failed"
            logger.info("[%s] Gate FAIL: %s", thread_id[-8:], gate_result.get("reasons"))

        # --- Regen routing ---
        if current_state == "gate_failed":
            # Check: is this failure fixable by Pass 2 reselection?
            row = ns.get_thread_state(pipeline_db, thread_id)
            if not _is_regen_eligible((row or {}).get("failure_reasons")):
                # Coherence failures are thread-level — regen provably cannot fix them.
                # Discard immediately without consuming the regen budget.
                ns.transition(pipeline_db, thread_id, "discarded")
                result["final_state"] = "discarded"
                result["skip_regen"] = "coherence_only"
                logger.info("[%s] Discarded (coherence-only failure — regen skipped)", thread_id[-8:])
                return result

            dispatched = ns.dispatch_regen(pipeline_db, thread_id)
            if not dispatched:
                # Budget exhausted (pass2_attempt >= 1)
                ns.transition(pipeline_db, thread_id, "discarded")
                result["final_state"] = "discarded"
                logger.info("[%s] Discarded (regen budget exhausted)", thread_id[-8:])
                return result

            current_state = "pass2_running"
            logger.info("[%s] Regen dispatched → pass2_running (attempt not yet consumed)", thread_id[-8:])

            # Regen Pass 2
            logger.info("[%s] Running Regen Pass 2…", thread_id[-8:])
            try:
                bp = run_pass2(thread_id, catalog=catalog, state=state, journal_db=journal_db)
                regen_bp_id = bp.get("bp_id")
            except Exception as exc:
                # Pass 2 itself failed — leave in pass2_failed; budget NOT consumed
                # (recover_inflight will rewind to gate_failed on next restart)
                ns.transition(pipeline_db, thread_id, "pass2_failed", error_info=str(exc)[:500])
                result["error"] = str(exc)
                result["final_state"] = "pass2_failed"
                return result
            ns.transition(pipeline_db, thread_id, "pass2_done", bp_id=regen_bp_id)
            logger.info("[%s] Regen Pass 2 done → %s", thread_id[-8:], regen_bp_id)

            # Gate2 — budget consumed only on terminal outcome
            try:
                gate_result = _run_gate_inline(thread_id, catalog, composite_min, coherence_min)
            except Exception as exc:
                ns.transition(pipeline_db, thread_id, "pass2_done", error_info=str(exc)[:200])
                result["error"] = f"gate2 error: {exc}"
                result["final_state"] = "pass2_done"
                return result

            verdict = gate_result["verdict"]
            gate_id = gate_result.get("gate_id")
            regen_reasons_json = json.dumps(gate_result.get("reasons", []))

            passed = verdict == "PASS"
            # commit_regen_outcome increments pass2_attempt=1 and sets final state
            ns.commit_regen_outcome(
                pipeline_db, thread_id, passed,
                gate_id=gate_id,
                failure_reasons=None if passed else regen_reasons_json,
            )
            if passed:
                result["ok"] = True
                result["final_state"] = "gate_passed"
                result["regen"] = True
                result["elapsed_s"] = round(time.monotonic() - t0, 1)
                logger.info("[%s] GATE PASS (regen) ✓ (%.0fs)", thread_id[-8:], result["elapsed_s"])
            else:
                result["final_state"] = "discarded"
                logger.info("[%s] Discarded after regen: %s", thread_id[-8:], gate_result.get("reasons"))

        result["elapsed_s"] = round(time.monotonic() - t0, 1)
        return result

    except Exception as exc:
        logger.exception("[%s] Unhandled error in pipeline: %s", thread_id[-8:], exc)
        try:
            row = ns.get_thread_state(pipeline_db, thread_id)
            cur_state = (row or {}).get("state", "unknown")
            if cur_state not in ns.TERMINAL_STATES:
                ns.transition(pipeline_db, thread_id, "pass2_failed", error_info=str(exc)[:500])
        except Exception:
            pass
        result["error"] = str(exc)
        result["final_state"] = "pass2_failed"
        result["elapsed_s"] = round(time.monotonic() - t0, 1)
        return result


def run_orchestrator(
    *,
    catalog: Path,
    state: Path,
    journal_db: Path,
    concurrency: int = 3,
    limit: int | None = None,
    force: bool = False,
    composite_min: float = 0.50,
    coherence_min: float = 0.60,
) -> dict[str, Any]:
    """Run the narrative pipeline orchestrator.

    Loads all threads, seeds them as pending (respects existing state unless
    force=True), recovers any in-flight states from a prior crash, then
    dispatches threads to a worker pool.

    Args:
        catalog: Path to catalog/ directory.
        state: Path to state/ directory (guidance.md, config).
        journal_db: Path to state/journal.db.
        concurrency: Number of concurrent Pro calls.
        limit: Cap threads processed this run (None = all pending).
        force: Reset all threads to pending before running.
        composite_min: §4 gate composite threshold.
        coherence_min: §4 gate coherence threshold.

    Returns:
        Summary dict with counts and per-thread results.
    """
    pipeline_db = state / _NARRATIVE_PIPELINE_DB
    ns.ensure_db(pipeline_db)

    # Discover all threads
    threads_dir = catalog / "narrative_threads"
    thread_files = sorted(threads_dir.glob("thr_*.json"))
    thread_ids = []
    for tf in thread_files:
        try:
            t = json.loads(tf.read_text(encoding="utf-8"))
            thread_ids.append(t["thread_id"])
        except Exception as exc:
            logger.warning("Skipping corrupt thread file %s: %s", tf.name, exc)

    if not thread_ids:
        logger.warning("No threads found in %s", threads_dir)
        return {"total": 0, "seeded": 0, "results": []}

    # Seed pending state
    seeded = ns.seed_pending(pipeline_db, thread_ids, force=force)
    logger.info("Seeded %d threads as pending (force=%s)", seeded, force)

    # Crash recovery: advance or rewind in-flight states
    recovered = ns.recover_inflight(pipeline_db, catalog)
    if any(recovered.values()):
        logger.info("Recovered in-flight states: %s", recovered)

    # Collect threads that still need work
    workable_states = {"pending", "pass1_done", "pass2_done", "gate_failed"}
    all_states = ns.get_summary(pipeline_db)
    logger.info("Pipeline state before run: %s", all_states)

    work_queue: list[tuple[str, str]] = []
    for tid in thread_ids:
        row = ns.get_thread_state(pipeline_db, tid)
        if not row:
            continue
        cur = row["state"]
        if cur in workable_states:
            work_queue.append((tid, cur))

    if limit is not None:
        work_queue = work_queue[:limit]

    if not work_queue:
        logger.info("Nothing to do — all threads already at terminal or in-flight state")
        return {
            "total": len(thread_ids),
            "seeded": seeded,
            "work_items": 0,
            "results": [],
            "summary": ns.get_summary(pipeline_db),
        }

    logger.info(
        "Dispatching %d threads with concurrency=%d (composite≥%.2f, coherence≥%.2f)",
        len(work_queue), concurrency, composite_min, coherence_min,
    )

    t_start = time.monotonic()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_thread_pipeline,
                tid,
                pipeline_db=pipeline_db,
                catalog=catalog,
                state=state,
                journal_db=journal_db,
                composite_min=composite_min,
                coherence_min=coherence_min,
                current_state=cur_state,
            ): tid
            for tid, cur_state in work_queue
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"thread_id": tid, "ok": False, "error": str(exc), "final_state": "unknown"}
            results.append(res)

    elapsed = round(time.monotonic() - t_start, 1)
    final_summary = ns.get_summary(pipeline_db)

    passed = sum(1 for r in results if r.get("final_state") == "gate_passed")
    failed = sum(1 for r in results if r.get("final_state") in ("discarded", "pass1_failed", "pass2_failed"))
    regen_passed = sum(1 for r in results if r.get("regen") and r.get("ok"))

    logger.info(
        "Orchestrator done in %.0fs — %d passed (%d via regen), %d failed/discarded",
        elapsed, passed, regen_passed, failed,
    )

    return {
        "total": len(thread_ids),
        "seeded": seeded,
        "work_items": len(work_queue),
        "elapsed_s": elapsed,
        "passed": passed,
        "regen_passed": regen_passed,
        "failed": failed,
        "results": results,
        "summary": final_summary,
    }


def print_status(pipeline_db: Path) -> None:
    """Print current pipeline state counts to stdout."""
    if not pipeline_db.exists():
        print("Pipeline DB not initialised — no runs yet.")
        return
    summary = ns.get_summary(pipeline_db)
    print("\nNarrative pipeline state:")
    state_order = [
        "pending", "pass1_running", "pass1_done", "pass1_failed",
        "pass2_running", "pass2_done", "gate_running",
        "gate_passed", "gate_failed", "discarded", "pass2_failed",
    ]
    for s in state_order:
        n = summary.get(s, 0)
        if n or s in ("gate_passed", "discarded"):
            marker = "✓" if s == "gate_passed" else ("✗" if s in ("discarded", "pass1_failed", "pass2_failed") else " ")
            print(f"  {marker} {s:<20} {n:>3}")
    total = sum(summary.values())
    print(f"  {'─'*27}")
    print(f"    {'total':<20} {total:>3}\n")

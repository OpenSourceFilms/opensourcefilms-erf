#!/usr/bin/env python3
"""timeline_audit — per-clip forensic READ+WRITE ledger (MASTER §5.2.2).

For one or more media_ids, reconstruct the complete story of what the system did
to the clip:
  1. WRITE ledger  — every stage: did it write an artifact? did it record an
     outcome? do the two agree? (surfaces SILENT-DEFECT / PHANTOM / CONTRADICTION)
  2. EXPECTED-BUT-MISSING — given the has_speech/has_faces flags + the DAG, which
     stages SHOULD have produced/recorded for this clip but didn't.
  3. READ ledger   — who consumed each artifact. NOTE: reads are NOT logged by the
     pipeline; consumption is INFERRED from the DAG (a downstream stage that has its
     own artifact must have read its inputs). That inference gap is itself a finding.
  4. META-CONSUMERS — the log-readers: is this clip in status_projection (auditor)
     and does it show in health.json critical_gaps (health consumer)?

Single source of truth for "artifact exists" and "what stages exist": the
src/erf/stages.py registry (D-Log-04 Phase 2) — this file no longer imports
reconcile_status; it derives its own artifact predicates from the same
registry reconcile_status.py does, independently, per stage.

Run: PYTHONPATH=src venvs/control/bin/python bin/timeline_audit.py <media_id> [...] [--json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"
DB = ROOT / "state" / "journal.db"
HEALTH = ROOT / "logs.all" / "_latest" / "health.json"

sys.path.insert(0, str(ROOT / "src"))
from erf import outcomes as oc                       # noqa: E402
from erf import stages as st                          # noqa: E402  (the ONE stage universe)


def _ne(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _predicate_fn(stage: "st.Stage"):
    preds = st.artifact_predicates(stage)

    def check(m: str) -> bool:
        for kind, tmpl in preds:
            p = ROOT / tmpl.format(mid=m)
            ok = _ne(p) if kind == "ne" else p.exists()
            if not ok:
                return False
        return True
    return check


# stage -> predicate(media_id) -> True iff the canonical artifact is present on
# disk. Phase 2 (D-Log-04): derived from the registry, same order/scope as
# reconcile_status.STAGE_ARTIFACTS (st.RECONCILE_ORDER) — independently built
# here rather than imported, per the Phase 2 report (decouples the two tools;
# both now derive from stages.py instead of one importing the other's derived
# state).
STAGE_ARTIFACTS: dict = {
    name: _predicate_fn(st.STAGES[name]) for name in st.RECONCILE_ORDER
}

# DAG read-edges: consumer -> upstream stages whose artifacts it reads (MASTER §5).
# Phase 2 (D-Log-06): VALUES now come from the registry's `depends` (deleting the
# duplicated dependency-list literals this dict used to hardcode) — see
# tests/test_stages_registry.py::test_read_edges_equivalence for the proof this
# reproduces the pre-Phase-2 READ_EDGES values exactly for every key below.
#
# FINDING (scope, not deleted silently — task item 6): the KEY SET here is
# frozen to the original 25 consumers, not "every registry stage with
# depends". The registry now also carries real `depends` for P, Q0, E2, tags
# (never in READ_EDGES — a genuine pre-existing gap in this file's own
# hardcoded DAG, the "fourth universe" the redesign doc names) and for
# REDIARIZE/CLUSTERS (documented in stages.py as inferred, not sourced from
# here). Expanding to the full registry would add new READERS entries (e.g.
# P becomes a reader of B3, tags becomes a reader of B0/E0) and change this
# tool's printed output for every media — out of scope for "do not change
# what any consumer does" (task item 8); left as a Phase-3-or-later follow-up.
_READ_LEDGER_CONSUMERS = (
    "B0", "B1", "B2", "B3", "B4", "C", "E0", "F0", "F1", "F2", "D0", "P0", "W",
    "RG", "RP", "DX", "SEV2", "SEV2_ADJ", "PSTREAM", "T0", "EV", "LABEL", "V0",
    "E1", "CARD",
)
READ_EDGES = {name: list(st.STAGES[name].depends) for name in _READ_LEDGER_CONSUMERS}
# invert -> stage: who reads it
READERS: dict[str, list[str]] = {}
for consumer, inputs in READ_EDGES.items():
    for src in inputs:
        READERS.setdefault(src, []).append(consumer)

SPEECH_SPINE = set(st.speech_spine_stages().keys())
VISUAL = {k for k, v in st.STAGES.items() if v.visual}
TERMINAL = {"LABEL", "C", "EV"}  # legitimately consumed by nothing downstream here


def _exists(stage: str, m: str) -> bool:
    pred = STAGE_ARTIFACTS.get(stage)
    try:
        return bool(pred(m)) if pred else False
    except Exception:
        return False


def _journal_events(m: str) -> list[tuple]:
    """Per-stage journal-row tally. Phase 1 (D-Log-04): journal activity for a
    media_id now spans two tables — `events` (outcome dispositions, §3.1) and
    `events_trace` (start/warn/sub-events, old shape). Union both, normalised to
    (at_utc, stage, type, run_id), to preserve this function's original meaning
    (schema-compatibility fix only — no change to what callers do with the rows)."""
    try:
        conn = sqlite3.connect(str(DB))
        rows = conn.execute(
            "SELECT at_utc, stage, status, run_id FROM events WHERE media_id=? "
            "UNION ALL "
            "SELECT at_utc, stage, event_type, run_id FROM events_trace WHERE media_id=? "
            "ORDER BY at_utc",
            (m, m)).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _projection_stages(m: str) -> dict[str, int]:
    try:
        conn = sqlite3.connect(str(DB))
        rows = conn.execute(
            "SELECT stage, ok FROM status_projection WHERE media_id=?", (m,)).fetchall()
        conn.close()
        return {s: ok for s, ok in rows}
    except Exception:
        return {}


def _verdict(exists: bool, status: str | None) -> str:
    if exists and status in ("ok", "empty"):
        return "WROTE+RECORDED"
    if exists and status is None:
        return "SILENT-DEFECT"        # ran/wrote but never declared an outcome
    if exists and status in ("skip", "error"):
        return "CONTRADICTION"        # says no-artifact yet artifact is on disk
    if exists and status == "partial":
        return "PARTIAL"
    if not exists and status in ("ok", "empty"):
        return "PHANTOM"              # declared success but no artifact
    if not exists and status in ("skip", "error"):
        return "RECORDED-SKIP"        # concluded no-artifact, with a reason
    return "ABSENT"                   # neither artifact nor outcome


def audit(m: str) -> dict:
    doc = oc.for_media(m)
    stages_rec = doc.get("stages", {})
    flags = {k: v.get("value") for k, v in doc.get("flags", {}).items()}
    ev = _journal_events(m)
    ev_by_stage: dict[str, int] = {}
    for _at, st, _et, _rid in ev:
        ev_by_stage[st] = ev_by_stage.get(st, 0) + 1
    proj = _projection_stages(m)

    ledger = []
    for stage in STAGE_ARTIFACTS:
        exists = _exists(stage, m)
        rec = stages_rec.get(stage)
        status = rec.get("status") if rec else None
        ledger.append({
            "stage": stage, "exists": exists, "status": status,
            "reason": (rec or {}).get("reason"), "counts": (rec or {}).get("counts"),
            "producer": (rec or {}).get("producer"), "events": ev_by_stage.get(stage, 0),
            "verdict": _verdict(exists, status),
        })

    has_speech = flags.get("has_speech")
    has_faces = flags.get("has_faces")
    missing = []
    for row in ledger:
        st, v = row["stage"], row["verdict"]
        if v in ("SILENT-DEFECT", "PHANTOM", "CONTRADICTION"):
            missing.append({"stage": st, "why": "integrity defect", "actual": v})
        elif st in SPEECH_SPINE and has_speech is not False and v == "ABSENT":
            missing.append({"stage": st, "why": "speech media, spine stage expected", "actual": v})
        elif st in VISUAL and has_faces is not False and v == "ABSENT" and has_faces is not None:
            missing.append({"stage": st, "why": "has_faces, visual stage expected", "actual": v})

    return {"media_id": m, "flags": flags, "ledger": ledger, "missing": missing,
            "projection": proj, "n_journal_events": len(ev)}


def _print(a: dict) -> None:
    m = a["media_id"]
    print(f"\n{'='*74}\nCLIP {m}   flags={a['flags'] or '{}'}   journal_events={a['n_journal_events']}\n{'='*74}")

    print("\n--- 1. WRITE LEDGER (stage | artifact | outcome | jrnl | verdict) ---")
    print(f"{'stage':<10}{'artifact':>9}{'outcome':>12}{'jrnl':>6}  verdict")
    for r in a["ledger"]:
        art = "on-disk" if r["exists"] else "-"
        out = r["status"] or "NONE"
        if r["reason"]:
            out = f"{out}:{r['reason']}"[:11]
        flag = "" if r["verdict"] in ("WROTE+RECORDED", "RECORDED-SKIP", "ABSENT") else "  <--"
        print(f"{r['stage']:<10}{art:>9}{out:>12}{r['events']:>6}  {r['verdict']}{flag}")

    print("\n--- 2. EXPECTED-BUT-MISSING ---")
    if a["flags"].get("has_speech") is False:
        print("  has_speech=False → speech spine legitimately absent (exempt).")
    if not a["missing"]:
        print("  none — every expected stage wrote AND recorded.")
    for x in a["missing"]:
        print(f"  {x['stage']:<10} expected: {x['why']:<44} actual: {x['actual']}")

    print("\n--- 3. READ LEDGER (INFERRED from DAG; reads are NOT logged) ---")
    exists_map = {r["stage"]: r["exists"] for r in a["ledger"]}
    for r in a["ledger"]:
        st = r["stage"]
        if not r["exists"]:
            continue
        readers = READERS.get(st, [])
        if not readers:
            tag = "(terminal — consumed by nothing here)" if st in TERMINAL else "(no known downstream reader)"
            print(f"  {st:<10} -> {tag}")
            continue
        parts = []
        for rd in readers:
            if exists_map.get(rd):
                parts.append(f"{rd}=CONSUMED")
            else:
                parts.append(f"{rd}=DID-NOT-READ")
        print(f"  {st:<10} -> {', '.join(parts)}")

    print("\n--- 4. META-CONSUMERS (the log-readers) ---")
    proj = a["projection"]
    print(f"  reconcile/status_projection: {len(proj)} stage rows recorded for this clip")
    try:
        health = json.loads(HEALTH.read_text())
        gaps = health.get("critical_gaps", {})
        if m in gaps:
            print(f"  pipeline_status/health.json: FLAGGED — critical gaps {gaps[m]}")
        else:
            print(f"  pipeline_status/health.json: clean (not in critical_gaps; snapshot {health.get('at_utc','?')[:19]})")
    except Exception as e:
        print(f"  pipeline_status/health.json: unreadable ({e})")
    print("  orchestrator read_back: aggregate-level (see logs.all/rebuild_report_*.json)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media_ids", nargs="+")
    ap.add_argument("--json", action="store_true", help="dump machine-readable dict instead of tables")
    args = ap.parse_args()
    results = [audit(m) for m in args.media_ids]
    if args.json:
        print(json.dumps(results, indent=1, default=str))
    else:
        for a in results:
            _print(a)


if __name__ == "__main__":
    main()

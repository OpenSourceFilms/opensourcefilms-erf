#!/usr/bin/env python3
"""pipeline_status — the consumer that reads the log to drive the pipeline.

Turns state/journal.db (the event log) into an actionable health view: per-stage
status, current failures, stuck in-progress runs, media missing critical stages,
and continuous-loop heartbeats. Writes a machine-readable snapshot to
logs.all/_latest/health.json so other tooling (and future me) can consume it.

Phase 1 (D-Log-04): journal.db's `events` table is now the outcome-only surface
(§3.1: status/reason/metrics/...), with a companion `events_trace` table for
start/warn/sub-events (old event_type/payload shape) — see docs/
OBSERVABILITY_REDESIGN.md. health.json is DERIVED from `events` (never asserted
independently); the old `--project` writer (which independently asserted
catalog/status/*.json from status_projection — one of the redesign's named root
causes) is replaced by erf.outcomes.rebuild_status_files(). CRITICAL/SPEECH_SPINE
below are now sourced from the stages.py registry (critical_stages() /
speech_spine_stages()) instead of being hardcoded here — Phase 0's own
consistency test already proved these hardcoded sets equal the registry's;
Phase 2 deletes the literals here entirely in favour of iterating the registry.

Usage: python bin/pipeline_status.py [--failures-only] [--stage T0] [--gaps]
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/erf")
DB = ROOT / "state" / "journal.db"
HB_DIR = ROOT / "logs.all" / "_latest" / "heartbeat"
HEALTH = ROOT / "logs.all" / "_latest" / "health.json"

sys.path.insert(0, str(ROOT / "src"))
from erf import stages as _stages_registry  # noqa: E402
from erf import outcomes as _oc              # noqa: E402
from erf import atomicio as _atomicio        # noqa: E402

# Critical = the whole who's-speaking spine, not just endpoints (MASTER §5.2.2).
# A media missing any of these has a real attribution gap the health view must
# surface — endpoints-only (A/T0/EV) let a mid-spine failure read green.
# Sourced from the stages.py registry (D-Log-04 §3.3/§4 Phase 0), not hardcoded.
CRITICAL = sorted(_stages_registry.critical_stages().keys())
# The speech-only subset: a no-speech media (has_speech=False) legitimately produces
# none of these, so the consumer must read that flag and exempt them — otherwise the
# gap view cries wolf on every no-speech media (reading the log correctly, §5.2.2).
SPEECH_SPINE = set(_stages_registry.speech_spine_stages().keys())
STUCK_MIN = 30


def _no_speech_media() -> set[str]:
    """Media explicitly recorded has_speech=False — exempt from the speech spine."""
    try:
        return set(_oc.media_by_flag("has_speech", False))
    except Exception:
        return set()


def _age_min(iso: str) -> float:
    try:
        t = datetime.fromisoformat(iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 60.0
    except Exception:
        return -1.0


def _latest_status() -> dict:
    """Return {(media_id, stage): (status, at_utc, extra)} — latest outcome per
    (media_id, stage) from `events` (§3.1: the outcome surface), with
    "in_progress" synthesized from `events_trace` when a stage's most recent
    trace is a "start" newer than its latest recorded outcome. Phase 1: "start"
    is trace-only and no longer shares a table/id-space with the disposition, so
    recency is compared by at_utc (ISO8601, sorts correctly as text) across the
    two tables rather than by a single shared autoincrement id."""
    conn = sqlite3.connect(str(DB))
    out: dict = {}
    for media_id, stage, status, at_utc, reason in conn.execute(
        "SELECT e.media_id, e.stage, e.status, e.at_utc, e.reason FROM events e "
        "JOIN (SELECT media_id, stage, MAX(id) AS mid FROM events GROUP BY media_id, stage) latest "
        "ON e.media_id = latest.media_id AND e.stage = latest.stage AND e.id = latest.mid"
    ).fetchall():
        out[(media_id, stage)] = (status, at_utc, reason)
    for media_id, stage, at_utc in conn.execute(
        "SELECT t.media_id, t.stage, t.at_utc FROM events_trace t "
        "JOIN (SELECT media_id, stage, MAX(id) AS mid FROM events_trace "
        "      WHERE event_type='start' GROUP BY media_id, stage) latest "
        "ON t.media_id = latest.media_id AND t.stage = latest.stage AND t.id = latest.mid"
    ).fetchall():
        key = (media_id, stage)
        prior = out.get(key)
        if prior is None or at_utc > prior[1]:
            out[key] = ("in_progress", at_utc, None)
    conn.close()
    return out


def _projection_ok() -> dict[str, int]:
    """Authoritative per-stage completion from status_projection.

    status_projection is a materialized view folded from `events` (§3.4:
    `erf.outcomes.rebuild_projection`) — never independently asserted (Invariant
    1 / D-Log-04 P1). The raw-events tally below can show phantom completion (a
    stage journalled ok but wrote nothing); this is the count that reflects what
    is actually on disk, per the last `reconcile_status.py --reconcile` pass
    (drift between the two is exactly what `reconcile_status.py` (no flags)
    surfaces).
    """
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT stage, COUNT(*) FROM status_projection WHERE ok=1 "
        "AND media_id LIKE 'm_%' AND media_id NOT LIKE 'm_unknown%' GROUP BY stage").fetchall()
    conn.close()
    return {stage: n for stage, n in rows}


def _recent_failures(latest: dict, limit: int = 20) -> list[dict]:
    """Errors that are STILL that (media_id, stage)'s latest recorded outcome.
    A raw `events` query for status='error' (the prior approach) surfaces every
    historical error row ever logged, including ones a later ok/skip/empty run
    already superseded — misleading (found 2026-07-16). `latest` is the same
    latest-per-(media_id,stage) fold `_latest_status()` already computed for the
    stage tallies, so a since-recovered error correctly drops off here too."""
    rows = [
        (at_utc, media_id, stage, reason)
        for (media_id, stage), (status, at_utc, reason) in latest.items()
        if status == "error"
    ]
    rows.sort(key=lambda r: r[0], reverse=True)
    rows = rows[:limit]
    return [{"at_utc": at_utc, "media_id": media_id, "stage": stage, "error": str(reason or "")[:160]}
            for at_utc, media_id, stage, reason in rows]


def _heartbeats() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(HB_DIR / "*.json"))):
        try:
            r = json.loads(Path(f).read_text(encoding="utf-8"))
            r["age_min"] = round(_age_min(r.get("at_utc", "")), 1)
            out.append(r)
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures-only", action="store_true")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--project", action="store_true",
                    help="materialize catalog/status/*.json from the journal, then exit")
    args = ap.parse_args()

    if args.project:
        # Phase 1 (D-Log-04): this used to independently assert catalog/status/*.json
        # from status_projection (root cause #2/#3 — a second writer of the same
        # surface). Now derived, via the single outcomes.rebuild_status_files(),
        # which itself folds status_projection from `events` first.
        n = _oc.rebuild_status_files(DB)
        print(f"derived journal status -> {n} catalog/status/*.json files")
        return

    latest = _latest_status()
    # per-stage tallies — over REAL media only. Synthetic pre-id failures
    # (m_unknown_*, logged before a real media_id could be computed) and
    # job-scoped ids (thr_*, _corpus, story titles) are bucketed separately so
    # the media view stays honest.
    stages: dict[str, dict] = {}
    media_set = set()
    stage_media: dict[str, set] = {}
    stuck = []
    pre_id_failures = 0
    # Corpus-scoped stage verdicts (GATE, CLUSTERS "__corpus__" rows). These are
    # filtered out of the per-media view below (not m_*) but MUST still roll up
    # into the corpus verdict — spec P5 says "health rolls up to one verdict,
    # gate included", and before this fix a degraded GATE/CLUSTERS corpus row
    # was invisible in health.json entirely (W2 checkpoint review finding,
    # D-Log-12: the last-mile instance of the green-but-wrong class).
    corpus_stages: dict[str, dict] = {}
    for (media_id, stage), (status, at_utc, _payload) in latest.items():
        if media_id == "__corpus__":
            corpus_stages[stage] = {"status": status, "at_utc": at_utc,
                                    "reason": _payload}
            continue
        if str(media_id).startswith("m_unknown"):
            if status == "error":
                pre_id_failures += 1
            continue
        if not str(media_id).startswith("m_"):
            continue  # job-scoped (threads/corpus/story) — not a media stage
        media_set.add(media_id)
        stage_media.setdefault(stage, set()).add(media_id)
        # "degraded" explicit in the default (Phase 3, D-Log-04 §4/P5): a
        # health-predicate failure on an "ok" row is not-green (see
        # erf.outcomes.NOT_GREEN / record()'s ok->degraded derivation) and must
        # be visible in health.json even at 0, not only appear dynamically the
        # first time one occurs.
        s = stages.setdefault(stage, {"ok": 0, "error": 0, "skip": 0, "in_progress": 0, "degraded": 0})
        s[status] = s.get(status, 0) + 1
        if status == "in_progress" and _age_min(at_utc) > STUCK_MIN:
            stuck.append({"media_id": media_id, "stage": stage, "age_min": round(_age_min(at_utc), 1)})

    failures = _recent_failures(latest)
    proj_ok = _projection_ok()  # disk-reconciled completion (authoritative)
    hbs = _heartbeats()
    # media missing critical stages
    no_speech = _no_speech_media()
    gaps = {}
    for m in sorted(media_set):
        # ok = produced; skip = no artifact expected (e.g. no-speech W); empty =
        # explicit-empty artifact written by a spine layer for no-speech media.
        # All three are "concluded, not a gap". No-speech media (explicit
        # has_speech=False) are exempt from the speech spine — they concluded
        # without speech, so absence there is correct, not a gap.
        crit = [st for st in CRITICAL if not (m in no_speech and st in SPEECH_SPINE)]
        miss = [st for st in crit if latest.get((m, st), ("missing",))[0] not in ("ok", "skip", "empty")]
        if miss:
            gaps[m] = miss

    corpus_not_green = sorted(
        st for st, v in corpus_stages.items() if v["status"] in _oc.NOT_GREEN
    )
    snapshot = {"at_utc": datetime.now(timezone.utc).isoformat(),
                "n_media": len(media_set), "stages": stages,
                "stuck_in_progress": stuck, "recent_failures": failures,
                "heartbeats": hbs, "critical_gaps": gaps,
                "pre_id_failures": pre_id_failures,
                "corpus_stages": corpus_stages,
                "corpus_not_green": corpus_not_green}
    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    # Phase 1 (D-Log-04 constraint: all derived-view writes go through atomicio).
    _atomicio.write_json(HEALTH, snapshot, indent=1)

    if args.failures_only:
        print(f"=== recent failures ({len(failures)}) ===")
        for f in failures:
            print(f"  {f['at_utc'][:19]} {f['media_id']} [{f['stage']}] {f['error']}")
        return

    print(f"PIPELINE HEALTH @ {snapshot['at_utc'][:19]}  ({len(media_set)} media)")
    if pre_id_failures:
        print(f"  pre-id failures (synthetic m_unknown, pre-media_id errors): {pre_id_failures}")
    if corpus_stages:
        print("\ncorpus-scoped stages (GATE / CLUSTERS __corpus__ verdicts):")
        for st, v in sorted(corpus_stages.items()):
            flag = "  <-- NOT GREEN" if st in corpus_not_green else ""
            print(f"  {st:14} {v['status']:10} {v['at_utc'][:19]}"
                  f"{('  ' + str(v['reason'])) if v['reason'] else ''}{flag}")
    print("\nheartbeats:")
    for h in hbs:
        print(f"  {h.get('component'):16} {h['age_min']}m ago  {({k: v for k, v in h.items() if k not in ('component','at_utc','age_min')})}")
    if not hbs:
        print("  (none)")
    print("\nstage status  (ok = disk-reconciled status_projection; err/skip/inprog = live events):")
    all_stages = sorted(set(stages) | set(proj_ok), key=lambda x: (args.stage != x, x))
    for st in all_stages:
        if args.stage and st != args.stage:
            continue
        s = stages.get(st, {})
        ok = proj_ok.get(st, 0)
        flag = "  <-- ERRORS" if s.get("error") else ("  <-- IN-PROGRESS" if s.get("in_progress") else "")
        print(f"  {st:14} ok={ok:4} err={s.get('error',0):3} skip={s.get('skip',0):4} "
              f"inprog={s.get('in_progress',0):3}{flag}")
    if stuck:
        print(f"\nSTUCK in-progress > {STUCK_MIN}min ({len(stuck)}):")
        for s in stuck[:15]:
            print(f"  {s['media_id']} [{s['stage']}] {s['age_min']}m")
    if failures:
        print(f"\nrecent failures ({len(failures)}):")
        for f in failures[:10]:
            print(f"  {f['at_utc'][:19]} {f['media_id']} [{f['stage']}] {f['error']}")
    if args.gaps and gaps:
        print(f"\nmedia missing critical stages {CRITICAL} ({len(gaps)}):")
        for m, miss in list(gaps.items())[:20]:
            print(f"  {m}: {miss}")
    print(f"\nsnapshot -> {HEALTH}")


if __name__ == "__main__":
    main()

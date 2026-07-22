#!/usr/bin/env python3
"""Reconcile pipeline status from canonical disk artifacts (Invariant 1).

Status must be DERIVED, never asserted by a stage independently of what it
produced. This eliminates two drift classes seen in the journal: phantom-ok (a
stage journalled ok but wrote no artifact) and under-counting (a stage wrote
artifacts but never journalled).

Phase 1 (D-Log-04 P1): status_projection is now DERIVED from journal.db's
`events` table (the ONE outcome surface, §3.1), not independently asserted from
disk — `erf.outcomes.rebuild_projection()` is the fold. `verify()`/no-flag below
compares that fold against disk (still the drift check this tool exists for);
`--reconcile` now just re-folds the projection (see `reconcile()` docstring for
why it can no longer hand-write rows from disk).

Modes:
  --verify           (default) print journal-vs-disk drift per stage; no writes
  --reconcile        re-fold status_projection from events (see reconcile() docstring)
  --purge-synthetic  delete m_unknown% rows from status_projection, events, events_trace
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("reconcile_status")

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"
DB = ROOT / "state" / "journal.db"

sys.path.insert(0, str(ROOT / "src"))
from erf import stages as _st  # noqa: E402  (registry — the ONE stage universe, D-Log-04 Phase 2)


def _ne(path: Path) -> bool:
    """Artifact exists and is non-empty."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _predicate_fn(stage: "_st.Stage"):
    """Build predicate(media_id) -> bool from the registry's artifact_predicates
    (primary + also_requires), ANDed together — reproduces the exact compound
    checks A/D0/DX used to hardcode inline (media_manifest+words, words.wx+vad,
    community1+community1_excl)."""
    preds = _st.artifact_predicates(stage)

    def check(m: str) -> bool:
        for kind, tmpl in preds:
            p = ROOT / tmpl.format(mid=m)
            ok = _ne(p) if kind == "ne" else p.exists()
            if not ok:
                return False
        return True
    return check


# stage -> predicate(media_id) -> True iff the canonical artifact is present on disk.
# Phase 2 (D-Log-04): derived from src/erf/stages.py — the ONE stage universe —
# instead of being an independently hardcoded literal (this WAS the second of the
# three drifting universes the redesign names). Index/process stages (C, C_vis
# materialise to LanceDB; RETRANSCRIBE overwrites A's words) have no per-media
# artifact_path in the registry and are intentionally excluded — left untouched
# by --reconcile, exactly as before. CLUSTERS is also excluded here even though
# it has a registry predicate — see stages.RECONCILE_ORDER's docstring for why:
# on-spine as of Phase 3, but the ~250 pre-Phase-3 identity_map files on disk
# have no matching events rows, so including it now would break the drift-308
# invariant this file exists to preserve; include after the W3a corpus rebuild
# re-runs CLUSTERS and records events for every identity_map file.
STAGE_ARTIFACTS: "OrderedDict[str, callable]" = OrderedDict(
    (name, _predicate_fn(_st.STAGES[name])) for name in _st.RECONCILE_ORDER
)


def _media_ids() -> list[str]:
    return sorted(p.stem for p in (CAT / "media_manifest").glob("m_*.json"))


def _current_fp(mid: str, tmpl: str) -> str | None:
    """Recompute the same content-fingerprint the `inputs=` write side uses
    (sha256(content)[:16] — `build_speech_events_v2._identity_map_fp` /
    `build_transcript._identity_map_fp`) for the upstream artifact `tmpl`
    resolves to for this media. None if the artifact is absent (matches the
    write side's convention of omitting the key entirely, not a placeholder)."""
    p = ROOT / tmpl.format(mid=mid)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _stale_media(conn: sqlite3.Connection, stage: str, candidates: set[str]) -> set[str]:
    """D-Log-20: the read side of the `inputs=` fingerprint groundwork MASTER
    §5.2 Phase 3 wrote (SEV2/T0 populate `inputs={"identity_map": <fp>}`) but
    nothing read (stages.py's own docstring: "no staleness-marking logic reads
    `inputs` yet"). A media is stale for `stage` if the upstream artifact whose
    fingerprint `stage` recorded when it last ran no longer matches that
    artifact's CURRENT content — the recorded outcome is internally consistent
    (journal agrees with disk, see `candidates`) but stale relative to an
    upstream change since (e.g. identity_map re-resolved after SEV2/T0 ran on
    the old version). Read-only detection, mirroring `verify()`'s existing
    degraded surfacing — does not write a `stale` event itself; the fix is to
    re-run the stage (same discipline as drift, MASTER §5.2)."""
    tmpls = _st.STAGES[stage].fingerprint_inputs
    if not tmpls or not candidates:
        return set()
    rows = conn.execute(
        "SELECT e.media_id, e.inputs FROM events e "
        "JOIN (SELECT media_id, MAX(id) AS mid FROM events WHERE stage=? GROUP BY media_id) latest "
        "ON e.media_id = latest.media_id AND e.id = latest.mid",
        (stage,),
    ).fetchall()
    recorded = {}
    for m, raw in rows:
        try:
            recorded[m] = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            recorded[m] = {}
    stale = set()
    for m in candidates:
        inputs = recorded.get(m, {})
        for tmpl in tmpls:
            key = Path(tmpl.format(mid=m)).parent.name
            if inputs.get(key) != _current_fp(m, tmpl):
                stale.add(m)
                break
    return stale


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.execute("PRAGMA journal_mode=TRUNCATE")
    return conn


def audit_outcomes() -> None:
    """Audit the explicit-outcome layer (Outcome Contract / MASTER §5.2.1).

    Checks the outcome layer BOTH directions (the auditor must be symmetric — a
    one-way check is a read-side blind spot, MASTER §5.2.2):
      SILENT  = artifact on disk but NO recorded outcome (stage produced silently).
      PHANTOM = outcome recorded ok/empty but NO artifact on disk (the catalog/status
                surface claims a deliverable that reconcile already purged from
                status_projection — the two surfaces have diverged; a stage reading
                oc.is_complete/status_of would get a false 'produced' signal).
    Also tallies recorded outcomes and media-level flags so coverage is visible.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from erf import outcomes as oc  # reads the per-media status files
    media = _media_ids()
    print(f"{'stage':<12}{'artifacts':>10}{'outcomes':>10}{'silent':>8}{'phantom':>9}")
    print("-" * 49)
    total_silent = total_phantom = 0
    phantom_by_stage: dict[str, int] = {}
    for stage, pred in STAGE_ARTIFACTS.items():
        arts = {m for m in media if pred(m)}
        rec = {m for m in media if oc.get(m, stage) is not None}
        silent = arts - rec
        # phantom: outcome asserts a deliverable (ok/empty) but no artifact on disk.
        phantom = {m for m in media
                   if (oc.get(m, stage) or {}).get("status") in ("ok", "empty") and m not in arts}
        total_silent += len(silent)
        total_phantom += len(phantom)
        if phantom:
            phantom_by_stage[stage] = len(phantom)
        flag = "  <-- SILENT" if silent else ("  <-- PHANTOM" if phantom else "")
        print(f"{stage:<12}{len(arts):>10}{len(rec):>10}{len(silent):>8}{len(phantom):>9}{flag}")
    print("-" * 49)
    print(f"total silent  (artifact without recorded outcome): {total_silent}")
    print(f"total phantom (ok/empty outcome without artifact): {total_phantom}")
    if phantom_by_stage:
        print(f"  phantom by stage: {phantom_by_stage}")
        print("  → catalog/status surface diverged from disk (stale ok/empty records).")
        print("    A full rebuild re-runs these stages and overwrites them; until then")
        print("    stages reading oc.is_complete on these (media,stage) get a false True.")
    # media-level flag coverage
    print("\nmedia-level flags:")
    for fl in ("has_speech", "has_faces", "has_events", "has_identity", "has_transcript"):
        n_true = len(oc.media_by_flag(fl, True))
        n_false = len(oc.media_by_flag(fl, False))
        print(f"  {fl:16} set={n_true + n_false:4}  (true={n_true} false={n_false})")
    # contradiction check: has_speech=False must not have chunks
    bad = [m for m in oc.media_by_flag("has_speech", False) if _ne(CAT / f"chunks/{m}.jsonl")]
    if bad:
        print(f"\n  CONTRADICTION: {len(bad)} media flagged has_speech=False but have a chunks file: {bad[:5]}")


def verify(conn: sqlite3.Connection) -> None:
    media = _media_ids()
    # status_projection marks ok=1 for ANY definite conclusion (ok/skip/empty —
    # erf.outcomes.COMPLETED), but skip concludes "no artifact expected", and
    # empty only implies an artifact for the explicit-empty spine layers.
    # Subtract those media via the recorded outcome, else every no_speech skip
    # counts as a phantom (T0 showed 53 such false phantoms, 2026-07-03).
    sys.path.insert(0, str(ROOT / "src"))
    from erf import outcomes as oc
    # Phase 1 (D-Log-04 P1): status_projection is DERIVED from `events` — fold it
    # fresh before comparing, so this never compares against a stale projection.
    oc.rebuild_projection(DB)
    # stages whose "empty" outcome still writes an explicit-empty artifact
    # (A writes a 1-byte words.jsonl placeholder for no-speech media)
    explicit_empty_stages = {"A", "SEV2", "SEV2_ADJ", "PSTREAM"}
    print(f"{'stage':<12}{'journal_ok':>11}{'disk':>7}{'agree':>7}{'phantom':>9}{'unrecorded':>12}{'degraded':>10}{'stale':>8}")
    print("-" * 76)
    drift = 0
    n_degraded = 0
    n_stale = 0
    for stage, pred in STAGE_ARTIFACTS.items():
        rows = conn.execute(
            "SELECT DISTINCT media_id FROM status_projection WHERE stage=? AND ok=1", (stage,)).fetchall()
        # D-Log-17: corpus-scoped stages (CLUSTERS, GATE) also write a synthetic
        # __corpus__ row; it is not a media and has no per-media artifact —
        # comparing it against the disk predicate manufactured a permanent
        # 1-phantom. Restrict the journal side to the real media universe.
        jset = {r[0] for r in rows} & set(media)
        no_artifact = {"skip"} if stage in explicit_empty_stages else {"skip", "empty"}
        jset = {m for m in jset if oc.status_of(m, stage) not in no_artifact}
        dset = {m for m in media if pred(m)}
        # D-Log-17: a media recorded `degraded` (Phase 3 semantic health) with
        # its artifact on disk is RECORDED, not drift — jset only holds ok=1 so
        # it fell out and was mislabeled "unrecorded" (m_a835b1b57cb6). Surface
        # it in its own column instead: still visibly red via pipeline_status's
        # critical_gaps, but not conflated with record/disk divergence.
        degraded = {m for m in dset - jset if oc.status_of(m, stage) == "degraded"}
        phantom = jset - dset
        unrecorded = dset - jset - degraded
        # D-Log-20: staleness is orthogonal to drift/degraded — only meaningful
        # where journal and disk already agree an artifact exists.
        stale = _stale_media(conn, stage, (jset & dset) | degraded)
        drift += len(phantom) + len(unrecorded)
        n_degraded += len(degraded)
        n_stale += len(stale)
        flags = []
        if phantom or unrecorded:
            flags.append("DRIFT")
        if degraded:
            flags.append("DEGRADED")
        if stale:
            flags.append("STALE")
        flag = "  <-- " + "/".join(flags) if flags else ""
        print(f"{stage:<12}{len(jset):>11}{len(dset):>7}{len(jset & dset):>7}"
              f"{len(phantom):>9}{len(unrecorded):>12}{len(degraded):>10}{len(stale):>8}{flag}")
    print("-" * 76)
    print(f"total drift (phantom + unrecorded across tracked stages): {drift}")
    if n_degraded:
        print(f"degraded (recorded, artifact present, semantic-health floor failed): {n_degraded}")
    if n_stale:
        print(f"stale (recorded+disk agree, but fingerprinted upstream input has since changed): {n_stale}")
    print(f"media universe: {len(media)}")


def reconcile(conn: sqlite3.Connection) -> None:
    """Phase 1 (D-Log-04 P1): status_projection is now DERIVED from `events` —
    this can no longer assert rows straight from disk (that write would just be
    clobbered by the next fold, and asserting ok=1 with no backing event is
    exactly the "asserted independently" root cause the redesign eliminates).
    `--reconcile` now folds the projection fresh from events. Where that still
    disagrees with disk (see `verify()`/no-flag output above), the fix is to
    RE-RUN the stage so it records a fresh event — not to hand-edit the
    projection. `conn` is accepted for CLI-signature compatibility but unused;
    rebuild_projection() opens its own connection."""
    sys.path.insert(0, str(ROOT / "src"))
    from erf import outcomes as oc
    n = oc.rebuild_projection(DB)
    print(f"reconciled: status_projection refolded from events -> {n} (media,stage) rows")
    print("  (disk-vs-events disagreement is no longer patchable here — re-run the "
          "stage to record a fresh event; see `--audit-outcomes` / no-flag drift above)")


def purge_synthetic(conn: sqlite3.Connection) -> None:
    sp = conn.execute("DELETE FROM status_projection WHERE media_id LIKE 'm_unknown%'").rowcount
    # Phase 1: the outcome-only `events` table + the new `events_trace` table both
    # need purging of synthetic pre-id rows. `events_legacy` (the frozen, verbatim
    # pre-migration snapshot) is deliberately NOT touched — it's an audit record,
    # not a live surface.
    ev = conn.execute("DELETE FROM events WHERE media_id LIKE 'm_unknown%'").rowcount
    tr = 0
    if _table_exists(conn, "events_trace"):
        tr = conn.execute("DELETE FROM events_trace WHERE media_id LIKE 'm_unknown%'").rowcount
    conn.commit()
    print(f"purged synthetic rows: status_projection={sp}, events={ev}, events_trace={tr}")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reconcile", action="store_true", help="re-fold status_projection from events (see reconcile())")
    ap.add_argument("--purge-synthetic", action="store_true", help="delete m_unknown%% rows")
    ap.add_argument("--audit-outcomes", action="store_true",
                    help="flag silent stages (artifact without explicit outcome) + flag coverage")
    args = ap.parse_args()

    if args.audit_outcomes:
        audit_outcomes()
        return

    conn = _connect()
    try:
        if args.reconcile:
            reconcile(conn)
        elif args.purge_synthetic:
            purge_synthetic(conn)
        else:
            verify(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

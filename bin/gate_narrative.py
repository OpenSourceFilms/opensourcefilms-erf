#!/usr/bin/env python3
"""G-NARRATIVE — §4 gate for the narrative short-form pipeline.

Evaluates all candidate narrative threads against two criteria:
  (i)  composite_score ≥ COMPOSITE_MIN  (settled: D-Narrative-E1-03)
  (ii) coherence sub-score ≥ COHERENCE_MIN  (calibrated: 0.60, see calibration note)

Also validates structural integrity of the beat_plan:
  - beat_plan exists in catalog/beat_plans/
  - all beats have chunk_id, media_id, in_ms, out_ms
  - broll.keyframe_id refs exist when enabled

Outputs:
  catalog/narrative_gate/gate_{run_id}.json  — per-thread PASS/FAIL with reasons
  catalog/narrative_gate/latest.json         — symlink to most recent gate run

Usage:
    PYTHONPATH=src venvs/embed/bin/python bin/gate_narrative.py
    PYTHONPATH=src venvs/embed/bin/python bin/gate_narrative.py --composite-min 0.50 --coherence-min 0.60
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from erf.atomicio import write_tmp_rename
from erf.contracts import validate

_CATALOG = _ROOT / "catalog"
_JOURNAL_DB = _ROOT / "state" / "journal.db"

# D-Narrative-E1-03: criterion (i) settled at composite ≥ 0.50.
# criterion (ii): coherence ≥ 0.60, calibrated against 20-thread corpus (min=0.573,
#   p50=0.700, max=0.959). Gap at 0.573→0.649 makes 0.60 a natural breakpoint.
_DEFAULT_COMPOSITE_MIN = 0.50
_DEFAULT_COHERENCE_MIN = 0.60

_CALIBRATION_NOTE = (
    "coherence_min=0.60 calibrated 2026-06-12 against 20-thread corpus "
    "(min=0.573, p50=0.700, max=0.959). Gap at 0.573→0.649 makes 0.60 a natural "
    "breakpoint; 19/20 threads pass. composite_min=0.50 settled in D-Narrative-E1-03 "
    "(all 20 top-N threads pass; threshold anchors the quality floor for §5 orchestration)."
)


def _load_index(catalog: Path) -> dict[str, str]:
    path = catalog / "beat_plans" / "_thread_index.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_beat_plan(bp_id: str, catalog: Path) -> dict | None:
    path = catalog / "beat_plans" / f"{bp_id}.jsonl"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _kf_exists(keyframe_id: str, catalog: Path, _cache: dict[str, set] = {}) -> bool:
    media_id = keyframe_id.split(":")[0]
    if media_id not in _cache:
        ids: set[str] = set()
        kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
        if kf_path.exists():
            for line in kf_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        ids.add(json.loads(line)["keyframe_id"])
                    except Exception:
                        pass
        _cache[media_id] = ids
    return keyframe_id in _cache[media_id]


def _check_beat_plan(bp: dict, catalog: Path) -> list[str]:
    """Return a list of structural failure reasons for this beat_plan, or [] if OK."""
    issues: list[str] = []
    beats = bp.get("beats", [])
    if not beats:
        issues.append("beat_plan has no beats")
        return issues

    missing_required = 0
    for beat in beats:
        if not beat.get("chunk_id") or not beat.get("media_id"):
            missing_required += 1
        if not isinstance(beat.get("in_ms"), int) or not isinstance(beat.get("out_ms"), int):
            missing_required += 1
    if missing_required:
        issues.append(f"{missing_required} beat(s) missing required fields (chunk_id/media_id/in_ms/out_ms)")

    bad_refs = []
    for beat in beats:
        broll = beat.get("broll")
        if isinstance(broll, dict) and broll.get("enabled") and broll.get("keyframe_id"):
            kid = broll["keyframe_id"]
            if not _kf_exists(kid, catalog):
                bad_refs.append(kid)
    if bad_refs:
        issues.append(f"{len(bad_refs)} broll keyframe_id(s) not in catalog: {bad_refs[:3]}")

    return issues


def gate_thread(
    thread_id: str,
    *,
    catalog: Path,
    composite_min: float = _DEFAULT_COMPOSITE_MIN,
    coherence_min: float = _DEFAULT_COHERENCE_MIN,
) -> dict:
    """Evaluate §4 gate for a single thread. Returns result dict (no file write).

    Used by the orchestrator to gate each thread inline without triggering a
    full-corpus gate artifact write.
    """
    thread_path = catalog / "narrative_threads" / f"{thread_id}.json"
    thread = json.loads(thread_path.read_text(encoding="utf-8"))
    scores = thread.get("scores", {})
    composite = scores.get("composite", 0.0)
    coherence = scores.get("coherence", 0.0)

    reasons: list[str] = []
    if composite < composite_min:
        reasons.append(f"composite {composite:.3f} < threshold {composite_min}")
    if coherence < coherence_min:
        reasons.append(f"coherence {coherence:.3f} < threshold {coherence_min}")

    bp_index = _load_index(catalog)
    bp_id = bp_index.get(thread_id)
    if not bp_id:
        reasons.append("no beat_plan in index")
    else:
        bp = _load_beat_plan(bp_id, catalog)
        if bp is None:
            reasons.append(f"beat_plan file missing: {bp_id}")
        else:
            reasons.extend(_check_beat_plan(bp, catalog))

    return {
        "thread_id": thread_id,
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "scores": {"composite": composite, "coherence": coherence},
        "bp_id": bp_id,
    }


def run_gate(
    *,
    catalog: Path,
    composite_min: float = _DEFAULT_COMPOSITE_MIN,
    coherence_min: float = _DEFAULT_COHERENCE_MIN,
) -> dict:
    threads_dir = catalog / "narrative_threads"
    thread_files = sorted(threads_dir.glob("thr_*.json"))
    if not thread_files:
        print("ERROR: no narrative_threads found", file=sys.stderr)
        sys.exit(1)

    bp_index = _load_index(catalog)

    results: list[dict] = []
    for tf in thread_files:
        try:
            thread = json.loads(tf.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: skipping corrupt thread file {tf}: {exc}", file=sys.stderr)
            continue

        thread_id = thread["thread_id"]
        scores = thread.get("scores", {})
        composite = scores.get("composite", 0.0)
        coherence = scores.get("coherence", 0.0)
        hook = scores.get("hook", 0.0)
        arc = scores.get("arc", 0.0)
        quote = scores.get("quote", 0.0)

        reasons: list[str] = []

        # Criterion (i): composite ≥ composite_min
        if composite < composite_min:
            reasons.append(f"composite {composite:.3f} < threshold {composite_min}")

        # Criterion (ii): coherence ≥ coherence_min
        if coherence < coherence_min:
            reasons.append(f"coherence {coherence:.3f} < threshold {coherence_min}")

        # Structural: beat_plan must exist
        bp_id = bp_index.get(thread_id)
        n_beats = None
        if not bp_id:
            reasons.append("no beat_plan in index")
        else:
            bp = _load_beat_plan(bp_id, catalog)
            if bp is None:
                reasons.append(f"beat_plan file missing: {bp_id}")
            else:
                n_beats = len(bp.get("beats", []))
                bp_issues = _check_beat_plan(bp, catalog)
                reasons.extend(bp_issues)

        verdict = "PASS" if not reasons else "FAIL"
        results.append({
            "thread_id": thread_id,
            "verdict": verdict,
            "reasons": reasons,
            "scores": {
                "composite": composite,
                "coherence": coherence,
                "hook": hook,
                "arc": arc,
                "quote": quote,
            },
            "bp_id": bp_id,
            "n_beats": n_beats,
        })

    shortlist = [r["thread_id"] for r in results if r["verdict"] == "PASS"]
    n_pass = len(shortlist)
    n_fail = len(results) - n_pass

    gate_id = f"gate_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    gate_record = {
        "schema_version": "1.0.0",
        "gate_id": gate_id,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "composite_min": composite_min,
            "coherence_min": coherence_min,
        },
        "results": results,
        "shortlist": shortlist,
        "summary": {
            "total": len(results),
            "pass": n_pass,
            "fail": n_fail,
            "calibration_notes": _CALIBRATION_NOTE,
        },
    }

    validate(gate_record, "narrative_gate")
    return gate_record


def _write_gate(gate_record: dict, catalog: Path) -> Path:
    out_dir = catalog / "narrative_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_id = gate_record["gate_id"]
    out_path = out_dir / f"{gate_id}.json"
    write_tmp_rename(out_path, json.dumps(gate_record, indent=2, ensure_ascii=False).encode("utf-8"))

    # Atomic latest pointer
    latest_path = out_dir / "latest.json"
    write_tmp_rename(latest_path, json.dumps({"gate_id": gate_id, "path": str(out_path)}).encode("utf-8"))

    return out_path


def _print_report(gate_record: dict) -> None:
    t = gate_record["thresholds"]
    s = gate_record["summary"]
    results = gate_record["results"]

    print(f"\n{'='*70}")
    print(f"NARRATIVE GATE — {gate_record['gate_id']}")
    print(f"thresholds: composite ≥ {t['composite_min']} | coherence ≥ {t['coherence_min']}")
    print(f"{'='*70}")
    print(f"\n{'verdict':<6} {'thread':<14} {'composite':>9} {'coherence':>9} {'beats':>5}  reasons")
    print(f"{'-'*80}")

    for r in sorted(results, key=lambda x: x["scores"]["composite"], reverse=True):
        vchar = "✓" if r["verdict"] == "PASS" else "✗"
        sc = r["scores"]
        reason_str = "; ".join(r["reasons"]) if r["reasons"] else "—"
        print(
            f"  {vchar}    {r['thread_id'][-12:]:<14} {sc['composite']:>9.3f} {sc['coherence']:>9.3f} "
            f"{str(r['n_beats'] or '—'):>5}  {reason_str[:50]}"
        )

    print(f"\nResult: {s['pass']}/{s['total']} PASS | {s['fail']}/{s['total']} FAIL")
    print(f"Shortlist ({s['pass']} threads):")
    for tid in gate_record["shortlist"]:
        print(f"  {tid}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Narrative §4 gate")
    parser.add_argument("--composite-min", type=float, default=_DEFAULT_COMPOSITE_MIN)
    parser.add_argument("--coherence-min", type=float, default=_DEFAULT_COHERENCE_MIN)
    parser.add_argument("--catalog", type=Path, default=_CATALOG)
    parser.add_argument("--dry-run", action="store_true", help="Don't write output")
    args = parser.parse_args()

    gate_record = run_gate(
        catalog=args.catalog,
        composite_min=args.composite_min,
        coherence_min=args.coherence_min,
    )

    _print_report(gate_record)

    if not args.dry_run:
        out_path = _write_gate(gate_record, args.catalog)
        print(f"Gate output written: {out_path}")
    else:
        print("(dry-run: no output written)")


if __name__ == "__main__":
    main()

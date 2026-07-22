#!/usr/bin/env python3
# Python 3.11 script bin/eval_attribution.py
"""Attribution evaluation gate for the ERF pipeline."""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
#  Path configuration – script is assumed to live in <ROOT>/bin/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DEFAULT = ROOT / "state/eval/attribution_golden.jsonl"
BASELINE_PATH = ROOT / "state/eval/baseline/ATTR.json"
RUNS_DIR = Path(os.environ.get("ERF_EVAL_RUNS_DIR") or (ROOT / "state/eval/runs"))  # D-Log-35 seam
LEAK_STRATIFICATION = ROOT / "state/eval/golden_leak_stratification.json"

# Phase 4 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §4/P5): gate-as-stage — the
# gate emits its own verdict into the same log as any other stage ("green" means
# meaning-intact, not merely file-exists). erf.outcomes is a HARD dependency here
# (no `if outcomes is not None` escape, matching CLUSTERS'/T0's/SEV2's Phase 3/4
# pattern) — this IS the correctness gate; it must never silently fail to log its
# own verdict.
sys.path.insert(0, str(ROOT / "src"))
from erf import outcomes
from erf.paths import catalog_root

CATALOG = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam); unset = ROOT/catalog

# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------
def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON payload atomically via a temporary file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _word_stream(word: Dict[str, Any]) -> str:
    return word.get("stream", "A")


def assertion_governs(word: Dict[str, Any], assertion: Dict[str, Any]) -> bool:
    """Check whether `assertion` governs the given transcript word."""
    if "stream" in assertion and _word_stream(word) != assertion["stream"]:
        return False
    if "speaker_label" in assertion and word.get("speaker_label") != assertion["speaker_label"]:
        return False
    return True


def assertion_passes(assertion: Dict[str, Any], word: Dict[str, Any]) -> bool:
    """Return True if `assertion` correctly identifies the word."""
    if "person_id" in assertion:
        return word.get("person_id") == assertion["person_id"]
    if "anonymous" in assertion:
        return word.get("person_id") is None
    # not_person_ids
    pid = word.get("person_id")
    return pid not in assertion["not_person_ids"]  # None passes


def load_transcript(media_id: str, cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Load (and cache) the transcript JSONL for `media_id`."""
    if media_id in cache:
        return cache[media_id]
    path = CATALOG / "asr" / f"{media_id}.transcript.jsonl"
    words: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words.append(json.loads(line))
    except FileNotFoundError:
        logging.warning("Transcript file missing for media %s; treating its windows as empty.", media_id)
        words = []  # will be cached as empty
    cache[media_id] = words
    return words


# ---------------------------------------------------------------------------
#  Main evaluation logic
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate speaker attribution against golden truth.")
    parser.add_argument("--golden", type=Path, default=GOLDEN_DEFAULT, help="Path to golden JSONL.")
    parser.add_argument("--include-unverified", action="store_true", help="Score unverified golden entries as well.")
    parser.add_argument("--pass-threshold", type=float, default=0.90, help="Window accuracy threshold to pass.")
    parser.add_argument("--set-baseline", action="store_true", help="Replace the baseline with the current run and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ------------------------------------------------------------------
    # Load golden entries
    # ------------------------------------------------------------------
    golden_path = args.golden.resolve()
    golden_entries: List[Dict[str, Any]] = []
    try:
        with open(golden_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not args.include_unverified and not entry.get("verified", False):
                    continue
                golden_entries.append(entry)
    except FileNotFoundError:
        logging.error("Golden file not found: %s", golden_path)
        sys.exit(2)

    if not golden_entries:
        logging.warning("No golden entries to evaluate.")
        return

    logging.info("Processing %d golden entries.", len(golden_entries))

    # ------------------------------------------------------------------
    # Per‑window processing
    # ------------------------------------------------------------------
    transcript_cache: Dict[str, List[Dict[str, Any]]] = {}
    missing_media_warned: set[str] = set()

    # Per‑stratum accumulators
    stratum_windows: Dict[str, int] = {}
    stratum_empty: Dict[str, int] = {}
    stratum_passed: Dict[str, int] = {}
    stratum_total_scored: Dict[str, int] = {}
    stratum_total_correct: Dict[str, int] = {}

    # Per‑window records for the run JSON
    window_records: List[Dict[str, Any]] = []

    for entry in golden_entries:
        gid = entry["golden_id"]
        media_id = entry["media_id"]
        stratum = entry["stratum"]
        start_ms = entry["start_ms"]
        end_ms = entry["end_ms"]
        assertions = entry["expected"]

        # Ensure once‑per‑media warning
        if media_id not in missing_media_warned and media_id not in transcript_cache:
            # load_transcript will issue its own warning on missing file;
            # avoid double‑warning by checking cache after load.
            pass
        words = load_transcript(media_id, transcript_cache)
        if not words and media_id not in missing_media_warned:
            missing_media_warned.add(media_id)

        # Words falling inside the window
        window_words = [w for w in words if w["start_ms"] < end_ms and w["end_ms"] > start_ms]

        # Score words
        n_scored = 0
        n_correct = 0
        failures: List[Dict[str, Any]] = []
        for w in window_words:
            governing = [a for a in assertions if assertion_governs(w, a)]
            if not governing:
                continue  # no assertion applies => word not scored
            n_scored += 1
            all_pass = all(assertion_passes(a, w) for a in governing)
            if all_pass:
                n_correct += 1
            else:
                # collect up to 3 failure samples for this window
                if len(failures) < 3:
                    # pick the first failing assertion
                    failing = next(a for a in governing if not assertion_passes(a, w))
                    failures.append({
                        "text": w["text"],
                        "got_person_id": w.get("person_id"),
                        "assertion": failing,
                    })

        # Window statistics
        accuracy = n_correct / n_scored if n_scored > 0 else None
        passed = n_scored > 0 and accuracy is not None and accuracy >= args.pass_threshold
        if n_scored == 0:
            # empty window – pass/fail flag is irrelevant, but we store False
            passed = False

        window_record = {
            "golden_id": gid,
            "stratum": stratum,
            "n_scored": n_scored,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "passed": passed,
            "failures": failures,
        }
        window_records.append(window_record)

        # Update stratum counters
        stratum_windows[stratum] = stratum_windows.get(stratum, 0) + 1
        if n_scored == 0:
            stratum_empty[stratum] = stratum_empty.get(stratum, 0) + 1
        else:
            stratum_total_scored[stratum] = stratum_total_scored.get(stratum, 0) + n_scored
            stratum_total_correct[stratum] = stratum_total_correct.get(stratum, 0) + n_correct
            if passed:
                stratum_passed[stratum] = stratum_passed.get(stratum, 0) + 1

    # ------------------------------------------------------------------
    # Compute metrics per stratum and overall
    # ------------------------------------------------------------------
    all_strata = sorted(stratum_windows.keys())  # order 'A'..'D'
    metrics: Dict[str, Dict[str, float]] = {}

    for stratum in all_strata:
        w = stratum_windows.get(stratum, 0)
        e = stratum_empty.get(stratum, 0)
        p = stratum_passed.get(stratum, 0)
        scored = stratum_total_scored.get(stratum, 0)
        correct = stratum_total_correct.get(stratum, 0)
        wa = correct / scored if scored > 0 else 1.0
        non_empty = w - e
        pr = p / non_empty if non_empty > 0 else 1.0
        metrics[stratum] = {
            "windows": w,
            "empty": e,
            "passed": p,
            "word_acc": wa,
            "pass_rate": pr,
        }

    # Overall (micro‑average across strata)
    total_w = sum(stratum_windows.values())
    total_e = sum(stratum_empty.values())
    total_p = sum(stratum_passed.values())
    total_score = sum(stratum_total_scored.values())
    total_correct = sum(stratum_total_correct.values())
    overall_wa = total_correct / total_score if total_score > 0 else 1.0
    non_empty_overall = total_w - total_e
    overall_pr = total_p / non_empty_overall if non_empty_overall > 0 else 1.0
    metrics["OVERALL"] = {
        "windows": total_w,
        "empty": total_e,
        "passed": total_p,
        "word_acc": overall_wa,
        "pass_rate": overall_pr,
    }

    # ------------------------------------------------------------------
    # Print compact table
    # ------------------------------------------------------------------
    print(f"{'stratum':<10}{'windows':>8}{'passed':>8}{'empty':>8}{'word_acc':>10}{'pass_rate':>10}")
    for stratum in all_strata + ["OVERALL"]:
        m = metrics[stratum]
        print(
            f"{stratum:<10}{m['windows']:>8}{m['passed']:>8}{m['empty']:>8}"
            f"{m['word_acc']:>10.3f}{m['pass_rate']:>10.3f}"
        )

    # ------------------------------------------------------------------
    # Additive golden-leak clean/contaminated breakdown (bin/eval_golden_leak.py).
    # Reporting only -- never affects the gate's pass/fail or exit code, and
    # the full-63 OVERALL metric above remains the regression guard. Skipped
    # silently if the sidecar hasn't been generated yet.
    # ------------------------------------------------------------------
    leak_metrics: Optional[Dict[str, Dict[str, float]]] = None
    if LEAK_STRATIFICATION.exists():
        try:
            leak_data = json.loads(LEAK_STRATIFICATION.read_text(encoding="utf-8"))
            leak_by_gid = {gid: w["status"] for gid, w in leak_data.get("windows", {}).items()}
        except (json.JSONDecodeError, KeyError):
            leak_by_gid = {}
        if leak_by_gid:
            group_scored: Dict[str, int] = {"clean": 0, "contaminated": 0}
            group_correct: Dict[str, int] = {"clean": 0, "contaminated": 0}
            group_windows: Dict[str, int] = {"clean": 0, "contaminated": 0}
            group_passed: Dict[str, int] = {"clean": 0, "contaminated": 0}
            group_empty: Dict[str, int] = {"clean": 0, "contaminated": 0}
            for wr in window_records:
                status = leak_by_gid.get(wr["golden_id"])
                if status not in ("clean", "contaminated"):
                    continue
                group_windows[status] += 1
                if wr["n_scored"] == 0:
                    group_empty[status] += 1
                else:
                    group_scored[status] += wr["n_scored"]
                    group_correct[status] += wr["n_correct"]
                    if wr["passed"]:
                        group_passed[status] += 1
            leak_metrics = {}
            for status in ("clean", "contaminated"):
                w, e = group_windows[status], group_empty[status]
                scored, correct = group_scored[status], group_correct[status]
                non_empty = w - e
                leak_metrics[status] = {
                    "windows": w, "empty": e, "passed": group_passed[status],
                    "word_acc": correct / scored if scored > 0 else 1.0,
                    "pass_rate": group_passed[status] / non_empty if non_empty > 0 else 1.0,
                }
            print(f"\n{'golden-leak':<14}{'windows':>8}{'passed':>8}{'empty':>8}{'word_acc':>10}{'pass_rate':>10}")
            for status in ("clean", "contaminated"):
                m = leak_metrics[status]
                print(
                    f"{status:<14}{m['windows']:>8}{m['passed']:>8}{m['empty']:>8}"
                    f"{m['word_acc']:>10.3f}{m['pass_rate']:>10.3f}"
                )

    # ------------------------------------------------------------------
    # Atomic run record
    # ------------------------------------------------------------------
    run_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_record = {
        "at_utc": run_utc,
        "golden_file": str(golden_path),
        "include_unverified": args.include_unverified,
        "pass_threshold": args.pass_threshold,
        "metrics": metrics,
        "leak_metrics": leak_metrics,
        "windows": window_records,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RUNS_DIR / f"attr_{run_utc}.json", run_record)

    # ------------------------------------------------------------------
    # Baseline operations
    # ------------------------------------------------------------------
    if args.set_baseline:
        baseline_data = {
            "at_utc": run_utc,
            "golden_file": str(golden_path),
            "n_windows": len(golden_entries),
            "metrics": metrics,
        }
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(BASELINE_PATH, baseline_data)
        print("Baseline set.")
        return  # exit 0 implicitly

    if not BASELINE_PATH.exists():
        print("NO BASELINE (run --set-baseline)")
        return

    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            baseline = json.load(f)
        base_metrics = baseline["metrics"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        logging.error("Cannot load baseline: %s", exc)
        sys.exit(2)

    # ------------------------------------------------------------------
    # Regression checks
    # ------------------------------------------------------------------
    regressions: List[str] = []

    # overall word accuracy drop > 0.01
    delta_wa = overall_wa - base_metrics["OVERALL"]["word_acc"]
    print(f"OVERALL word_acc: {base_metrics['OVERALL']['word_acc']:.3f} -> {overall_wa:.3f} delta {delta_wa:+.3f}")
    if delta_wa < -0.01:
        regressions.append(f"OVERALL word_acc {delta_wa:+.3f}")

    # overall pass rate drop > 0.02
    delta_pr = overall_pr - base_metrics["OVERALL"]["pass_rate"]
    print(f"OVERALL pass_rate: {base_metrics['OVERALL']['pass_rate']:.3f} -> {overall_pr:.3f} delta {delta_pr:+.3f}")
    if delta_pr < -0.02:
        regressions.append(f"OVERALL pass_rate {delta_pr:+.3f}")

    # per‑stratum word accuracy drop > 0.02 (strata present in both)
    for stratum in sorted(set(all_strata) & set(base_metrics.keys()) - {"OVERALL"}):
        base_wa = base_metrics[stratum]["word_acc"]
        curr_wa = metrics[stratum]["word_acc"]
        d = curr_wa - base_wa
        print(f"{stratum} word_acc: {base_wa:.3f} -> {curr_wa:.3f} delta {d:+.3f}")
        if d < -0.02:
            regressions.append(f"{stratum} word_acc {d:+.3f}")

    # Phase 4 (D-Log-04 §4): emit the gate's own verdict as a first-class
    # outcome — "green" now means meaning-intact (no attribution regression
    # vs the loaded baseline), not merely "the script didn't crash". Reuses
    # the regression check above rather than recomputing pass/fail.
    gate_status = "degraded" if regressions else "ok"
    gate_reason = "; ".join(regressions) if regressions else None
    outcomes.record(
        "__corpus__", "GATE", gate_status, reason=gate_reason,
        metrics={
            "word_acc": overall_wa,
            "window_pass": overall_pr,
            "n_windows": total_w,
            "per_stratum": {s: metrics[s] for s in all_strata},
        },
        producer="eval_attribution.py",
    )

    if regressions:
        for msg in regressions:
            print(f"REGRESSION: {msg}")
        sys.exit(1)
    else:
        print("PASS vs baseline")


if __name__ == "__main__":
    main()

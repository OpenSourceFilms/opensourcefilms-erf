"""Phase 4 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md P4 / §2 root cause #6):
per-media error isolation for RG (`bin/build_recall_gaps.py`) and RP
(`bin/recall_pass.py`), plus a smoke test for GATE-as-stage
(`bin/eval_attribution.py`).

Before this phase, `main()`'s per-media loop in both RG and RP had no
try/except: one media's exception killed the whole batch and recorded no
`error` outcome for the failing media or any media scheduled after it. These
tests monkeypatch the per-media worker function to raise for one synthetic
media_id and prove: (a) the batch does not abort, (b) an `error` outcome is
recorded for the failing media, (c) later media still get processed.

All tests use `monkeypatch` on the loaded module's `outcomes.record` (a spy,
never the real DB — same pattern as `tests/test_semantic_health.py`'s
`record_spy` fixture) and a `tmp_path` catalog — nothing here touches the
live `state/journal.db` or `catalog/outcomes/`.

bin/ is not a package (no __init__.py) — modules are loaded via
importlib.util.spec_from_file_location, matching the pattern
tests/test_stages_registry.py and tests/test_semantic_health.py already use.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rg = _load_bin_module("rg_isolation", "build_recall_gaps.py")
rp = _load_bin_module("rp_isolation", "recall_pass.py")
ea = _load_bin_module("ea_gate", "eval_attribution.py")


# ──────────────────────────────────────────────────────────────────────────
# RG — bin/build_recall_gaps.py main() per-media isolation
# ──────────────────────────────────────────────────────────────────────────

def _rg_spy(monkeypatch):
    calls = []

    def fake_record(media_id, stage, status, **kw):
        calls.append((media_id, stage, status, kw))

    monkeypatch.setattr(rg.outcomes, "record", fake_record)
    return calls


def test_rg_per_media_exception_isolated_and_recorded(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    asr_dir = catalog / "asr"
    asr_dir.mkdir(parents=True)
    # media discovery scans catalog/asr/*.words.jsonl
    (asr_dir / "m_bad.words.jsonl").write_text("", encoding="utf-8")
    (asr_dir / "m_good.words.jsonl").write_text("", encoding="utf-8")

    def fake_generate_gaps(media_id, catalog_path):
        if media_id == "m_bad":
            raise RuntimeError("synthetic per-media failure")
        return []

    monkeypatch.setattr(rg, "generate_gaps", fake_generate_gaps)
    calls = _rg_spy(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["build_recall_gaps.py", "--catalog", str(catalog)])

    rg.main()  # must NOT raise — the synthetic exception is isolated per-media

    by_mid = {c[0]: c for c in calls}
    assert by_mid["m_bad"][1] == "RG"
    assert by_mid["m_bad"][2] == "error"
    assert "synthetic per-media failure" in by_mid["m_bad"][3]["reason"]
    assert by_mid["m_bad"][3]["producer"] == "build_recall_gaps.py"

    # the batch continued: m_good still got its normal `ok` outcome + artifact
    assert by_mid["m_good"][2] == "ok"
    assert (catalog / "asr_recall_gaps" / "m_good.jsonl").exists()
    # m_bad's exception happened before the write — no artifact for it
    assert not (catalog / "asr_recall_gaps" / "m_bad.jsonl").exists()


def test_rg_missing_asr_dir_still_fails_the_whole_run(tmp_path, monkeypatch):
    """Setup/arg-parsing failures (not the per-media body) must still abort —
    the per-media try/except must not have been widened to swallow this."""
    catalog = tmp_path / "catalog_missing"
    monkeypatch.setattr(sys, "argv", ["build_recall_gaps.py", "--catalog", str(catalog)])
    with pytest.raises(SystemExit):
        rg.main()


# ──────────────────────────────────────────────────────────────────────────
# RP — bin/recall_pass.py main() per-media isolation
# ──────────────────────────────────────────────────────────────────────────

def _rp_spy(monkeypatch):
    calls = []

    def fake_record(media_id, stage, status, **kw):
        calls.append((media_id, stage, status, kw))

    monkeypatch.setattr(rp.outcomes, "record", fake_record)
    return calls


def test_rp_per_media_exception_isolated_and_recorded(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    gaps_dir = catalog / "asr_recall_gaps"
    gaps_dir.mkdir(parents=True)
    (gaps_dir / "m_bad.jsonl").write_text("", encoding="utf-8")
    (gaps_dir / "m_good.jsonl").write_text("", encoding="utf-8")
    bts_root = tmp_path / "bts_missing"  # deliberately absent -> skip index build

    def fake_process_media(media_id, catalog_path, bts_root_path, bts_index, min_priority):
        if media_id == "m_bad":
            raise RuntimeError("synthetic crash process_media itself didn't catch")
        return (1, 1, 0)  # (n_windows, n_recovered, n_dropped)

    monkeypatch.setattr(rp, "process_media", fake_process_media)
    calls = _rp_spy(monkeypatch)
    monkeypatch.setattr(sys, "argv",
                        ["recall_pass.py", "--catalog", str(catalog), "--bts-root", str(bts_root)])

    rp.main()  # must NOT raise

    by_mid = {c[0]: c for c in calls}
    assert by_mid["m_bad"][1] == "RP"
    assert by_mid["m_bad"][2] == "error"
    assert "synthetic crash" in by_mid["m_bad"][3]["reason"]
    assert by_mid["m_bad"][3]["producer"] == "recall_pass.py"
    # m_good never hit the except branch (process_media "succeeded" per the fake) —
    # its OWN outcomes.record calls (skip/ok/error) live inside the real
    # process_media, untouched here, so the spy legitimately sees nothing for it.
    assert "m_good" not in by_mid


def test_rp_process_media_signature_unchanged():
    """Ground-truth sanity: process_media's OWN outcomes.record calls on its
    documented skip/ok/error exit paths (lines ~137-181) are untouched by this
    phase — this phase only adds a loop-level except for an exception
    process_media itself doesn't catch."""
    import inspect
    sig = inspect.signature(rp.process_media)
    assert list(sig.parameters) == ["media_id", "catalog", "bts_root", "bts_index", "min_priority"]


# ──────────────────────────────────────────────────────────────────────────
# GATE — bin/eval_attribution.py records its own verdict (D-Log-04 §4 P5)
# ──────────────────────────────────────────────────────────────────────────

def _write_golden(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _write_baseline(path: Path, overall_word_acc: float, overall_pass_rate: float,
                    stratum_word_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "at_utc": "2026-01-01T00:00:00Z",
        "golden_file": "synthetic",
        "n_windows": 1,
        "metrics": {
            "A": {"windows": 1, "empty": 0, "passed": 1,
                  "word_acc": stratum_word_acc, "pass_rate": 1.0},
            "OVERALL": {"windows": 1, "empty": 0, "passed": 1,
                        "word_acc": overall_word_acc, "pass_rate": overall_pass_rate},
        },
    }), encoding="utf-8")


def _gate_env(tmp_path, monkeypatch):
    """Redirect every disk surface eval_attribution.py touches to tmp_path —
    never the real catalog/state/journal.db."""
    monkeypatch.setattr(ea, "ROOT", tmp_path)
    monkeypatch.setattr(ea, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(ea, "RUNS_DIR", tmp_path / "state" / "eval" / "runs")
    monkeypatch.setattr(ea, "BASELINE_PATH", tmp_path / "state" / "eval" / "baseline" / "ATTR.json")
    monkeypatch.setattr(ea, "LEAK_STRATIFICATION", tmp_path / "nonexistent_leak.json")
    calls = []

    def fake_record(media_id, stage, status, **kw):
        calls.append((media_id, stage, status, kw))

    monkeypatch.setattr(ea.outcomes, "record", fake_record)
    return calls


def _make_transcript_and_golden(tmp_path, person_id_written):
    transcript_dir = tmp_path / "catalog" / "asr"
    transcript_dir.mkdir(parents=True)
    word = {"word_index": 0, "text": "hi", "start_ms": 0, "end_ms": 500,
            "speaker_label": "SPEAKER_00", "person_id": person_id_written}
    (transcript_dir / "m_gate.transcript.jsonl").write_text(
        json.dumps(word) + "\n", encoding="utf-8")

    golden_path = tmp_path / "golden.jsonl"
    _write_golden(golden_path, [{
        "golden_id": "g1", "media_id": "m_gate", "stratum": "A",
        "start_ms": 0, "end_ms": 500, "verified": True,
        "expected": [{"person_id": "p1"}],
    }])
    return golden_path


def test_gate_records_ok_when_no_regression(tmp_path, monkeypatch):
    calls = _gate_env(tmp_path, monkeypatch)
    golden_path = _make_transcript_and_golden(tmp_path, person_id_written="p1")  # correct -> word_acc 1.0
    _write_baseline(ea.BASELINE_PATH, overall_word_acc=0.5, overall_pass_rate=0.5, stratum_word_acc=0.5)
    monkeypatch.setattr(sys, "argv", ["eval_attribution.py", "--golden", str(golden_path)])

    ea.main()  # PASS path: main() returns normally, no sys.exit call at all

    assert len(calls) == 1
    media_id, stage, status, kw = calls[0]
    assert media_id == "__corpus__"
    assert stage == "GATE"
    assert status == "ok"
    assert kw["reason"] is None
    assert kw["metrics"]["word_acc"] == pytest.approx(1.0)
    assert kw["metrics"]["window_pass"] == pytest.approx(1.0)
    assert kw["producer"] == "eval_attribution.py"


def test_gate_records_degraded_on_regression(tmp_path, monkeypatch):
    calls = _gate_env(tmp_path, monkeypatch)
    golden_path = _make_transcript_and_golden(tmp_path, person_id_written="WRONG")  # -> word_acc 0.0
    _write_baseline(ea.BASELINE_PATH, overall_word_acc=0.954, overall_pass_rate=0.754, stratum_word_acc=0.943)
    monkeypatch.setattr(sys, "argv", ["eval_attribution.py", "--golden", str(golden_path)])

    with pytest.raises(SystemExit) as exc:
        ea.main()
    assert exc.value.code == 1  # gate exits 1 on regression

    assert len(calls) == 1
    media_id, stage, status, kw = calls[0]
    assert media_id == "__corpus__"
    assert stage == "GATE"
    assert status == "degraded"
    assert kw["reason"]  # required + must cite the regressed metric
    assert "word_acc" in kw["reason"]
    assert kw["metrics"]["word_acc"] == pytest.approx(0.0)
    assert kw["metrics"]["n_windows"] == 1
    assert "A" in kw["metrics"]["per_stratum"]
    assert kw["producer"] == "eval_attribution.py"


# ──────────────────────────────────────────────────────────────────────────
# Double-failure guard (W2 checkpoint review finding, D-Log-12): the recovery
# record() inside the per-media except handler can ITSELF raise (P4 — no
# silent swallow, e.g. journal.db lock contention). That secondary failure
# must not kill the batch; it surfaces as stderr + a non-zero exit at the end.
# ──────────────────────────────────────────────────────────────────────────

def test_rg_record_failure_inside_handler_does_not_kill_batch(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    asr_dir = catalog / "asr"
    asr_dir.mkdir(parents=True)
    (asr_dir / "m_bad.words.jsonl").write_text("", encoding="utf-8")
    (asr_dir / "m_good.words.jsonl").write_text("", encoding="utf-8")

    def fake_generate_gaps(media_id, catalog_path):
        if media_id == "m_bad":
            raise RuntimeError("primary failure")
        return []

    calls = []

    class Sqlite3LikeError(Exception):
        pass

    def failing_record(media_id, stage, status, **kw):
        if media_id == "m_bad" and status == "error":
            raise Sqlite3LikeError("journal locked")  # secondary failure
        calls.append((media_id, stage, status))

    monkeypatch.setattr(rg, "generate_gaps", fake_generate_gaps)
    monkeypatch.setattr(rg.outcomes, "record", failing_record)
    monkeypatch.setattr(sys, "argv", ["build_recall_gaps.py", "--catalog", str(catalog)])

    # The batch must complete (m_good processed) AND the run must exit non-zero
    # (an unloggable outcome is a failed run, P4).
    with pytest.raises(SystemExit) as exc_info:
        rg.main()
    assert exc_info.value.code == 1
    assert ("m_good", "RG", "ok") in calls  # batch continued past the double failure
    assert (catalog / "asr_recall_gaps" / "m_good.jsonl").exists()

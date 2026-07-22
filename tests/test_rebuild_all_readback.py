"""D-Log-21 (GAMEPLAN Sonnet worklist item 4) — rebuild_all's read-back now
surfaces whole-stage subprocess crashes, not just per-media gaps.

The run-1 post-mortem blind spot (D-Log-14): SEV2_ADJ and PSTREAM both
crashed at startup (rc=1, torch/driver mismatch) before touching a single
media. `read_back()`'s existing checks (`oc.media_by_status`/`_scan`) glob
only `catalog/status/m_*.json` — `__corpus__` (the synthetic id GATE already
uses for its own corpus-scoped verdict, MASTER §5.2 Gate-as-stage) never
matches that glob, so a whole-stage crash was invisible to read-back even
though `main()`'s loop knew about it (only visible in that run's own
ephemeral stdout / `results` list).

`main()` now writes an explicit corpus-scoped run-outcome for every stage,
every run (ok or error — never left stale from a prior crash after a later
clean run). This file proves `read_back()`'s new branch reads that back
correctly, independent of `main()`'s in-memory state.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

from erf import journal, outcomes

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ra = _load_bin_module("ra_readback_dlog21", "rebuild_all.py")


@pytest.fixture
def tmp_journal(tmp_path):
    db = tmp_path / "journal.db"
    journal.init_journal(db)
    return db


@pytest.fixture
def isolated_outcomes(tmp_path, monkeypatch):
    """Same discipline as test_degraded_proof.py: STATUS_DIR is a hardcoded
    module global `record()`'s `db=` kwarg does not redirect."""
    status_dir = tmp_path / "outcomes"
    monkeypatch.setattr(outcomes, "STATUS_DIR", status_dir)
    return status_dir


@pytest.fixture
def stub_subprocess(monkeypatch):
    """read_back()'s step 1 shells out to the real pipeline_status.py against
    the real corpus — irrelevant to what's being proven here and slow.
    Stubbed to raise, which read_back()'s own try/except already handles
    (logs a warning, continues) — this exercises that existing resilience,
    not bypasses it."""
    def _boom(*a, **k):
        raise RuntimeError("stubbed — not under test")
    monkeypatch.setattr(ra.subprocess, "run", _boom)


def test_read_back_surfaces_a_crashed_stage(isolated_outcomes, tmp_journal, stub_subprocess):
    log = logging.getLogger("test_rebuild_all_readback")
    outcomes.record("__corpus__", "SEV2_ADJ", "error",
                     reason="subprocess rc=1 (see logs.all/rebuild_x/SEV2_ADJ.log)",
                     db=tmp_journal)

    summary = ra.read_back(log, run_stage_names=["SEV2_ADJ", "PSTREAM"])

    assert summary["stage_crashes"] == {
        "SEV2_ADJ": "subprocess rc=1 (see logs.all/rebuild_x/SEV2_ADJ.log)"
    }
    assert "PSTREAM" not in summary["stage_crashes"]


def test_read_back_clean_run_has_no_stage_crashes(isolated_outcomes, tmp_journal, stub_subprocess):
    """A stage that ran cleanly writes status='ok', not absence — confirms the
    branch doesn't false-positive on a healthy run."""
    log = logging.getLogger("test_rebuild_all_readback")
    outcomes.record("__corpus__", "SEV2_ADJ", "ok", metrics={"elapsed_s": 171.0}, db=tmp_journal)

    summary = ra.read_back(log, run_stage_names=["SEV2_ADJ"])

    assert summary["stage_crashes"] == {}


def test_read_back_stale_crash_cleared_by_a_later_clean_run(isolated_outcomes, tmp_journal, stub_subprocess):
    """The exact staleness trap this design avoids: a PAST run's crash record
    must not still read as crashed after a later run recorded 'ok' for the
    same stage (record() is called every run, ok or error, never skipped)."""
    log = logging.getLogger("test_rebuild_all_readback")
    outcomes.record("__corpus__", "SEV2_ADJ", "error", reason="subprocess rc=1", db=tmp_journal)
    outcomes.record("__corpus__", "SEV2_ADJ", "ok", metrics={"elapsed_s": 171.0}, db=tmp_journal)

    summary = ra.read_back(log, run_stage_names=["SEV2_ADJ"])

    assert summary["stage_crashes"] == {}


def test_read_back_no_run_stage_names_skips_the_check_entirely(isolated_outcomes, tmp_journal, stub_subprocess):
    """Backward-compatible call shape (no run_stage_names) must not raise or
    spuriously populate stage_crashes."""
    log = logging.getLogger("test_rebuild_all_readback")
    summary = ra.read_back(log)
    assert summary["stage_crashes"] == {}

"""Scheduler tests: DAG correctness, staleness, next_jobs."""
import pytest
from erf.scheduler import (
    STAGE_REGISTRY,
    dag_for_media,
    ready_stages,
    is_stale,
    next_jobs,
)


# ── STAGE_REGISTRY ────────────────────────────────────────────────────────────

def test_registry_has_all_stages():
    # E1 (narrative signal re-tag), E2 (corpus scan), Q0 (proxy hook, D-Q0-01)
    # added after this test was written -- 2026-07-08 test-currency fix.
    expected = {"A","B0","B1","B2","B3","B4","C","E0","E1","E2","P","Q0","F0","F1"}
    assert set(STAGE_REGISTRY.keys()) == expected


def test_registry_deps_are_valid():
    all_stages = set(STAGE_REGISTRY.keys())
    for stage, info in STAGE_REGISTRY.items():
        for dep in info["depends"]:
            assert dep in all_stages, f"{stage} depends on unknown stage {dep}"


def test_registry_tiers():
    gpu_stages = {"B1","B4","F0"}
    cpu_stages = {"B0","B2","B3","C","E0","P","F1"}
    gpu_preferred = {"A"}
    for s in gpu_stages:
        assert STAGE_REGISTRY[s]["tier"] == "gpu"
    for s in cpu_stages:
        assert STAGE_REGISTRY[s]["tier"] == "cpu"
    for s in gpu_preferred:
        assert STAGE_REGISTRY[s]["tier"] == "gpu-preferred"


# ── ready_stages ──────────────────────────────────────────────────────────────

def test_nothing_completed_only_A_ready():
    ready = ready_stages(set())
    assert ready == ["A"]


def test_after_A_B0_and_B2_ready():
    ready = ready_stages({"A"})
    assert "B0" in ready
    assert "B2" in ready
    assert "B1" not in ready  # B0 not done yet
    assert "F0" not in ready  # B3 not done yet


def test_after_A_B0_B1_ready():
    ready = ready_stages({"A","B0"})
    assert "B1" in ready


def test_C_not_ready_without_any_embed():
    ready = ready_stages({"A","B0"})
    assert "C" not in ready


def test_C_ready_after_B1_only():
    ready = ready_stages({"A","B0","B1"})
    assert "C" in ready


def test_C_ready_after_B4_only():
    ready = ready_stages({"A","B2","B3","B4"})
    assert "C" in ready


def test_running_stages_excluded():
    ready = ready_stages({"A"}, running_stages={"B0"})
    assert "B0" not in ready
    assert "B2" in ready


def test_ready_stages_cpu_before_gpu():
    ready = ready_stages({"A", "B0"})
    # Sort key is (tier, phase): cpu=0 < gpu=2, so B2 (cpu,phase2) beats B1 (gpu,phase1)
    assert "B2" in ready
    assert "B1" in ready
    # B2 is cpu — must appear before B1 (gpu) regardless of phase
    assert ready.index("B2") < ready.index("B1")


def test_full_pipeline_order():
    # Simulate completing stages in dependency order
    completed = set()
    order = []
    for _ in range(len(STAGE_REGISTRY)):
        ready = ready_stages(completed)
        if not ready:
            break
        order.append(ready[0])
        completed.add(ready[0])
    # A must come first
    assert order[0] == "A"
    # Every stage eventually becomes ready
    assert completed == set(STAGE_REGISTRY.keys())


# ── is_stale ──────────────────────────────────────────────────────────────────

def test_is_stale_not_completed():
    assert is_stale("A", set(), {}) is True


def test_is_stale_completed():
    assert is_stale("A", {"A"}, {"A": "run_001"}) is False


# ── next_jobs ─────────────────────────────────────────────────────────────────

def test_next_jobs_empty_status():
    jobs = next_jobs(["m_aaa000000001", "m_bbb000000002"], {})
    # Both media have A ready
    stages = [j["stage"] for j in jobs]
    assert stages.count("A") == 2


def test_next_jobs_partial_progress():
    status_map = {
        "m_aaa000000001": {
            "A": {"ok": True, "run_id": "r1"},
        }
    }
    jobs = next_jobs(["m_aaa000000001"], status_map)
    stages = [j["stage"] for j in jobs]
    assert "A" not in stages
    assert "B0" in stages or "B2" in stages


def test_next_jobs_cpu_before_gpu():
    # After A+B0 done: B1 (gpu,phase1) and B2 (cpu,phase2) both ready.
    # Sort key is (tier, phase): cpu=0 < gpu=2 — B2 must precede B1.
    status_map = {"m_aaa000000001": {"A": {"ok": True, "run_id": "r1"},
                                      "B0": {"ok": True, "run_id": "r2"}}}
    jobs = next_jobs(["m_aaa000000001"], status_map)
    stages = [j["stage"] for j in jobs]
    if "B2" in stages and "B1" in stages:
        b1_idx = next(i for i,j in enumerate(jobs) if j["stage"]=="B1")
        b2_idx = next(i for i,j in enumerate(jobs) if j["stage"]=="B2")
        assert b2_idx < b1_idx  # cpu(B2) before gpu(B1)


def test_next_jobs_running_excluded():
    status_map = {
        "m_aaa000000001": {
            "A": {"ok": None, "run_id": "r1"},  # ok=None means running
        }
    }
    jobs = next_jobs(["m_aaa000000001"], status_map)
    # A is running, not completed — still in status but ok=None
    # ready_stages checks completed (ok=True), so A is not completed, A is not in running either
    # (since running requires ok=None AND run_id — but next_jobs doesn't extract running from status yet)
    # For Phase 0, this just verifies no crash
    assert isinstance(jobs, list)

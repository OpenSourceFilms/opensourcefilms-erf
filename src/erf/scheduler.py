"""DAG-aware stage scheduler for ERF.

Computes which stages are ready to run given the current completion state.
Does not touch any DB — receives pre-queried data, returns job descriptors.

Stage DAG from MASTER.md §5:
  A:  depends=[]          tier=gpu-preferred  phase=1
  B0: depends=[A]         tier=cpu            phase=1
  B1: depends=[B0]        tier=gpu            phase=1
  B2: depends=[A]         tier=cpu            phase=2
  B3: depends=[B2]        tier=cpu            phase=2
  B4: depends=[B3]        tier=gpu            phase=2
  C:  depends=[B1, B4]    tier=cpu            phase=1  (materialises whatever is ready)
  E0: depends=[B0, B2]    tier=cpu            phase=2
  P:  depends=[B3]        tier=cpu            phase=2  (produces proxy)
  Q0: depends=[P]         tier=cpu            phase=2  (requires proxy — see D-Q0-01)
  F0: depends=[A, B3]     tier=gpu            phase=3  (requires proxy — see D-Q0-01)
  F1: depends=[F0]        tier=cpu            phase=3

consumes_hard: artifact types that must physically exist before the stage runs,
beyond the stage-ok dependency. Currently informational; not enforced by the
scheduler — callers are responsible for passing correct paths. See D-Q0-01.
"""
from __future__ import annotations

from typing import Any

from . import stages as _stages_registry

# ──────────────────────────────────────────────────────────────────────────────
# Stage registry
# ──────────────────────────────────────────────────────────────────────────────
#
# Phase 2 (D-Log-04 §4): derived from src/erf/stages.py — the single stage
# universe — instead of being an independently hardcoded literal. This is the
# "scheduled" subset (14 stages: exactly what scheduler.py's DAG functions
# below actually schedule); Stage.scheduled marks membership in stages.py.
# Only `depends`/`tier`/`phase` (and `corpus_scoped` for E2) are reproduced —
# the only fields _deps_met/ready_stages/dag_for_media/next_jobs below read.
# The old literal also carried informational-only `consumes_hard`/
# `consumes_soft`/`produces`/`idempotent` keys that no code in this file (or
# anywhere else — checked repo-wide) ever reads; dropped as dead weight, not
# behavior.

STAGE_REGISTRY: dict[str, dict[str, Any]] = {
    name: {
        "depends": list(s.depends),
        "tier": s.tier,
        "phase": s.phase,
        **({"corpus_scoped": True} if s.scope == "corpus" else {}),
    }
    for name, s in _stages_registry.STAGES.items()
    if s.scheduled
}

# C can materialise from either text or visual embeddings alone.
# Track which partial combinations allow C to proceed.
_C_PARTIAL_DEPS: list[set[str]] = [{"B1"}, {"B4"}, {"B1", "B4"}]

_TIER_ORDER = {"cpu": 0, "gpu-preferred": 1, "gpu": 2}


# ──────────────────────────────────────────────────────────────────────────────
# DAG helpers
# ──────────────────────────────────────────────────────────────────────────────

def _deps_met(stage: str, completed: set[str]) -> bool:
    """True if all dependencies for stage are in completed (ok=True)."""
    deps = STAGE_REGISTRY[stage]["depends"]
    if stage == "C":
        return any(partial.issubset(completed) for partial in _C_PARTIAL_DEPS)
    return all(d in completed for d in deps)


def dag_for_media(completed_stages: set[str]) -> dict[str, list[str]]:
    """Return {stage: [deps]} for all stages whose deps are met or already complete.

    Includes both pending and completed stages to give the full reachable subgraph.
    """
    return {
        stage: info["depends"]
        for stage, info in STAGE_REGISTRY.items()
        if _deps_met(stage, completed_stages) or stage in completed_stages
    }


def ready_stages(
    completed_stages: set[str],
    running_stages: set[str] | None = None,
) -> list[str]:
    """Return stages ready to run: deps met, not yet done, not currently running.

    Ordered: cpu before gpu, then by phase ascending.
    """
    running = running_stages or set()
    candidates = [
        stage for stage in STAGE_REGISTRY
        if _deps_met(stage, completed_stages)
        and stage not in completed_stages
        and stage not in running
    ]
    return sorted(
        candidates,
        key=lambda s: (_TIER_ORDER.get(STAGE_REGISTRY[s]["tier"], 99), STAGE_REGISTRY[s]["phase"]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Staleness
# ──────────────────────────────────────────────────────────────────────────────

def is_stale(
    stage: str,
    completed_stages: set[str],
    lineage: dict[str, str],  # {stage: run_id}
) -> bool:
    """True if stage needs reprocessing.

    Phase 0 simplified: a stage is stale if it is not in completed_stages.
    Full lineage-based staleness (comparing run_ids of deps vs what this stage
    was built from) is implemented in Phase 1+ when lineage tracking is wired up.
    """
    return stage not in completed_stages


# ──────────────────────────────────────────────────────────────────────────────
# Multi-media job planner
# ──────────────────────────────────────────────────────────────────────────────

def next_jobs(
    media_ids: list[str],
    status_map: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return ready jobs across all media, deduplicated, sorted cpu-first then phase.

    status_map: {media_id: {stage: {ok, run_id, at_utc, ...}}}
    Returns: [{media_id, stage, tier, phase}]
    """
    jobs: list[dict[str, Any]] = []
    for mid in media_ids:
        stage_status = status_map.get(mid, {})
        completed = {s for s, v in stage_status.items() if v.get("ok")}
        running = {s for s, v in stage_status.items() if v.get("ok") is None and v.get("run_id")}
        for stage in ready_stages(completed, running):
            info = STAGE_REGISTRY[stage]
            jobs.append({
                "media_id": mid,
                "stage": stage,
                "tier": info["tier"],
                "phase": info["phase"],
            })

    # Sort: cpu < gpu-preferred < gpu, then phase, then media_id for stability
    jobs.sort(key=lambda j: (_TIER_ORDER.get(j["tier"], 99), j["phase"], j["media_id"]))
    return jobs

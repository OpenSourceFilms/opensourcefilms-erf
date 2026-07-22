"""Phase 2 (D-Log-04, docs/OBSERVABILITY_REDESIGN.md §4): the registry is now
the ONLY stage universe. Phase 0's three hardcoded literals this file used to
cross-check (scheduler.STAGE_REGISTRY, reconcile_status.STAGE_ARTIFACTS,
pipeline_status.CRITICAL/SPEECH_SPINE) no longer exist as independent
literals — scheduler.py, reconcile_status.py, and bin/timeline_audit.py now
all DERIVE their stage universes from src/erf/stages.py. This file asserts
the new invariant: every consumer's derived universe still equals what the
registry says, and no file outside stages.py hardcodes a stage-universe
literal any more (the anti-drift guarantee, made structurally checkable).

bin/ is not a package (no __init__.py, no pytest sys.path entry) — consumer
modules are loaded via importlib.util.spec_from_file_location, matching the
pattern bin/timeline_audit.py itself used to import reconcile_status.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from erf import stages as st
from erf.scheduler import STAGE_REGISTRY

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


reconcile_status = _load_bin_module("reconcile_status", "reconcile_status.py")
pipeline_status = _load_bin_module("pipeline_status", "pipeline_status.py")
timeline_audit = _load_bin_module("timeline_audit", "timeline_audit.py")

CRITICAL = pipeline_status.CRITICAL
SPEECH_SPINE = pipeline_status.SPEECH_SPINE


# ── registry sanity (unchanged from Phase 0) ────────────────────────────────

def test_registry_keys_unique():
    keys = list(st.STAGES.keys())
    assert len(keys) == len(set(keys))


def test_every_stage_with_predicate_kind_has_artifact_path():
    for name, s in st.STAGES.items():
        if s.predicate_kind is not None:
            assert s.artifact_path is not None, f"{name}: predicate_kind set but artifact_path is None"
        if s.artifact_path is not None:
            assert s.predicate_kind in ("ne", "exists"), f"{name}: artifact_path set but bad predicate_kind"


def test_retranscribe_excluded():
    assert "RETRANSCRIBE" not in st.STAGES


def test_build_identity_clusters_present():
    assert "CLUSTERS" in st.STAGES
    assert st.STAGES["CLUSTERS"].artifact_path is not None


# ── also_requires / artifact_predicates (Phase 2, task item 1) ─────────────

def test_also_requires_reproduces_compound_predicates():
    # A/D0/DX had compound `X and Y` predicates in the pre-Phase-2 reconcile
    # lambdas; also_requires must reproduce the second branch exactly.
    assert st.artifact_predicates(st.STAGES["A"]) == (
        ("ne", "catalog/asr/{mid}.words.jsonl"),
        ("ne", "catalog/media_manifest/{mid}.json"),
    )
    assert st.artifact_predicates(st.STAGES["D0"]) == (
        ("ne", "catalog/asr/{mid}.words.wx.jsonl"),
        ("ne", "catalog/asr/{mid}.vad.jsonl"),
    )
    assert st.artifact_predicates(st.STAGES["DX"]) == (
        ("exists", "catalog/diar_hypotheses/{mid}/community1.jsonl"),
        ("exists", "catalog/diar_hypotheses/{mid}/community1_excl.jsonl"),
    )


def test_artifact_predicates_empty_for_no_predicate_stages():
    assert st.artifact_predicates(st.STAGES["C"]) == ()
    assert st.artifact_predicates(st.STAGES["E2"]) == ()


def test_reconcile_predicate_reproduces_compound_and(tmp_path, monkeypatch):
    """Integration proof (not just data-shape): reconcile_status.STAGE_ARTIFACTS
    is built generically from artifact_predicates(); this exercises the actual
    callable for stage A end-to-end against a fake filesystem, proving the
    generic builder reproduces the original hand-written `_ne(x) and _ne(y)`
    AND semantics — not just that the (kind, path) tuples look right."""
    monkeypatch.setattr(reconcile_status, "ROOT", tmp_path)
    (tmp_path / "catalog" / "asr").mkdir(parents=True)
    (tmp_path / "catalog" / "media_manifest").mkdir(parents=True)
    words = tmp_path / "catalog" / "asr" / "m_test.words.jsonl"
    manifest = tmp_path / "catalog" / "media_manifest" / "m_test.json"
    pred = reconcile_status.STAGE_ARTIFACTS["A"]

    assert pred("m_test") is False           # neither branch
    words.write_text("x")
    assert pred("m_test") is False            # one branch only -- AND must fail
    manifest.write_text("x")
    assert pred("m_test") is True             # both branches -- AND succeeds


# ── scheduler.STAGE_REGISTRY (Phase 2: derived, not a literal) ─────────────

def test_scheduler_registry_matches_scheduled_subset():
    assert set(STAGE_REGISTRY.keys()) == set(st.scheduled_stages().keys())
    assert len(STAGE_REGISTRY) == 14


def test_scheduler_fields_match():
    for stage, info in STAGE_REGISTRY.items():
        s = st.STAGES[stage]
        assert info["depends"] == list(s.depends), f"{stage}: depends mismatch"
        assert info["tier"] == s.tier, f"{stage}: tier mismatch"
        assert info["phase"] == s.phase, f"{stage}: phase mismatch"
        assert info.get("corpus_scoped", False) == (s.scope == "corpus"), f"{stage}: corpus_scoped mismatch"


# ── reconcile_status.STAGE_ARTIFACTS (Phase 2: derived, not a literal) ─────

def test_reconcile_stage_artifacts_matches_reconcile_order():
    assert list(reconcile_status.STAGE_ARTIFACTS.keys()) == list(st.RECONCILE_ORDER)
    assert len(reconcile_status.STAGE_ARTIFACTS) == 30


def test_clusters_included_in_reconcile():
    # D-Log-15 (2026-07-12): CLUSTERS was off-spine through W2/early W3a (its
    # ~250 pre-existing identity_map files predated the outcomes.record() write
    # path, so including it would have inflated "unrecorded" drift and broken
    # the drift-308 invariant). The W3a run 3 corpus rebuild re-ran CLUSTERS,
    # every identity_map file now has a matching events row, and it's on-spine.
    assert "CLUSTERS" in st.RECONCILE_ORDER
    assert "CLUSTERS" in reconcile_status.STAGE_ARTIFACTS


def test_reconcile_index_process_stages_excluded():
    # C (LanceDB materialise) and E2 (corpus scan) have no per-media artifact.
    assert "C" not in reconcile_status.STAGE_ARTIFACTS
    assert "E2" not in reconcile_status.STAGE_ARTIFACTS


# ── pipeline_status.CRITICAL / SPEECH_SPINE (already registry-sourced) ─────

def test_critical_set_matches():
    registry_critical = {k for k, v in st.STAGES.items() if v.critical}
    assert registry_critical == set(CRITICAL)


def test_speech_spine_set_matches():
    registry_spine = {k for k, v in st.STAGES.items() if v.speech_spine}
    assert registry_spine == set(SPEECH_SPINE)


def test_speech_spine_is_subset_of_critical():
    assert set(SPEECH_SPINE) <= set(CRITICAL)


# ── timeline_audit.py (Phase 2 / D-Log-06) ──────────────────────────────────

# Pre-Phase-2 READ_EDGES literal, frozen here verbatim for the historical
# equivalence proof the task required BEFORE deleting it from
# bin/timeline_audit.py (it no longer exists there as a literal -- this is the
# permanent regression record that the deletion was safe).
_FROZEN_READ_EDGES = {
    "B0": ["A"], "B1": ["B0"], "B2": ["A"], "B3": ["B2"], "B4": ["B3"],
    "C": ["B1", "B4"], "E0": ["B0", "B2"], "F0": ["A", "B3"], "F1": ["F0"],
    "F2": ["F0", "P"], "D0": ["P"], "P0": ["A"], "W": ["D0", "A"], "RG": ["W", "D0"],
    "RP": ["RG"], "DX": ["P"], "SEV2": ["W", "RP", "DX"], "SEV2_ADJ": ["SEV2"],
    "PSTREAM": ["SEV2_ADJ", "RP"], "T0": ["PSTREAM", "SEV2_ADJ"],
    "EV": ["T0", "V0", "E1", "CARD", "F2", "F1"], "LABEL": ["EV"],
    "V0": ["B3"], "E1": ["E0"], "CARD": ["E0"],
}


def test_read_edges_equivalence():
    """D-Log-06 equivalence check, run before READ_EDGES was deleted from
    bin/timeline_audit.py: for every stage READ_EDGES used to describe, the
    registry's `depends` matches exactly (zero VALUE mismatches found)."""
    for name, deps in _FROZEN_READ_EDGES.items():
        assert list(st.STAGES[name].depends) == deps, (
            f"{name}: registry depends {st.STAGES[name].depends} != frozen READ_EDGES {deps}"
        )


def test_read_edges_scope_gap_is_a_documented_finding():
    """FINDING (Phase 2 report): the pre-Phase-2 READ_EDGES literal never had
    keys for P, Q0, E2, or tags -- even though scheduler.STAGE_REGISTRY (P, Q0)
    and stages.py's own sourced comments (E2, tags) already knew their real
    depends. This is a pre-existing incompleteness in timeline_audit's own
    hardcoded DAG (the "fourth universe" the redesign doc names), not
    introduced by Phase 2. Phase 2 preserves this SCOPE deliberately (task
    item 8: don't change what a consumer does) rather than silently widening
    READERS/the READ LEDGER output -- tracked here so it can't vanish quietly."""
    gap = {"P", "Q0", "E2", "tags"}
    assert gap.isdisjoint(_FROZEN_READ_EDGES.keys())
    for name in gap:
        assert st.STAGES[name].depends, f"{name}: expected the registry to know real depends"
    assert gap.isdisjoint(timeline_audit.READ_EDGES.keys()), (
        "timeline_audit.READ_EDGES scope was expected to stay frozen to the pre-Phase-2 key set"
    )


def test_timeline_audit_read_edges_values_match_registry():
    for name, deps in timeline_audit.READ_EDGES.items():
        assert deps == list(st.STAGES[name].depends)


def test_timeline_audit_speech_spine_matches_registry():
    assert timeline_audit.SPEECH_SPINE == {k for k, v in st.STAGES.items() if v.speech_spine}


def test_timeline_audit_visual_matches_registry():
    assert timeline_audit.VISUAL == {k for k, v in st.STAGES.items() if v.visual}
    # exact pre-Phase-2 VISUAL literal, for regression:
    assert timeline_audit.VISUAL == {"B2", "B3", "B4", "F0", "F1", "F2", "V0"}


def test_timeline_audit_stage_artifacts_matches_reconcile_scope():
    # Independently derived (Phase 2: timeline_audit no longer imports
    # reconcile_status), but must land on the identical stage set/order.
    assert list(timeline_audit.STAGE_ARTIFACTS.keys()) == list(reconcile_status.STAGE_ARTIFACTS.keys())
    assert list(timeline_audit.STAGE_ARTIFACTS.keys()) == list(st.RECONCILE_ORDER)


# ── accessor helpers ─────────────────────────────────────────────────────────

def test_critical_stages_accessor():
    assert set(st.critical_stages().keys()) == {k for k, v in st.STAGES.items() if v.critical}


def test_speech_spine_stages_accessor():
    assert set(st.speech_spine_stages().keys()) == {k for k, v in st.STAGES.items() if v.speech_spine}


def test_scheduled_stages_accessor():
    assert set(st.scheduled_stages().keys()) == {k for k, v in st.STAGES.items() if v.scheduled}
    assert len(st.scheduled_stages()) == 14


def test_visual_stages_accessor():
    assert set(st.visual_stages().keys()) == {k for k, v in st.STAGES.items() if v.visual}


def test_media_and_corpus_scoped_partition():
    # D-Log-26: scope grew a third value ("on_demand" — SHORT/STORY/SURVEY/
    # ORIENT, worklist item 3(c)); media/corpus/on_demand now partition STAGES.
    media = set(st.media_scoped().keys())
    corpus = set(st.corpus_scoped().keys())
    on_demand = set(st.on_demand_scoped().keys())
    assert media | corpus | on_demand == set(st.STAGES.keys())
    assert media & corpus == set()
    assert media & on_demand == set()
    assert corpus & on_demand == set()


def test_on_demand_scoped_accessor():
    assert set(st.on_demand_scoped().keys()) == {"SHORT", "STORY", "SURVEY", "ORIENT"}
    for name in st.on_demand_scoped():
        stage = st.STAGES[name]
        assert not stage.scheduled, f"{name} must stay off the rebuild DAG"
        assert stage.artifact_path is None, f"{name} must stay out of RECONCILE_ORDER"


def test_get_accessor():
    for name in st.STAGES:
        assert st.get(name) is st.STAGES[name]


# ── anti-drift guard: no hardcoded stage-universe literal outside stages.py ─

# Matches `NAME = {"...` or `NAME: <type annotation> = {"...` -- a literal
# dict/set opening with a quoted key -- for the names Phase 2 retired as
# independent universes. Deliberately does NOT match dict/set
# *comprehensions* (`NAME = {k: v for ...}` / `{k for k in ...}`), which is
# exactly the replacement pattern every rewired consumer now uses, so this
# stays a guard against reintroducing a literal, not a ban on deriving one.
_UNIVERSE_LITERAL_RE = re.compile(
    r'\b(STAGE_ARTIFACTS|READ_EDGES|SPEECH_SPINE|STAGE_REGISTRY|CRITICAL|VISUAL)'
    r'\s*(:[^=\n]+)?=\s*\{\s*"'
)


def test_no_hardcoded_stage_universe_literals_outside_stages_py():
    offenders = []
    candidates = list(BIN.glob("*.py")) + list((ROOT / "src" / "erf").glob("*.py"))
    for py in candidates:
        if py.name == "stages.py":
            continue
        text = py.read_text(encoding="utf-8")
        if _UNIVERSE_LITERAL_RE.search(text):
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"hardcoded stage-universe literal(s) found outside stages.py: {offenders}"

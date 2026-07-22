"""Phase 5 (D-Log-04 §4 Phase 5) — the "degraded-proof" decisive verification.

`docs/OBSERVABILITY_REDESIGN.md` §6 item 4 says, verbatim: "null-out one media's
identity_map, rerun SEV2->T0, assert the row disposition is degraded and the
corpus verdict flips to not-green." **This is imprecise as literally written,
given the predicate design Phase 3 actually built (D-Log-09).**

The `pct_person_id > 0` predicate is CONDITIONAL: it is only evaluated when a
media's `identity_map` has >=1 slot with `source=="gallery"`. If you null/delete
`identity_map` entirely, `has_gallery_slot` is False, the calling stage omits
`pct_person_id` from the `values` dict it hands to `evaluate_health()`, and the
predicate is never evaluated — not defaulted, not failed. The row stays `"ok"`.
This is intentional (it is what stops rando-only/b-roll media from crying wolf
when they legitimately have zero gallery-anchored attribution), not a bug.

So the literal spec instruction, followed literally, demonstrates a VACUOUS
PASS, not a degraded flip — the opposite of "decisive". This file implements
BOTH the corrected understanding and the actual decisive proof, exactly as
directed for this phase (mirroring the Phase 0 F1-collision correction pattern
— see `src/erf/stages.py`'s module docstring — this is a spec correction,
documented explicitly, not a silent deviation):

  * **Test A** — reproduces the spec's literal scenario (identity_map absent)
    and proves it is a vacuous pass: `evaluate_health()` returns `{}`, the row
    stays `"ok"`, not `"degraded"`.
  * **Test B** — the actual failure mode that motivated this entire redesign
    (D-Identity-04: gallery-based identity resolution silently collapsed while
    everything stayed green). A gallery-anchored `identity_map` slot IS present
    for the synthetic media, but it is keyed to a speaker label that never
    appears in that media's words — so real word-level `person_id` resolution
    genuinely collapses to zero despite the gallery slot existing. This is
    exercised through the REAL `build_speech_events_v2.build_media()` function
    (path (a) per the Phase 5 brief — a full `build_transcript.process_media()`
    fixture was judged too fragile/heavy for the marginal proof value, since T0
    shares the identical `evaluate_health()`/predicate mechanism already proven
    real on SEV2; T0 is recorded with the same real-derived collapse value and
    carried through the full record()->rebuild_projection()->pipeline_status
    integration chain).

Isolation discipline (this phase is explicitly about proving a "corruption"
scenario — a previous phase's agent accidentally touched live corpus data
once, self-corrected but flagged as a real risk): every test here uses a
tmp_path-scoped sqlite journal.db (`db=` kwarg) AND monkeypatches
`erf.outcomes.STATUS_DIR` (a hardcoded module global that `db=` does NOT
redirect — the "known smell (a)" flagged in D-Log-07's report; `catalog/status/
{mid}.json` writes go through it regardless of the `db=` kwarg) to a tmp
directory, so the real `catalog/outcomes/*.json` surface is NEVER touched, not
even transiently. `erf.build_speech_events_v2.CAT` is likewise monkeypatched to
tmp_path for every fixture. `bin/pipeline_status.py`'s module-level `DB`
constant is monkeypatched before calling its real `_latest_status()`.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from erf import journal, outcomes, stages

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bs = _load_bin_module("bs_degraded_proof", "build_speech_events_v2.py")
ps = _load_bin_module("ps_degraded_proof", "pipeline_status.py")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


@pytest.fixture
def tmp_journal(tmp_path):
    db = tmp_path / "journal.db"
    journal.init_journal(db)
    return db


@pytest.fixture
def isolated_outcomes(tmp_path, monkeypatch):
    """`outcomes.STATUS_DIR` is a hardcoded module global; `record()`'s `db=`
    kwarg redirects the events/status_projection surface but NOT the per-media
    `catalog/status/{mid}.json` merge, which always goes through STATUS_DIR
    directly. Monkeypatch it to a tmp dir for the lifetime of the test so
    nothing in this file ever touches the real catalog/outcomes/, even
    transiently."""
    status_dir = tmp_path / "outcomes"
    monkeypatch.setattr(outcomes, "STATUS_DIR", status_dir)
    return status_dir


# ════════════════════════════════════════════════════════════════════════════
# Test A — the spec's literal scenario: nulled identity_map -> VACUOUS PASS
# ════════════════════════════════════════════════════════════════════════════

def test_a_nulled_identity_map_is_vacuous_pass_not_degraded(
    tmp_path, monkeypatch, tmp_journal, isolated_outcomes,
):
    """OBSERVABILITY_REDESIGN.md §6 item 4's literal wording ("null-out one
    media's identity_map ... assert the row disposition is degraded") does
    NOT hold given D-Log-09's conditional predicate design. Absent
    identity_map -> idmap == {} -> has_gallery_slot is False -> the calling
    stage omits "pct_person_id" from evaluate_health()'s `values` -> the
    predicate is never evaluated -> record()'s ok->degraded derivation has
    nothing to flip on -> the row stays "ok". Asserted explicitly below, with
    a real record() call, not a hand-wave."""
    monkeypatch.setattr(bs, "CAT", tmp_path)
    mid = "m_test_degraded_a"

    # Words exist (ASR ran fine) but identity_map/{mid}.json is deliberately
    # ABSENT -- the "null-out identity_map" scenario from the spec's literal text.
    _write_jsonl(tmp_path / "asr" / f"{mid}.words.jsonl", [
        {"start_ms": 0, "end_ms": 200, "text": "hello", "prob": 0.9, "speaker_label": "SPEAKER_00"},
        {"start_ms": 200, "end_ms": 400, "text": "world", "prob": 0.9, "speaker_label": "SPEAKER_00"},
    ])
    assert not (tmp_path / "identity_map" / f"{mid}.json").exists()

    r = bs.build_media(mid, tmp_path / "speech_events_v2", write=False)
    assert r["has_gallery_slot"] is False  # nothing to anchor to -- idmap == {}
    assert r["pct_person_id"] == 0.0       # attribution genuinely produced no person_id

    # Exactly how build_speech_events_v2.py decides whether the predicate is
    # in scope (see its module docstring / build() around the record() call).
    values = {"pct_person_id": r["pct_person_id"]} if r["has_gallery_slot"] else {}
    health = stages.evaluate_health("SEV2", values)
    assert health == {}  # THE FINDING: predicate SKIPPED, not evaluated -- no verdict exists at all

    outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": r["pct_person_id"]},
                    health=health, db=tmp_journal)

    conn = sqlite3.connect(str(tmp_journal))
    row = conn.execute(
        "SELECT status, health FROM events WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    conn.close()
    assert row[0] == "ok"  # NOT "degraded" -- the vacuous pass, confirmed against the real write path
    assert json.loads(row[1]) == {}

    # Zero live-surface pollution check (belt-and-braces on top of tmp_path autoclean):
    assert not (ROOT / "catalog" / "outcomes" / f"{mid}.json").exists()


# ════════════════════════════════════════════════════════════════════════════
# Test B — the actual decisive proof: gallery slot present, resolution
# collapses anyway (the real D-Identity-04 failure mode)
# ════════════════════════════════════════════════════════════════════════════

def test_b_gallery_slot_present_but_resolution_collapses_flips_corpus_verdict(
    tmp_path, monkeypatch, tmp_journal, isolated_outcomes,
):
    """Path (a): REAL `build_speech_events_v2.build_media()` invocation.
    identity_map anchors a gallery slot to label "SPEAKER_99" -- a label that
    never appears in this media's ASR words (every word's speaker_label is
    "SPEAKER_00"). `has_gallery_slot` is True (the identity_map DOES have a
    source=="gallery" entry) but `_resolve_identity()` can never find
    "SPEAKER_00" in the idmap, so every event's person_id is None -- a genuine,
    fixture-content-driven collapse of word-level attribution despite the
    gallery slot's presence. This reproduces D-Identity-04's real failure
    mode (identity resolution present upstream, silently lost downstream),
    not the vacuous "no gallery slot at all" case covered by Test A.

    Then proves the INTEGRATION claim end to end against the REAL consumer
    code: (1) record() lands status="degraded" via a real write, (2)
    rebuild_projection() folds it, (3) pipeline_status.py's actual
    _latest_status() + the actual critical_gaps boolean expression (quoted
    verbatim from bin/pipeline_status.py's main(), ~lines 205-206) now flags
    this media at stage T0 (and SEV2) -- "the corpus verdict flips to
    not-green", proven against the real consumer, not asserted in isolation.
    """
    monkeypatch.setattr(bs, "CAT", tmp_path)
    mid = "m_test_degraded_b"

    _write_jsonl(tmp_path / "asr" / f"{mid}.words.jsonl", [
        {"start_ms": 0, "end_ms": 200, "text": "hello", "prob": 0.9, "speaker_label": "SPEAKER_00"},
        {"start_ms": 200, "end_ms": 400, "text": "world", "prob": 0.9, "speaker_label": "SPEAKER_00"},
    ])
    (tmp_path / "identity_map").mkdir(parents=True)
    (tmp_path / "identity_map" / f"{mid}.json").write_text(json.dumps(
        {"speakers": {"SPEAKER_99": {"source": "gallery", "person_id": "p_ghost", "gallery_score": 0.9}}}
    ))

    r = bs.build_media(mid, tmp_path / "speech_events_v2", write=False)
    assert r["has_gallery_slot"] is True  # identity_map DID anchor a gallery slot for this media
    assert r["n_events"] >= 1             # words were attributed to a real speaker track (SPEAKER_00)
    assert r["pct_person_id"] == 0.0      # ...but NONE of it resolved to a person_id -- the collapse

    sev2_health = stages.evaluate_health("SEV2", {"pct_person_id": r["pct_person_id"]})
    assert sev2_health["pct_person_id"]["verdict"] == "fail"

    # ── (1) REAL record() call lands status="degraded" (not hand-constructed SQL) ──
    outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": r["pct_person_id"]},
                    health=sev2_health, db=tmp_journal)
    # T0 shares the IDENTICAL conditional pct_person_id>0 predicate (D-Log-09) --
    # record it too, reusing the SAME real-derived collapse value, to cover both
    # spine points the redesign doc names ("rerun SEV2->T0"). evaluate_health()
    # is the single home for the floor comparison (stages.py) -- calling it again
    # for "T0" is exercising the real mechanism a second time, not re-deriving it.
    t0_health = stages.evaluate_health("T0", {"pct_person_id": r["pct_person_id"]})
    assert t0_health["pct_person_id"]["verdict"] == "fail"
    outcomes.record(mid, "T0", "ok", metrics={"pct_person_id": r["pct_person_id"]},
                    health=t0_health, db=tmp_journal)

    conn = sqlite3.connect(str(tmp_journal))
    sev2_row = conn.execute(
        "SELECT status, reason FROM events WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    t0_row = conn.execute(
        "SELECT status, reason FROM events WHERE media_id=? AND stage='T0'", (mid,)
    ).fetchone()
    conn.close()
    assert sev2_row[0] == "degraded"
    assert "pct_person_id" in sev2_row[1]
    assert t0_row[0] == "degraded"
    assert "pct_person_id" in t0_row[1]

    # ── (2) rebuild_projection() folds it correctly (the AUTHORITY fold) ──
    n = outcomes.rebuild_projection(db=tmp_journal)
    assert n >= 2
    conn = sqlite3.connect(str(tmp_journal))
    proj_sev2 = conn.execute(
        "SELECT ok FROM status_projection WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    proj_t0 = conn.execute(
        "SELECT ok FROM status_projection WHERE media_id=? AND stage='T0'", (mid,)
    ).fetchone()
    conn.close()
    assert proj_sev2[0] == 0  # degraded is NOT in COMPLETED -> not-green in the projection
    assert proj_t0[0] == 0

    # ── (3) pipeline_status.py's ACTUAL gap-computation logic flags it ──
    monkeypatch.setattr(ps, "DB", tmp_journal)
    latest = ps._latest_status()  # real function, real query, only DB redirected
    assert latest[(mid, "SEV2")][0] == "degraded"
    assert latest[(mid, "T0")][0] == "degraded"

    no_speech = ps._no_speech_media()  # real call (reads outcomes.STATUS_DIR, isolated above)
    assert mid not in no_speech        # our synthetic media never set has_speech=False

    # EXACT replica of bin/pipeline_status.py main()'s critical_gaps predicate
    # (quoted verbatim, ~lines 205-206):
    #   crit = [st for st in CRITICAL if not (m in no_speech and st in SPEECH_SPINE)]
    #   miss = [st for st in crit if latest.get((m, st), ("missing",))[0] not in ("ok", "skip", "empty")]
    # Using ps.CRITICAL / ps.SPEECH_SPINE (the REAL, registry-derived module
    # constants) and the REAL `latest` dict from the REAL _latest_status() call
    # above -- this is the actual boolean expression pipeline_status.py runs,
    # not a paraphrase.
    crit = [st for st in ps.CRITICAL if not (mid in no_speech and st in ps.SPEECH_SPINE)]
    miss = [st for st in crit if latest.get((mid, st), ("missing",))[0] not in ("ok", "skip", "empty")]

    assert "SEV2" in miss
    assert "T0" in miss
    # THE decisive assertion: the corpus verdict flips to not-green for this
    # media, specifically at the two stages whose real downstream attribution
    # collapsed, computed via pipeline_status.py's REAL, unmodified
    # gap-computation code -- not a reimplementation, not an isolated
    # evaluate_health() unit check. "Today it stays green" (per the spec's own
    # framing) would mean SEV2/T0 absent from `miss`; here they are present.

    # Zero live-surface pollution check (belt-and-braces on top of tmp_path autoclean):
    assert not (ROOT / "catalog" / "outcomes" / f"{mid}.json").exists()


# ════════════════════════════════════════════════════════════════════════════
# Test C — corpus-scoped verdicts roll up into health.json (W2 checkpoint
# review finding, D-Log-12: spec P5 says "health rolls up to one verdict,
# gate included", but pipeline_status's m_* filter silently dropped
# "__corpus__" rows — a degraded GATE/CLUSTERS verdict was invisible in the
# corpus health surface. The last-mile instance of green-but-wrong.)
# ════════════════════════════════════════════════════════════════════════════

def test_c_degraded_corpus_stage_rolls_up_into_health_json(
    tmp_path, monkeypatch, tmp_journal, isolated_outcomes,
):
    """A degraded corpus-scoped row (here: GATE) must surface in health.json's
    new corpus_stages/corpus_not_green fields — proven through the REAL
    pipeline_status.main() write path (DB/HEALTH monkeypatched to tmp)."""
    # A real degraded GATE row via the real writer (explicit degraded needs reason):
    outcomes.record("__corpus__", "GATE", "degraded",
                    reason="OVERALL word_acc -0.100",
                    metrics={"word_acc": 0.854}, db=tmp_journal)
    # And one healthy media row so the media view isn't empty:
    outcomes.record("m_test_degraded_c", "A", "ok", db=tmp_journal)

    health_path = tmp_path / "health.json"
    monkeypatch.setattr(ps, "DB", tmp_journal)
    monkeypatch.setattr(ps, "HEALTH", health_path)
    monkeypatch.setattr(sys, "argv", ["pipeline_status.py"])
    ps.main()

    snap = json.loads(health_path.read_text())
    assert "GATE" in snap["corpus_stages"]
    assert snap["corpus_stages"]["GATE"]["status"] == "degraded"
    assert snap["corpus_not_green"] == ["GATE"]
    # And the media view is unpolluted by the corpus row:
    assert "GATE" not in snap["stages"]
    assert not (ROOT / "catalog" / "outcomes" / "m_test_degraded_c.json").exists()

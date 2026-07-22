"""Phase 3 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §3.1/§3.2): semantic
health predicates — the "green-but-wrong killer". Stub; built out via Edit.
"""
from __future__ import annotations

import importlib.util
import json
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


bic = _load_bin_module("bic_health", "build_identity_clusters.py")
bs = _load_bin_module("bs_health", "build_speech_events_v2.py")
bt = _load_bin_module("bt_health", "build_transcript.py")


@pytest.fixture
def real_status_cleanup():
    """Delete catalog/outcomes/{media_id}.json for every media_id appended to
    this fixture's returned list, after the test — for tests that call the
    REAL erf.outcomes.record() (STATUS_DIR is a hardcoded global; db=<tmp>
    does not redirect it)."""
    created: list[str] = []
    yield created
    for mid in created:
        p = outcomes.STATUS_DIR / f"{mid}.json"
        if p.exists():
            p.unlink()


@pytest.fixture
def tmp_journal(tmp_path):
    db = tmp_path / "journal.db"
    journal.init_journal(db)
    return db


# ──────────────────────────────────────────────────────────────────────────
# evaluate_health — pass/fail/boundary/conditional/unknown
# ──────────────────────────────────────────────────────────────────────────

def test_evaluate_health_ge_boundary_pass():
    h = stages.evaluate_health("CLUSTERS", {"gallery_share": 0.45})
    assert h["gallery_share"] == {"value": 0.45, "floor": 0.45, "verdict": "pass"}


def test_evaluate_health_ge_just_below_fails():
    h = stages.evaluate_health("CLUSTERS", {"gallery_share": 0.4499})
    assert h["gallery_share"]["verdict"] == "fail"


def test_evaluate_health_le_boundary_pass():
    h = stages.evaluate_health("CLUSTERS", {"no_embedding_share": 0.15})
    assert h["no_embedding_share"] == {"value": 0.15, "floor": 0.15, "verdict": "pass"}


def test_evaluate_health_le_just_above_fails():
    h = stages.evaluate_health("CLUSTERS", {"no_embedding_share": 0.1501})
    assert h["no_embedding_share"]["verdict"] == "fail"


def test_evaluate_health_gt_strict_boundary_fails():
    # SEV2/T0 predicate is `>` 0.0, strict — 0.0 itself must fail.
    h = stages.evaluate_health("SEV2", {"pct_person_id": 0.0})
    assert h["pct_person_id"]["verdict"] == "fail"
    h2 = stages.evaluate_health("T0", {"pct_person_id": 0.1})
    assert h2["pct_person_id"]["verdict"] == "pass"


def test_evaluate_health_both_cluster_predicates_together():
    h = stages.evaluate_health("CLUSTERS", {"gallery_share": 0.585, "no_embedding_share": 0.053})
    assert h["gallery_share"]["verdict"] == "pass"
    assert h["no_embedding_share"]["verdict"] == "pass"


def test_evaluate_health_omitted_predicate_is_skipped_not_defaulted():
    """The conditional-coupling mechanism: a predicate name absent from
    `values` is not evaluated at all (not defaulted to pass OR fail)."""
    h = stages.evaluate_health("SEV2", {})
    assert h == {}
    h2 = stages.evaluate_health("CLUSTERS", {"gallery_share": 0.0})
    assert "no_embedding_share" not in h2
    assert "gallery_share" in h2


def test_evaluate_health_unknown_stage_returns_empty():
    assert stages.evaluate_health("NOT_A_REAL_STAGE", {"anything": 1}) == {}


def test_evaluate_health_stage_with_no_predicates_returns_empty():
    # "A" has no declared health_predicates (Phase 0/1/2 stage, untouched here).
    assert stages.evaluate_health("A", {"pct_person_id": 0.0}) == {}


def test_evaluate_health_unknown_op_raises():
    bogus = stages.Stage(name="_TEST_BADOP", health_predicates=(("x", "!=", 1),))
    stages.STAGES["_TEST_BADOP"] = bogus
    try:
        with pytest.raises(ValueError):
            stages.evaluate_health("_TEST_BADOP", {"x": 5})
    finally:
        del stages.STAGES["_TEST_BADOP"]


# ──────────────────────────────────────────────────────────────────────────
# record() — ok -> degraded derivation
# ──────────────────────────────────────────────────────────────────────────

def test_record_ok_with_failing_health_derives_degraded(tmp_journal, real_status_cleanup):
    mid = "m_test_degraded_1"
    real_status_cleanup.append(mid)
    health = stages.evaluate_health("SEV2", {"pct_person_id": 0.0})  # fails: 0 not > 0
    outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": 0.0}, health=health,
                    db=tmp_journal)

    import sqlite3
    conn = sqlite3.connect(str(tmp_journal))
    row = conn.execute(
        "SELECT status, reason FROM events WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    proj = conn.execute(
        "SELECT ok FROM status_projection WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    conn.close()
    assert row[0] == "degraded"
    assert "pct_person_id" in row[1]
    assert proj[0] == 0  # degraded is NOT in COMPLETED -> not-green in the projection

    status_doc = json.loads((outcomes.STATUS_DIR / f"{mid}.json").read_text())
    assert status_doc["stages"]["SEV2"]["status"] == "degraded"


def test_record_ok_with_passing_health_stays_ok(tmp_journal, real_status_cleanup):
    mid = "m_test_degraded_2"
    real_status_cleanup.append(mid)
    health = stages.evaluate_health("SEV2", {"pct_person_id": 55.0})  # passes
    outcomes.record(mid, "SEV2", "ok", metrics={"pct_person_id": 55.0}, health=health,
                    db=tmp_journal)

    import sqlite3
    conn = sqlite3.connect(str(tmp_journal))
    row = conn.execute(
        "SELECT status FROM events WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    conn.close()
    assert row[0] == "ok"


def test_record_ok_with_no_health_stays_ok(tmp_journal, real_status_cleanup):
    mid = "m_test_degraded_3"
    real_status_cleanup.append(mid)
    outcomes.record(mid, "SEV2", "ok", metrics={"n_words": 10}, db=tmp_journal)

    import sqlite3
    conn = sqlite3.connect(str(tmp_journal))
    row = conn.execute(
        "SELECT status FROM events WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    conn.close()
    assert row[0] == "ok"


def test_record_explicit_degraded_keeps_own_reason_no_double_derivation(tmp_journal, real_status_cleanup):
    mid = "m_test_degraded_4"
    real_status_cleanup.append(mid)
    failing_health = stages.evaluate_health("SEV2", {"pct_person_id": 0.0})
    outcomes.record(mid, "SEV2", "degraded", reason="explicit caller reason",
                    health=failing_health, db=tmp_journal)

    import sqlite3
    conn = sqlite3.connect(str(tmp_journal))
    row = conn.execute(
        "SELECT status, reason FROM events WHERE media_id=? AND stage='SEV2'", (mid,)
    ).fetchone()
    conn.close()
    assert row[0] == "degraded"
    assert row[1] == "explicit caller reason"  # NOT overwritten — derivation only fires for status="ok"


def test_record_degraded_without_reason_or_health_raises(tmp_journal):
    with pytest.raises(ValueError):
        outcomes.record("m_test_degraded_5", "SEV2", "degraded", db=tmp_journal)


# ──────────────────────────────────────────────────────────────────────────
# T0 — conditional pct_person_id predicate, pure-function level
# ──────────────────────────────────────────────────────────────────────────

def test_t0_pct_person_id_mixed():
    rows = [{"person_id": "p1"}, {"person_id": None}, {"person_id": "p2"}, {"person_id": None}]
    assert bt._pct_person_id(rows) == 50.0


def test_t0_pct_person_id_empty_rows():
    assert bt._pct_person_id([]) == 0.0


def test_t0_pct_person_id_all_zero():
    rows = [{"person_id": None}, {"person_id": None}]
    assert bt._pct_person_id(rows) == 0.0


def test_t0_has_gallery_slot_true(tmp_path):
    catalog = tmp_path
    (catalog / "identity_map").mkdir(parents=True)
    (catalog / "identity_map" / "m_x.json").write_text(json.dumps(
        {"media_id": "m_x", "speakers": {"SPEAKER_00": {"source": "gallery", "person_id": "p1"}}}))
    assert bt._has_gallery_slot(catalog, "m_x") is True


def test_t0_has_gallery_slot_false_no_gallery_source(tmp_path):
    catalog = tmp_path
    (catalog / "identity_map").mkdir(parents=True)
    (catalog / "identity_map" / "m_x.json").write_text(json.dumps(
        {"media_id": "m_x", "speakers": {"SPEAKER_00": {"source": "cluster_unknown"}}}))
    assert bt._has_gallery_slot(catalog, "m_x") is False


def test_t0_has_gallery_slot_false_missing_file(tmp_path):
    assert bt._has_gallery_slot(tmp_path, "m_missing") is False


def test_t0_identity_map_fp_present_and_absent(tmp_path):
    catalog = tmp_path
    (catalog / "identity_map").mkdir(parents=True)
    p = catalog / "identity_map" / "m_x.json"
    p.write_text('{"a": 1}')
    fp = bt._identity_map_fp(catalog, "m_x")
    assert "identity_map" in fp
    assert len(fp["identity_map"]) == 16
    assert bt._identity_map_fp(catalog, "m_missing") == {}


def test_t0_conditional_predicate_end_to_end_via_evaluate_health(tmp_path):
    """The full conditional-coupling contract as T0's process_media applies it:
    include the key only when has_gallery_slot, else the predicate is never
    evaluated (vacuous pass)."""
    catalog = tmp_path
    (catalog / "identity_map").mkdir(parents=True)
    (catalog / "identity_map" / "m_gal.json").write_text(json.dumps(
        {"speakers": {"S0": {"source": "gallery"}}}))
    (catalog / "identity_map" / "m_rando.json").write_text(json.dumps(
        {"speakers": {"S0": {"source": "cluster_unknown"}}}))

    # gallery media with zero person_id coverage -> predicate evaluated, fails
    pct = 0.0
    has_gal = bt._has_gallery_slot(catalog, "m_gal")
    values = {"pct_person_id": pct} if has_gal else {}
    h = stages.evaluate_health("T0", values)
    assert h["pct_person_id"]["verdict"] == "fail"

    # rando-only media with zero person_id coverage -> predicate never evaluated
    has_gal2 = bt._has_gallery_slot(catalog, "m_rando")
    values2 = {"pct_person_id": pct} if has_gal2 else {}
    h2 = stages.evaluate_health("T0", values2)
    assert h2 == {}


# ──────────────────────────────────────────────────────────────────────────
# SEV2 — conditional pct_person_id predicate via build_media() (monkeypatched
# CAT -> tmp_path; never touches the real catalog)
# ──────────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def test_sev2_pct_person_id_and_gate_gallery_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "CAT", tmp_path)
    mid = "m_sev2test"
    _write_jsonl(tmp_path / "asr" / f"{mid}.words.jsonl", [
        {"start_ms": 0, "end_ms": 200, "text": "hello", "prob": 0.9, "speaker_label": "SPEAKER_00"},
        {"start_ms": 200, "end_ms": 400, "text": "world", "prob": 0.9, "speaker_label": "SPEAKER_00"},
    ])
    (tmp_path / "identity_map").mkdir(parents=True)
    (tmp_path / "identity_map" / f"{mid}.json").write_text(json.dumps(
        {"speakers": {"SPEAKER_00": {"source": "gallery", "person_id": "p_test", "gallery_score": 0.9}}}))

    r = bs.build_media(mid, tmp_path / "speech_events_v2", write=False)
    assert r["has_gallery_slot"] is True
    assert r["n_events"] == 1
    assert r["pct_person_id"] == 100.0

    health = bs._stages_registry.evaluate_health("SEV2", {"pct_person_id": r["pct_person_id"]})
    assert health["pct_person_id"]["verdict"] == "pass"


def test_sev2_no_gallery_slot_vacuous_pass_even_with_zero_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "CAT", tmp_path)
    mid = "m_sev2test2"
    _write_jsonl(tmp_path / "asr" / f"{mid}.words.jsonl", [
        {"start_ms": 0, "end_ms": 200, "text": "hi", "prob": 0.9, "speaker_label": "SPEAKER_00"},
    ])
    (tmp_path / "identity_map").mkdir(parents=True)
    (tmp_path / "identity_map" / f"{mid}.json").write_text(json.dumps(
        {"speakers": {"SPEAKER_00": {"source": "cluster_unknown", "cluster_id": "vc_x"}}}))

    r = bs.build_media(mid, tmp_path / "speech_events_v2", write=False)
    assert r["has_gallery_slot"] is False
    assert r["pct_person_id"] == 0.0  # unenrolled -> no person_id, legitimately 0

    values = {"pct_person_id": r["pct_person_id"]} if r["has_gallery_slot"] else {}
    health = bs._stages_registry.evaluate_health("SEV2", values)
    assert health == {}  # predicate never evaluated -> vacuous pass


def test_sev2_empty_words_returns_zero_pct_and_gallery_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "CAT", tmp_path)
    mid = "m_sev2empty"
    (tmp_path / "identity_map").mkdir(parents=True)
    (tmp_path / "identity_map" / f"{mid}.json").write_text(json.dumps(
        {"speakers": {"SPEAKER_00": {"source": "gallery", "person_id": "p_test"}}}))
    r = bs.build_media(mid, tmp_path / "speech_events_v2", write=False)
    assert r["n_words"] == 0
    assert r["pct_person_id"] == 0.0
    assert r["has_gallery_slot"] is True


def test_sev2_identity_map_fp(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "CAT", tmp_path)
    (tmp_path / "identity_map").mkdir(parents=True)
    p = tmp_path / "identity_map" / "m_x.json"
    p.write_text('{"a": 1}')
    fp = bs._identity_map_fp("m_x")
    assert len(fp["identity_map"]) == 16
    assert bs._identity_map_fp("m_missing") == {}


# ──────────────────────────────────────────────────────────────────────────
# CLUSTERS — histogram math + on-spine record() wiring
# (erf.outcomes.record monkeypatched to a spy: NEVER touches the real
# journal.db or catalog/outcomes/. build()'s write path, when exercised, is
# pointed at a tmp_path CAT — NEVER the real catalog.)
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def record_spy(monkeypatch):
    calls = []

    def fake_record(media_id, stage, status, **kw):
        calls.append((media_id, stage, status, kw))

    monkeypatch.setattr(bic.outcomes, "record", fake_record)
    return calls


def _media_map_fixture():
    return {
        "m_a": {
            "S0": {"person_id": "p_gal", "cluster_id": "vc_1", "source": "gallery", "gallery_score": 0.9},
            "S1": {"person_id": "p_unk_1", "cluster_id": "vc_2", "source": "cluster_unknown"},
        },
        "m_b": {
            "S0": {"person_id": None, "cluster_id": None, "source": "no_embedding", "reason": "zero"},
            "S1": {"person_id": "p_unk_3", "cluster_id": "vc_3", "source": "demoted_overmerge"},
        },
    }


def test_clusters_source_histogram():
    hist = bic._source_histogram(_media_map_fixture())
    assert hist == {"gallery": 1, "cluster_unknown": 1, "no_embedding": 1, "demoted_overmerge": 1}


def test_clusters_corpus_shares():
    hist = bic._source_histogram(_media_map_fixture())
    total, gallery_share, no_embedding_share = bic._corpus_shares(hist)
    assert total == 4
    assert gallery_share == 0.25
    assert no_embedding_share == 0.25


def test_clusters_corpus_shares_empty_histogram():
    assert bic._corpus_shares({}) == (0, 0.0, 0.0)


def test_clusters_corpus_shares_matches_real_corpus_healthy_values():
    """Sanity check against DECISIONS.md D-Log-08's measured healthy values
    (gallery_share=0.585, no_embedding_share=0.053) using a histogram shaped
    like the real corpus (230 gallery / 393 total, 21 no_embedding / 393)."""
    hist = {"gallery": 230, "cluster_unknown": 134, "no_embedding": 21, "demoted_overmerge": 8}
    total, gallery_share, no_embedding_share = bic._corpus_shares(hist)
    assert total == 393
    assert gallery_share == pytest.approx(0.585, abs=0.001)
    assert no_embedding_share == pytest.approx(0.053, abs=0.001)
    health = stages.evaluate_health("CLUSTERS", {
        "gallery_share": gallery_share, "no_embedding_share": no_embedding_share})
    assert health["gallery_share"]["verdict"] == "pass"
    assert health["no_embedding_share"]["verdict"] == "pass"


def test_clusters_record_media_emits_one_row_per_media(record_spy):
    bic._record_media(_media_map_fixture())
    mids = {c[0] for c in record_spy}
    assert mids == {"m_a", "m_b"}
    for mid, stage, status, kw in record_spy:
        assert stage == "CLUSTERS"
        assert status == "ok"
        assert kw["metrics"]["n_slots"] == 2


def test_clusters_record_corpus_ok_healthy_histogram(record_spy):
    hist = {"gallery": 230, "cluster_unknown": 134, "no_embedding": 21, "demoted_overmerge": 8}
    bic._record_corpus("ok", hist, reason=None, n_media=250, n_clusters=116)
    assert len(record_spy) == 1
    mid, stage, status, kw = record_spy[0]
    assert (mid, stage, status) == ("__corpus__", "CLUSTERS", "ok")
    assert kw["health"]["gallery_share"]["verdict"] == "pass"
    assert kw["health"]["no_embedding_share"]["verdict"] == "pass"
    assert kw["metrics"]["total_slots"] == 393


def test_clusters_record_corpus_unhealthy_histogram_fails_health(record_spy):
    # gallery starved, no_embedding heavy -> both predicates fail.
    hist = {"gallery": 10, "cluster_unknown": 40, "no_embedding": 50}
    bic._record_corpus("ok", hist, reason=None, n_media=10, n_clusters=5)
    _, _, _, kw = record_spy[0]
    assert kw["health"]["gallery_share"]["verdict"] == "fail"
    assert kw["health"]["no_embedding_share"]["verdict"] == "fail"


def test_clusters_record_corpus_empty_status_has_no_health(record_spy):
    bic._record_corpus("empty", {}, reason="no_slots", n_media=0)
    _, _, status, kw = record_spy[0]
    assert status == "empty"
    assert kw["health"] is None


def test_clusters_build_no_slots_exit_records_corpus_empty(record_spy):
    bic.build(0.55, False, slots=[], degenerate=[], gallery=[])
    assert len(record_spy) == 1
    mid, stage, status, kw = record_spy[0]
    assert (mid, stage, status) == ("__corpus__", "CLUSTERS", "empty")
    assert kw["reason"] == "no_slots"


def test_clusters_build_report_only_exit_records_corpus_skip_no_per_media(record_spy, monkeypatch):
    import numpy as np
    rng = np.random.default_rng(0)
    gal_vec = rng.normal(size=256)
    slots = [
        ("m_x", "S0", gal_vec + rng.normal(scale=0.01, size=256)),
        ("m_y", "S0", rng.normal(size=256)),
    ]
    bic.build(0.55, True, slots=slots, degenerate=[], gallery=[("p_test", gal_vec)])
    assert len(record_spy) == 1
    mid, stage, status, kw = record_spy[0]
    assert (mid, stage, status) == ("__corpus__", "CLUSTERS", "skip")
    assert kw["reason"] == "report_only"


def test_clusters_build_full_write_exit_records_per_media_and_corpus(record_spy, monkeypatch, tmp_path):
    monkeypatch.setattr(bic, "CAT", tmp_path)  # never touch the real catalog
    import numpy as np
    rng = np.random.default_rng(1)
    gal_vec = rng.normal(size=256)
    slots = [
        ("m_x", "S0", gal_vec + rng.normal(scale=0.01, size=256)),
        ("m_y", "S0", rng.normal(size=256)),
    ]
    bic.build(0.55, False, slots=slots, degenerate=[("m_z", "S0", "zero")],
             gallery=[("p_test", gal_vec)])

    statuses = [c[2] for c in record_spy]
    assert statuses.count("ok") == len(record_spy)  # all per-media + corpus rows are "ok" here
    mids = {c[0] for c in record_spy}
    assert mids == {"m_x", "m_y", "m_z", "__corpus__"}
    assert (tmp_path / "identity_clusters" / "clusters.json").exists()
    assert (tmp_path / "identity_map" / "m_x.json").exists()
    assert (tmp_path / "identity_map" / "m_z.json").exists()
    # nothing written to the real catalog
    assert not (ROOT / "catalog" / "identity_map" / "m_x.json").exists()

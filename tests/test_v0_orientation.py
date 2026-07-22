"""D-Log-25 (worklist item 3(b)): V0 visual-grounding orientation field +
render's b-roll orientation flag (journal-only — see render.py's D-Log-25
comment for why auto-correction was rejected after a real-corpus spot-check).
bin/ is not a package — v0_visual_grounding.py is loaded via importlib.util,
matching tests/test_stage_empty_dispositions.py. Importing it does not touch
the GPU/VLM weights: model loading lives inside _load_model_and_processor(),
never at import time.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v0 = _load_bin_module("v0_visual_grounding_test", "v0_visual_grounding.py")


def test_schema_lives_in_tracked_schemas_dir_not_gitignored_catalog():
    # D-Log-25 stress-test finding: this schema previously existed only under
    # gitignored catalog/schemas/ — never committed, invisible to git, despite
    # DECISIONS.md already flagging it "absent from schemas/". Moved to the
    # tracked schemas/ dir alongside this commit; guard against it drifting
    # back to the untracked location.
    assert (ROOT / "schemas" / "visual_grounding.schema.json").exists()
    assert not (ROOT / "catalog" / "schemas" / "visual_grounding.schema.json").exists()


def test_real_corpus_records_validate_against_schema():
    from erf.contracts import validate

    vg_dir = ROOT / "catalog" / "visual_grounding"
    if not vg_dir.exists():
        return  # no corpus in this environment
    records = list(vg_dir.glob("*.json"))
    if not records:
        return
    for f in records:
        validate(json.loads(f.read_text()), "visual_grounding")


def test_parse_vlm_response_accepts_valid_orientation():
    raw = json.dumps({
        "footage_type": "b_roll", "setting": "office", "on_screen_subjects": [],
        "shot_type": "wide", "visual_summary": "desk", "orientation": "rotated_90_cw",
    })
    parsed = v0._parse_vlm_response(raw)
    assert parsed["orientation"] == "rotated_90_cw"


def test_parse_vlm_response_defaults_missing_orientation_to_upright():
    raw = json.dumps({
        "footage_type": "b_roll", "on_screen_subjects": [], "visual_summary": "x",
    })
    parsed = v0._parse_vlm_response(raw)
    assert parsed["orientation"] == "upright"


def test_parse_vlm_response_rejects_invalid_orientation_value():
    raw = json.dumps({
        "footage_type": "b_roll", "on_screen_subjects": [], "visual_summary": "x",
        "orientation": "sideways-ish",
    })
    parsed = v0._parse_vlm_response(raw)
    assert parsed["orientation"] == "upright"


def test_null_grounding_orientation_is_none():
    record = v0._null_grounding("m_test", {"model": "none"})
    assert record["orientation"] is None


def test_error_grounding_orientation_is_none():
    record = v0._error_grounding("m_test", {"model": "none"}, "boom")
    assert record["orientation"] is None


erf_render = __import__("erf.render", fromlist=["_load_broll_orientation", "_ORIENTATION_FIX_VF"])


def test_load_broll_orientation_missing_file_defaults_upright(tmp_path):
    catalog = tmp_path
    (catalog / "visual_grounding").mkdir()
    assert erf_render._load_broll_orientation(catalog, "m_absent") == "upright"


def test_load_broll_orientation_reads_real_value(tmp_path):
    catalog = tmp_path
    vg_dir = catalog / "visual_grounding"
    vg_dir.mkdir()
    (vg_dir / "m_rot.json").write_text(json.dumps({"orientation": "upside_down"}))
    assert erf_render._load_broll_orientation(catalog, "m_rot") == "upside_down"


def test_load_broll_orientation_rejects_unknown_value(tmp_path):
    catalog = tmp_path
    vg_dir = catalog / "visual_grounding"
    vg_dir.mkdir()
    (vg_dir / "m_bad.json").write_text(json.dumps({"orientation": "diagonal"}))
    assert erf_render._load_broll_orientation(catalog, "m_bad") == "upright"


def test_orientation_fix_vf_covers_all_non_upright_values():
    assert set(erf_render._ORIENTATION_FIX_VF.keys()) == {
        "rotated_90_cw", "rotated_90_ccw", "upside_down",
    }
    assert erf_render._ORIENTATION_FIX_VF["rotated_90_cw"] == "transpose=2"
    assert erf_render._ORIENTATION_FIX_VF["rotated_90_ccw"] == "transpose=1"


# ── Real render integration: flag-only, no pixel correction ────────────────
# m_af69ff9a010d is a real corpus media whose V0 orientation verdict
# (rotated_90_cw) was visually spot-checked during this build and confirmed
# genuinely rotated (unlike two other flagged clips in the same run, which
# were upright footage misread from motion blur / an odd low-angle
# composition — see render.py's D-Log-25 comment). Used here as b-roll to
# exercise the real render()->journal path, not a synthetic fixture.
ROT_CATALOG = ROOT / "catalog"
ROT_PROXY_DIR = ROOT / "media" / "proxies"
ROT_MEDIA_ID = "m_af69ff9a010d"

pytestmark_rot = __import__("pytest").mark.skipif(
    not (ROT_PROXY_DIR / f"{ROT_MEDIA_ID}.mp4").exists()
    or not (ROT_CATALOG / "visual_grounding" / f"{ROT_MEDIA_ID}.json").exists(),
    reason="rotated corpus fixture media not present in this environment",
)


@pytestmark_rot
def test_broll_orientation_flags_without_correcting_pixels(tmp_path, render_catalog):
    import json as _json
    import sqlite3
    from erf.journal import init_journal
    from erf.render import render

    orientation = _json.loads(
        (ROT_CATALOG / "visual_grounding" / f"{ROT_MEDIA_ID}.json").read_text()
    )["orientation"]
    assert orientation == "rotated_90_cw", "fixture media's V0 verdict drifted — re-pick a fixture"

    journal_db = tmp_path / "journal.db"
    init_journal(journal_db)

    result = render(
        beat_plan={"beats": [{"in_ms": 0, "out_ms": 3000}], "format_fit": "youtube_shorts"},
        clips=[{"media_id": ROT_MEDIA_ID, "in_ms": 0, "out_ms": 3000, "_is_broll": True}],
        export_id="g_orient_flag_test",
        catalog=render_catalog,
        proxy_dir=ROT_PROXY_DIR,
        output_dir=tmp_path,
        reframe_strategy="face_centroid",
        caption_style="none",
        journal_db=journal_db,
    )
    assert Path(result["output_mp4_path"]).exists()

    with sqlite3.connect(str(journal_db)) as con:
        rows = con.execute(
            "SELECT payload FROM events_trace WHERE event_type = ?",
            ("render/broll_orientation/flagged",),
        ).fetchall()
    assert len(rows) == 1
    payload = _json.loads(rows[0][0])
    assert payload["media_id"] == ROT_MEDIA_ID
    assert payload["orientation"] == "rotated_90_cw"

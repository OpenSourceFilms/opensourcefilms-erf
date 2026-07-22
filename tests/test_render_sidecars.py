"""D-Log-24: render sidecars (captions.ass + beats_debug.json) are always-on,
no longer gated behind ERF_RENDER_DEBUG. Real render against corpus media —
mirrors render_gate_test's own real-proxy pattern (render.py:1700)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog"
PROXY_DIR = ROOT / "media" / "proxies"
MEDIA_ID = "m_01d094906b30"  # 16 words, all within [0, 3880]ms — short, fast fixture

pytestmark = pytest.mark.skipif(
    not (PROXY_DIR / f"{MEDIA_ID}.mp4").exists()
    or not (CATALOG / "asr" / f"{MEDIA_ID}.transcript.jsonl").exists(),
    reason="corpus fixture media not present in this environment",
)


def test_render_writes_both_sidecars_without_debug_env(tmp_path, monkeypatch, render_catalog):
    from erf.render import render

    monkeypatch.delenv("ERF_RENDER_DEBUG", raising=False)

    export_id = "g_sidecar_test"
    result = render(
        beat_plan={"beats": [{"in_ms": 0, "out_ms": 4000}], "format_fit": "youtube_shorts"},
        clips=[{"media_id": MEDIA_ID, "in_ms": 0, "out_ms": 4000}],
        export_id=export_id,
        catalog=render_catalog,
        proxy_dir=PROXY_DIR,
        output_dir=tmp_path,
        reframe_strategy="face_centroid",
        caption_style="centre",
    )

    assert Path(result["output_mp4_path"]).exists()

    captions_path = tmp_path / f"{export_id}.captions.ass"
    debug_path = tmp_path / f"{export_id}.beats_debug.json"
    assert captions_path.exists(), "captions.ass sidecar must always write (no debug gate)"
    assert debug_path.exists(), "beats_debug.json sidecar must always write (no debug gate)"

    import json
    dbg = json.loads(debug_path.read_text())
    assert isinstance(dbg, list) and len(dbg) == 1
    assert dbg[0]["beat"] == 0
    assert any(ev.get("t") for ev in dbg[0]["events"])

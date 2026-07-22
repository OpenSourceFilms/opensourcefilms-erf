#!/usr/bin/env python3
"""Run the full Phase 1-3a pipeline on one media file.
Must be invoked from /workspace/erf with the embed venv (audio venv for stage A is handled via subprocess).
Usage: python bin/ingest_one.py <media_path>
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AUDIO_PYTHON = "/workspace/erf/venvs/audio/bin/python"
ERF_ROOT = Path(__file__).resolve().parents[1]

def run_stage_a(media_path: str) -> dict:
    """Run stage A in audio venv via subprocess."""
    script = f"""
import sys; sys.path.insert(0, 'src')
from erf.asr import run
result = run('{media_path}', config_path='state/config.toml')
import json; print(json.dumps({{'media_id': result.media_id, 'ok': result.ok, 'counts': result.counts, 'error': result.error}}))
"""
    r = subprocess.run(
        [AUDIO_PYTHON, "-c", script],
        capture_output=True, text=True, cwd=str(ERF_ROOT),
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/root",
             "PYTHONWARNINGS": "ignore"}
    )
    # Find the JSON line
    for line in r.stdout.strip().splitlines():
        try:
            return json.loads(line)
        except Exception:
            continue
    print(f"  Stage A stderr snippet: {r.stderr[-500:]}", file=sys.stderr)
    return {"ok": False, "error": r.stderr[-200:], "media_id": None, "counts": {}}


def run_stages(media_path: str) -> None:
    from erf.chunk import run as run_b0
    from erf.embed_text import run as run_b1
    from erf.materialise import run as run_c, run_visual as run_c_vis
    from erf.scenes import run as run_b2
    from erf.keyframes import run as run_b3
    from erf.embed_visual import run as run_b4
    from erf.quality import run as run_q0
    from erf.events import run as run_e0
    from erf.tags import run as run_tags
    from erf.proxy import run as run_p
    from erf.fts import index_media

    import tomllib
    with open("state/config.toml", "rb") as f:
        cfg = tomllib.load(f)
    catalog = Path(cfg["paths"]["catalog"])
    fts_db = Path(cfg["paths"]["erf_fts_db"])

    print(f"\n{'='*60}")
    print(f"Pipeline: {Path(media_path).name}")
    print(f"{'='*60}")

    # Stage A
    print("\n[A] ASR + Diarization...")
    a = run_stage_a(media_path)
    print(f"  ok={a['ok']}  counts={a['counts']}  id={a['media_id']}")
    if not a["ok"]:
        print(f"  ERROR: {a['error']}")
        return
    mid = a["media_id"]

    # B0
    print(f"\n[B0] Chunking...")
    r = run_b0(mid, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")
    if not r.ok: return

    # X0 (FTS)
    print(f"\n[X0] FTS index...")
    try:
        counts = index_media(mid, catalog, fts_db)
        print(f"  {counts}")
    except Exception as e:
        print(f"  WARN: {e}")

    # B1
    print(f"\n[B1] Text embedding...")
    r = run_b1(mid, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")
    if not r.ok: return

    # C
    print(f"\n[C] Materialise text_v1...")
    r = run_c(mid, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")

    # B2
    print(f"\n[B2] Scene detection...")
    r = run_b2(mid, media_path, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")
    if not r.ok: return

    # B3
    print(f"\n[B3] Keyframe extraction...")
    r = run_b3(mid, media_path, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")
    if not r.ok: return

    # B4
    print(f"\n[B4] Visual embedding...")
    r = run_b4(mid, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")
    if not r.ok: return

    # C_vis
    print(f"\n[C_vis] Materialise visual_v1...")
    r = run_c_vis(mid, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")

    # Q0
    print(f"\n[Q0] Quality metrics...")
    r = run_q0(mid, media_path, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")

    # E0
    print(f"\n[E0] Beat tagging...")
    r = run_e0(mid, config_path="state/config.toml", llm_tag=True)
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")

    # tags
    print(f"\n[tags] Scoring...")
    r = run_tags(mid, config_path="state/config.toml")
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")

    # P
    print(f"\n[P] Proxy + thumb...")
    mm = json.loads((catalog / "media_manifest" / f"{mid}.json").read_text())
    r = run_p(mid, media_path, config_path="state/config.toml", duration_ms=mm["duration_ms"])
    print(f"  ok={r.ok}  counts={r.counts}  error={r.error}")

    print(f"\n[DONE] media_id={mid}")
    return mid


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    media_path = sys.argv[1]
    run_stages(media_path)

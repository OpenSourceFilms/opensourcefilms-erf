#!/usr/bin/env python3
"""make_story — cross-media narrative shorts.

DS Pro surveys the WHOLE corpus, proposes self-contained stories (a mix of punchy
single-clip and multi-part narratives drawing from many clips), then builds and
renders each as an ordered, grounded beat plan. Nothing is siloed.

Usage: PYTHONPATH=src DEEPSEEK_API_KEY=.. venvs/control/bin/python bin/make_story.py [--n 8] [--min-clarity 0.1]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from erf.grounded_editor import condense, story_ideas, build_story
from erf.render import render as do_render
try:
    from erf.runlog import stage_run
except Exception:
    stage_run = None

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"
CFG = ROOT / "state" / "config.toml"


def _load_packages(min_clarity: float) -> dict[str, dict]:
    pkgs = {}
    for f in glob.glob(str(CAT / "evidence" / "*.json")):
        try:
            p = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not p.get("sentences"):
            continue
        clarity = (p.get("av_quality") or {}).get("speech_clarity_median") or 0
        if clarity < min_clarity:
            continue
        pkgs[p["media_id"]] = p
    return pkgs


def render_story(story: dict, idea: dict) -> dict | None:
    if not story["beats"]:
        print(f"  '{idea.get('title')}': no usable beats — {story.get('rationale','')[:80]}")
        return None
    if story["total_duration_s"] < 12:
        print(f"  '{idea.get('title')}': only {story['total_duration_s']}s — skip")
        return None
    clips = [{"media_id": b["media_id"], "in_ms": b["in_ms"], "out_ms": b["out_ms"],
              "label": b["role"], "_is_broll": False, "_crosstalk_trimmed": False}
             for b in story["beats"]]
    beat_plan = {"topic": story["title"], "logline": story.get("logline"),
                 "beats": [{"edit_role": b["role"], "note": (b.get("text") or "")[:60],
                            "media_id": b["media_id"], "in_ms": b["in_ms"], "out_ms": b["out_ms"]}
                           for b in story["beats"]]}
    export_id = f"story_{int(time.time())}_{abs(hash(story['title'])) % 10**6:06d}"
    res = do_render(beat_plan=beat_plan, clips=clips, export_id=export_id, catalog=CAT,
                    proxy_dir=ROOT / "media" / "proxies", output_dir=CAT / "exports",
                    reframe_strategy="active_speaker", journal_db=ROOT / "state" / "journal.db",
                    config_path=CFG)
    (CAT / "stories").mkdir(exist_ok=True)
    media_ids = sorted({b["media_id"] for b in story["beats"]})
    (CAT / "stories" / f"{export_id}.json").write_text(
        json.dumps({"export_id": export_id, "mp4": res.get("output_mp4_path"),
                    "media_ids": media_ids, "story": story, "idea": idea}, indent=2), encoding="utf-8")
    return {"export_id": export_id, **res, "n_clips": len(clips), "n_media": len(media_ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-clarity", type=float, default=0.1)
    a = ap.parse_args()

    pkgs = _load_packages(a.min_clarity)
    entries = [condense(p) for p in pkgs.values()]
    print(f"corpus: {len(entries)} clips → ideating stories...")
    ideas = story_ideas(entries, n=a.n)
    print(f"got {len(ideas)} story ideas")
    for i, idea in enumerate(ideas, 1):
        print(f"[{i}/{len(ideas)}] {idea.get('kind')}: '{idea.get('title')}' "
              f"({len(idea.get('media_ids', []))} clips) — {idea.get('through_line','')[:60]}")
        try:
            story = build_story(idea, pkgs)
            # Duration floor lives in render_story (<12s skip). It now works correctly
            # because ground_story dedups overlapping beats, so total_duration_s is honest
            # — pre-fix, a degenerate 43-beat plan summed to a fake 23.4s and slipped past.
            if stage_run:
                with stage_run("STORY", idea.get("title", "?")[:40]):
                    r = render_story(story, idea)
            else:
                r = render_story(story, idea)
            if r:
                print(f"    -> {r.get('output_mp4_path')} ({r.get('duration_s')}s, "
                      f"{r['n_clips']} clips / {r['n_media']} media)")
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

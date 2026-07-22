#!/usr/bin/env python3
"""make_short — evidence package → grounded editor (DS Pro) → render → mp4.

The clean end-to-end path that replaces scan+Pass1+Pass2+gate+batch_render.
Beat in/out are grounded sentence boundaries from the evidence, so cuts land
cleanly and captions are speaker-coloured (unified transcript).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from erf.grounded_editor import edit_single, ground_edit
from erf.render import render as do_render
try:
    from erf.runlog import stage_run, log_event
except Exception:
    stage_run = None

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"
CFG = ROOT / "state" / "config.toml"


def make_short(media_id: str, target_s: int = 30, reframe: str = "active_speaker") -> dict | None:
    pkg = json.loads((CAT / "evidence" / f"{media_id}.json").read_text(encoding="utf-8"))
    edit = edit_single(pkg, target_s=target_s)
    g = ground_edit(pkg, edit)
    if not g["beats"]:
        print(f"{media_id}: editor returned no usable beats — {g.get('rationale')}")
        return None
    if g["total_duration_s"] < 8:
        print(f"{media_id}: best moment only {g['total_duration_s']}s — too short for a viable short, skipping")
        return None
    clips = [{"media_id": b["media_id"], "in_ms": b["in_ms"], "out_ms": b["out_ms"],
              "label": b["role"], "_is_broll": False, "_crosstalk_trimmed": False}
             for b in g["beats"]]
    beat_plan = {
        "topic": g["title"], "logline": g.get("logline"),
        "beats": [{"edit_role": b["role"], "note": (b.get("text") or "")[:60],
                   "media_id": b["media_id"], "in_ms": b["in_ms"], "out_ms": b["out_ms"]}
                  for b in g["beats"]],
    }
    export_id = f"short_{int(time.time())}_{media_id[2:8]}"
    res = do_render(beat_plan=beat_plan, clips=clips, export_id=export_id, catalog=CAT,
                    proxy_dir=ROOT / "media" / "proxies", output_dir=CAT / "exports",
                    reframe_strategy=reframe, journal_db=ROOT / "state" / "journal.db",
                    config_path=CFG)
    (CAT / "shorts").mkdir(exist_ok=True)
    (CAT / "shorts" / f"{export_id}.json").write_text(
        json.dumps({"media_id": media_id, "export_id": export_id,
                    "mp4": res.get("output_mp4_path"), "edit": g}, indent=2), encoding="utf-8")
    return {"export_id": export_id, **res, "edit": g}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("media_ids", nargs="+")
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--reframe", default="active_speaker")
    a = ap.parse_args()
    for mid in a.media_ids:
        try:
            if stage_run:
                with stage_run("SHORT", mid):
                    r = make_short(mid, a.target, a.reframe)
            else:
                r = make_short(mid, a.target, a.reframe)
            if r:
                print(f"{mid}: '{r['edit']['title']}' -> {r.get('output_mp4_path')} "
                      f"({r.get('duration_s')}s)")
        except Exception as e:
            print(f"{mid}: FAILED {e}")
            traceback.print_exc()

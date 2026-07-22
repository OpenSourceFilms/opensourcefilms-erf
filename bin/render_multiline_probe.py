#!/usr/bin/env python3
"""Multi-line crosstalk subtitle prototype — renders directly from speech_events_v2.

Two speaker events that genuinely overlap in time render as TWO simultaneous lines:
the longer/dominant event on the bottom (\\an2), the shorter/interjecting event on
top (\\an8). Non-overlapping events render as a single bottom line. Colour is by
person_id (known) or a stable fallback per unknown voice cluster; ambiguous tracks
render grey. This validates the two-line concept end-to-end on a chosen window.

Usage:
  python bin/render_multiline_probe.py --mid m_5a8bedaa054a --start 43 --end 61 --out /path/out.mp4
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAT = ROOT / "catalog"

PERSON_COLORS = {}  # populate per-project, e.g. {"p_alice": "00FFFF"}
FALLBACK = ["FF00FF", "0000FF", "E6D8AD", "FFC0CB", "800080", "00D7FF"]
AMBIG = "AAAAAA"


def _load_events(mid: str, subdir: str = "speech_events_v2") -> list[dict]:
    p = CAT / subdir / f"{mid}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _bts_index(root="/workspace/BTS") -> dict:
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            full = os.path.join(dp, fn)
            if fn not in idx or len(full) < len(idx[fn]):
                idx[fn] = full
    return idx


def _source(mid: str) -> str | None:
    mp = CAT / "media_manifest" / f"{mid}.json"
    if not mp.exists():
        return None
    rel = json.loads(mp.read_text()).get("relpath", "")
    idx = _bts_index()
    return idx.get(os.path.basename(rel))


def _color(e: dict) -> str:
    if e["speaker_track_id"] == "unknown_overlap_speaker":
        return AMBIG
    if e.get("person_id") and e["person_id"] in PERSON_COLORS:
        return PERSON_COLORS[e["person_id"]]
    key = e.get("stable_unknown_id") or e.get("person_id") or e["speaker_track_id"]
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return FALLBACK[h % len(FALLBACK)]


def _ms_to_ass(ms: int) -> str:
    ms = max(0, ms)
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000; ms %= 60000
    s = ms // 1000; cs = (ms % 1000) // 10
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,54,&H00FFFFFF,&H00000000,&H90000000,1,3,1,2,80,80,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(events: list[dict], t0_ms: int, t1_ms: int) -> str:
    win = [e for e in events if e["start_ms"] < t1_ms and e["end_ms"] > t0_ms]
    win.sort(key=lambda e: e["start_ms"])
    lines = [HEADER]
    for e in win:
        a = max(e["start_ms"], t0_ms) - t0_ms
        b = min(e["end_ms"], t1_ms) - t0_ms
        if b <= a:
            continue
        dur = e["end_ms"] - e["start_ms"]
        # overlapping different-track event with longer duration -> this is secondary
        secondary = any(
            f is not e and f["speaker_track_id"] != e["speaker_track_id"]
            and f["start_ms"] < e["end_ms"] and f["end_ms"] > e["start_ms"]
            and (f["end_ms"] - f["start_ms"]) > dur
            for f in win)
        an = 8 if secondary else 2
        mv = 900 if secondary else 90
        col = _color(e)
        txt = e["text"].replace("{", "(").replace("}", ")")
        lines.append(
            f"Dialogue: {1 if secondary else 0},{_ms_to_ass(a)},{_ms_to_ass(b)},"
            f"Default,,0,0,{mv},,{{\\an{an}\\c&H{col}&}}{txt}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--events-dir", default="speech_events_v2")
    a = ap.parse_args()

    events = _load_events(a.mid, a.events_dir)
    t0, t1 = int(a.start * 1000), int(a.end * 1000)
    ass = build_ass(events, t0, t1)
    src = _source(a.mid)
    if not src:
        raise SystemExit(f"source not found for {a.mid}")

    tmp_ass = Path(tempfile.mktemp(suffix=".ass", dir="/tmp/claude-0"))
    tmp_ass.write_text(ass)
    dur = a.end - a.start
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
           "-ss", f"{a.start:.3f}", "-t", f"{dur:.3f}", "-i", src,
           "-vf", f"subtitles={tmp_ass}", "-c:a", "aac", "-c:v", "libx264",
           "-preset", "fast", "-crf", "20", a.out]
    subprocess.run(cmd, check=True)
    n = sum(1 for l in ass.splitlines() if l.startswith("Dialogue"))
    print(f"rendered {a.out}  ({n} cue lines, window {a.start}-{a.end}s)")
    print(ass)


if __name__ == "__main__":
    main()

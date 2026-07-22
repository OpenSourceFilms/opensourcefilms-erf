#!/usr/bin/env python3
"""QC — objective checks on produced shorts.

For each catalog/shorts/*.json (the edit spec + mp4), verify the rendered short is
viable: duration in range, correct 1080x1920, loudness near target, file exists.
Writes catalog/shorts/qc_report.json and prints a table. This is the gate on the
ACTUAL artifact (not upstream scores).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"


def _probe(mp4: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height:format=duration",
                        "-of", "json", str(mp4)], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
        s = d["streams"][0]
        return {"w": s.get("width"), "h": s.get("height"),
                "dur": float(d["format"]["duration"])}
    except Exception:
        return {}


def _lufs(mp4: Path):
    r = subprocess.run(["ffmpeg", "-i", str(mp4), "-af", "ebur128", "-f", "null", "-",
                        "-hide_banner"], capture_output=True, text=True)
    m = re.findall(r"I:\s*(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def qc_one(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    mp4 = Path(spec.get("mp4") or "")
    if not mp4.is_absolute():
        mp4 = ROOT / mp4
    title = (spec.get("edit") or {}).get("title")
    res = {"media_id": spec.get("media_id"), "title": title, "mp4": mp4.name}
    if not mp4.exists():
        res["verdict"] = "MISSING"
        return res
    p = _probe(mp4)
    lufs = _lufs(mp4)
    res.update({"dur": round(p.get("dur", 0), 1), "res": f"{p.get('w')}x{p.get('h')}", "lufs": lufs})
    issues = []
    if not (8 <= p.get("dur", 0) <= 60):
        issues.append(f"dur={p.get('dur')}")
    if (p.get("w"), p.get("h")) != (1080, 1920):
        issues.append(f"res={p.get('w')}x{p.get('h')}")
    if lufs is not None and not (-20 <= lufs <= -12):
        issues.append(f"lufs={lufs}")
    res["verdict"] = "OK" if not issues else "CHECK:" + ",".join(issues)
    return res


def main() -> None:
    specs = sorted((CAT / "shorts").glob("short_*.json"))
    rows = [qc_one(s) for s in specs]
    ok = sum(1 for r in rows if r.get("verdict") == "OK")
    print(f"QC {len(rows)} shorts — {ok} OK\n")
    for r in sorted(rows, key=lambda x: x.get("verdict", "")):
        print(f"  [{r.get('verdict'):<22}] {r.get('dur','?')}s {r.get('res','?')} "
              f"{r.get('lufs','?')}LUFS  '{r.get('title')}'")
    (CAT / "shorts" / "qc_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

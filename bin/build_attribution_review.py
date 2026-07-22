#!/usr/bin/env python3
"""Build a human-review pack for unverified attribution golden candidates.

Emits state/eval/attribution_review/review.md — one section per unverified
window with text, proposed expectation, provenance, a mid-window frame
(extracted from the proxy), and a ready-to-run ffplay command for listening.
Review workflow: play the clip, then edit attribution_candidates.jsonl —
set verified=true (correct), fix `expected` then verified=true, or delete
the line. Re-run eval_attribution.py --set-baseline after freezing.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "state/eval/attribution_candidates.jsonl"
OUT_DIR = ROOT / "state/eval/attribution_review"
PROXIES = ROOT / "media/proxies"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "frames").mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(CAND, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if not r.get("verified")]
    lines = ["# Attribution golden set — review pack",
             f"\n{len(todo)} unverified windows (of {len(rows)} candidates)."
             " For each: listen with the ffplay command, check the proposed"
             " speaker(s), then in attribution_candidates.jsonl set"
             " verified=true / fix expected / delete the line.\n"]
    n_frames = 0
    for r in sorted(todo, key=lambda x: (x["stratum"], x["media_id"], x["start_ms"])):
        mid, s, e = r["media_id"], r["start_ms"], r["end_ms"]
        gid = r["golden_id"]
        mp4 = PROXIES / f"{mid}.mp4"
        frame = OUT_DIR / "frames" / f"{gid}.jpg"
        if mp4.exists() and not frame.exists():
            mt = (s + e) / 2000.0
            p = subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{mt:.2f}", "-i", str(mp4),
                 "-frames:v", "1", "-q:v", "4", str(frame)],
                capture_output=True)
            if p.returncode == 0:
                n_frames += 1
        pad_s = max(0.0, s / 1000.0 - 1.0)
        dur = (e - s) / 1000.0 + 2.0
        lines += [
            f"\n## {gid}  [{r['stratum']}]  {mid}  {s / 1000:.2f}–{e / 1000:.2f}s",
            f"- text: `{r.get('text_preview', '')}`",
            f"- proposed: `{json.dumps(r['expected'])}`",
            f"- provenance: `{json.dumps(r.get('provenance', {}))}`",
            f"- listen: `ffplay -ss {pad_s:.2f} -t {dur:.2f} {mp4}`",
        ]
        if frame.exists():
            lines.append(f"- frame: ![{gid}](frames/{gid}.jpg)")
    out = OUT_DIR / "review.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"review pack: {out}  windows={len(todo)} frames={n_frames}")


if __name__ == "__main__":
    main()

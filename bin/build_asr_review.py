#!/usr/bin/env python3
"""Axis-1 (text-fidelity) review pack for the ASR bake-off — D-Log-39 step 5 + D-Log-41 decision 6.

Per golden window on the 3 golden-bearing probe media: a numbered audio clip, each arm's RAW
ASR text (words.jsonl — NOT the recall-homogenised T0 transcript, which is identical-ish across
arms), audio stats (RMS/peak dBFS), and a gate-empty marker per arm (governed Axis-2 score == 0
on that window, i.e. the label-tied rejection D-Log-41 found). Plus the two text-canary media
(clean-audio + near-inaudible) as whole-media raw transcripts per arm, for clean-text and
hallucination judgment. Emits review.md + VERDICTS.txt skeleton. Read-only wrt canon; writes
only under state/eval/asr_review/. A human reviewer adjudicates; no adoption action here (T-ERF-09).
"""
import json
import re
import subprocess
from pathlib import Path

ERF = Path(__file__).resolve().parent.parent
ARMS = ["control", "canary", "parakeet", "voxtral"]
GOLDEN = ERF / "state/eval/attribution_golden.jsonl"
PROXIES = ERF / "media/proxies"
OUT = ERF / "state/eval/asr_review"
CLIPS = OUT / "clips"
GOLD_MEDIA = {"m_5a8bedaa054a", "m_e1b299626700", "m_0e36ebd54774"}
CANARY_MEDIA = ["m_9caf6ce31df4", "m_01d094906b30"]  # 0 golden windows; Axis-1 only


def governs(w, a):
    if "stream" in a and w.get("stream", "A") != a["stream"]:
        return False
    if "speaker_label" in a and w.get("speaker_label") != a["speaker_label"]:
        return False
    return True


def load(arm, mid, layer):
    p = ERF / f"state/bakeoff/asr_shadow/{arm}/catalog/asr/{mid}.{layer}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] \
        if p.exists() and p.stat().st_size else []


def wtext(words, s, e):
    ws = [w for w in words if w["start_ms"] < e and w["end_ms"] > s]
    return " ".join(w["text"] for w in ws), len(ws)


def audio_stats(mid, s, e):
    mp4 = PROXIES / f"{mid}.mp4"
    if not mp4.exists():
        return "n/a"
    ss = max(0.0, s / 1000.0)
    args = ["ffmpeg", "-hide_banner", "-ss", f"{ss:.2f}"]
    if e is not None:
        args += ["-t", f"{max(0.1, (e - s) / 1000.0):.2f}"]
    args += ["-i", str(mp4), "-af", "volumedetect", "-f", "null", "-"]
    p = subprocess.run(args, capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*([-\d.]+)", p.stderr)
    x = re.search(r"max_volume:\s*([-\d.]+)", p.stderr)
    return f"RMS {m.group(1) if m else '?'} / peak {x.group(1) if x else '?'} dBFS"


def clip(mid, s, e, name):
    mp4 = PROXIES / f"{mid}.mp4"
    out = CLIPS / name
    if not mp4.exists() or out.exists():
        return out.exists()
    ss = max(0.0, s / 1000.0 - 1.0)
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{ss:.2f}"]
    if e is not None:
        args += ["-t", f"{(e - s) / 1000.0 + 2.0:.2f}"]
    args += ["-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000", str(out)]
    return subprocess.run(args, capture_output=True).returncode == 0


golden = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]
G = sorted([g for g in golden if g.get("verified") and g["media_id"] in GOLD_MEDIA],
           key=lambda x: (x["media_id"], x["start_ms"]))

raw = {arm: {} for arm in ARMS}
t0 = {arm: {} for arm in ARMS}
for arm in ARMS:
    for mid in GOLD_MEDIA | set(CANARY_MEDIA):
        raw[arm][mid] = load(arm, mid, "words")
        t0[arm][mid] = load(arm, mid, "transcript")

CLIPS.mkdir(parents=True, exist_ok=True)
L = ["# ASR bake-off — Axis-1 (text-fidelity) review pack",
     "",
     "The reviewer's ear adjudicates which arm's TEXT is most accurate, and flags hallucinations.",
     "Each arm's line is its RAW ASR output (what the engine transcribed), not the recall-filled",
     "final transcript. `gate-empty arms` = the Axis-2 gate scored 0 words there (usually a",
     "label-tied rejection per D-Log-41, NOT missing words — check the text). Voxtral abstained",
     "(0 raw words) on m_5a8bedaa054a — visible below. Fill verdicts in VERDICTS.txt.",
     ""]
V = ["# ASR bake-off Axis-1 verdicts — fill best_text (control/canary/parakeet/voxtral or 'tie'),",
     "# hallucination (arms that invented words, comma-sep or 'none'), free-text notes.",
     ""]

n = 0
for g in G:
    n += 1
    mid, s, e, A = g["media_id"], g["start_ms"], g["end_ms"], g["expected"]
    gid, strat = g["golden_id"], g["stratum"]
    name = f"{n:02d}_{gid}.mp3"
    clip(mid, s, e, name)
    empties = []
    for arm in ARMS:
        ww = [w for w in t0[arm][mid] if w["start_ms"] < e and w["end_ms"] > s]
        if sum(1 for w in ww if any(governs(w, a) for a in A)) == 0:
            empties.append(arm)
    ff = f"ffplay -autoexit -ss {max(0.0, s / 1000 - 1):.1f} -t {(e - s) / 1000 + 2:.1f} {PROXIES / (mid + '.mp4')}"
    L += [f"## [{n:02d}] {gid} [{strat}] {mid} {s / 1000:.1f}-{e / 1000:.1f}s",
          f"- audio: {audio_stats(mid, s, e)}",
          f"- clip: `clips/{name}`  ·  or `{ff}`",
          f"- gate-empty arms: {', '.join(empties) if empties else 'none'}"]
    for arm in ARMS:
        txt, cnt = wtext(raw[arm][mid], s, e)
        L.append(f"- **{arm}** ({cnt}w): {txt or '_(no raw ASR words)_'}")
    L.append("")
    V.append(f"[{n:02d}] {gid} {strat} {mid}: best_text=____  hallucination=____  notes=____")

for mid in CANARY_MEDIA:
    n += 1
    name = f"{n:02d}_canary_{mid}.mp3"
    clip(mid, 0, None, name)
    L += [f"## [{n:02d}] text-canary {mid} (whole media, 0 golden windows)",
          f"- audio: {audio_stats(mid, 0, None)}",
          f"- clip: `clips/{name}`  ·  or `ffplay -autoexit {PROXIES / (mid + '.mp4')}`"]
    for arm in ARMS:
        words = raw[arm][mid]
        txt = " ".join(w["text"] for w in words)
        L.append(f"- **{arm}** ({len(words)}w): {txt or '_(no raw ASR words)_'}")
    L.append("")
    V.append(f"[{n:02d}] text-canary {mid}: best_text=____  hallucination=____  notes=____")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "review.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(OUT / "VERDICTS.txt").write_text("\n".join(V) + "\n", encoding="utf-8")
print(f"wrote {OUT / 'review.md'} and VERDICTS.txt  ({n} pack items, {len(G)} windows + {len(CANARY_MEDIA)} canaries)")

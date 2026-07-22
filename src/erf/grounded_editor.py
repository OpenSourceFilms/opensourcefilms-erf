"""Grounded editor — DeepSeek Pro as the editorial brain, reading real evidence.

Replaces the old scan(HDBSCAN)+Pass1(brief)+Pass2(slot-fill)+ideate chain. The
editor reads an EVIDENCE PACKAGE (the actual sentence-segmented, speaker-attributed,
timed transcript fused with visual grounding, quality, identity, signals) and
selects the strongest self-contained moment for a vertical marketing short.

Grounding guarantee: the LLM chooses by SENTENCE INDEX; concrete in/out timestamps
are taken from the evidence (ground_edit), never from the model — so it cannot
hallucinate timings or invent footage that doesn't exist.
"""
from __future__ import annotations

import json
import logging
import re

from .llm import complete

log = logging.getLogger("grounded_editor")

_SYSTEM = """\
You are a world-class short-form video editor. You cut vertical (9:16) social clips
whose single job is to MARKET A FILM — make a stranger scrolling stop, watch, and
want to see more of this project and these people.

What makes a great short here:
- A hook in the first ~2 seconds: a bold claim, a funny line, a vivid image, tension.
- One clean, self-contained idea or moment with a satisfying mini-arc (setup → payoff).
- A quotable, human, shareable line. Personality over information.
- It must stand alone with zero outside context.

You are given the FULL evidence for one source clip: the real transcript split into
numbered sentences with timestamps and per-sentence signals (emotional beat, hook and
quote potential), plus what the footage looks like, who is in it, and audio quality.

Select the SINGLE strongest self-contained moment, roughly 15–45 seconds, by choosing a
contiguous (or tightly adjacent) range of sentences. Prefer moments with a real hook and
a quotable payoff. Avoid sentences that trail into crosstalk, dead air, or unintelligible
audio. Do not stitch unrelated topics together.

Return ONLY strict JSON, no markdown:
{
  "title": "short punchy title",
  "logline": "one sentence on what this short delivers",
  "hook_line": "the exact opening sentence text you will lead with",
  "rationale": "why this works as a scroll-stopping marketing short",
  "target_duration_s": <int>,
  "beats": [
    {"sentence_start_index": <int>, "sentence_end_index": <int>,
     "role": "hook" | "body" | "button",
     "caption_emphasis": ["word", ...],
     "broll_intent": "<what a cutaway here should show, or null>"}
  ]
}
Beats reference sentence indices from the evidence. Keep beats in playback order.
If nothing in this clip works as a standalone short, return {"beats": []} with a rationale.\
"""


def format_evidence(pkg: dict) -> str:
    people = ", ".join(f"{p['name']}" for p in pkg.get("people", [])) or "unknown"
    avq = pkg.get("av_quality", {})
    dur = (pkg.get("duration_ms") or 0) / 1000.0
    head = [
        f"CLIP {pkg.get('media_id')} | footage={pkg.get('footage_type')} | setting={pkg.get('setting')}",
        f"people={people} | unidentified={pkg.get('unidentified_speakers')} | duration={dur:.0f}s",
        f"synopsis: {pkg.get('synopsis')}",
        f"looks like: {pkg.get('visual_summary')}",
        f"audio: speech_clarity={avq.get('speech_clarity_median')} top_class={avq.get('audio_top_class')} clipping={avq.get('clipping_max')}",
        "",
        "SENTENCES (idx [start_ms-end_ms] speaker: text  | signals):",
    ]
    lines = []
    for s in pkg.get("sentences", []):
        lines.append(
            f"  {s['sentence_index']} [{s['start_ms']}-{s['end_ms']}] {s.get('speaker')}: "
            f"\"{s['text']}\"  | emo={s.get('emotional_beat')} hook={s.get('hook')} quote={s.get('quote')}"
            + ("  [b-roll moment]" if s.get("b_roll_flag") else "")
        )
    sil = pkg.get("silences", [])
    ov = pkg.get("overlaps", [])
    tail = [
        "",
        f"SILENCES (clean cut points), ms: {[(s['start_ms'], s['end_ms']) for s in sil][:20]}",
        f"OVERLAPS (crosstalk — avoid cutting inside), ms: {[(o['start_ms'], o['end_ms']) for o in ov][:20]}",
    ]
    return "\n".join(head + lines + tail)


def _parse_json(raw: str) -> dict:
    if isinstance(raw, dict):
        return raw
    text = (raw or "").strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # salvage the first {...} block
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


def edit_single(pkg: dict, *, target_s: int = 30, llm=None) -> dict:
    """Ask the editor to select the strongest moment. Returns the raw edit dict."""
    llm = llm or complete
    user = (
        f"Target duration: about {target_s} seconds.\n\n"
        + format_evidence(pkg)
        + "\n\nReturn ONLY the JSON edit object."
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
    raw = llm(messages, json_only=True)
    edit = _parse_json(raw)
    return edit


_SURVEY_SYSTEM = """\
You are the commissioning editor for a film's social channel. You receive a CATALOG of
every source clip we have — each with what it is, who's in it, its mood, and its 2-3
strongest candidate lines (with scores). Your job: pick the moments most likely to make
a great vertical short that MARKETS THE FILM — scroll-stopping, funny or striking or
human, quotable, shareable. Pick the strongest moment wherever it lives (an interview
confession, an on-set joke, a candid reaction).

Return ONLY strict JSON:
{"picks": [
  {"media_id": "...", "why": "what makes this a scroll-stopper", "angle": "the short's hook/angle",
   "strength": <0-1>}
]}
Rank picks strongest first. Only include moments you genuinely believe could perform.\
"""


def condense(pkg: dict, top_k: int = 3) -> dict:
    """Condensed corpus-survey entry for one media."""
    sents = sorted(pkg.get("sentences", []),
                   key=lambda s: (s.get("hook", 0) + s.get("quote", 0)), reverse=True)
    top = [{"start_ms": s["start_ms"], "text": s["text"][:160],
            "hook": s.get("hook"), "quote": s.get("quote")} for s in sents[:top_k]]
    return {
        "media_id": pkg.get("media_id"),
        "footage_type": pkg.get("footage_type"),
        "duration_s": round((pkg.get("duration_ms") or 0) / 1000.0),
        "people": [p["name"] for p in pkg.get("people", [])] or pkg.get("unidentified_speakers", []),
        "emotional_tenor": pkg.get("emotional_tenor"),
        "synopsis": pkg.get("synopsis"),
        "speech_clarity": (pkg.get("av_quality") or {}).get("speech_clarity_median"),
        "top_moments": top,
    }


def format_survey(entries: list[dict]) -> str:
    lines = ["CATALOG (one block per clip):"]
    for e in entries:
        lines.append(
            f"- {e['media_id']} | {e['footage_type']} | {e['duration_s']}s | people={e['people']} "
            f"| mood={e['emotional_tenor']} | clarity={e['speech_clarity']}\n"
            f"    synopsis: {e['synopsis']}\n"
            f"    moments: " + " || ".join(
                f"[{m['start_ms']}ms h{m['hook']}/q{m['quote']}] \"{m['text']}\"" for m in e['top_moments'])
        )
    return "\n".join(lines)


def survey_select(entries: list[dict], *, n: int = 12, llm=None) -> list[dict]:
    """Ask DS Pro to pick the strongest moments across the whole corpus."""
    llm = llm or complete
    user = format_survey(entries) + f"\n\nPick the {n} strongest. Return ONLY the JSON."
    messages = [{"role": "system", "content": _SURVEY_SYSTEM}, {"role": "user", "content": user}]
    out = _parse_json(llm(messages, json_only=True))
    return out.get("picks", [])


_STORY_IDEATE_SYSTEM = """\
You are the showrunner for a film's social channel. From the WHOLE corpus below (every clip
we have — interviews and on-set moments), find self-contained STORIES worth telling as vertical
shorts that market the film. A story is a through-line with an arc, not just one funny line.

Nothing is siloed: a single story may draw moments from MANY different clips (e.g. an interview
sets up an idea, on-set footage pays it off; two people across two clips form a back-and-forth).
It may also live in a single clip when that's genuinely the best telling.

Give a MIX: some short & punchy (1 clip, ~15-25s), some multi-part narratives (2-5 clips, ~35-90s).

Return ONLY strict JSON:
{"stories": [
  {"title": "...", "logline": "the arc in one sentence", "kind": "punchy" | "narrative",
   "media_ids": ["m_...", ...], "through_line": "what connects these moments",
   "est_duration_s": <int>, "strength": <0-1>}
]}
Rank strongest first. media_ids must come from the corpus below.\
"""

_STORY_BUILD_SYSTEM = """\
You are editing ONE self-contained narrative short for a film's social channel. You have a story
concept and the FULL sentence-level evidence for each relevant clip. Build the edit as an ordered
sequence of beats that tells the story with a real arc: a hook that grabs in 2s, development, and a
payoff/button. You may cut between clips and speakers freely to build the narrative — that's
encouraged (multiple angles, call-and-response, setup in one clip and payoff in another).

Each beat selects a contiguous sentence range from ONE clip. Order beats for the BEST storytelling,
not source order. Keep it tight — every beat must earn its place. Avoid crosstalk/dead-air sentences.

Return ONLY strict JSON:
{"title": "...", "logline": "...", "hook_line": "...", "rationale": "why this lands as a story",
 "beats": [
   {"media_id": "m_...", "sentence_start_index": <int>, "sentence_end_index": <int>,
    "role": "hook" | "setup" | "development" | "turn" | "payoff" | "button",
    "caption_emphasis": ["word", ...]}
 ]}
Beats reference sentence indices from the named clip's evidence. If the concept doesn't hold
together, return {"beats": []} with a rationale.\
"""


def story_ideas(entries: list[dict], *, n: int = 8, llm=None) -> list[dict]:
    """Propose cross-media story concepts over the whole corpus."""
    llm = llm or complete
    user = format_survey(entries) + f"\n\nPropose the {n} strongest stories (mix punchy + narrative). Return ONLY JSON."
    messages = [{"role": "system", "content": _STORY_IDEATE_SYSTEM}, {"role": "user", "content": user}]
    return _parse_json(llm(messages, json_only=True)).get("stories", [])


def build_story(idea: dict, packages: dict[str, dict], *, llm=None) -> dict:
    """Build a grounded cross-media beat plan for one story concept.
    `packages` maps media_id -> full evidence package."""
    llm = llm or complete
    blocks = []
    for mid in idea.get("media_ids", []):
        pkg = packages.get(mid)
        if pkg:
            blocks.append(format_evidence(pkg))
    user = (f"STORY CONCEPT: {idea.get('title')} — {idea.get('logline')}\n"
            f"through-line: {idea.get('through_line')}\n"
            f"target ~{idea.get('est_duration_s', 45)}s\n\n"
            + "\n\n========\n\n".join(blocks)
            + "\n\nBuild the narrative. Return ONLY the JSON edit object.")
    messages = [{"role": "system", "content": _STORY_BUILD_SYSTEM}, {"role": "user", "content": user}]
    edit = _parse_json(llm(messages, json_only=True))
    return ground_story(packages, edit)


def _dedup_overlapping_beats(beats: list[dict]) -> list[dict]:
    """Drop beats whose source window substantially overlaps an already-kept
    same-media window (>50% of the shorter window). A degenerate edit — e.g. a
    'comedic loop' concept over a 4-sentence clip — can emit dozens of beats that
    all ground onto the same 1-2 windows; the renderer already drops these, but if
    grounding keeps them the stored story artifact lies about its structure and
    total_duration_s (observed: 43 beats / 23.4s claimed vs 5.9s actual). Dedup
    here so the artifact equals what renders. Order-preserving (keeps first)."""
    kept: list[dict] = []
    for b in beats:
        dup = False
        for k in kept:
            if k["media_id"] != b["media_id"]:
                continue
            ov = min(b["out_ms"], k["out_ms"]) - max(b["in_ms"], k["in_ms"])
            if ov > 0 and ov > 0.5 * min(b["out_ms"] - b["in_ms"], k["out_ms"] - k["in_ms"]):
                dup = True
                break
        if not dup:
            kept.append(b)
    if len(kept) < len(beats):
        log.warning("ground: collapsed %d beats -> %d distinct windows (degenerate edit)",
                    len(beats), len(kept))
    return kept


def ground_story(packages: dict[str, dict], edit: dict) -> dict:
    """Ground cross-media beats: in/out ms come from each clip's own evidence sentences."""
    beats = []
    for b in edit.get("beats", []):
        mid = b.get("media_id")
        pkg = packages.get(mid)
        if not pkg:
            log.warning("story beat references unknown media %s — drop", mid)
            continue
        sents = pkg.get("sentences", [])
        n = len(sents)
        try:
            i0 = int(b["sentence_start_index"])
            i1 = int(b["sentence_end_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if i0 > i1:
            i0, i1 = i1, i0
        if i0 < 0 or i1 >= n:
            log.warning("[%s] story beat out of range [%d,%d] n=%d — drop", mid, i0, i1, n)
            continue
        in_ms, out_ms = sents[i0]["start_ms"], sents[i1]["end_ms"]
        if out_ms <= in_ms:
            continue
        beats.append({
            "media_id": mid, "chunk_id": f"{mid}:c:{in_ms}-{out_ms}",
            "in_ms": in_ms, "out_ms": out_ms, "role": b.get("role", "development"),
            "caption_emphasis": b.get("caption_emphasis", []),
            "text": " ".join(sents[i]["text"] for i in range(i0, i1 + 1)),
        })
    beats = _dedup_overlapping_beats(beats)
    if beats:
        beats[-1]["out_ms"] += 400
    total = sum(b["out_ms"] - b["in_ms"] for b in beats)
    return {"title": edit.get("title"), "logline": edit.get("logline"),
            "hook_line": edit.get("hook_line"), "rationale": edit.get("rationale"),
            "total_duration_s": round(total / 1000.0, 1), "beats": beats}


def ground_edit(pkg: dict, edit: dict) -> dict:
    """Convert sentence-index beats to GROUNDED ms taken from the evidence itself."""
    sents = pkg.get("sentences", [])
    n = len(sents)
    media_id = pkg.get("media_id")
    beats = []
    for b in edit.get("beats", []):
        try:
            i0 = int(b["sentence_start_index"])
            i1 = int(b["sentence_end_index"])
        except (KeyError, TypeError, ValueError):
            log.warning("[%s] beat missing sentence indices: %s", media_id, b)
            continue
        if i0 > i1:
            i0, i1 = i1, i0
        if i0 < 0 or i1 >= n:
            log.warning("[%s] beat indices out of range [%d,%d] (n=%d) — drop", media_id, i0, i1, n)
            continue
        in_ms = sents[i0]["start_ms"]
        out_ms = sents[i1]["end_ms"]
        if out_ms <= in_ms:
            continue
        text = " ".join(sents[i]["text"] for i in range(i0, i1 + 1))
        beats.append({
            "media_id": media_id,
            "chunk_id": f"{media_id}:c:{in_ms}-{out_ms}",
            "in_ms": in_ms,
            "out_ms": out_ms,
            "role": b.get("role", "body"),
            "caption_emphasis": b.get("caption_emphasis", []),
            "broll_intent": b.get("broll_intent"),
            "text": text,
        })
    beats = _dedup_overlapping_beats(beats)
    # Tail pad on the final beat: Whisper/wx word-end timestamps are tight and the
    # renderer's VAD snap can pull the cut back INTO the closing word, making shorts
    # feel like they stop mid-sentence. Give the last word room to finish + a breath.
    if beats:
        beats[-1]["out_ms"] += 400

    total = sum(b["out_ms"] - b["in_ms"] for b in beats)
    return {
        "title": edit.get("title"),
        "logline": edit.get("logline"),
        "hook_line": edit.get("hook_line"),
        "rationale": edit.get("rationale"),
        "total_duration_s": round(total / 1000.0, 1),
        "beats": beats,
    }

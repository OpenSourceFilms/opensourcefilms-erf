#!/usr/bin/env python3
"""Story Engine Phase 3 vertical slice — chain completion + adapt + render.

Selects the richest editorial-labeled clips, calls DS Pro in thinking mode,
parses candidates, validates (C2), runs each through se_adapter, and renders.

Usage:
    PYTHONPATH=src DEEPSEEK_API_KEY=... venvs/embed/bin/python bin/se_chain.py \
        [--n-clips 8] [--lane single|mini|long|all] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"
sys.path.insert(0, str(ROOT / "src"))

from erf.llm import complete as llm_complete
from erf.se_adapter import adapt as se_adapt
from erf.render import render as do_render
from erf import runlog

log = logging.getLogger("se_chain")

_LANE_BANDS_S = {
    "single_clip_15_25s": (15, 25),
    "mini_cross_media_30_60s": (30, 60),
    "long_cross_media_60_120s": (60, 120),
}

_CANDIDATE_SCHEMA = """\
{
  "candidate_id": "<unique string>",
  "source_type": "pressure|viewer_question|payoff",
  "primary_format_lane": "single_clip_15_25s|mini_cross_media_30_60s|long_cross_media_60_120s",
  "verdict": "complete|reject|partial_open_thread",
  "title": "<punchy title>",
  "central_question": "<the viewer question this short answers>",
  "specific_tension": "<the tension driving this story>",
  "archetype": "<one archetype>",
  "confidence_tier": "pro_scored",
  "known_weakness": "<honest weak point>",
  "beats": [
    {
      "role": "HOOK|WANT|OBSTACLE|ATTEMPT|DECISION|TURN|RESULT|PAYOFF|BUTTON",
      "clip_id": "<one of the given media_ids>",
      "sentence_range": [start_idx, end_idx],
      "evidence_ids": ["<clip_id>.s<idx>" for idx in range(start_idx, end_idx+1)],
      "why_this_beat": "<one sentence rationale>",
      "audio_mode": "sync_dialogue|broll_audio|music_only|silent_visual",
      "sync_audio_required": true_if_sync_dialogue_else_false,
      "dialogue_required": true_if_sync_dialogue_else_false,
      "visual_payoff_required": false,
      "payoff_type": "none|visual_proof|funny_button|technical_reveal|human_reaction|failure_that_teaches|before_after|open_loop"
    }
  ],
  "missing_beats": [],
  "future_footage_needed": []
}"""


def _load_editorial(media_id: str) -> dict | None:
    manifest = CAT / "editorial" / media_id / "manifest.json"
    if not manifest.exists():
        return None
    try:
        m = json.loads(manifest.read_text())
        job_id = m.get("latest_job_id")
        if not job_id:
            return None
        return json.loads((CAT / "editorial" / media_id / f"{job_id}.json").read_text())
    except Exception:
        return None


def select_top_clips(n: int = 8, lane: str = "all") -> list[dict]:
    scored = []
    ev_dir = CAT / "evidence"
    for ev_path in ev_dir.glob("*.json"):
        media_id = ev_path.stem
        ed = _load_editorial(media_id)
        if not ed:
            continue
        cl = ed.get("clip_level", {})
        forces = cl.get("story_forces", {})
        archetypes = cl.get("archetype_candidates", [])
        tension = cl.get("central_tension")
        score = sum(1 for v in forces.values() if v) + len(archetypes) + (1 if tension else 0)
        scored.append({"media_id": media_id, "score": score, "cl": cl, "ed_sentences": ed.get("sentences", [])})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n]


def load_clip_data(clip: dict) -> dict | None:
    mid = clip["media_id"]
    ev_path = CAT / "evidence" / f"{mid}.json"
    if not ev_path.exists():
        return None
    ev = json.loads(ev_path.read_text())
    # Build sentence_index → editorial-signals lookup
    ed_map = {s["sentence_index"]: s for s in clip["ed_sentences"] if "sentence_index" in s}
    merged_sents = []
    for s in ev.get("sentences", []):
        idx = s.get("sentence_index", 0)
        ed_s = ed_map.get(idx, {})
        ca = ed_s.get("chain_affordances", {})
        merged_sents.append({
            "idx": idx,
            "start_ms": s.get("start_ms", 0),
            "end_ms": s.get("end_ms", 0),
            "speaker": s.get("speaker", "?"),
            "text": s.get("text", ""),
            "functions": ed_s.get("functions", []),
            "tension_tags": ed_s.get("tension_tags", []),
            "can_start": ca.get("can_start_story", False),
            "can_payoff": ca.get("can_payoff_story", False),
        })
    cl = clip["cl"]
    return {
        "media_id": mid,
        "synopsis": ev.get("synopsis", ""),
        "duration_ms": ev.get("duration_ms", 0),
        "story_forces": cl.get("story_forces", {}),
        "central_tension": cl.get("central_tension"),
        "archetypes": cl.get("archetype_candidates", []),
        "sentences": merged_sents,
    }


def _format_clip(cd: dict, max_sents: int = 60) -> str:
    sf = cd["story_forces"]
    lines = [
        f"CLIP {cd['media_id']} | {cd['synopsis'][:120]} | dur={cd['duration_ms']//1000}s"
        f" | tension={cd['central_tension']}",
        f"want: {sf.get('want','')}",
        f"obstacle: {sf.get('obstacle','')}",
        f"payoff: {sf.get('payoff','')}",
        f"archetypes: {cd['archetypes']}",
        "SENTENCES (idx [start_ms-end_ms] speaker: text | fn | can_start can_payoff):",
    ]
    sents = cd["sentences"]
    if len(sents) > max_sents:
        # Show sentences with functions or chain_affordances
        sents = [s for s in sents if s["functions"] or s["can_start"] or s["can_payoff"]] or sents[:max_sents]
    for s in sents:
        fn = ",".join(s["functions"]) if s["functions"] else "-"
        tt = ",".join(s["tension_tags"]) if s["tension_tags"] else ""
        lines.append(
            f'  {s["idx"]} [{s["start_ms"]}-{s["end_ms"]}ms] {s["speaker"]}: '
            f'"{s["text"]}" | {fn}{(" |"+tt) if tt else ""}'
            f'{" START" if s["can_start"] else ""}{" PAYOFF" if s["can_payoff"] else ""}'
        )
    return "\n".join(lines)


def build_prompt(clips: list[dict], lane: str) -> str:
    clip_ids = [c["media_id"] for c in clips]
    lane_guidance = {
        "single_clip_15_25s": "One clip per candidate, beats span 15–25s of sentences total.",
        "mini_cross_media_30_60s": "2–4 different clips, 30–60s total.",
        "long_cross_media_60_120s": "3+ clips, 60–120s total.",
        "all": "Mix all three lanes: produce at least 1 candidate per lane.",
    }.get(lane, "Mix all three lanes.")

    clips_text = "\n\n".join(_format_clip(c) for c in clips)

    return f"""\
You are a world-class short-form editor structuring BTS footage for Life®, an indie documentary.
Your goal: produce 3–5 candidates for vertical social shorts (9:16) that a filmmaker will actually watch and feel.

LANE GUIDANCE: {lane_guidance}

VALID clip_ids (ONLY these): {clip_ids}

CANDIDATE SCHEMA (return a strict JSON array of these, no prose):
{_CANDIDATE_SCHEMA}

CONSTRAINTS (violation = rejection):
- clip_id must be exactly one of the valid clip_ids above
- sentence_range: [start_idx, end_idx] inclusive; both indices must exist in the evidence below
- evidence_ids: ["{'{clip_id}'}.s{'{idx}'}" for idx in range(start_idx, end_idx+1)]
- confidence_tier: MUST be "pro_scored" (this is a Pro thinking call)
- audio_mode=sync_dialogue → sync_audio_required=true AND dialogue_required=true
- audio_mode≠sync_dialogue → sync_audio_required=false, dialogue_required=false
- single_clip_15_25s: exactly 1 distinct clip_id across all beats, total sentence time 15–25s
- mini_cross_media_30_60s: 2–4 distinct clip_ids, total 30–60s
- long_cross_media_60_120s: 3+ clip_ids, total 60–120s
- Every complete candidate must have a real HOOK beat and a real PAYOFF beat
- verdict=complete only when all story beats are present and all evidence exists
- Never invent timestamps or clip_ids — only reference what's below

EVIDENCE:
{clips_text}

Return ONLY a JSON array. No markdown, no prose."""


def validate_c2(cand: dict, valid_ids: set, ev_idx: dict[str, dict]) -> list[str]:
    errs = []
    for f in ("candidate_id", "verdict", "primary_format_lane", "confidence_tier"):
        if not cand.get(f):
            errs.append(f"missing {f}")
    verdict = cand.get("verdict")
    if verdict not in ("complete", "reject", "partial_open_thread"):
        errs.append(f"bad verdict: {verdict}")
    tier = cand.get("confidence_tier")
    if tier not in ("pro_scored", "pro_scored_tournament_ranked"):
        errs.append(f"bad confidence_tier: {tier}")
    lane = cand.get("primary_format_lane")
    if lane not in _LANE_BANDS_S:
        errs.append(f"bad lane: {lane}")
    if verdict != "complete":
        return errs
    beats = cand.get("beats") or []
    if not beats:
        errs.append("complete but no beats")
        return errs
    lo, hi = _LANE_BANDS_S.get(lane, (0, 9999))
    total_s = 0.0
    clip_ids_used: set = set()
    for i, b in enumerate(beats):
        cid = b.get("clip_id")
        if not cid or cid not in valid_ids:
            errs.append(f"beat {i}: bad clip_id {cid!r}")
            continue
        clip_ids_used.add(cid)
        sr = b.get("sentence_range")
        if not (isinstance(sr, list) and len(sr) == 2 and all(isinstance(x, int) for x in sr)):
            errs.append(f"beat {i}: sentence_range must be [int,int]")
            continue
        a, z = sr
        if a > z:
            errs.append(f"beat {i}: sentence_range[0] > [1]")
            continue
        sent_map = ev_idx.get(cid, {})
        for idx in range(a, z + 1):
            if idx not in sent_map:
                errs.append(f"beat {i}: sentence {idx} not in evidence for {cid}")
                break
        if a in sent_map and z in sent_map:
            total_s += (sent_map[z]["end_ms"] - sent_map[a]["start_ms"]) / 1000
        am = b.get("audio_mode")
        sr_ok = b.get("sync_audio_required", False)
        dr_ok = b.get("dialogue_required", False)
        if am == "silent_visual" and sr_ok:
            errs.append(f"beat {i}: silent_visual + sync_audio_required")
        if am == "sync_dialogue" and not sr_ok:
            errs.append(f"beat {i}: sync_dialogue requires sync_audio_required=true")
    if not (lo <= total_s <= hi):
        errs.append(f"total duration {total_s:.1f}s outside {lane} band ({lo}-{hi}s)")
    if lane == "single_clip_15_25s" and len(clip_ids_used) != 1:
        errs.append(f"single_clip lane needs 1 clip_id, got {len(clip_ids_used)}")
    if lane == "mini_cross_media_30_60s" and not (2 <= len(clip_ids_used) <= 4):
        errs.append(f"mini lane needs 2-4 clip_ids, got {len(clip_ids_used)}")
    return errs


def _read_last_jsonl(path: Path) -> dict:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1])


def render_candidate(cand: dict, export_prefix: str) -> dict:
    result = se_adapt(cand)
    if not result.get("ok"):
        return {"ok": False, "reason": result.get("reason")}
    bp_path = Path(result["path"])
    bp = _read_last_jsonl(bp_path)
    clips = [
        {"media_id": b["media_id"], "in_ms": b["in_ms"], "out_ms": b["out_ms"],
         "label": b.get("edit_role", "body"), "_is_broll": False, "_crosstalk_trimmed": False}
        for b in bp["beats"]
    ]
    export_id = f"{export_prefix}_{result['bp_id'][-6:]}"
    rendered = do_render(
        beat_plan=bp, clips=clips, export_id=export_id,
        catalog=CAT, proxy_dir=ROOT / "media" / "proxies",
        output_dir=CAT / "exports",
        journal_db=ROOT / "state" / "journal.db",
        config_path=ROOT / "state" / "config.toml",
        reframe_strategy="active_speaker",
    )
    return {"ok": True, "bp_id": result["bp_id"], "export_id": export_id, **rendered}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=8)
    ap.add_argument("--lane", choices=["all", "single", "mini", "long"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--candidates-file", default="", help="Re-render from an existing candidates.jsonl (skips LLM)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    lane_map = {"single": "single_clip_15_25s", "mini": "mini_cross_media_30_60s",
                "long": "long_cross_media_60_120s", "all": "all"}
    lane = lane_map[args.lane]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs" / f"se_chain_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Selecting top %d clips…", args.n_clips)
    clips_meta = select_top_clips(args.n_clips, lane)
    clips = [cd for m in clips_meta if (cd := load_clip_data(m)) is not None]
    if not clips:
        log.error("No clip data loaded")
        sys.exit(1)
    log.info("Loaded %d clips: %s", len(clips), [c["media_id"] for c in clips])

    prompt = build_prompt(clips, lane)
    (out_dir / "prompt.txt").write_text(prompt)

    if args.dry_run:
        log.info("Dry-run: prompt written to %s/prompt.txt — skipping LLM call", out_dir)
        log.info("Clips selected: %s", [c["media_id"] for c in clips])
        return

    # Re-use existing candidates (skips LLM call — useful to re-render with code fixes)
    if args.candidates_file:
        cands_path = Path(args.candidates_file)
        log.info("Loading candidates from %s (skipping LLM)", cands_path)
        candidates = [json.loads(l) for l in cands_path.read_text().splitlines() if l.strip()]
        latency_ms = 0.0
    else:
        runlog.log_event("SE:C", "_corpus", "start",
                         {"n_clips": len(clips), "lane": lane, "run_id": run_id})
        log.info("Calling DS Pro thinking mode…")
        t0 = time.time()
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = llm_complete(messages, model="deepseek-v4-pro", temperature=0.6)
        except Exception as e:
            runlog.log_event("SE:C", "_corpus", "error", {"error": str(e), "run_id": run_id})
            log.error("LLM call failed: %s", e)
            sys.exit(1)
        latency_ms = (time.time() - t0) * 1000
        log.info("LLM done in %.0fms", latency_ms)
        (out_dir / "raw_response.txt").write_text(raw)

        # Parse candidates
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            candidates = json.loads(text)
            if not isinstance(candidates, list):
                raise ValueError("expected JSON array")
        except Exception as e:
            runlog.log_event("SE:C", "_corpus", "error",
                             {"error": "parse_failed", "detail": str(e), "latency_ms": latency_ms})
            log.error("Parse failed: %s", e)
            sys.exit(1)

    # Validate (C2)
    ev_idx: dict[str, dict] = {
        c["media_id"]: {s["idx"]: s for s in c["sentences"]}
        for c in clips
    }
    valid_ids = set(ev_idx)
    valid: list[dict] = []
    rejected = []
    for cand in candidates:
        errs = validate_c2(cand, valid_ids, ev_idx)
        cid = cand.get("candidate_id", "?")
        if errs:
            log.warning("C2 reject %s: %s", cid, errs)
            rejected.append({"candidate_id": cid, "errors": errs})
        else:
            valid.append(cand)
            log.info("C2 pass: %s | %s | %s", cid, cand.get("primary_format_lane"), cand.get("title",""))

    runlog.log_event("SE:C", "_corpus", "ok", {
        "model": "deepseek-v4-pro", "mode": "thinking",
        "latency_ms": round(latency_ms), "temperature": 0.6,
        "n_candidates": len(candidates), "n_valid": len(valid),
        "decision_summary": f"{len(valid)}/{len(candidates)} candidates passed C2",
    })

    (out_dir / "candidates.jsonl").write_text("\n".join(json.dumps(c) for c in candidates))
    if rejected:
        (out_dir / "rejections.json").write_text(json.dumps(rejected, indent=2))
    log.info("%d/%d candidates valid after C2", len(valid), len(candidates))

    if args.dry_run or not valid:
        if not valid:
            log.warning("No valid candidates — nothing to render")
        else:
            log.info("Dry-run: skipping adapt+render")
        return

    # Adapt + render
    export_prefix = f"se_{run_id[:8]}"
    results = []
    for cand in valid:
        if cand.get("verdict") != "complete":
            continue
        cid = cand.get("candidate_id", "?")
        log.info("Rendering %s…", cid)
        try:
            res = render_candidate(cand, export_prefix)
            results.append({"candidate_id": cid, **res})
            if res["ok"]:
                log.info("  → %s (%s)", res.get("output_mp4_path"), res.get("duration_s"))
            else:
                log.warning("  → adapter/render failed: %s", res.get("reason"))
        except Exception as e:
            log.exception("Render error for %s", cid)
            results.append({"candidate_id": cid, "ok": False, "reason": str(e)})

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    ok_count = sum(1 for r in results if r.get("ok"))
    log.info("Done: %d/%d rendered. Output: %s", ok_count, len(results), out_dir)
    for r in results:
        if r.get("ok"):
            print(r.get("output_mp4_path", "?"))


if __name__ == "__main__":
    main()

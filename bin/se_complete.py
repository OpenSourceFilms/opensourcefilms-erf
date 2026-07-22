"""SE:C — Pro chain completion on pre-mined, pre-validated ingredient sets.

Reads SE:B2 ingredients (output of se_mine.py), builds Pro chain-completion
prompts (ingredients are grounded; Pro assembles story structure but must NOT
originate new clip_ids/sentence_ranges), validates with se_floor (SE:C2),
deduplicates with se_dedup (SE:C3), writes candidate JSONL.

Gate: SE-3 taste sign-off by the project lead before running with --render.
Do NOT run with render flags until SE-3 gate passes.
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from erf.llm import complete as llm_complete
from erf.runlog import log_event
from erf.se_floor import validate_candidate, build_ev_idx
from erf.se_dedup import cluster_candidates

CAT = ROOT / "catalog"
RUNS = ROOT / "runs"
LANES = ["single_clip_15_25s", "mini_cross_media_30_60s", "long_cross_media_60_120s"]

_CANDIDATE_SCHEMA = """{
  "candidate_id": "c_{uuid4_short}",
  "verdict": "complete|reject|partial_open_thread",
  "primary_format_lane": "<the lane you were given>",
  "confidence_tier": "pro_scored",
  "title": "string",
  "central_question": "string — what a viewer wants resolved",
  "specific_tension": "string",
  "archetype": "string",
  "known_weakness": "string or null",
  "beats": [
    {
      "role": "HOOK|WANT|OBSTACLE|ATTEMPT|DECISION|TURN|RESULT|PAYOFF|BUTTON",
      "clip_id": "<MUST be from the provided ingredient clip_ids>",
      "sentence_range": [start_idx, end_idx],
      "evidence_ids": ["clip_id.sN", ...],
      "why_this_beat": "string",
      "audio_mode": "sync_dialogue|vo|broll_audio|music_only|silent_visual",
      "sync_audio_required": false,
      "dialogue_required": false,
      "visual_payoff_required": false,
      "visual_support": "string or null",
      "payoff_type": "none|visual_proof|funny_button|technical_reveal|human_reaction|failure_that_teaches|before_after|open_loop"
    }
  ],
  "missing_beats": [],
  "future_footage_needed": []
}"""

log = logging.getLogger("se_complete")


def load_ingredients(ing_dir: Path, lane: str) -> list[dict]:
    path = ing_dir / f"ingredients_{lane}_b2.jsonl"
    if not path.exists():
        log.warning("No B2 ingredients file: %s", path)
        return []
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    ings = []
    for line in lines:
        try:
            ings.append(json.loads(line))
        except json.JSONDecodeError as e:
            log.warning("Skipping invalid JSONL line: %s", e)
    return ings


def load_evidence(cat: Path, clip_ids: list[str]) -> dict[str, dict]:
    ev = {}
    for cid in clip_ids:
        ev_path = cat / "evidence" / f"{cid}.json"
        if ev_path.exists():
            ev[cid] = json.loads(ev_path.read_text())
    return ev


def build_chain_prompt(ingredients: list[dict], evidence: dict, lane: str) -> str:
    lane_guidance = {
        "single_clip_15_25s": "single clip, 15-25s, one self-contained quote or micro-story",
        "mini_cross_media_30_60s": "2-4 clips, 30-60s, setup→obstacle→attempt/result→payoff",
        "long_cross_media_60_120s": "3+ clips, 60-120s, technical arc with viewer question, turn, payoff",
    }[lane]

    clip_ids = list({ing["clip_id"] for ing in ingredients if ing.get("clip_id")})

    lane_bands = {
        "single_clip_15_25s": (15, 25),
        "mini_cross_media_30_60s": (30, 60),
        "long_cross_media_60_120s": (60, 120),
    }
    lo, hi = lane_bands.get(lane, (0, 999))

    ing_lines = []
    for ing in ingredients:
        cid = ing.get("clip_id", "?")
        sr = ing.get("sentence_range", [])
        src = ing.get("source_type", "?")
        claim = ing.get("claim", "")
        # Compute actual duration from evidence so Pro can pick ranges that fit the band
        ing_dur = ""
        ev_sents = {s["sentence_index"]: s for s in evidence.get(cid, {}).get("sentences", [])}
        if len(sr) == 2 and sr[0] in ev_sents and sr[1] in ev_sents:
            dur_s = (ev_sents[sr[1]]["end_ms"] - ev_sents[sr[0]]["start_ms"]) / 1000
            ing_dur = f" [{dur_s:.1f}s]"
        ing_lines.append(f"  [{src}]{ing_dur} clip_id={cid} sentence_range={sr} — {claim}")

    ev_lines = []
    for cid in clip_ids:
        ev_data = evidence.get(cid, {})
        ev_lines.append(f"\n### clip_id: {cid}")
        for s in ev_data.get("sentences", []):
            idx = s["sentence_index"]
            dur = (s.get("end_ms", 0) - s.get("start_ms", 0)) / 1000
            txt = s.get("text", "")
            ev_lines.append(f"  [{idx}] {dur:.1f}s | {txt[:120]}")

    return f"""You are assembling a BTS short film story arc for Life®, an indie documentary.

LANE: {lane} ({lane_guidance})
DURATION: total sentence time across ALL beats MUST be between {lo} and {hi} seconds.
          Ingredient durations are shown in brackets [Xs] — pick ranges that sum to {lo}–{hi}s.

PRE-MINED INGREDIENTS (grounded — your raw material):
{chr(10).join(ing_lines)}

YOUR TASK: Assemble ONE complete candidate story arc from these ingredients.
You are chain-completing a structure — not inventing from scratch.

HARD CONSTRAINTS:
- clip_id values in beats MUST be from this list ONLY: {clip_ids}
- sentence_range indices MUST exist in the evidence below for that clip_id
- evidence_ids: ["clip_id.sN" for N in range(start_idx, end_idx+1)]
- Do NOT invent new clip_ids or sentence indices
- verdict=complete requires a real HOOK beat and a real PAYOFF beat
- confidence_tier MUST be "pro_scored"
- primary_format_lane MUST be "{lane}"
- DURATION: sum of all beat durations MUST be {lo}–{hi}s (check ingredient [Xs] labels)
- AUDIO RULES (violation = rejection):
    * audio_mode="sync_dialogue" → sync_audio_required=true AND dialogue_required=true
    * audio_mode="silent_visual" → dialogue_required=false
    * audio_mode ∈ {{vo, broll_audio, music_only}} → sync_audio_required=false, dialogue_required=false

CANDIDATE SCHEMA:
{_CANDIDATE_SCHEMA}

EVIDENCE:
{"".join(ev_lines)}

Return ONLY a single JSON object (the candidate). No array, no markdown, no prose."""


def run_lane(ingredients: list[dict], evidence: dict, lane: str, max_per_lane: int,
             valid_ids: set[str], *, dry_run: bool = False) -> list[dict]:
    """Run Pro chain completion for one lane. Returns valid, deduped candidates."""
    if dry_run or not ingredients:
        return []

    ingredient_clip_ids = {ing["clip_id"] for ing in ingredients if ing.get("clip_id")}
    ev_idx: dict[str, dict[int, dict]] = {
        cid: build_ev_idx(evidence.get(cid, {})) for cid in ingredient_clip_ids
    }

    candidates = []
    for attempt in range(max_per_lane):
        log.info("  Calling Pro for lane=%s attempt=%d", lane, attempt + 1)
        prompt = build_chain_prompt(ingredients, evidence, lane)
        t0 = time.monotonic()
        try:
            raw = llm_complete(
                [{"role": "user", "content": prompt}],
                model="deepseek-v4-pro",
                temperature=0.6,
                json_only=True,
            )
            latency = time.monotonic() - t0
            cand = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(cand, dict):
                raise ValueError("expected JSON object")
        except Exception as exc:
            log_event("SE:C", "_corpus", "pro_error", {"lane": lane, "attempt": attempt, "error": str(exc)})
            log.warning("Pro call failed: %s", exc)
            continue

        if not cand.get("candidate_id"):
            cand["candidate_id"] = f"c_{uuid.uuid4().hex[:8]}"

        errs = validate_candidate(cand, valid_ids, ev_idx, ingredient_clip_ids=ingredient_clip_ids)
        verdict = cand.get("verdict")
        log_event("SE:C", "_corpus", "candidate_validated", {
            "lane": lane, "candidate_id": cand.get("candidate_id"),
            "verdict": verdict, "errors": errs, "latency": round(latency, 2),
        })

        if errs:
            log.warning("Candidate rejected (%d errors): %s", len(errs), errs[:3])
            continue
        if verdict != "complete":
            log.info("Candidate verdict=%s (not complete) — skipping adapter", verdict)
            continue

        candidates.append(cand)

    deduped = cluster_candidates(candidates, lane)
    log.info("Lane %s: %d valid → %d after dedup", lane, len(candidates), len(deduped))
    return deduped


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in records) + ("\n" if records else ""))
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingredients-dir", required=True, help="Path to se_mine run dir")
    ap.add_argument("--catalog", default=str(CAT))
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--lanes", nargs="+", choices=LANES, default=LANES)
    ap.add_argument("--max-per-lane", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cat = Path(args.catalog)
    ing_dir = Path(args.ingredients_dir)
    out_dir = (Path(args.output_dir) if args.output_dir
               else RUNS / f"se_complete_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_ids = {p.stem for p in (cat / "evidence").glob("*.json")}
    all_candidates: list[dict] = []

    log_event("SE:C", "_corpus", "start", {
        "ingredients_dir": str(ing_dir), "lanes": args.lanes,
        "max_per_lane": args.max_per_lane, "dry_run": args.dry_run,
    })

    for lane in args.lanes:
        log.info("=== Lane: %s ===", lane)
        ingredients = load_ingredients(ing_dir, lane)
        log.info("  Loaded %d B2 ingredients", len(ingredients))
        clip_ids = list({ing["clip_id"] for ing in ingredients if ing.get("clip_id")})
        evidence = load_evidence(cat, clip_ids)

        lane_cands = run_lane(ingredients, evidence, lane, args.max_per_lane, valid_ids,
                               dry_run=args.dry_run)
        atomic_write_jsonl(out_dir / f"candidates_{lane}.jsonl", lane_cands)
        all_candidates.extend(lane_cands)

    atomic_write_jsonl(out_dir / "candidates_all.jsonl", all_candidates)
    log_event("SE:C", "_corpus", "complete_done", {"out_dir": str(out_dir), "n_candidates": len(all_candidates)})
    log.info("Done. %d total candidates → %s", len(all_candidates), out_dir)
    print(f"OUTPUT: {out_dir}")


if __name__ == "__main__":
    main()

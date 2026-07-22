"""SE:B ingredient miner + SE:B2 dedup/rank.

Mines the BTS corpus for story ingredients (grounded clip_id + sentence_range
references) using Flash ideation, then deterministically ranks + deduplicates
to produce top-8/lane inputs for SE:C chain completion.

Outputs:
  runs/se_mine_{date}/ingredients_{lane}.jsonl   — all valid ingredients
  runs/se_mine_{date}/ingredients_{lane}_b2.jsonl — top-8 after B2 dedup/rank
  runs/se_mine_{date}/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from erf.llm import complete as llm_complete
from erf.runlog import log_event

CAT = ROOT / "catalog"
RUNS = ROOT / "runs"

LANES = ["single_clip_15_25s", "mini_cross_media_30_60s", "long_cross_media_60_120s"]
LANE_ABBR = {"single_clip_15_25s": "sc", "mini_cross_media_30_60s": "mc", "long_cross_media_60_120s": "lc"}
LANE_GUIDANCE = {
    "single_clip_15_25s": "single clip, 15-25s total, one complete quote or tiny self-contained story",
    "mini_cross_media_30_60s": "2-4 different clips, 30-60s, setup→obstacle→attempt/result→payoff",
    "long_cross_media_60_120s": "3+ clips, 60-120s, technical/contextual material with viewer question, turn, payoff",
}
COUNTS_PER_BATCH = {"story_pressure": 4, "viewer_question": 4, "payoff": 3}

log = logging.getLogger("se_mine")


def load_corpus(cat: Path) -> list[dict]:
    """Load all media with editorial labels + evidence. Filters low-quality."""
    ev_dir = cat / "evidence"
    ed_dir = cat / "editorial"
    corpus = []
    for ev_path in sorted(ev_dir.glob("*.json")):
        mid = ev_path.stem
        ev = json.loads(ev_path.read_text())
        if ev.get("n_sentences", 0) < 3:
            continue
        aq = (ev.get("av_quality") or {}).get("speech_clarity_median", 0)
        if aq < 0.25:
            continue
        manifest_path = ed_dir / mid / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        job_id = manifest.get("latest_job_id")
        label_path = ed_dir / mid / f"{job_id}.json"
        if not label_path.exists():
            continue
        label = json.loads(label_path.read_text())
        if label.get("parse_error"):
            continue
        corpus.append({"media_id": mid, "editorial": label, "evidence": ev})
    return corpus


def _format_clips_for_prompt(batch: list[dict]) -> str:
    lines = []
    for item in batch:
        mid = item["media_id"]
        ed = item["editorial"]
        ev = item["evidence"]
        cl = ed.get("clip_level", {})
        sf = cl.get("story_forces", {})
        arch = cl.get("archetype_candidates", [])
        tension = cl.get("central_tension", "")
        lines.append(f"\n### clip_id: {mid}")
        lines.append(f"  archetype: {', '.join(arch) or 'unknown'}")
        lines.append(f"  tension: {tension}")
        lines.append(f"  story_forces: want={sf.get('want')} obstacle={sf.get('obstacle')} payoff={sf.get('payoff')}")
        lines.append(f"  av_quality(speech): {ev.get('av_quality',{}).get('speech_clarity_median',0):.2f}")
        lines.append("  sentences:")
        for s in ev.get("sentences", []):
            idx = s["sentence_index"]
            hook = s.get("hook", 0)
            quote = s.get("quote", 0)
            txt = s.get("text", "")
            start = s.get("start_ms", 0)
            end = s.get("end_ms", 0)
            dur = (end - start) / 1000
            lines.append(f"    [{idx}] {dur:.1f}s hook={hook:.2f} quote={quote:.2f} | {txt[:120]}")
    return "\n".join(lines)


def mine_batch(batch: list[dict], lane: str, source_type: str, n: int, *, dry_run: bool = False) -> list[dict]:
    """Call Flash to mine N ingredients of source_type for lane. Returns valid ingredients only."""
    batch_ids = [item["media_id"] for item in batch]
    clips_text = _format_clips_for_prompt(batch)
    source_guidance = {
        "story_pressure": "A conflict driver: something forcing a decision or action. Find moments of genuine obstacle, friction, or forced choice.",
        "viewer_question": "A curiosity hook: something a viewer would want resolved. Find moments that create an open question or mystery.",
        "payoff": "A resolution: a moment that satisfies, reveals, or lands. Find moments of emotional or informational closure.",
    }[source_type]

    prompt = f"""You are mining BTS footage for story ingredients — grounded references to compelling moments.

LANE: {lane} ({LANE_GUIDANCE[lane]})
SOURCE TYPE: {source_type}
{source_guidance}

TASK: Produce exactly {n} ingredients of type "{source_type}" for the "{lane}" lane.

HARD CONSTRAINTS (violation = ingredient is discarded):
- clip_id MUST be exactly one of: {batch_ids}
- sentence_range [start_idx, end_idx]: BOTH indices must appear in the sentence list below
- evidence_ids: ["clip_id.sN" for N in range(start_idx, end_idx+1)]
- Do NOT invent clip_ids or sentence indices not shown below
- hook_score, quote_score: use max score from the sentences in the range
- av_quality: use the speech clarity value shown for that clip

Return ONLY a JSON array of {n} objects with these fields:
  clip_id, sentence_range, evidence_ids, claim, archetype, hook_score, quote_score,
  av_quality, tension_tags, can_start_story, can_payoff_story

No markdown, no prose, no explanation.

EVIDENCE:
{clips_text}"""

    if dry_run:
        return []

    t0 = time.monotonic()
    try:
        raw = llm_complete([{"role": "user", "content": prompt}], model="deepseek-v4-flash",
                           temperature=0.75, json_only=True)
        latency = time.monotonic() - t0
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, list):
            raise ValueError("not a list")
    except Exception as exc:
        log_event("SE:B", "_corpus", "flash_error", {"source_type": source_type, "lane": lane, "error": str(exc)})
        log.warning("Flash mining failed (%s/%s): %s", lane, source_type, exc)
        return []

    log_event("SE:B", "_corpus", "flash_call",
              {"source_type": source_type, "lane": lane, "latency": round(latency, 2), "n_raw": len(parsed)})

    # Validate each ingredient — only accept those with real clip_ids and valid sentence ranges
    sent_idx_cache: dict[str, set[int]] = {
        item["media_id"]: {s["sentence_index"] for s in item["evidence"].get("sentences", [])}
        for item in batch
    }
    batch_by_id = {item["media_id"]: item for item in batch}

    valid = []
    for ing in parsed:
        cid = ing.get("clip_id")
        if cid not in batch_ids:
            log.debug("discarded: unknown clip_id %r", cid)
            continue
        sr = ing.get("sentence_range")
        if not (isinstance(sr, list) and len(sr) == 2 and all(isinstance(x, int) for x in sr)):
            log.debug("discarded: bad sentence_range %r", sr)
            continue
        a, z = sr
        if a > z or a not in sent_idx_cache.get(cid, set()) or z not in sent_idx_cache.get(cid, set()):
            log.debug("discarded: sentence range OOB for %s[%d,%d]", cid, a, z)
            continue

        item = batch_by_id[cid]
        sents = [s for s in item["evidence"].get("sentences", []) if a <= s["sentence_index"] <= z]
        ed_sents = [s for s in item["editorial"].get("sentences", []) if a <= s.get("sentence_index", -1) <= z]

        # Overwrite scores with ground truth from evidence
        ing["hook_score"] = max((s.get("hook", 0) for s in sents), default=0)
        ing["quote_score"] = max((s.get("quote", 0) for s in sents), default=0)
        ing["av_quality"] = item["evidence"].get("av_quality", {}).get("speech_clarity_median", 0)
        ing["can_start_story"] = any(
            s.get("chain_affordances", {}).get("can_start_story") for s in ed_sents
        )
        ing["can_payoff_story"] = any(
            s.get("chain_affordances", {}).get("can_payoff_story") for s in ed_sents
        )
        ing["evidence_ids"] = [f"{cid}.s{i}" for i in range(a, z + 1)]
        ing["primary_format_lane"] = lane
        ing["source_type"] = source_type
        valid.append(ing)

    return valid


def dedup_rank_b2(ingredients: list[dict], lane: str, top_n: int = 8) -> list[dict]:
    """SE:B2: dedup by (clip_id, sentence_range), rank by composite score with tension diversity."""
    seen: set[tuple] = set()
    unique = []
    for ing in ingredients:
        key = (ing.get("clip_id"), tuple(ing.get("sentence_range", [])))
        if key not in seen:
            seen.add(key)
            unique.append(ing)

    selected: list[dict] = []
    used_tensions: set[str] = set()
    remaining = list(unique)
    for _ in range(top_n):
        if not remaining:
            break
        best = max(remaining, key=lambda g: (
            0.4 * g.get("hook_score", 0)
            + 0.3 * g.get("quote_score", 0)
            + 0.2 * g.get("av_quality", 0)
            + 0.1 * (1 if g.get("can_start_story") else 0)
            - 0.15 * (1 if any(t in used_tensions for t in g.get("tension_tags", [])) else 0)
        ))
        selected.append(best)
        remaining.remove(best)
        used_tensions.update(best.get("tension_tags", []))

    return selected


def assign_ids(ingredients: list[dict], lane: str, source_type: str, start: int) -> list[dict]:
    abbr = LANE_ABBR[lane]
    for i, ing in enumerate(ingredients):
        ing["ingredient_id"] = f"ing_{abbr}_{source_type}_{start + i:04d}"
    return ingredients


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in records) + ("\n" if records else ""))
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(CAT))
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--lanes", nargs="+", choices=LANES, default=LANES)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--clips-per-batch", type=int, default=15)
    ap.add_argument("--b2-top", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cat = Path(args.catalog)
    out_dir = (Path(args.output_dir) if args.output_dir
               else RUNS / f"se_mine_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    corpus = load_corpus(cat)
    log.info("Loaded %d media items from corpus", len(corpus))
    if not corpus:
        log.error("No corpus items — check editorial + evidence dirs")
        sys.exit(1)

    log_event("SE:B", "_corpus", "start", {
        "out_dir": str(out_dir), "lanes": args.lanes,
        "corpus_items": len(corpus), "dry_run": args.dry_run,
    })
    summary: dict = {"lanes": {}, "total_flash_calls": 0, "total_rejected": 0, "out_dir": str(out_dir)}

    for lane in args.lanes:
        log.info("=== Lane: %s ===", lane)
        all_ingredients: list[dict] = []
        flash_calls = 0
        rejected = 0
        id_counter = 0

        for src_type, n_per_batch in COUNTS_PER_BATCH.items():
            for batch_idx in range(args.batches):
                batch_size = min(args.clips_per_batch, len(corpus))
                batch = rng.sample(corpus, batch_size)
                log.info("  %s / %s / batch %d: %d clips", lane, src_type, batch_idx, len(batch))
                mined = mine_batch(batch, lane, src_type, n_per_batch, dry_run=args.dry_run)
                flash_calls += 1
                rejected += max(0, n_per_batch - len(mined))
                mined = assign_ids(mined, lane, src_type, id_counter)
                id_counter += len(mined)
                all_ingredients.extend(mined)
                log.info("    got %d valid ingredients", len(mined))

        b2 = dedup_rank_b2(all_ingredients, lane, top_n=args.b2_top)
        atomic_write_jsonl(out_dir / f"ingredients_{lane}.jsonl", all_ingredients)
        atomic_write_jsonl(out_dir / f"ingredients_{lane}_b2.jsonl", b2)
        log.info("  Lane %s: %d total ingredients, %d b2", lane, len(all_ingredients), len(b2))

        summary["lanes"][lane] = {
            "total": len(all_ingredients),
            "b2": len(b2),
            "flash_calls": flash_calls,
            "rejected": rejected,
        }
        summary["total_flash_calls"] += flash_calls
        summary["total_rejected"] += rejected

    summary_path = out_dir / "summary.json"
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.rename(summary_path)

    log_event("SE:B", "_corpus", "mine_complete", {"out_dir": str(out_dir), **summary})
    log.info("Done. Output: %s", out_dir)
    print(f"OUTPUT: {out_dir}")


if __name__ == "__main__":
    main()

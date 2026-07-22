#!/usr/bin/env python3
"""bin/se_probe.py — Story Engine SE-0c PROBE harness.

Measures DeepSeek V4 pro (thinking) vs flash (non-thinking) on the REAL engine
workload (grounded candidate generation), so SE-4 budgets/concurrency/wall-clock
are sized from measurement, not assumption. No narrative judgement — pure metrics.

Writes runs/probe_<UTCstamp>/{measurements.json, summary.txt}.

Usage:
    PYTHONPATH=src python3 bin/se_probe.py --dry-run         # build prompt, no API
    DEEPSEEK_API_KEY=... python3 bin/se_probe.py --pro-n 8 --flash-n 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/erf")
CAT = ROOT / "catalog"
sys.path.insert(0, str(ROOT / "src"))

from erf.llm import complete as llm_complete  # noqa: E402

log = logging.getLogger("se_probe")

_PRO = "deepseek-v4-pro"
_FLASH = "deepseek-v4-flash"


# ---------------------------------------------------------------- clip gathering
def gather_clips(max_clips: int):
    """Editorial-labelled media → [(media_id, [{sentence_index,text}])], set(real_ids).

    media_id is the editorial DIRECTORY NAME (robust — does not depend on manifest
    fields). Sentences come from catalog/evidence/{mid}.json; text is best-effort
    (never required, so a clip is not dropped just for missing text)."""
    base = CAT / "editorial"
    if not base.exists():
        log.error("no editorial dir at %s", base)
        sys.exit(3)
    clips = []
    for ed_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        mid = ed_dir.name
        if not mid.startswith("m_"):
            continue
        ev_path = CAT / "evidence" / f"{mid}.json"
        if not ev_path.exists():
            continue
        try:
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("bad evidence json %s", ev_path)
            continue
        sents = ev.get("sentences")
        if not isinstance(sents, list) or not sents:
            continue
        snippets = []
        for s in sents[:6]:
            if not isinstance(s, dict) or "sentence_index" not in s:
                continue
            txt = s.get("text") or s.get("asr_text") or s.get("sentence") or ""
            snippets.append({"sentence_index": s["sentence_index"], "text": txt})
        if not snippets:
            continue
        clips.append((mid, snippets))
        if len(clips) >= max_clips:
            break
    if not clips:
        log.error("no editorial-labelled clips with evidence sentences found")
        sys.exit(3)
    return clips, {c[0] for c in clips}


def build_messages(clips):
    """One grounded chain-completion-style prompt, reused for every call."""
    payload = [{"clip_id": mid, "sentences": sn} for mid, sn in clips]
    system = {
        "role": "system",
        "content": (
            "You are a short-form story editor. Given source clips (each a clip_id with "
            "0-indexed sentences), propose ONE candidate as STRICT JSON only — no prose, "
            "no markdown fences."
        ),
    }
    user = {
        "role": "user",
        "content": (
            "Source clips:\n" + json.dumps(payload, indent=2) + "\n\n"
            "Return exactly this JSON object:\n"
            '{"candidate_id":"str","primary_format_lane":'
            '"single_clip_15_25s|mini_cross_media_30_60s|long_cross_media_60_120s",'
            '"verdict":"complete|reject|partial_open_thread","title":"str",'
            '"central_question":"str","beats":[{"role":"str",'
            '"clip_id":"<EXACTLY one provided clip_id>","sentence_range":[a,b],'
            '"why_this_beat":"str"}]}\n\n'
            "HARD RULES:\n"
            "- every beat clip_id MUST be exactly one of the provided clip_id strings;\n"
            "- never output an integer or digit-only string as a clip_id;\n"
            "- sentence_range = two integers [a,b], a<=b, both existing for that clip."
        ),
    }
    return [system, user]


# ---------------------------------------------------------------- one call + classify
def run_one(messages, model, temperature):
    """Return (latency_s, parsed_or_None, error_kind, detail).

    error_kind ∈ {None, 'transport', 'json', 'empty'} — distinguishes API/transport
    failures (reliability) from malformed model output (quality)."""
    t0 = time.time()
    try:
        raw = llm_complete(messages, json_only=True, model=model, temperature=temperature)
    except Exception as e:  # urllib/HTTP/timeout from erf.llm (after its own retries)
        return time.time() - t0, None, "transport", f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    if not raw or not raw.strip():
        return dt, None, "empty", "empty response"
    try:
        return dt, json.loads(raw), None, None
    except json.JSONDecodeError as e:
        return dt, None, "json", f"JSONDecodeError: {e}"


def classify(parsed, real_ids):
    """Classify a successfully-parsed candidate: ok | missing_beats |
    fabricated_id | array_index_as_id | bad_range. (Transport/json handled upstream.)"""
    if not isinstance(parsed, dict):
        return "non_json"
    beats = parsed.get("beats")
    if not isinstance(beats, list) or not beats:
        return "missing_beats"
    worst = "ok"
    rank = {"ok": 0, "bad_range": 1, "array_index_as_id": 2, "fabricated_id": 3}
    for b in beats:
        if not isinstance(b, dict):
            return "missing_beats"
        cid = b.get("clip_id")
        if isinstance(cid, int) or (isinstance(cid, str) and (cid.isdigit() or len(cid) <= 3)):
            cur = "array_index_as_id"
        elif cid not in real_ids:
            cur = "fabricated_id"
        else:
            sr = b.get("sentence_range")
            ok = (isinstance(sr, list) and len(sr) == 2
                  and all(isinstance(x, int) for x in sr) and sr[0] <= sr[1])
            cur = "ok" if ok else "bad_range"
        if rank[cur] > rank[worst]:
            worst = cur
    return worst


def outcome_of(parsed, error_kind, real_ids):
    if error_kind == "transport":
        return "error"
    if error_kind in ("json", "empty"):
        return "non_json"
    return classify(parsed, real_ids)


def pctl(vals, q):
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    return s[int(q * (len(s) - 1) + 0.5)]


def run_model(model, n, messages, real_ids):
    lats, outcomes, details = [], {}, []
    for i in range(n):
        dt, parsed, ek, detail = run_one(messages, model, 0.0)
        lats.append(dt)
        oc = outcome_of(parsed, ek, real_ids)
        outcomes[oc] = outcomes.get(oc, 0) + 1
        if detail:
            details.append({"i": i, "kind": ek, "detail": detail[:200]})
        log.info("%s call %d/%d: %.1fs -> %s", model, i + 1, n, dt, oc)
    return {
        "n": n,
        "latency_p50": round(pctl(lats, 0.50), 2),
        "latency_p95": round(pctl(lats, 0.95), 2),
        "latency_min": round(min(lats), 2),
        "latency_max": round(max(lats), 2),
        "outcome_breakdown": outcomes,
        "ok_rate": round(outcomes.get("ok", 0) / n, 3),
        "failures": details[:10],
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="SE-0c probe")
    ap.add_argument("--pro-n", type=int, default=8)
    ap.add_argument("--flash-n", type=int, default=8)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--diversity-n", type=int, default=3)
    ap.add_argument("--diversity-temp", type=float, default=0.6)
    ap.add_argument("--no-concurrency", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    clips, real_ids = gather_clips(args.clips)
    messages = build_messages(clips)
    log.info("selected %d clips: %s", len(clips), sorted(real_ids))

    if args.dry_run:
        print("REAL_IDS:", sorted(real_ids))
        print("--- system ---\n" + messages[0]["content"])
        print("--- user (head) ---\n" + messages[1]["content"][:700])
        return

    if not os.environ.get("DEEPSEEK_API_KEY"):
        log.error("DEEPSEEK_API_KEY not set")
        sys.exit(2)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = ROOT / "runs" / f"probe_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    m = {
        "schema": "se_probe/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": vars(args),
        "real_ids": sorted(real_ids),
        "per_model": {},
        "concurrency": {},
        "diversity": {},
        "notes": ["token usage unavailable via erf.llm.complete; latency+reliability+diversity only",
                  "error=transport failure (reliability); non_json=malformed model output (quality)"],
    }

    log.info("=== PRO model: %d calls ===", args.pro_n)
    m["per_model"]["pro"] = run_model(_PRO, args.pro_n, messages, real_ids)
    log.info("=== FLASH model: %d calls ===", args.flash_n)
    m["per_model"]["flash"] = run_model(_FLASH, args.flash_n, messages, real_ids)

    if not args.no_concurrency:
        log.info("=== concurrency: 2 concurrent PRO ===")
        calls, ok = [], True
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(run_one, messages, _PRO, 0.0): i for i in range(2)}
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                try:
                    dt, parsed, ek, detail = fut.result()
                    calls.append({"i": i, "latency": round(dt, 2),
                                  "outcome": outcome_of(parsed, ek, real_ids),
                                  "error": detail[:200] if detail else None})
                    if ek == "transport":
                        ok = False
                except Exception as e:
                    calls.append({"i": i, "outcome": "error", "error": str(e)[:200]})
                    ok = False
        m["concurrency"] = {"tested": 2, "concurrency_ok": ok and len(calls) == 2, "calls": calls}

    if args.diversity_n > 0:
        log.info("=== diversity: %d PRO @ temp %.2f ===", args.diversity_n, args.diversity_temp)
        sets, titles, fails = [], [], 0
        for i in range(args.diversity_n):
            dt, parsed, ek, detail = run_one(messages, _PRO, args.diversity_temp)
            if parsed is None:
                fails += 1
                continue
            beats = parsed.get("beats") or []
            sets.append({b.get("clip_id") for b in beats if isinstance(b, dict)})
            if parsed.get("title"):
                titles.append(parsed["title"])
        collapse = None
        if len(sets) >= 2:
            js = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    u = len(sets[i] | sets[j])
                    js.append(len(sets[i] & sets[j]) / u if u else 0.0)
            collapse = round(statistics.mean(js), 3) if js else None
        m["diversity"] = {"n": args.diversity_n, "temp": args.diversity_temp,
                          "parsed_ok": len(sets), "failed": fails,
                          "collapse_metric_jaccard": collapse,
                          "distinct_titles": sorted(set(titles))}

    tmp = outdir / ".measurements.json.tmp"
    tmp.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, outdir / "measurements.json")

    def line(s): return s
    p, f = m["per_model"].get("pro", {}), m["per_model"].get("flash", {})
    summary = "\n".join([
        f"PROBE {ts}",
        f"clips={sorted(real_ids)}",
        "",
        f"PRO   n={p.get('n')} ok_rate={p.get('ok_rate')} "
        f"p50={p.get('latency_p50')}s p95={p.get('latency_p95')}s "
        f"min/max={p.get('latency_min')}/{p.get('latency_max')}s",
        f"      outcomes={p.get('outcome_breakdown')}",
        f"FLASH n={f.get('n')} ok_rate={f.get('ok_rate')} "
        f"p50={f.get('latency_p50')}s p95={f.get('latency_p95')}s",
        f"      outcomes={f.get('outcome_breakdown')}",
        "",
        f"concurrency_ok={m['concurrency'].get('concurrency_ok')}",
        f"diversity collapse(jaccard)={m['diversity'].get('collapse_metric_jaccard')} "
        f"distinct_titles={len(m['diversity'].get('distinct_titles', []))}",
    ])
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\n→ {outdir/'measurements.json'}")


if __name__ == "__main__":
    main()

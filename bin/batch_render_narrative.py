#!/usr/bin/env python3
"""Batch render all gate_passed narrative threads through the Wave 2b render pipeline.

Reads gate_passed threads from state/narrative_pipeline.db, loads each beat_plan,
runs the same assembly + b-roll expansion + render pipeline as /shorts/assemble
(render=true), and writes MP4s to catalog/exports/.

Usage:
    python bin/batch_render_narrative.py [--concurrency N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Bootstrap: add erf src and bin to path                                      #
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_render")

_CONFIG_PATH = _ROOT / "state" / "config.toml"
_PIPELINE_DB = _ROOT / "state" / "narrative_pipeline.db"

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

def _load_cfg() -> dict:
    with open(_CONFIG_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    p = raw.get("paths", {})
    return {
        "catalog": Path(p.get("catalog", "/workspace/erf/catalog")),
        "state":   Path(p.get("state",   "/workspace/erf/state")),
        "erf_root": Path(p.get("erf_root", "/workspace/erf")),
    }


# --------------------------------------------------------------------------- #
# Beat plan loading                                                            #
# --------------------------------------------------------------------------- #

def _load_latest_beat_plan(bp_id: str, catalog: Path) -> dict:
    """Return the latest (last-line) revision from the beat_plan JSONL."""
    bp_path = catalog / "beat_plans" / f"{bp_id}.jsonl"
    lines = [l.strip() for l in bp_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"Empty beat_plan file: {bp_path}")
    return json.loads(lines[-1])


# --------------------------------------------------------------------------- #
# Clip-building helpers (mirrors api.py shorts_assemble)                      #
# --------------------------------------------------------------------------- #

def _get_shortform_chunks(mid: str, catalog: Path, cache: dict) -> list[dict]:
    if mid not in cache:
        from erf.atomicio import read_jsonl
        cp = catalog / "chunks" / f"{mid}.jsonl"
        all_c = read_jsonl(cp) if cp.exists() else []
        cache[mid] = [c for c in all_c if c.get("chunk_rule", "").endswith("_short")]
    return cache[mid]


def _resolve_beat_boundaries(b: dict, catalog: Path, shortform_cache: dict) -> tuple[int, int]:
    _SHORTFORM_ROLES = {"hook", "call_to_action"}
    roles = set(b.get("edit_role") or [])
    if not roles.intersection(_SHORTFORM_ROLES):
        return b["in_ms"], b["out_ms"]
    mid = b["media_id"]
    beat_in, beat_out = b["in_ms"], b["out_ms"]
    best_overlap = 0
    best_in, best_out = beat_in, beat_out
    for sc in _get_shortform_chunks(mid, catalog, shortform_cache):
        ov = max(0, min(beat_out, sc["end_ms"]) - max(beat_in, sc["start_ms"]))
        if ov > best_overlap:
            best_overlap = ov
            best_in, best_out = sc["start_ms"], sc["end_ms"]
    return best_in, best_out


def _get_diar_turns(mid: str, catalog: Path, cache: dict) -> list[dict]:
    if mid not in cache:
        dp = catalog / "asr" / f"{mid}.diar.jsonl"
        cache[mid] = (
            [json.loads(l) for l in dp.read_text().splitlines() if l.strip()]
            if dp.exists() else []
        )
    return cache[mid]


def _get_words(mid: str, catalog: Path, cache: dict) -> list[dict]:
    if mid not in cache:
        wp = catalog / "asr" / f"{mid}.transcript.jsonl"
        cache[mid] = (
            [json.loads(l) for l in wp.read_text().splitlines() if l.strip()]
            if wp.exists() else []
        )
    return cache[mid]


def _trim_crosstalk_out(mid: str, in_ms: int, out_ms: int, catalog: Path,
                        diar_cache: dict, lookahead_ms: int = 2000,
                        words_cache: dict | None = None) -> tuple[int, bool]:
    # `<=` (not `<`) makes re-detection idempotent across re-renders of a stored
    # beat plan whose out_ms was already pulled back to a foreign-speaker onset:
    # the foreign turn then starts exactly at out_ms. See api.py for the full note.
    turns = [t for t in _get_diar_turns(mid, catalog, diar_cache)
             if t["start_ms"] <= out_ms and t["end_ms"] > in_ms]
    if not turns:
        return out_ms, False
    spk_ms: dict[str, int] = {}
    for t in turns:
        spk_ms[t["speaker_label"]] = spk_ms.get(t["speaker_label"], 0) + (
            min(t["end_ms"], out_ms) - max(t["start_ms"], in_ms)
        )
    primary = max(spk_ms, key=spk_ms.__getitem__)
    words = _get_words(mid, catalog, words_cache) if words_cache is not None else []
    for t in sorted(turns, key=lambda x: x["start_ms"]):
        if out_ms - lookahead_ms <= t["start_ms"] <= out_ms and t["speaker_label"] != primary:
            # Skip PHANTOM diar turns: a foreign-speaker turn with no transcribed words
            # of that speaker is a breath/noise the diarizer mislabeled (D-Render-09).
            # Trimming on it chops the primary speaker's own continuing sentence. Only
            # trim when the foreign speaker has real words that would bleed in.
            if words and not any(
                w.get("speaker_label") == t["speaker_label"]
                and t["start_ms"] <= int(w["start_ms"]) < t["end_ms"]
                for w in words
            ):
                continue
            return t["start_ms"], True
    return out_ms, False


def _build_clips(beats: list[dict], catalog: Path) -> tuple[list[dict], list[dict]]:
    """Build (clips, beats_with_trimmed_out) from raw beat_plan beats."""
    shortform_cache: dict = {}
    diar_cache: dict = {}
    words_cache: dict = {}
    clips = []
    adjusted_beats = []
    for b in beats:
        in_ms, out_ms = _resolve_beat_boundaries(b, catalog, shortform_cache)
        lookahead = 4000 if b.get("edit_role") == "quote" else 2000
        out_ms, _ct = _trim_crosstalk_out(b["media_id"], in_ms, out_ms, catalog, diar_cache,
                                          lookahead_ms=lookahead, words_cache=words_cache)
        b = dict(b)
        b["out_ms"] = out_ms
        adjusted_beats.append(b)
        raw_label = f"{b.get('edit_role', '')} {(b.get('note') or '')[:40]}".strip()
        safe_label = raw_label.replace("'", "’")  # OTIO FCPXML adapter uses label in XPath [@name='...']
        clips.append({
            "media_id": b["media_id"],
            "in_ms": in_ms,
            "out_ms": out_ms,
            "label": safe_label,
            "source_relpath": b.get("source_relpath"),
            "_is_broll": b.get("edit_role") == "b_roll",
            "_crosstalk_trimmed": _ct,
        })
    return clips, adjusted_beats


# --------------------------------------------------------------------------- #
# B-roll expansion (mirrors api.py)                                           #
# --------------------------------------------------------------------------- #

def _expand_broll(beats: list[dict], clips: list[dict], proxy_dir: Path) -> tuple[list[dict], list[dict]]:
    """Inject b-roll clips mid-beat when a candidate qualifies and its proxy exists."""
    _MIN_SCORE = 0.3
    _MIN_DUR_MS = 6000
    out_clips: list[dict] = []
    out_beats: list[dict] = []
    broll_count = 0

    for b, c in zip(beats, clips):
        cands = b.get("b_roll_candidates") or []
        valid = [x for x in cands if x.get("combined_score", 0) >= _MIN_SCORE]
        c_in = int(c["in_ms"])
        c_out = int(c["out_ms"])
        c_dur = c_out - c_in

        if valid and c_dur >= _MIN_DUR_MS:
            top = valid[0]
            br_mid = top["media_id"]
            br_t = int(top["time_ms"])
            br_in = max(0, br_t - 2000)
            br_out = br_t + 2000
            br_kf = top["keyframe_id"]
            br_proxy = proxy_dir / f"{br_mid}.mp4"
            if br_proxy.exists():
                mid_ms = c_in + c_dur // 2
                # first half dialogue
                out_clips.append({**c, "out_ms": mid_ms})
                out_beats.append({**b, "out_ms": mid_ms})
                # b-roll overlay
                out_clips.append({
                    "media_id": br_mid, "in_ms": br_in, "out_ms": br_out,
                    "label": f"broll {br_kf}", "_is_broll": True,
                })
                out_beats.append({
                    "beat_index": -1, "chunk_id": f"broll:{br_kf}",
                    "media_id": br_mid, "in_ms": br_in, "out_ms": br_out,
                    "edit_role": "b_roll",
                })
                # second half dialogue
                out_clips.append({**c, "in_ms": mid_ms})
                out_beats.append({**b, "in_ms": mid_ms})
                broll_count += 1
                continue

        out_clips.append(c)
        out_beats.append(b)

    return out_clips, out_beats


# --------------------------------------------------------------------------- #
# Review queue row                                                             #
# --------------------------------------------------------------------------- #

def _insert_review_queue(
    state: Path,
    review_id: str,
    export_id: str,
    bp_id: str,
    bp: dict,
    export_relpath: str,
) -> None:
    from erf.review import _connect as rv_connect, _emit_journal as rv_journal
    now = datetime.now(timezone.utc).isoformat()
    platform = bp.get("format_fit", "any")
    revision = bp.get("revision", 0)
    tmpl_id = bp.get("template_id")
    conn = rv_connect(state / "erf.db")
    conn.execute(
        "INSERT INTO review_queue "
        "(review_id, export_id, beat_plan_id, beat_plan_revision, template_id, "
        "platform, state, created_at_utc, output_otio_relpath) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
        (review_id, export_id, bp_id, revision, tmpl_id, platform, now, export_relpath),
    )
    conn.commit()
    conn.close()
    rv_journal(state, review_id, "create", {"export_id": export_id, "beat_plan_id": bp_id})


def _update_review_mp4(state: Path, review_id: str, mp4_relpath: str) -> None:
    from erf.review import _connect as rv_connect
    conn = rv_connect(state / "erf.db")
    conn.execute(
        "UPDATE review_queue SET output_mp4_relpath=? WHERE review_id=?",
        (mp4_relpath, review_id),
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Per-thread render                                                            #
# --------------------------------------------------------------------------- #

def _render_thread(
    thread_id: str,
    bp_id: str,
    catalog: Path,
    state: Path,
    proxy_dir: Path,
    journal_db: Path,
    dry_run: bool,
) -> dict[str, Any]:
    t0 = time.monotonic()
    result: dict[str, Any] = {"thread_id": thread_id, "bp_id": bp_id, "ok": False}

    try:
        from erf.export import export as do_export
        from erf.render import render as do_render
        from erf.review import _make_review_id
        from erf.atomicio import write_json

        # 1. Load beat plan
        bp = _load_latest_beat_plan(bp_id, catalog)
        beats = bp.get("beats", [])
        if not beats:
            result["error"] = "beat_plan has no beats"
            return result

        log.info("[%s] beat_plan loaded: %d beats, topic=%r", thread_id[-8:], len(beats), bp.get("topic", "")[:60])

        # 2. Build clips
        clips, adjusted_beats = _build_clips(beats, catalog)
        bp_adjusted = {**bp, "beats": adjusted_beats}

        if dry_run:
            result["ok"] = True
            result["dry_run"] = True
            result["beats"] = len(beats)
            result["elapsed_s"] = round(time.monotonic() - t0, 1)
            log.info("[%s] DRY RUN — skipping export+render", thread_id[-8:])
            return result

        # 3. Export (OTIO/FCPXML manifest)
        export_manifest = do_export(
            clips, fmt="fcpxml", query=bp.get("topic"), config_path=_CONFIG_PATH
        )
        export_id = export_manifest["export_id"]
        export_relpath = export_manifest["output_relpath"]
        log.info("[%s] export_id=%s", thread_id[-8:], export_id)

        # 4. Review queue row
        review_id = _make_review_id()
        _insert_review_queue(state, review_id, export_id, bp_id, bp, export_relpath)

        # 5. B-roll expansion
        render_clips, render_beats = _expand_broll(adjusted_beats, clips, proxy_dir)
        broll_injected = len(render_clips) - len(clips)
        if broll_injected > 0:
            log.info("[%s] B-roll expanded: +%d clips", thread_id[-8:], broll_injected)
        render_bp = {**bp_adjusted, "beats": render_beats}

        # 6. Render
        exports_dir = catalog / "exports"
        log.info("[%s] Starting render (%d clips)…", thread_id[-8:], len(render_clips))
        render_result = do_render(
            beat_plan=render_bp,
            clips=render_clips,
            export_id=export_id,
            catalog=catalog,
            proxy_dir=proxy_dir,
            output_dir=exports_dir,
            # active_speaker centres the crop on the speaking face (falls back to
            # face_centroid when no ASD data); avoids the "half of a non-speaker's
            # head" framing that face_centroid produces on multi-person shots.
            reframe_strategy="active_speaker",
            music_bed_path=None,
            journal_db=journal_db,
            config_path=_CONFIG_PATH,
        )

        mp4_path = render_result["output_mp4_path"]
        _mp4_abs = Path(mp4_path)
        if not _mp4_abs.is_absolute():
            _mp4_abs = _ROOT / _mp4_abs
        mp4_relpath = str(_mp4_abs.resolve().relative_to(_ROOT.resolve()))
        duration_s = render_result["duration_s"]

        # 7. Patch export manifest and review_queue with mp4 path
        manifest_path = catalog / "exports" / f"{export_id}.json"
        if manifest_path.exists():
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            mdata["output_mp4_relpath"] = mp4_relpath
            write_json(manifest_path, mdata, indent=2)

        _update_review_mp4(state, review_id, mp4_relpath)

        elapsed = round(time.monotonic() - t0, 1)
        log.info("[%s] DONE in %.0fs — %s (%.1fs)", thread_id[-8:], elapsed, mp4_relpath, duration_s)

        result.update({
            "ok": True,
            "review_id": review_id,
            "export_id": export_id,
            "mp4_relpath": mp4_relpath,
            "video_duration_s": duration_s,
            "broll_clips_injected": broll_injected,
            "elapsed_s": elapsed,
        })

    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 1)
        log.exception("[%s] FAILED after %.0fs: %s", thread_id[-8:], elapsed, exc)
        result["error"] = str(exc)
        result["elapsed_s"] = elapsed

    return result


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch render narrative gate_passed threads")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Concurrent renders (default 3; DF3 runs as subprocess so GPU pressure is per-process)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load beat plans and build clips but skip export+render")
    args = ap.parse_args()

    cfg = _load_cfg()
    catalog = cfg["catalog"]
    state = cfg["state"]
    proxy_dir = _ROOT / "media" / "proxies"
    journal_db = state / "journal.db"

    # Load gate_passed threads
    conn = sqlite3.connect(str(_PIPELINE_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT thread_id, bp_id FROM narr_pipeline_state WHERE state='gate_passed' ORDER BY thread_id"
    ).fetchall()
    conn.close()

    if not rows:
        log.error("No gate_passed threads found in %s", _PIPELINE_DB)
        sys.exit(1)

    work = [(r["thread_id"], r["bp_id"]) for r in rows]
    log.info("Found %d gate_passed threads to render (concurrency=%d)", len(work), args.concurrency)

    t_start = time.monotonic()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                _render_thread,
                thread_id, bp_id,
                catalog, state, proxy_dir, journal_db,
                args.dry_run,
            ): thread_id
            for thread_id, bp_id in work
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"thread_id": tid, "ok": False, "error": str(exc)}
            results.append(res)

    elapsed_total = round(time.monotonic() - t_start, 1)

    # ---------- Report -------------------------------------------------------
    passed = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    print("\n" + "=" * 70)
    print(f"  Batch render complete — {len(passed)}/{len(results)} OK  ({elapsed_total:.0f}s total)")
    print("=" * 70)

    print(f"\n{'Thread':<22}  {'BP':<32}  {'MP4 / Error'}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["thread_id"]):
        tid = r["thread_id"][-12:]
        bp = (r.get("bp_id") or "")[-20:]
        if r.get("ok"):
            mp4 = r.get("mp4_relpath", "")
            dur = r.get("video_duration_s", 0)
            broll = r.get("broll_clips_injected", 0)
            broll_tag = f" [+{broll} broll]" if broll else ""
            print(f"  {tid:<22}  {bp:<32}  {mp4}  ({dur:.1f}s){broll_tag}")
        else:
            err = (r.get("error") or "unknown")[:60]
            print(f"  {tid:<22}  {bp:<32}  FAIL: {err}")

    if failed:
        print(f"\n  {len(failed)} failures:")
        for r in failed:
            print(f"    {r['thread_id']}  {(r.get('error') or '')[:80]}")

    print(f"\n  Wall time: {elapsed_total:.0f}s   Avg per thread: {elapsed_total/len(results):.0f}s")
    print()

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

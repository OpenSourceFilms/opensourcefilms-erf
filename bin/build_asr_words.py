"""Stage W: Speaker-agnostic word list into catalog/asr_words/."""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger("build_asr_words")

# Outcome spine (MASTER §5.2.1): stage W records an explicit outcome per media so
# the log-readers (reconcile_status.py predicate on asr_words/, pipeline_status.py
# CRITICAL) are never blind to a who's-speaking-spine gap. Guarded so a missing
# import can never break the stage.
try:
    from erf.paths import catalog_root
    from erf import outcomes
except Exception:  # pragma: no cover
    outcomes = None
    def catalog_root() -> Path:
        return ROOT / "catalog"

CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam; was missed — DECISIONS.md D-Log-40)


def build_asr_words(media_id: str, *, force: bool = False) -> dict:
    out = CAT / "asr_words" / f"{media_id}.jsonl"
    if out.exists() and not force:
        return {"ok": True, "status": "exists", "media_id": media_id}

    wx = CAT / "asr" / f"{media_id}.words.wx.jsonl"
    std = CAT / "asr" / f"{media_id}.words.jsonl"
    if wx.exists():
        src, kind = wx, "wx"
    elif std.exists():
        src, kind = std, "std"
    else:
        return {"ok": False, "reason": "no_source", "media_id": media_id}

    words = []
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            w = json.loads(line)
        except Exception:
            continue
        words.append({
            "word_id": w.get("word_id") or f"{media_id}:w:{i:06d}",
            "media_id": media_id,
            "text": (w.get("text") or "").strip(),
            "start_ms": int(w.get("start_ms", 0)),
            "end_ms": int(w.get("end_ms", 0)),
            "score": float(w.get("score") or w.get("prob") or 0.0),
        })

    if not words:
        return {"ok": False, "reason": "empty_source", "media_id": media_id}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(w, ensure_ascii=False) for w in words) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return {"ok": True, "status": "written", "media_id": media_id, "n_words": len(words)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.all:
        # Discover media with EITHER WhisperX words OR std words — build_asr_words
        # falls back wx→std, so keying --all on wx alone silently drops media whose
        # D0/WhisperX pass produced nothing but that still have canonical A words
        # (surfaced 2026-07-05 as 2 has_speech=True media missing W). No-speech
        # placeholders resolve to a `skip` outcome, not a false artifact.
        wx = {p.name.split(".")[0] for p in (CAT / "asr").glob("*.words.wx.jsonl")}
        std = {p.name.split(".")[0] for p in (CAT / "asr").glob("*.words.jsonl")}
        mids = sorted(wx | std)
    else:
        mids = args.media_ids
    if not mids:
        ap.print_help(); sys.exit(1)
    ok = skip = err = 0
    for mid in mids:
        r = build_asr_words(mid, force=args.force)
        if r.get("ok") and r.get("status") == "written":
            log.info("  %s → %d words", mid, r.get("n_words", 0)); ok += 1
            if outcomes:
                outcomes.record(mid, "W", "ok", counts={"n_words": r.get("n_words", 0)},
                                artifacts=[f"catalog/asr_words/{mid}.jsonl"], producer="build_asr_words.py")
        elif r.get("ok"):  # already present (exists) — artifact on disk, work skipped
            skip += 1
            if outcomes:
                outcomes.record(mid, "W", "ok", artifacts=[f"catalog/asr_words/{mid}.jsonl"],
                                producer="build_asr_words.py")
        else:  # no upstream ASR words to project → no artifact expected (skip, not error)
            log.warning("  %s no artifact: %s", mid, r.get("reason")); err += 1
            if outcomes:
                outcomes.record(mid, "W", "skip", reason=r.get("reason") or "no_source",
                                producer="build_asr_words.py")
    log.info("Done: %d written, %d exist, %d no-artifact", ok, skip, err)


if __name__ == "__main__":
    main()

"""Stage D: Run pyannote/speaker-diarization-community-1 on media audio.

Requires: venvs/diarize (pyannote.audio 4.x, torch, torchaudio)
Output:
  catalog/diar_hypotheses/{mid}/community1.jsonl       — standard diarization
  catalog/diar_hypotheses/{mid}/community1_excl.jsonl  — exclusive diarization (better for word alignment)

Usage:
  python venvs/diarize/bin/python bin/run_diar_community1.py [media_ids...] [--all]
"""
from __future__ import annotations
import argparse, json, logging, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAT = ROOT / "catalog"
PROXY_DIR = ROOT / "media" / "proxies"
MODEL_ID = "pyannote/speaker-diarization-community-1"

log = logging.getLogger("run_diar_community1")

# Outcome spine (MASTER §5.2.1): stage DX records an explicit outcome per media so
# reconcile_status.py (predicate on diar_hypotheses/{m}/community1{,_excl}.jsonl)
# and pipeline_status.py CRITICAL are never blind to a spine gap. Guarded import.
try:
    sys.path.insert(0, str(ROOT / "src"))
    from erf import outcomes
except Exception:  # pragma: no cover
    outcomes = None


def _extract_wav(media_id: str, tmp_wav: Path) -> bool:
    proxy = PROXY_DIR / f"{media_id}.mp4"
    if not proxy.exists():
        log.warning("%s: no proxy at %s", media_id, proxy)
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(proxy),
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(tmp_wav),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and tmp_wav.exists() and tmp_wav.stat().st_size > 0


def _annotation_to_jsonl(annotation, hypothesis_name: str) -> list[dict]:
    turns = []
    for i, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        turns.append({
            "hypothesis": hypothesis_name,
            "turn_index": i,
            "speaker_label": str(speaker),
            "start_ms": int(turn.start * 1000),
            "end_ms": int(turn.end * 1000),
        })
    return turns


def _write_hypothesis(turns: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in turns) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)


def run_community1(media_id: str, pipeline, *, force: bool = False) -> dict:
    out_dir = CAT / "diar_hypotheses" / media_id
    out_std = out_dir / "community1.jsonl"
    out_excl = out_dir / "community1_excl.jsonl"

    if out_std.exists() and out_excl.exists() and not force:
        return {"ok": True, "status": "exists", "media_id": media_id}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = Path(f.name)
    try:
        if not _extract_wav(media_id, tmp_wav):
            return {"ok": False, "reason": "audio_extract_failed", "media_id": media_id}

        import torch
        with torch.no_grad():
            out = pipeline(str(tmp_wav))

        std_turns = _annotation_to_jsonl(out.speaker_diarization, "community1")
        excl_turns = _annotation_to_jsonl(out.exclusive_speaker_diarization, "community1_excl")

        _write_hypothesis(std_turns, out_std)
        _write_hypothesis(excl_turns, out_excl)

        return {
            "ok": True, "status": "written", "media_id": media_id,
            "n_std": len(std_turns), "n_excl": len(excl_turns),
        }
    except Exception as exc:
        log.error("%s: error: %s", media_id, exc)
        return {"ok": False, "reason": str(exc)[:200], "media_id": media_id}
    finally:
        tmp_wav.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all:
        mids = sorted({p.stem for p in PROXY_DIR.glob("*.mp4")})
    else:
        mids = args.media_ids
    if not mids:
        ap.print_help(); sys.exit(1)

    log.info("Loading %s...", MODEL_ID)
    try:
        from pyannote.audio import Pipeline
        import torch
        pipeline = Pipeline.from_pretrained(MODEL_ID)
        if args.device == "cuda" and torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            log.info("Running on CUDA")
        else:
            log.info("Running on CPU")
    except Exception as exc:
        log.error("Pipeline load failed: %s", exc)
        sys.exit(1)

    ok = skip = err = 0
    arts = lambda m: [f"catalog/diar_hypotheses/{m}/community1.jsonl",
                      f"catalog/diar_hypotheses/{m}/community1_excl.jsonl"]
    for mid in mids:
        log.info("  %s...", mid)
        r = run_community1(mid, pipeline, force=args.force)
        if r.get("ok") and r.get("status") == "written":
            log.info("    → std=%d excl=%d turns", r.get("n_std", 0), r.get("n_excl", 0))
            ok += 1
            if outcomes:
                outcomes.record(mid, "DX", "ok",
                                counts={"n_std": r.get("n_std", 0), "n_excl": r.get("n_excl", 0)},
                                artifacts=arts(mid), producer="run_diar_community1.py")
        elif r.get("ok"):  # already present (exists)
            skip += 1
            if outcomes:
                outcomes.record(mid, "DX", "ok", artifacts=arts(mid), producer="run_diar_community1.py")
        else:  # extract failure or diarization exception → genuine error (no artifact)
            log.warning("    FAILED: %s", r.get("reason"))
            err += 1
            if outcomes:
                outcomes.record(mid, "DX", "error", reason=r.get("reason") or "unknown",
                                producer="run_diar_community1.py")

    log.info("Done: %d ok, %d skipped, %d errors", ok, skip, err)


if __name__ == "__main__":
    main()

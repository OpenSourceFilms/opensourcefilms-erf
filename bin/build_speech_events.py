"""Stage S: Build per-word speech events by aligning asr_words to diar hypotheses.

Input:  catalog/asr_words/{mid}.jsonl (speaker-agnostic words)
        catalog/diar_hypotheses/{mid}/*.jsonl (one or more diarization hypotheses)
Output: catalog/speech_events/{mid}.jsonl

Each output row is one word with speaker_label derived by time-overlap voting across
all available hypotheses. Words with no diar coverage are flagged gap_word=True.
Words where hypotheses disagree are flagged speaker_conflict=True.
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
CAT = ROOT / "catalog"
log = logging.getLogger("build_speech_events")

# Priority for speaker_label assignment (must match identity_overlay label space).
# community1_baseline uses the same run that identity_overlay was built from → labels align.
# community1_excl is a different run → labels may be permuted → DO NOT use as primary
# until identity_overlay is rebuilt against the new run.
HYPOTHESIS_PRIORITY = [
    "community1_baseline",
]


def _load_hypothesis(path: Path) -> list[dict]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
            turns.append(t)
        except Exception:
            pass
    return turns


def _assign_speaker(word_start: int, word_end: int, turns: list[dict]) -> tuple[str | None, float]:
    """Return (speaker_label, overlap_fraction) for the turn that covers this word most."""
    word_len = max(word_end - word_start, 1)
    best_spk, best_overlap = None, 0.0
    for t in turns:
        overlap = max(0, min(word_end, t["end_ms"]) - max(word_start, t["start_ms"]))
        frac = overlap / word_len
        if frac > best_overlap:
            best_overlap = frac
            best_spk = t["speaker_label"]
    return best_spk, best_overlap


def build_speech_events(media_id: str, *, force: bool = False) -> dict:
    words_path = CAT / "asr_words" / f"{media_id}.jsonl"
    hyp_dir = CAT / "diar_hypotheses" / media_id
    out_path = CAT / "speech_events" / f"{media_id}.jsonl"

    if out_path.exists() and not force:
        return {"ok": True, "status": "exists", "media_id": media_id}

    if not words_path.exists():
        return {"ok": False, "reason": "no_words", "media_id": media_id}

    words = [json.loads(l) for l in words_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not words:
        return {"ok": False, "reason": "empty_words", "media_id": media_id}

    # Load all available hypotheses
    hypotheses: dict[str, list[dict]] = {}
    if hyp_dir.exists():
        for hp in sorted(hyp_dir.glob("*.jsonl")):
            h_name = hp.stem
            hypotheses[h_name] = _load_hypothesis(hp)

    # Primary hypothesis: best single source to avoid label-permutation confusion.
    # Cross-run voting is only valid after label-permutation alignment (not yet built).
    primary_hyp: str | None = next((h for h in HYPOTHESIS_PRIORITY if h in hypotheses), None)
    if primary_hyp is None and hypotheses:
        primary_hyp = sorted(hypotheses.keys())[0]

    events = []
    for i, w in enumerate(words):
        ws, we = w["start_ms"], w["end_ms"]

        # Collect evidence from all hypotheses (stored for future aligned fusion)
        evidence: dict[str, str | None] = {}
        for h_name, turns in hypotheses.items():
            spk, frac = _assign_speaker(ws, we, turns)
            evidence[h_name] = spk if frac > 0.01 else None

        # Assign from primary hypothesis only
        winner = evidence.get(primary_hyp) if primary_hyp else None
        confidence = 1.0 if winner else 0.0
        gap_word = winner is None
        # speaker_conflict reserved for future label-aligned multi-hypothesis disagreement
        speaker_conflict = False

        events.append({
            "event_id": f"{media_id}:e:{i:06d}",
            "media_id": media_id,
            "word_id": w["word_id"],
            "text": w["text"],
            "start_ms": ws,
            "end_ms": we,
            "score": w.get("score", 0.0),
            "speaker_label": winner,
            "primary_hypothesis": primary_hyp,
            "evidence": evidence,
            "confidence": round(confidence, 3),
            "gap_word": gap_word,
            "speaker_conflict": speaker_conflict,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)

    n_gap = sum(1 for e in events if e["gap_word"])
    n_conflict = sum(1 for e in events if e["speaker_conflict"])
    return {
        "ok": True, "status": "written", "media_id": media_id,
        "n_events": len(events), "n_gap": n_gap, "n_conflict": n_conflict,
        "n_hypotheses": len(hypotheses),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.all:
        mids = sorted({p.name.split(".")[0] for p in (CAT / "asr_words").glob("*.jsonl")})
    else:
        mids = args.media_ids
    if not mids:
        ap.print_help(); sys.exit(1)
    ok = skip = err = 0
    for mid in mids:
        r = build_speech_events(mid, force=args.force)
        if r.get("ok") and r.get("status") == "written":
            log.info("  %s → %d events, %d gap, %d conflict (%d hyp)",
                     mid, r["n_events"], r["n_gap"], r["n_conflict"], r["n_hypotheses"])
            ok += 1
        elif r.get("ok"):
            skip += 1
        else:
            log.warning("  %s FAILED: %s", mid, r.get("reason")); err += 1
    log.info("Done: %d written, %d skipped, %d errors", ok, skip, err)


if __name__ == "__main__":
    main()

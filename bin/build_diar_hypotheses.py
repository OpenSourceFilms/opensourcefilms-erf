"""Stage D (baseline): Migrate existing diar.jsonl into canonical diar_hypotheses/ format.

Source: catalog/asr/{mid}.diar.jsonl
Output: catalog/diar_hypotheses/{mid}/community1_baseline.jsonl

This preserves existing diarization as a named hypothesis, enabling multi-hypothesis fusion
without losing prior work. New hypothesis files (community1_excl, nemo_sortformer) are added
by their respective runners.
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAT = ROOT / "catalog"
ASR_DIR = CAT / "asr"
HYP_DIR = CAT / "diar_hypotheses"
HYPOTHESIS_NAME = "community1_baseline"

log = logging.getLogger("build_diar_hypotheses")


def build_baseline(media_id: str, *, force: bool = False) -> dict:
    src = ASR_DIR / f"{media_id}.diar.jsonl"
    if not src.exists():
        return {"ok": False, "reason": "no_source", "media_id": media_id}

    out_dir = HYP_DIR / media_id
    out = out_dir / f"{HYPOTHESIS_NAME}.jsonl"
    if out.exists() and not force:
        return {"ok": True, "status": "exists", "media_id": media_id}

    turns = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        turns.append({
            "hypothesis": HYPOTHESIS_NAME,
            "turn_index": t.get("turn_index", len(turns)),
            "speaker_label": t.get("speaker_label", "SPEAKER_??"),
            "start_ms": int(t.get("start_ms", 0)),
            "end_ms": int(t.get("end_ms", 0)),
        })

    if not turns:
        return {"ok": False, "reason": "empty_source", "media_id": media_id}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in turns) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return {"ok": True, "status": "written", "media_id": media_id, "n_turns": len(turns)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("media_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.all:
        mids = sorted({p.name.split(".")[0] for p in ASR_DIR.glob("*.diar.jsonl")})
    else:
        mids = args.media_ids
    if not mids:
        ap.print_help(); sys.exit(1)
    ok = skip = err = 0
    for mid in mids:
        r = build_baseline(mid, force=args.force)
        if r.get("ok") and r.get("status") == "written":
            log.info("  %s → %d turns", mid, r.get("n_turns", 0)); ok += 1
        elif r.get("ok"):
            skip += 1
        else:
            log.warning("  %s FAILED: %s", mid, r.get("reason")); err += 1
    log.info("Done: %d written, %d skipped, %d errors", ok, skip, err)


if __name__ == "__main__":
    main()

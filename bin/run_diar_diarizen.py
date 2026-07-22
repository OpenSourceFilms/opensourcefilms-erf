#!/usr/bin/env python3
"""Run the DiariZen diarizer on media and write hypothesis turns JSONL.

Bake-off tool (D-Log-35 decision 7a) — NOT a registry stage: stdout logging
only, no outcome-spine writes. Writes to the catalog root resolved by
erf.paths.catalog_root(), so a shadow bake-off arm sets ERF_CATALOG_ROOT and
production catalog/ is never touched.

Requires: venvs/diarizen (diarizen editable + vendored pyannote-audio fork).
Weights: cached HF hub at weights/diarizen/hub (offline; D-Log-30).

Usage:
  ERF_CATALOG_ROOT=<shadow>/catalog venvs/diarizen/bin/python \
      bin/run_diar_diarizen.py m_abc123 [m_def456 ...] [--force]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root

CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam)
PROXY_DIR = ROOT / "media" / "proxies"
MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md"


def _extract_wav(media_id: str, tmp_wav: Path) -> bool:
    proxy = PROXY_DIR / f"{media_id}.mp4"
    if not proxy.is_file():
        print(f"WARN {media_id}: proxy missing at {proxy}")
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(proxy),
           "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_wav)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and tmp_wav.is_file() and tmp_wav.stat().st_size > 0


def _write_jsonl(turns: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def main() -> None:
    # Offline cache must be set before diarizen/HF imports.
    os.environ.setdefault("HF_HOME", str(ROOT / "weights" / "diarizen"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("media_ids", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from diarizen.pipelines.inference import DiariZenPipeline
    pipeline = DiariZenPipeline.from_pretrained(MODEL_ID)

    failed = 0
    for mid in args.media_ids:
        out_path = CAT / "diar_hypotheses" / mid / "diarizen.jsonl"
        if out_path.exists() and not args.force:
            print(f"SKIP {mid}: exists")
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = Path(f.name)
        try:
            if not _extract_wav(mid, tmp_wav):
                print(f"FAIL {mid}: audio_extract_failed")
                failed += 1
                continue
            try:
                ann = pipeline(str(tmp_wav))
            except Exception as exc:
                print(f"FAIL {mid}: inference error: {exc}")
                failed += 1
                continue
            turns = [{"hypothesis": "diarizen", "turn_index": i,
                      "speaker_label": str(sp),
                      "start_ms": int(round(t.start * 1000)),
                      "end_ms": int(round(t.end * 1000))}
                     for i, (t, _, sp) in enumerate(ann.itertracks(yield_label=True))]
            _write_jsonl(turns, out_path)
            labels = {t["speaker_label"] for t in turns}
            span = (min(t["start_ms"] for t in turns), max(t["end_ms"] for t in turns)) if turns else (0, 0)
            print(f"OK {mid}: turns={len(turns)} labels={len(labels)} span_ms={span[0]}-{span[1]}")
        finally:
            tmp_wav.unlink(missing_ok=True)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

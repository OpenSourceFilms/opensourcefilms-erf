#!/usr/bin/env python3
"""RETRANSCRIBE — re-run Stage A (max-quality ASR + diarization) on SOURCE files
in place, then rebuild transcript + evidence per media.

Transcribes from the original source audio (not the lossy proxy) with the stronger
decoding params now in asr.py. media_id is content-hashed, so this overwrites
words/segments/diar for that media. Proxies are a separate stage and untouched.
Built to run continuously in the background as footage is digested.

Usage: PYTHONPATH=src venvs/embed/bin/python bin/retranscribe.py [--only-interviews] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from erf.asr import run as asr_run
try:
    from erf.identity import run as identity_run
except Exception:
    identity_run = None
try:
    from erf.runlog import log_event, heartbeat
except Exception:
    log_event = None
    heartbeat = None

ROOT = Path("/workspace/erf")
CFG = ROOT / "state" / "config.toml"
PY = sys.executable
INTERVIEW_TAG = "example_interview_14.6.26"


def sources(only_interviews: bool) -> list[Path]:
    files = [p for p in Path("/workspace/BTS").rglob("*")
             if p.suffix.lower() in (".mp4", ".mov")]
    files.sort(key=lambda p: (0 if INTERVIEW_TAG in str(p) else 1, str(p)))
    if only_interviews:
        files = [p for p in files if INTERVIEW_TAG in str(p)]
    return files


def _rebuild(mid: str) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    for script in ("bin/build_transcript.py", "bin/build_evidence.py"):
        subprocess.run([PY, script, "--media-ids", mid, "--force"],
                       cwd=str(ROOT), env=env, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-interviews", action="store_true")
    a = ap.parse_args()

    files = sources(a.only_interviews)
    if a.limit:
        files = files[: a.limit]
    print(f"retranscribe {len(files)} sources (max-quality ASR + diarization)…", flush=True)

    ok = err = 0
    for i, src in enumerate(files, 1):
        t0 = time.monotonic()
        if heartbeat:
            heartbeat("retranscribe", {"i": i, "n": len(files), "ok": ok, "err": err, "current": src.name})
        try:
            m = asr_run(src, config_path=CFG)
            mid = getattr(m, "media_id", None)
            # asr.run() CATCHES failures and returns a manifest with .error set
            # (e.g. a missing dependency). Read the outcome — never report ok when
            # the stage actually failed. (logging directive: surface silent failures.)
            asr_err = getattr(m, "error", None)
            if not mid or not getattr(m, "ok", True) or asr_err:
                err += 1
                if log_event and mid:
                    log_event("RETRANSCRIBE", mid, "error",
                              {"src": src.name, "asr_error": str(asr_err)[:300]})
                print(f"[{i}/{len(files)}] ASR FAILED {src.name}: {asr_err}", flush=True)
                continue
            # Re-resolve identity on the FRESH diarization so speaker→person (→caption
            # colour) stays consistent; pyannote labels aren't stable across runs.
            id_ok = True
            if identity_run is not None:
                try:
                    identity_run(mid, config_path=str(CFG))
                except Exception as ie:
                    id_ok = False
                    if log_event:
                        log_event("RETRANSCRIBE", mid, "error", {"step": "identity", "error": str(ie)[:300]})
            _rebuild(mid)
            if log_event:
                log_event("RETRANSCRIBE", mid, "ok",
                          {"src": src.name, "counts": getattr(m, "counts", {}), "identity_ok": id_ok})
            ok += 1
            print(f"[{i}/{len(files)}] ok {mid} {src.name} ({time.monotonic()-t0:.0f}s, id={id_ok})", flush=True)
        except Exception as e:
            err += 1
            print(f"[{i}/{len(files)}] FAIL {src.name}: {e}", flush=True)
    print(f"done ok={ok} err={err}", flush=True)


if __name__ == "__main__":
    main()

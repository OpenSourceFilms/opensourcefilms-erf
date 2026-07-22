#!/usr/bin/env python3
"""From-scratch ERF full-corpus rebuild orchestrator.

Runs each pipeline stage as a subprocess with the correct venv python (it imports
no *heavy* erf modules itself — pure launcher; it does import the dependency-free
src/erf/stages.py registry at module load to validate its stage universe, D-Log-04
Phase 2). The catalog move-aside MUST be done before running this; the
orchestrator does not wipe anything.

Stage order is the DAG: BASE (batch_ingest_bts: A→B0→FTS→B1→C→B2/B3/P→B4→C_vis→
Q0/E0/tags→F0→F1) then the creative chain, then voice enrollment, then a synthetic
purge + disk-truth reconcile of the journal.

Phase 2 report (docs/OBSERVABILITY_REDESIGN.md §4): the STAGES list below stays a
hand-ordered execution plan — it interleaves a composite bootstrap step (BASE =
batch_ingest_bts.py, itself running ~13 registry stages as one subprocess) with
per-stage subprocess invocations and two meta reconcile actions (PURGE,
RECONCILE) that are not registry stages at all. That interleaving + venv/cmd
mapping is execution config the registry doesn't (and shouldn't) encode, so it
is NOT auto-derived. What Phase 2 DOES enforce: every non-meta `name` in STAGES
is validated against the registry at import time (`_validate_stage_universe`
below) — an unknown/renamed stage now fails fast instead of silently drifting —
and READBACK_STAGES (a genuine unordered stage SET) is fully derived from
`stages.speech_spine_stages()`.

Usage:
  PYTHONPATH=src python bin/rebuild_all.py [--dry-run] [--only N ...] [--skip N ...] [--from N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/erf")
HB = ROOT / "logs.all" / "_latest" / "heartbeat" / "rebuild_all.json"

sys.path.insert(0, str(ROOT / "src"))
from erf import stages as _registry  # noqa: E402  (dependency-free; the ONE stage universe)
from erf import outcomes as oc  # noqa: E402  (D-Log-21: stage-scoped run-outcome + per-stage tee)


def VP(name: str) -> str:
    return str(ROOT / "venvs" / name / "bin" / "python")


# Ordered DAG. cmd[0] = venv python. fatal stages abort the run on nonzero rc.
STAGES = [
    {"name": "BASE",      "cmd": [VP("embed"),         "bin/batch_ingest_bts.py"],                       "fatal": True},
    {"name": "REDIARIZE", "cmd": [VP("audio"),         "bin/rediarize.py", "--all"],                     "fatal": False},
    {"name": "P0",        "cmd": [VP("groundingdino"), "bin/punctuation_overlay.py", "--all"],           "fatal": False},
    {"name": "D0",        "cmd": [VP("anyenhance"),    "bin/d0_preprocess.py", "--all", "--force"],      "fatal": False},
    # "Who's-speaking" spine (MASTER §5 DAG: W→RG→RP→DX→SEV2→SEV2_ADJ→PSTREAM).
    # T0 (build_transcript) is a projection of this spine, so all seven must
    # complete before T0 runs. Previously hand-run only; this makes the DAG faithful.
    {"name": "W",         "cmd": [VP("embed"),         "bin/build_asr_words.py", "--all"],               "fatal": False},
    {"name": "RG",        "cmd": [VP("embed"),         "bin/build_recall_gaps.py"],                      "fatal": False},
    # --min-priority low (D-Recall-01, 2026-07-08): the undocumented "high" CLI
    # default meant only 44/1415 corpus-wide detected gaps (3.1%) were ever
    # attempted -- 0 of 1070 low_conf_asr gaps ever reach "high" priority by
    # construction. recall_pass.py writes CANDIDATE words only + already gates
    # hallucinations (should_drop_segment); downstream adjudication owns
    # precision, so it's safe to attempt everything the detector already flagged.
    {"name": "RP",        "cmd": [VP("anyenhance"),    "bin/recall_pass.py", "--min-priority", "low"],   "fatal": False},
    {"name": "DX",        "cmd": [VP("diarize"),       "bin/run_diar_community1.py", "--all"],           "fatal": False},
    {"name": "SEV2",      "cmd": [VP("embed"),         "bin/build_speech_events_v2.py"],                 "fatal": False},
    {"name": "SEV2_ADJ",  "cmd": [VP("diarize"),       "bin/adjudicate_speech_events.py"],               "fatal": False},
    {"name": "PSTREAM",   "cmd": [VP("diarize"),       "bin/build_parallel_streams.py", "--all", "--device", "cuda"], "fatal": False},
    {"name": "T0",        "cmd": [VP("embed"),         "bin/build_transcript.py", "--all", "--force"],   "fatal": False},
    {"name": "F2",        "cmd": [VP("embed"),         "bin/run_f2_batch.py"],                           "fatal": False},
    {"name": "V0",        "cmd": [VP("anyenhance"),    "bin/v0_visual_grounding.py", "--force"],         "fatal": False},
    {"name": "E1",        "cmd": [VP("embed"),         "bin/run_e1.py", "--force"],                      "fatal": False},
    {"name": "CARD",      "cmd": [VP("embed"),         "bin/build_digest.py", "--only-cards"],           "fatal": False},
    {"name": "EV",        "cmd": [VP("embed"),         "bin/build_evidence.py", "--all", "--force"],     "fatal": False},
    {"name": "LABEL",     "cmd": [VP("embed"),         "bin/label_editorial.py", "--all", "--force"],    "fatal": False},
    # VOICE (bin/voice_enroll.py) removed 2026-07-08 (area-2 audit / D-Identity-05):
    # wrote identity_overlay person_id assignments from catalog/voice_gallery/, a
    # 256-dim WeSpeaker centroid gallery last built 2026-06-24 -- stale since
    # before either ReDimNet adoption, entirely disconnected from the current
    # gallery-anchored resolver (state/identity/gallery_redimnet2_b6) and its
    # guards (do-not-seed, dedup, same-clip-conflict). It only touched
    # method="none" slots so it couldn't clobber resolver-confirmed assignments,
    # but it could silently assign genuinely-unknown speakers using a stale,
    # unmaintained gallery outside every guard the current resolver enforces.
    # See DECISIONS.md 2026-07-08.
    {"name": "PURGE",     "cmd": [VP("embed"),         "bin/reconcile_status.py", "--purge-synthetic"],  "fatal": False},
    {"name": "RECONCILE", "cmd": [VP("embed"),         "bin/reconcile_status.py", "--reconcile"],        "fatal": False},
]

# Not registry stages: BASE is a composite bootstrap subprocess (batch_ingest_bts.py
# runs ~13 registry stages internally); PURGE/RECONCILE are reconcile_status.py
# CLI invocations, not pipeline stages.
_META_STEPS = {"BASE", "PURGE", "RECONCILE"}


def _validate_stage_universe() -> None:
    """Phase 2 (D-Log-04) guard: every non-meta STAGES[i]['name'] above must be a
    real stage in the src/erf/stages.py registry. This does not (and safely
    cannot) derive the ORDER/venv/cmd plan above from the registry — see the
    module docstring for why — but it does close the drift hole: an unknown or
    renamed stage name here now fails fast at import time instead of silently
    running (or silently never running) something the registry doesn't know
    about. Fails loud by design (P4: no silent swallow)."""
    unknown = [s["name"] for s in STAGES
               if s["name"] not in _META_STEPS and s["name"] not in _registry.STAGES]
    if unknown:
        raise RuntimeError(
            f"rebuild_all.py STAGES references unknown stage(s) not in the "
            f"erf.stages registry: {unknown}"
        )


_validate_stage_universe()


def setup_logging(stamp: str) -> logging.Logger:
    (ROOT / "logs.all").mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("rebuild_all")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(ROOT / "logs.all" / f"rebuild_all_{stamp}.log")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def write_hb(current: str, done: list[str], elapsed: float) -> None:
    try:
        HB.parent.mkdir(parents=True, exist_ok=True)
        HB.write_text(json.dumps({
            "component": "rebuild_all",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "current_stage": current, "done": done, "elapsed_s": round(elapsed, 1),
        }))
    except Exception:
        pass


# Who's-speaking spine — the stages whose per-media outcomes the orchestrator reads
# back after a run (MASTER §5.2.2: reading the log is the last step of every run).
# Phase 2 (D-Log-04): derived from the registry's speech_spine_stages() — a real
# stage SET, not execution config, so (unlike STAGES above) it IS fully rewired.
# Registry insertion order reproduces the original literal's order exactly
# (verified: W, RG, RP, DX, SEV2, SEV2_ADJ, PSTREAM, T0, EV).
READBACK_STAGES = list(_registry.speech_spine_stages().keys())


def read_back(log: logging.Logger, run_stage_names: list[str] | None = None) -> dict:
    """Read the outcome spine we just wrote: regenerate health.json and surface any
    errored/unrecorded spine media. A run that writes outcomes but never reads its
    own tail can 'succeed' over a corpus with silently-errored media.

    D-Log-21: also independently re-reads the corpus-scoped run-outcome `main()`'s
    loop now records for every stage (`__corpus__`, status ok/error by subprocess
    rc) — the run-1 post-mortem blind spot: a whole-stage startup crash (rc!=0
    before any media was touched, e.g. SEV2_ADJ/PSTREAM's torch/driver mismatch,
    D-Log-14) produces ZERO per-media outcomes, so the existing `errors`/
    `unrecorded` checks below (both `oc.media_by_status`-based, which globs only
    `m_*.json` status files) cannot see it — `__corpus__` isn't a real media id.
    Read from the journal fresh here rather than trusting this run's own in-memory
    `failures` list, so this actually proves the write path worked end to end."""
    summary: dict = {"errors": {}, "unrecorded": {}, "critical_gaps": {}, "stage_crashes": {}}
    if run_stage_names:
        for st in run_stage_names:
            rec = oc.get("__corpus__", st)
            if rec and rec.get("status") == "error":
                summary["stage_crashes"][st] = rec.get("reason")
                log.warning("read-back: %s crashed this run (rc!=0): %s", st, rec.get("reason"))
        if not summary["stage_crashes"]:
            log.info("read-back: no whole-stage subprocess crashes this run")
    # 1. regenerate the health snapshot (pipeline_status is the health consumer).
    try:
        subprocess.run([VP("embed"), "bin/pipeline_status.py"], cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": "src"}, timeout=300)
    except Exception as e:
        log.warning("read-back: pipeline_status failed: %s", e)
    # 2. read explicit outcomes for the spine (oc imported at module level).
    try:
        media = sorted(p.stem for p in (ROOT / "catalog" / "media_manifest").glob("m_*.json"))
        for st in READBACK_STAGES:
            errs = oc.media_by_status(st, "error")
            unrec = [m for m in media if oc.get(m, st) is None
                     and (ROOT / "catalog" / "media_manifest" / f"{m}.json").exists()]
            if errs:
                summary["errors"][st] = errs
                log.warning("read-back: %s has %d errored media: %s", st, len(errs), errs[:8])
            if unrec:
                summary["unrecorded"][st] = len(unrec)
    except Exception as e:
        log.warning("read-back: outcome read failed: %s", e)
    # 3. surface critical-stage gaps from the health snapshot.
    try:
        health = json.loads((ROOT / "logs.all" / "_latest" / "health.json").read_text())
        gaps = health.get("critical_gaps", {})
        if gaps:
            summary["critical_gaps"] = {m: g for m, g in list(gaps.items())[:20]}
            log.warning("read-back: %d media have critical-stage gaps", len(gaps))
    except Exception as e:
        log.warning("read-back: health.json read failed: %s", e)
    if not (summary["errors"] or summary["critical_gaps"]):
        log.info("read-back: clean — no errored media, no critical-stage gaps")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--skip", nargs="+")
    ap.add_argument("--from", dest="from_stage")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = STAGES[:]
    if args.from_stage:
        names = [s["name"] for s in stages]
        if args.from_stage not in names:
            print(f"unknown stage: {args.from_stage}", file=sys.stderr)
            sys.exit(2)
        stages = stages[names.index(args.from_stage):]
    if args.only:
        keep = set(args.only)
        stages = [s for s in stages if s["name"] in keep]
    if args.skip:
        drop = set(args.skip)
        stages = [s for s in stages if s["name"] not in drop]

    if args.dry_run:
        for s in stages:
            print(f"{s['name']:10} {'(fatal)' if s['fatal'] else '       '}  {' '.join(s['cmd'])}")
        return

    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    log = setup_logging(stamp)
    log.info("REBUILD start %s  stages=%s", started.isoformat(), [s["name"] for s in stages])

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    # All model weights/caches are present locally (whisper + pyannote on disk,
    # embedding/VLM weights at fixed paths). Offline mode skips HuggingFace etag
    # revalidation, which otherwise stalls model loads on this pod (observed multi-
    # minute hangs on a kept-alive HF CDN socket). Fail-fast beats hang.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    # onnxruntime-gpu (cuda-12 build) in the embed venv needs torch's bundled CUDA/cuDNN
    # libs on LD_LIBRARY_PATH to load its CUDA EP — used by F0/F1 (insightface face detect)
    # and F2 (ASD). Scoped to embed-venv stages ONLY, so the audio/anyenhance/groundingdino
    # venvs' own torch is never shadowed by these libs.
    import glob as _glob
    _embed = str(ROOT / "venvs" / "embed")
    _embed_cuda_libs = ":".join(
        sorted(_glob.glob(f"{_embed}/lib/python*/site-packages/nvidia/*/lib"))
        + _glob.glob(f"{_embed}/lib/python*/site-packages/torch/lib")
    )

    # D-Log-21: per-stage stdout/stderr, teed into logs.all instead of only
    # inheriting the orchestrator's own fd (which, under a background/nohup
    # launch, mixes every stage's output into one undifferentiated stream —
    # exactly what made the run-1 SEV2_ADJ/PSTREAM startup crash slow to spot).
    stage_log_dir = ROOT / "logs.all" / f"rebuild_{stamp}"
    stage_log_dir.mkdir(parents=True, exist_ok=True)

    results, failures, done = [], [], []
    for s in stages:
        name = s["name"]
        log.info("===== %s =====  %s", name, " ".join(s["cmd"]))
        t0 = datetime.now(timezone.utc)
        stage_env = env
        if _embed_cuda_libs and _embed in s["cmd"][0]:
            stage_env = dict(env)
            _prev = env.get("LD_LIBRARY_PATH", "")
            stage_env["LD_LIBRARY_PATH"] = _embed_cuda_libs + ((":" + _prev) if _prev else "")
        stage_log_path = stage_log_dir / f"{name}.log"
        try:
            with open(stage_log_path, "wb") as lf:
                proc = subprocess.Popen(s["cmd"], cwd=ROOT, env=stage_env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in proc.stdout:
                    sys.stdout.buffer.write(line)
                    lf.write(line)
                sys.stdout.flush()
                proc.wait()
                rc = proc.returncode
        except Exception as e:
            log.error("%s raised: %s", name, e)
            rc = -1
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        log.info("%s done rc=%d in %.0fs (log: %s)", name, rc, elapsed,
                 stage_log_path.relative_to(ROOT))
        results.append({"name": name, "rc": rc, "elapsed_s": round(elapsed, 1)})
        done.append(name)
        write_hb(name, done, (datetime.now(timezone.utc) - started).total_seconds())
        # D-Log-21: an explicit corpus-scoped run-outcome, written EVERY time
        # (ok or error) so a past crash can never look "still crashed" after a
        # later clean run -- mirrors GATE's existing __corpus__ precedent
        # (MASTER §5.2's Gate-as-stage). Read independently in read_back()
        # rather than trusting this run's own in-memory `failures` list.
        try:
            oc.record("__corpus__", name, "ok" if rc == 0 else "error",
                      reason=None if rc == 0 else f"subprocess rc={rc} (see {stage_log_path.relative_to(ROOT)})",
                      metrics={"elapsed_s": round(elapsed, 1)}, producer="rebuild_all.py")
        except Exception as e:
            log.error("%s: failed to record run-outcome: %s", name, e)
        if rc != 0:
            if s["fatal"]:
                log.error("FATAL stage %s failed (rc=%d) — aborting", name, rc)
                sys.exit(1)
            failures.append(name)

    log.info("===== SUMMARY =====")
    for r in results:
        log.info("  %-10s rc=%-3d %.0fs", r["name"], r["rc"], r["elapsed_s"])
    log.info("failures (non-fatal): %s", failures or "none")

    # Read the log we just wrote (MASTER §5.2.2 orchestrator obligation).
    log.info("===== READ-BACK =====")
    readback = read_back(log, run_stage_names=[s["name"] for s in stages])

    report = ROOT / "logs.all" / f"rebuild_report_{stamp}.json"
    report.write_text(json.dumps({
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "stages": results, "failures": failures,
        "read_back": readback,
    }, indent=2))
    log.info("report -> %s", report)


if __name__ == "__main__":
    main()

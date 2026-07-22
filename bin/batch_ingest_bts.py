#!/usr/bin/env python3
"""Phased batch ingest of BTS video files — full chain A→F1.

Phases (each GPU model loaded once across all files):
  Phase 1: A (audio subprocess, batch) + B0 + X0 — all files
  Phase 2: B1 (text embed, one model load) — all files
  Phase 3: C (text materialise) — all files
  Phase 4: B2 + B3 + P — all files
  Phase 5: B4 (visual embed, one model load) — all files
  Phase 6: C_vis — all files
  Phase 7: Q0 + E0 + tags — all files
  Phase 8: F0 (face detect, one model load) — all files
  Phase 9: F1 (identity) — all files

Run from /workspace/erf with the embed venv:
  PYTHONPATH=src venvs/embed/bin/python bin/batch_ingest_bts.py
"""
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["EMBED_IDLE_SECONDS"] = "3600"  # Hold models across all files in a phase

ERF_ROOT = Path("/workspace/erf")
AUDIO_PYTHON = str(ERF_ROOT / "venvs/audio/bin/python")
CONFIG = "state/config.toml"

try:
    from erf.runlog import heartbeat as _heartbeat
except Exception:
    _heartbeat = None

import tomllib
with open(CONFIG, "rb") as fh:
    _cfg = tomllib.load(fh)
CATALOG = Path(_cfg["paths"]["catalog"])
FTS_DB = Path(_cfg["paths"]["erf_fts_db"])

LOG_PATH = ERF_ROOT / "logs" / "batch_bts_ingest.log"
LOG_PATH.parent.mkdir(exist_ok=True)
_log_fh = open(LOG_PATH, "a")

STATE_FILE = Path("/tmp/bts_ingest_state.json")


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()
    if _heartbeat:
        _heartbeat("batch_ingest", {"msg": msg[:140]})


def vram_release() -> None:
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── State persistence ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"media_ids": {}, "failures": [], "phases_done": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Phase 1: A (batch subprocess) + B0 + X0 ───────────────────────────────────

def phase1_asr(videos: list[str], state: dict) -> dict[str, str]:
    """Run Stage A for all files (batch audio subprocess). Returns {path: media_id}."""
    if "phase1" in state["phases_done"]:
        log("Phase 1: already done, skipping")
        return {v: state["media_ids"].get(v) for v in videos if v in state["media_ids"]}

    log(f"\n=== PHASE 1: ASR + B0 + X0 for {len(videos)} files ===")

    # Write paths to a temp file so the subprocess can read them cleanly
    paths_file = Path("/tmp/bts_asr_paths.json")
    paths_file.write_text(json.dumps(videos))

    script = """
import sys, json
sys.path.insert(0, 'src')
import warnings; warnings.filterwarnings('ignore')
from erf.asr import run

paths = json.loads(open('/tmp/bts_asr_paths.json').read())
for p in paths:
    try:
        r = run(p, config_path='state/config.toml')
        print(json.dumps({'path': p, 'media_id': r.media_id, 'ok': r.ok,
                          'counts': r.counts, 'error': r.error}), flush=True)
    except Exception as e:
        print(json.dumps({'path': p, 'media_id': None, 'ok': False,
                          'counts': {}, 'error': str(e)[:200]}), flush=True)
"""
    log("  Starting batch ASR subprocess...")
    proc = subprocess.Popen(
        [AUDIO_PYTHON, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(ERF_ROOT),
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )

    from erf.chunk import run as run_b0
    from erf.fts import index_media

    path_to_mid: dict[str, str] = {}
    n = len(videos)
    done = 0

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue

        done += 1
        path = rec.get("path", "?")
        fname = Path(path).name
        mid = rec.get("media_id")

        if not rec.get("ok") or not mid:
            state["failures"].append({"file": fname, "stage": "A",
                                      "error": str(rec.get("error", ""))[:200]})
            log(f"  [{done}/{n}] FAIL A {fname}: {rec.get('error','')[:80]}")
            continue

        path_to_mid[path] = mid
        state["media_ids"][path] = mid
        log(f"  [{done}/{n}] A ok {fname} → {mid} {rec.get('counts')}")

        # B0 + X0 immediately (CPU, no VRAM)
        try:
            r = run_b0(mid, config_path=CONFIG)
            if r.ok:
                log(f"         B0 ok {r.counts}")
            else:
                state["failures"].append({"file": fname, "mid": mid, "stage": "B0",
                                          "error": str(r.error)[:200]})
                log(f"         FAIL B0: {r.error}")
        except Exception as e:
            state["failures"].append({"file": fname, "mid": mid, "stage": "B0",
                                      "error": str(e)[:200]})
            log(f"         FAIL B0: {e}")

        try:
            counts = index_media(mid, CATALOG, FTS_DB)
            log(f"         X0 ok {counts}")
        except Exception as e:
            log(f"         WARN X0: {e}")

        save_state(state)

    proc.wait()
    if proc.returncode not in (0, None):
        log(f"  Audio subprocess exited with {proc.returncode}")

    state["phases_done"].append("phase1")
    save_state(state)
    return path_to_mid


# ── Phase 2: B1 text embed ─────────────────────────────────────────────────────

def phase2_b1(mids: list[str], state: dict) -> None:
    if "phase2" in state["phases_done"]:
        log("Phase 2: already done, skipping")
        return

    log(f"\n=== PHASE 2: B1 text embed for {len(mids)} files ===")
    from erf.embed_text import run as run_b1
    failed_b1 = set()
    for i, mid in enumerate(mids, 1):
        try:
            r = run_b1(mid, config_path=CONFIG)
            if r.ok:
                log(f"  [{i}/{len(mids)}] B1 ok {mid} {r.counts}")
            else:
                failed_b1.add(mid)
                state["failures"].append({"mid": mid, "stage": "B1",
                                          "error": str(r.error)[:200]})
                log(f"  [{i}/{len(mids)}] FAIL B1 {mid}: {r.error}")
        except Exception as e:
            failed_b1.add(mid)
            state["failures"].append({"mid": mid, "stage": "B1", "error": str(e)[:200]})
            log(f"  [{i}/{len(mids)}] FAIL B1 {mid}: {e}")

    vram_release()
    state["phases_done"].append("phase2")
    save_state(state)


# ── Phase 3: C text materialise ───────────────────────────────────────────────

def phase3_c(mids: list[str], state: dict) -> None:
    if "phase3" in state["phases_done"]:
        log("Phase 3: already done, skipping")
        return

    log(f"\n=== PHASE 3: C materialise text_v1 for {len(mids)} files ===")
    from erf.materialise import run as run_c
    for i, mid in enumerate(mids, 1):
        try:
            r = run_c(mid, config_path=CONFIG)
            log(f"  [{i}/{len(mids)}] C {'ok' if r.ok else 'FAIL'} {mid} {r.counts or r.error}")
            if not r.ok:
                state["failures"].append({"mid": mid, "stage": "C",
                                          "error": str(r.error)[:200]})
        except Exception as e:
            state["failures"].append({"mid": mid, "stage": "C", "error": str(e)[:200]})
            log(f"  [{i}/{len(mids)}] FAIL C {mid}: {e}")
        if i % 20 == 0:
            save_state(state)

    state["phases_done"].append("phase3")
    save_state(state)


# ── Phase 4: B2 + B3 + P ──────────────────────────────────────────────────────

def phase4_b2b3p(path_to_mid: dict[str, str], state: dict) -> set[str]:
    if "phase4" in state["phases_done"]:
        log("Phase 4: already done, skipping")
        return set(m for m in path_to_mid.values() if m)

    log(f"\n=== PHASE 4: B2 + B3 + P for {len(path_to_mid)} files ===")
    from erf.scenes import run as run_b2
    from erf.keyframes import run as run_b3
    from erf.proxy import run as run_p

    b3_ok: set[str] = set()
    for i, (path, mid) in enumerate(path_to_mid.items(), 1):
        if not mid:
            continue
        fname = Path(path).name
        try:
            r = run_b2(mid, path, config_path=CONFIG)
            if not r.ok:
                state["failures"].append({"mid": mid, "stage": "B2",
                                          "error": str(r.error)[:200]})
                log(f"  [{i}] FAIL B2 {fname}: {r.error}")
                continue

            r = run_b3(mid, path, config_path=CONFIG)
            if not r.ok:
                state["failures"].append({"mid": mid, "stage": "B3",
                                          "error": str(r.error)[:200]})
                log(f"  [{i}] FAIL B3 {fname}: {r.error}")
                continue

            mm_path = CATALOG / "media_manifest" / f"{mid}.json"
            mm = json.loads(mm_path.read_text())
            r = run_p(mid, path, config_path=CONFIG, duration_ms=mm["duration_ms"])
            if r.ok:
                b3_ok.add(mid)
                log(f"  [{i}/{len(path_to_mid)}] B2+B3+P ok {fname} → {mid}")
            else:
                state["failures"].append({"mid": mid, "stage": "P",
                                          "error": str(r.error)[:200]})
                log(f"  [{i}] FAIL P {fname}: {r.error}")
        except Exception as e:
            state["failures"].append({"mid": mid, "stage": "B2/B3/P",
                                      "error": str(e)[:200]})
            log(f"  [{i}] FAIL B2/B3/P {fname}: {e}")
        if i % 20 == 0:
            save_state(state)

    state["phases_done"].append("phase4")
    save_state(state)
    return b3_ok


# ── Phase 5: B4 visual embed ──────────────────────────────────────────────────

def phase5_b4(mids: list[str], state: dict) -> set[str]:
    if "phase5" in state["phases_done"]:
        log("Phase 5: already done, skipping")
        return set(mids)

    log(f"\n=== PHASE 5: B4 visual embed for {len(mids)} files ===")
    # Release text model from Phase 2 before loading visual model (D-sess3-03/04)
    try:
        from erf import embed_text as _et
        _et._MODEL_CACHE.clear()
        import gc; gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log("  VRAM cleared: text model cache released before B4")
    except Exception as _e:
        log(f"  WARN: VRAM clear failed: {_e}")

    from erf.embed_visual import run as run_b4
    b4_ok: set[str] = set()
    for i, mid in enumerate(mids, 1):
        try:
            r = run_b4(mid, config_path=CONFIG)
            if r.ok:
                b4_ok.add(mid)
                log(f"  [{i}/{len(mids)}] B4 ok {mid} {r.counts}")
            else:
                state["failures"].append({"mid": mid, "stage": "B4",
                                          "error": str(r.error)[:200]})
                log(f"  [{i}/{len(mids)}] FAIL B4 {mid}: {r.error}")
        except Exception as e:
            state["failures"].append({"mid": mid, "stage": "B4", "error": str(e)[:200]})
            log(f"  [{i}/{len(mids)}] FAIL B4 {mid}: {e}")

    vram_release()
    state["phases_done"].append("phase5")
    save_state(state)
    return b4_ok


# ── Phase 6: C_vis ────────────────────────────────────────────────────────────

def phase6_cvis(mids: list[str], state: dict) -> None:
    if "phase6" in state["phases_done"]:
        log("Phase 6: already done, skipping")
        return

    log(f"\n=== PHASE 6: C_vis materialise visual_v1 for {len(mids)} files ===")
    from erf.materialise import run_visual as run_c_vis
    for i, mid in enumerate(mids, 1):
        try:
            r = run_c_vis(mid, config_path=CONFIG)
            log(f"  [{i}/{len(mids)}] C_vis {'ok' if r.ok else 'FAIL'} {mid} {r.counts or r.error}")
            if not r.ok:
                state["failures"].append({"mid": mid, "stage": "C_vis",
                                          "error": str(r.error)[:200]})
        except Exception as e:
            state["failures"].append({"mid": mid, "stage": "C_vis", "error": str(e)[:200]})
            log(f"  [{i}] FAIL C_vis {mid}: {e}")
        if i % 20 == 0:
            save_state(state)

    state["phases_done"].append("phase6")
    save_state(state)


# ── Phase 7: Q0 + E0 + tags ───────────────────────────────────────────────────

def phase7_q0e0tags(mids: list[str], state: dict) -> None:
    if "phase7" in state["phases_done"]:
        log("Phase 7: already done, skipping")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    from erf.quality import run as run_q0
    from erf.events import run as run_e0
    from erf.tags import run as run_tags
    from erf import outcomes as _oc

    # Q0/E0/tags are independent across media and dominated by network latency (E0
    # calls the DeepSeek API per event), so fan out across media with a thread pool.
    # Intra-media order (Q0→E0→tags) is preserved inside each task. Worker count is
    # env-tunable; default 8 balances E0's network concurrency against Q0's local
    # ffmpeg/CNN14 subprocesses (~12 vCPUs). All journal.db writes are serialised by
    # journal.DB_WRITE_LOCK, and per-media status files by outcomes._file_lock.
    workers = max(1, int(os.environ.get("ERF_PHASE7_WORKERS", "8")))

    # Resume idempotency: skip media that already recorded a terminal outcome for all
    # three stages (explicit-outcome check, never file presence). A relaunch resumes
    # instead of re-doing completed media.
    def _done(mid: str) -> bool:
        return (_oc.is_complete(mid, "Q0")
                and _oc.is_complete(mid, "E0")
                and _oc.is_complete(mid, "tags"))

    todo = [m for m in mids if not _done(m)]
    skipped = len(mids) - len(todo)
    total = len(todo)
    log(f"\n=== PHASE 7: Q0 + E0 + tags — {total} to do, "
        f"{skipped} already complete (resume), workers={workers} ===")

    lock = threading.Lock()
    progress = {"n": 0}

    def _one(mid: str) -> None:
        fails: list[dict] = []
        proxy_path = ERF_ROOT / "media" / "proxies" / f"{mid}.mp4"
        if proxy_path.exists():
            try:
                r = run_q0(mid, str(proxy_path), config_path=CONFIG)
                if not r.ok:
                    fails.append({"mid": mid, "stage": "Q0", "error": str(r.error)[:200]})
            except Exception as e:
                fails.append({"mid": mid, "stage": "Q0", "error": str(e)[:200]})
        else:
            fails.append({"mid": mid, "stage": "Q0", "error": "proxy not found"})

        try:
            r = run_e0(mid, config_path=CONFIG, llm_tag=True)
            if not r.ok:
                fails.append({"mid": mid, "stage": "E0", "error": str(r.error)[:200]})
        except Exception as e:
            fails.append({"mid": mid, "stage": "E0", "error": str(e)[:200]})

        try:
            r = run_tags(mid, config_path=CONFIG)
            if not r.ok:
                fails.append({"mid": mid, "stage": "tags", "error": str(r.error)[:200]})
        except Exception as e:
            fails.append({"mid": mid, "stage": "tags", "error": str(e)[:200]})

        with lock:
            state["failures"].extend(fails)
            progress["n"] += 1
            n = progress["n"]
            if n % 10 == 0 or n == total:
                log(f"  [{n}/{total}] Q0+E0+tags done")
                save_state(state)
                if _heartbeat:
                    _heartbeat("phase7_q0e0tags", {"done": n, "total": total})

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, m): m for m in todo}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                mid = futures[fut]
                with lock:
                    state["failures"].append({"mid": mid, "stage": "phase7",
                                              "error": f"worker crash: {str(exc)[:180]}"})
                log(f"  worker crashed on {mid}: {exc}")

    state["phases_done"].append("phase7")
    save_state(state)


# ── Phase 8: F0 face detect ───────────────────────────────────────────────────

def phase8_f0(mids: list[str], state: dict) -> set[str]:
    if "phase8" in state["phases_done"]:
        log("Phase 8: already done, skipping")
        return set(mids)

    log(f"\n=== PHASE 8: F0 face detection for {len(mids)} files ===")
    from erf.faces import run as run_f0
    f0_ok: set[str] = set()
    for i, mid in enumerate(mids, 1):
        try:
            r = run_f0(mid, config_path=CONFIG)
            if r.ok:
                f0_ok.add(mid)
                if i % 20 == 0 or r.counts.get("tracks", 0) > 0:
                    log(f"  [{i}/{len(mids)}] F0 ok {mid} {r.counts}")
            else:
                state["failures"].append({"mid": mid, "stage": "F0",
                                          "error": str(r.error)[:200]})
                log(f"  [{i}/{len(mids)}] FAIL F0 {mid}: {r.error}")
        except Exception as e:
            state["failures"].append({"mid": mid, "stage": "F0", "error": str(e)[:200]})
            log(f"  [{i}/{len(mids)}] FAIL F0 {mid}: {e}")

    vram_release()
    state["phases_done"].append("phase8")
    save_state(state)
    return f0_ok


# ── Phase 9: F1 identity ──────────────────────────────────────────────────────

def phase9_f1(mids: list[str], state: dict) -> None:
    if "phase9" in state["phases_done"]:
        log("Phase 9: already done, skipping")
        return

    log(f"\n=== PHASE 9: F1 identity for {len(mids)} files ===")
    from erf.identity import run as run_f1
    for i, mid in enumerate(mids, 1):
        try:
            r = run_f1(mid, config_path=CONFIG)
            if not r.ok:
                state["failures"].append({"mid": mid, "stage": "F1",
                                          "error": str(r.error)[:200]})
        except Exception as e:
            state["failures"].append({"mid": mid, "stage": "F1", "error": str(e)[:200]})
        if i % 20 == 0:
            log(f"  [{i}/{len(mids)}] F1 done so far...")
            save_state(state)

    state["phases_done"].append("phase9")
    save_state(state)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")

    videos_file = Path("/tmp/bts_videos.txt")
    if not videos_file.exists():
        print("ERROR: /tmp/bts_videos.txt not found")
        sys.exit(1)

    videos = [v.strip() for v in videos_file.read_text().splitlines() if v.strip()]
    log(f"\nPhased batch ingest: {len(videos)} files")

    state = load_state()
    t_start = time.time()

    # ── Phase 1: ASR + B0 + X0 ────────────────────────────────────────────────
    path_to_mid = phase1_asr(videos, state)
    all_mids = [m for m in path_to_mid.values() if m]
    log(f"  Phase 1 complete: {len(all_mids)} media_ids")

    # ── Phase 2: B1 text embed ────────────────────────────────────────────────
    phase2_b1(all_mids, state)

    # ── Phase 3: C materialise text ───────────────────────────────────────────
    phase3_c(all_mids, state)

    # ── Phase 4: B2 + B3 + P ──────────────────────────────────────────────────
    b3_ok = phase4_b2b3p(path_to_mid, state)
    b3_list = sorted(b3_ok)
    log(f"  Phase 4 complete: {len(b3_list)} files have keyframes + proxy")

    # ── Phase 5: B4 visual embed ──────────────────────────────────────────────
    b4_ok = phase5_b4(b3_list, state)
    b4_list = sorted(b4_ok)

    # ── Phase 6: C_vis ────────────────────────────────────────────────────────
    phase6_cvis(b4_list, state)

    # ── Phase 7: Q0 + E0 + tags ───────────────────────────────────────────────
    phase7_q0e0tags(all_mids, state)

    # ── Phase 8: F0 face detect ───────────────────────────────────────────────
    f0_ok = phase8_f0(b3_list, state)
    f0_list = sorted(f0_ok)

    # ── Phase 9: F1 identity ──────────────────────────────────────────────────
    phase9_f1(f0_list, state)

    elapsed = time.time() - t_start
    n_fail = len(state["failures"])
    log(f"\n=== BATCH COMPLETE in {elapsed/3600:.1f}h ===")
    log(f"Processed: {len(all_mids)}/{len(videos)} files")
    log(f"Failure entries: {n_fail}")
    for f in state["failures"][:30]:
        log(f"  FAIL {f.get('mid', f.get('file','?'))} @ {f['stage']}: "
            f"{f.get('error','')[:80]}")

    with open("/tmp/bts_results.json", "w") as fh:
        json.dump(state, fh, indent=2)

    _log_fh.close()


if __name__ == "__main__":
    main()

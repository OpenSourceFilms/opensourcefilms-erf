#!/usr/bin/env python3
"""REDIARIZE — re-run diarization only (no Whisper) on existing transcripts.

Usage:
  PYTHONPATH=src venvs/audio/bin/python bin/rediarize.py --all
  PYTHONPATH=src venvs/audio/bin/python bin/rediarize.py --has-words
  PYTHONPATH=src venvs/audio/bin/python bin/rediarize.py --media-ids m_abc m_def
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, tempfile, time, tomllib
from pathlib import Path

ROOT = Path("/workspace/erf")
sys.path.insert(0, str(ROOT / "src"))

from erf.asr import _load_hf_token, _merge_turns, _assign_speaker, _dedup_speakers
from erf.atomicio import jsonl_append, write_tmp_rename

try:
    from erf.runlog import log_event, heartbeat
    from erf import outcomes
except Exception:
    log_event = heartbeat = None

CFG_PATH = ROOT / "state" / "config.toml"
CATALOG = ROOT / "catalog"
PY = sys.executable


def _load_cfg() -> dict:
    cfg: dict = {"diar_min_speakers": None, "diar_max_speakers": None,
                 "diar_merge_gap_ms": 500, "diar_min_turn_ms": 200, "diar_fallback_ms": 3000,
                 "diar_dedup_threshold": 0.80}
    if CFG_PATH.exists():
        with open(CFG_PATH, "rb") as fh:
            raw = tomllib.load(fh)
        diar = raw.get("asr", {}).get("diarization", {})
        for k, ck in [("min_speakers","diar_min_speakers"),("max_speakers","diar_max_speakers"),
                      ("merge_gap_ms","diar_merge_gap_ms"),("min_turn_ms","diar_min_turn_ms"),
                      ("fallback_ms","diar_fallback_ms")]:
            if k in diar:
                cfg[ck] = int(diar[k])
        if "dedup_threshold" in diar:
            cfg["diar_dedup_threshold"] = float(diar["dedup_threshold"])
    return cfg


def _source_path(mid: str) -> Path | None:
    mp = CATALOG / "media_manifest" / f"{mid}.json"
    if not mp.exists():
        return None
    try:
        d = json.loads(mp.read_text())
        # Explicit full path first
        s = d.get("source_path") or d.get("path")
        if s:
            p = Path(s)
            if p.exists():
                return p
        # Search by relpath in BTS tree
        relpath = d.get("relpath")
        if relpath:
            for found in Path("/workspace/BTS").rglob(relpath):
                return found
    except Exception:
        pass
    return None


def _diarize(media_path: Path, cfg: dict) -> list[dict]:
    import numpy as np, scipy.io.wavfile as wf, static_ffmpeg, torch
    static_ffmpeg.add_paths()
    from pyannote.audio import Pipeline as P
    hf = _load_hf_token()
    pipeline = None
    for nm in ["pyannote/speaker-diarization-community-1","pyannote/speaker-diarization-3.1"]:
        try:
            pipeline = P.from_pretrained(nm, token=hf); break
        except Exception:
            continue
    if pipeline is None:
        raise RuntimeError("No pyannote pipeline")
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ffmpeg = shutil.which("ffmpeg")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    try:
        subprocess.run([ffmpeg,"-y","-i",str(media_path),"-ar","16000","-ac","1",wav_path],
                       capture_output=True, check=True)
        sr, wav_data = wf.read(wav_path)
        waveform = torch.from_numpy(wav_data.astype(np.float32)/32768.0).unsqueeze(0)
    finally:
        try: os.unlink(wav_path)
        except OSError: pass
    kw: dict = {}
    if cfg["diar_min_speakers"] is not None: kw["min_speakers"] = cfg["diar_min_speakers"]
    if cfg["diar_max_speakers"] is not None: kw["max_speakers"] = cfg["diar_max_speakers"]
    result = pipeline({"waveform": waveform, "sample_rate": sr}, **kw)
    ann = getattr(result, "speaker_diarization", result)
    emb_raw = getattr(result, "speaker_embeddings", None)
    embeddings: dict[str, list[float]] = {}
    if emb_raw is not None and hasattr(ann, "labels"):
        for i, label in enumerate(ann.labels()):
            if i < len(emb_raw):
                embeddings[label] = emb_raw[i].tolist()
    raw = [{"turn_index": i, "speaker_label": sp,
            "start_ms": int(round(t.start*1000)), "end_ms": int(round(t.end*1000))}
           for i,(t,_,sp) in enumerate(ann.itertracks(yield_label=True))]
    if embeddings:
        raw = _dedup_speakers(raw, embeddings, cfg.get("diar_dedup_threshold", 0.80))
    return _merge_turns(raw, cfg["diar_merge_gap_ms"], cfg["diar_min_turn_ms"]), embeddings


def rediarize_one(mid: str, cfg: dict) -> dict:
    asr = CATALOG / "asr"
    wp, sp, dp = asr/f"{mid}.words.jsonl", asr/f"{mid}.segments.jsonl", asr/f"{mid}.diar.jsonl"
    # Gate on A's explicit has_speech flag (cross-stage awareness), not file presence.
    if outcomes.flag(mid, "has_speech") is False:
        return {"ok": False, "error": "no_speech"}
    if not wp.exists():
        return {"ok": False, "error": "no words.jsonl"}
    words = [json.loads(l) for l in wp.read_text().splitlines() if l.strip()]
    segs  = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()] if sp.exists() else []
    src = _source_path(mid)
    if src is None or not src.exists():
        proxy = ROOT/"media"/"proxies"/f"{mid}.mp4"
        src = proxy if proxy.exists() else None
    if src is None:
        return {"ok": False, "error": "source not found"}
    turns, embeddings = _diarize(src, cfg)
    fb = cfg["diar_fallback_ms"]
    for w in words: w["speaker_label"] = _assign_speaker(w["start_ms"],w["end_ms"],turns,fb)
    for s in segs:  s["speaker_label"] = _assign_speaker(s["start_ms"],s["end_ms"],turns,fb)
    emb_path = CATALOG / "asr" / f"{mid}.diar_embeddings.json"
    for p in (wp, sp, dp):
        if p.exists(): p.unlink()
    for w in words: jsonl_append(wp, w)
    for s in segs:  jsonl_append(sp, s)
    for t in turns: jsonl_append(dp, t)
    if embeddings:
        write_tmp_rename(emb_path, json.dumps(embeddings, separators=(",", ":")).encode())
    elif emb_path.exists():
        emb_path.unlink()
    return {"ok": True, "turns": len(turns), "speakers": len({t["speaker_label"] for t in turns}),
            "deduped": bool(embeddings)}


def _rebuild(mid: str) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    for script in ("bin/build_transcript.py","bin/build_evidence.py"):
        p = ROOT/script
        if p.exists():
            subprocess.run([PY,str(p),"--media-ids",mid,"--force"],
                           cwd=str(ROOT), env=env, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--has-words", action="store_true")
    g.add_argument("--media-ids", nargs="+", metavar="MID")
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-speakers", type=int, default=None, metavar="N",
                    help="Override diar_min_speakers for this run only")
    ap.add_argument("--max-speakers", type=int, default=None, metavar="N",
                    help="Override diar_max_speakers for this run only")
    a = ap.parse_args()
    cfg = _load_cfg()
    if a.min_speakers is not None:
        cfg["diar_min_speakers"] = a.min_speakers
    if a.max_speakers is not None:
        cfg["diar_max_speakers"] = a.max_speakers
    print(f"[rediarize] merge_gap={cfg['diar_merge_gap_ms']}ms min_turn={cfg['diar_min_turn_ms']}ms "
          f"fallback={cfg['diar_fallback_ms']}ms dedup={cfg['diar_dedup_threshold']} "
          f"spk={cfg['diar_min_speakers']}-{cfg['diar_max_speakers']}", flush=True)
    if a.all:
        mids = sorted(f.stem.replace(".diar","") for f in (CATALOG/"asr").glob("*.diar.jsonl"))
    elif a.has_words:
        mids = [f.stem.replace(".words","") for f in sorted((CATALOG/"asr").glob("*.words.jsonl"))
                if any(l.strip() for l in f.read_text().splitlines())]
    else:
        mids = a.media_ids
    if a.limit: mids = mids[:a.limit]
    print(f"[rediarize] {len(mids)} files", flush=True)
    ok = err = skipped = empty = 0
    for i, mid in enumerate(mids, 1):
        t0 = time.monotonic()
        if heartbeat: heartbeat("rediarize", {"i":i,"n":len(mids),"ok":ok,"err":err,"current":mid})
        try:
            res = rediarize_one(mid, cfg)
            if not res["ok"]:
                skipped += 1
                outcomes.record(mid, "REDIARIZE", "skip", reason=res["error"])
                outcomes.set_flag(mid, "has_diar_embeddings", False, reason=res["error"])
                print(f"[{i}/{len(mids)}] SKIP {mid}: {res['error']}", flush=True); continue
            if res["turns"] == 0:
                # Diarization ran but produced 0 turns — rediarize_one unlinks
                # any stale diar.jsonl and never rewrites one in this case, so
                # no output file exists. Recording "ok" + has_diar_embeddings=True
                # here (the prior bug) is the same phantom pattern as drift-308:
                # a green outcome with nothing behind it, and a false flag that
                # would make F1/identity consumers trust embeddings that don't
                # exist. Record the honest terminal disposition instead.
                empty += 1
                outcomes.record(mid, "REDIARIZE", "empty", reason="no_turns")
                outcomes.set_flag(mid, "has_diar_embeddings", False, reason="no_turns")
                print(f"[{i}/{len(mids)}] EMPTY {mid}: no_turns", flush=True); continue
            if not a.no_rebuild: _rebuild(mid)
            if log_event: log_event("REDIARIZE", mid, "ok", {"turns":res["turns"],"speakers":res["speakers"]})
            outcomes.record(mid, "REDIARIZE", "ok", counts={"turns":res["turns"],"speakers":res["speakers"]})
            outcomes.set_flag(mid, "has_diar_embeddings", True)
            outcomes.set_flag(mid, "n_speakers", res["speakers"])
            ok += 1
            print(f"[{i}/{len(mids)}] ok {mid}: {res['turns']} turns {res['speakers']} spk ({time.monotonic()-t0:.0f}s)", flush=True)
        except Exception as e:
            err += 1
            if log_event: log_event("REDIARIZE", mid, "error", {"error": str(e)[:300]})
            outcomes.record(mid, "REDIARIZE", "error", reason=str(e)[:200])
            print(f"[{i}/{len(mids)}] FAIL {mid}: {e}", flush=True)
    print(f"[rediarize] done ok={ok} skip={skipped} empty={empty} err={err}", flush=True)


if __name__ == "__main__":
    main()

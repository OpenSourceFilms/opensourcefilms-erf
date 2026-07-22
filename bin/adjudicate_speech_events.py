#!/usr/bin/env python3
"""Targeted voiceprint adjudication of speech_events_v2.

Fixes the visible failure: exclusive-diar fine switching splits one speaker's
utterance across colours (e.g. SpeakerA->SpeakerB->SpeakerB->SpeakerA). Operates ONLY on the
conflict/ping-pong/low-confidence subset — not corpus-wide churn.

For each candidate event: crop its audio (+/-PAD) from the media wav, embed with
WeSpeaker, and compare (cosine) to the per-media speaker centroids of every
plausible local label (its exclusive label, its canonical-ASR label, and the
labels of the events immediately before/after). If one label wins by absolute
threshold + margin, the event is rewritten to that speaker. If not, an isolated
single-word flip folds to its surrounding speaker (coherence default) so
exclusive diarization can never force a mid-utterance flip on its own; a genuine
conflict with no voiceprint winner is marked ambiguous.

  reads : catalog/speech_events_v2/{mid}.jsonl
          catalog/asr/{mid}.diar_embeddings.json (+ diar_embeddings_repair sidecar)
          catalog/identity_map/{mid}.json
          media/proxies/{mid}.mp4
  writes: catalog/speech_events_v2_adjudicated/{mid}.jsonl
          catalog/speech_events_v2_adjudication_qc/{mid}.json
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam); unset = ROOT/catalog
try:
    from erf import outcomes
except Exception:
    outcomes = None
PROX = ROOT / "media" / "proxies"
PAD_S = 0.35
WIN_ABS = 0.30       # winning cosine must reach this
WIN_MARGIN = 0.06    # ...and beat the runner-up by this
MERGE_GAP = 700


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _load_centroids(mid: str) -> dict[str, np.ndarray]:
    """Per-label 256-d centroids, preferring the repair sidecar when degenerate."""
    out: dict[str, np.ndarray] = {}
    p = CAT / "asr" / f"{mid}.diar_embeddings.json"
    rep = {}
    rp = CAT / "diar_embeddings_repair" / f"{mid}.json"
    if rp.exists():
        try:
            rep = json.loads(rp.read_text()).get("embeddings", {})
        except Exception:
            rep = {}
    if p.exists():
        d = json.loads(p.read_text())
        for lbl, vec in d.items():
            v = np.asarray(vec, dtype=np.float64) if isinstance(vec, list) and len(vec) == 256 else None
            if v is None or not np.all(np.isfinite(v)) or float(v @ v) < 1e-12:
                rv = rep.get(lbl)
                if isinstance(rv, list) and len(rv) == 256:
                    rvn = np.asarray(rv, dtype=np.float64)
                    if np.all(np.isfinite(rvn)) and float(rvn @ rvn) >= 1e-12:
                        out[lbl] = rvn
                continue
            out[lbl] = v
    return out


def _load_identity(mid: str) -> dict:
    p = CAT / "identity_map" / f"{mid}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("speakers", {})
    except Exception:
        return {}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return -1.0
    return float(a @ b / (na * nb))


def _extract_wav(mid: str, dst: Path) -> bool:
    src = PROX / f"{mid}.mp4"
    if not src.exists():
        return False
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
                       capture_output=True)
    return r.returncode == 0 and dst.exists()


def _resolve_identity(label, idmap):
    if label is None:
        return None, None, 0.0
    info = idmap.get(label)
    if not info:
        return None, None, 0.0
    if info.get("source") == "gallery" and info.get("person_id"):
        return info["person_id"], None, round(float(info.get("gallery_score", 0.0)), 4)
    return None, info.get("cluster_id") or info.get("person_id"), 0.0


def _candidates(events):
    n = len(events)
    for i, e in enumerate(events):
        lab = e.get("diar_label")
        canon = (e.get("evidence") or {}).get("canonical_asr")
        prev = events[i - 1].get("diar_label") if i > 0 else None
        nxt = events[i + 1].get("diar_label") if i < n - 1 else None
        conflict = bool(canon and lab and canon != lab)
        flip = e.get("n_words") == 1 and prev and nxt and prev == nxt and prev != lab
        lowconf = (e.get("confidence", {}).get("speaker", 1.0) < 0.6 and e.get("overlap_group_id"))
        if conflict or flip or lowconf:
            cand = {x for x in (lab, canon, prev, nxt) if x}
            kind = "conflict" if conflict else ("flip" if flip else "lowconf")
            yield i, cand, kind, prev, nxt


def _merge_adjacent(events: list[dict], mid: str) -> list[dict]:
    out: list[dict] = []
    for e in events:
        if (out and out[-1]["speaker_track_id"] == e["speaker_track_id"]
                and out[-1].get("person_id") == e.get("person_id")
                and e["start_ms"] - out[-1]["end_ms"] <= MERGE_GAP):
            p = out[-1]
            p["end_ms"] = e["end_ms"]
            p["text"] = (p["text"] + " " + e["text"]).strip()
            p["n_words"] = p.get("n_words", 0) + e.get("n_words", 0)
            if p.get("words") is not None or e.get("words") is not None:
                p["words"] = (p.get("words") or []) + (e.get("words") or [])
            for k in ("asr", "speaker", "identity"):
                p["confidence"][k] = round(min(p["confidence"][k], e["confidence"][k]), 3)
        else:
            out.append(dict(e))
    for j, e in enumerate(out):
        e["event_id"] = f"{mid}:evt:{j:04d}"
    return out


def _qc(events: list[dict]) -> dict:
    n = len(events)
    conflict = sum(1 for e in events if (e.get("evidence") or {}).get("canonical_asr")
                   and e.get("diar_label") and e["evidence"]["canonical_asr"] != e["diar_label"])
    singleword = sum(1 for e in events if e.get("n_words") == 1)
    pp = 0
    for i in range(1, n - 1):
        if (events[i].get("n_words") == 1
                and events[i - 1]["speaker_track_id"] == events[i + 1]["speaker_track_id"]
                and events[i]["speaker_track_id"] != events[i - 1]["speaker_track_id"]):
            pp += 1
    amb = sum(1 for e in events if e["speaker_track_id"] == "unknown_overlap_speaker")
    return {"n_events": n, "identity_conflict_events": conflict,
            "singleword_events": singleword, "pingpong_residual": pp,
            "ambiguous_events": amb}


def adjudicate(mid: str, inf, Segment, write=True) -> dict:
    events = _load_jsonl(CAT / "speech_events_v2" / f"{mid}.jsonl")
    if not events:
        if write:
            # explicit empty output — absence is never a signal
            _write(mid, [], _qc([]), _qc([]), {})
        return {"media_id": mid, "skipped": "no_events"}
    centroids = _load_centroids(mid)
    idmap = _load_identity(mid)
    before = _qc(events)
    stats = {"canonical": 0, "exclusive": 0, "neighbor": 0, "identity_map": 0,
             "folded": 0, "ambiguous": 0, "no_audio": 0, "unchanged": 0}

    if len(centroids) < 1:
        if write:
            _write(mid, events, before, before, stats)
        return {"media_id": mid, "skipped": "no_centroids"}

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        if not _extract_wav(mid, wav):
            if write:
                _write(mid, events, before, before, stats)
            return {"media_id": mid, "skipped": "no_wav"}

        for i, cand, kind, prev, nxt in _candidates(events):
            e = events[i]
            a = max(0.0, e["start_ms"] / 1000.0 - PAD_S)
            b = e["end_ms"] / 1000.0 + PAD_S
            try:
                emb = np.asarray(inf.crop(str(wav), Segment(a, b))).reshape(-1)
            except Exception:
                stats["no_audio"] += 1
                continue
            if emb.shape[0] != 256 or not np.all(np.isfinite(emb)):
                stats["no_audio"] += 1
                continue
            sims = {lbl: _cos(emb, centroids[lbl]) for lbl in cand if lbl in centroids}
            if not sims:
                continue
            ranked = sorted(sims.items(), key=lambda kv: -kv[1])
            win_lbl, win_sim = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else -1.0
            orig = e.get("diar_label")
            decisive = win_sim >= WIN_ABS and (win_sim - runner) >= WIN_MARGIN

            if decisive and win_lbl != orig:
                _apply(e, win_lbl, mid, idmap, kind, sims, "voiceprint")
                canon = (e.get("evidence") or {}).get("canonical_asr")
                if win_lbl == canon:
                    stats["canonical"] += 1
                elif win_lbl in (prev, nxt):
                    stats["neighbor"] += 1
                else:
                    stats["exclusive"] += 1
            elif decisive and win_lbl == orig:
                stats["unchanged"] += 1
            elif kind == "flip" and prev and prev == nxt:
                # no voiceprint winner: fold isolated flip to surrounding speaker
                _apply(e, prev, mid, idmap, kind, sims, "coherence_fold")
                stats["folded"] += 1
            elif kind == "conflict":
                e["speaker_track_id"] = "unknown_overlap_speaker"
                e["diar_label"] = None
                e["person_id"] = None
                e["stable_unknown_id"] = None
                e.setdefault("adjudication", {})
                e["adjudication"] = {"decision": "ambiguous", "sims": {k: round(v, 3) for k, v in sims.items()}}
                stats["ambiguous"] += 1
            else:
                stats["unchanged"] += 1

    merged = _merge_adjacent(events, mid)
    after = _qc(merged)
    if write:
        _write(mid, merged, before, after, stats)
    return {"media_id": mid, "before": before, "after": after, "stats": stats}


def _apply(e, new_lbl, mid, idmap, kind, sims, method):
    pid, suid, iconf = _resolve_identity(new_lbl, idmap)
    e["diar_label"] = new_lbl
    e["speaker_track_id"] = f"{mid}:{new_lbl}"
    e["person_id"] = pid
    e["stable_unknown_id"] = suid
    e["confidence"]["identity"] = iconf
    e["adjudication"] = {"decision": method, "kind": kind, "won": new_lbl,
                         "sims": {k: round(v, 3) for k, v in sims.items()}}


def _write(mid, events, before, after, stats):
    od = CAT / "speech_events_v2_adjudicated"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{mid}.jsonl"
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    qd = CAT / "speech_events_v2_adjudication_qc"
    qd.mkdir(parents=True, exist_ok=True)
    (qd / f"{mid}.json").write_text(json.dumps(
        {"media_id": mid, "before": before, "after": after, "stats": stats},
        ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    import torch
    from pyannote.audio import Model, Inference
    from pyannote.core import Segment
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf = Inference(model, window="whole", device=dev)

    if a.only:
        mids = [a.only]
    else:
        mids = sorted({os.path.basename(p).split(".")[0]
                       for p in glob.glob(str(CAT / "speech_events_v2" / "*.jsonl"))})
    if a.limit:
        mids = mids[:a.limit]

    agg = {"canonical": 0, "exclusive": 0, "neighbor": 0, "identity_map": 0,
           "folded": 0, "ambiguous": 0, "no_audio": 0, "unchanged": 0}
    b_conf = a_conf = b_pp = a_pp = b_sw = a_sw = 0
    done = 0
    for mid in mids:
        try:
            r = adjudicate(mid, inf, Segment)
        except Exception as exc:
            # Per-media isolation: one bad media must not kill the batch, and the
            # failure must be an explicit error outcome (never silent absence).
            if outcomes is not None:
                outcomes.record(mid, "SEV2_ADJ", "error", reason=str(exc)[:200],
                                producer="adjudicate_speech_events.py")
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue
        if outcomes is not None:
            if "stats" in r:
                outcomes.record(
                    mid, "SEV2_ADJ", "ok", counts=r["stats"],
                    artifacts=[f"catalog/speech_events_v2_adjudicated/{mid}.jsonl"],
                    producer="adjudicate_speech_events.py")
            else:
                skipped = r.get("skipped", "unknown")
                if skipped == "no_events":
                    outcomes.record(mid, "SEV2_ADJ", "empty", reason="no_events",
                                    producer="adjudicate_speech_events.py")
                else:
                    # no_centroids / no_wav: the pass-through copy IS the
                    # artifact — record ok (unadjudicated), never skip
                    # (skip means "no artifact expected"; 13 false drift
                    # rows in reconcile, found 2026-07-03)
                    outcomes.record(mid, "SEV2_ADJ", "ok",
                                    reason=skipped,
                                    counts={"adjudicated_events": 0, "passthrough": 1},
                                    artifacts=[f"catalog/speech_events_v2_adjudicated/{mid}.jsonl"],
                                    producer="adjudicate_speech_events.py")
        if "stats" not in r:
            continue
        done += 1
        for k in agg:
            agg[k] += r["stats"].get(k, 0)
        b_conf += r["before"]["identity_conflict_events"]; a_conf += r["after"]["identity_conflict_events"]
        b_pp += r["before"]["pingpong_residual"]; a_pp += r["after"]["pingpong_residual"]
        b_sw += r["before"]["singleword_events"]; a_sw += r["after"]["singleword_events"]
        if a.only:
            print(json.dumps(r, indent=2))
    print(f"\nMEDIA adjudicated={done}")
    print(f"identity_conflict_events {b_conf} -> {a_conf}")
    print(f"pingpong_residual        {b_pp} -> {a_pp}")
    print(f"singleword_events        {b_sw} -> {a_sw}")
    print(f"decisions: {agg}")


if __name__ == "__main__":
    main()

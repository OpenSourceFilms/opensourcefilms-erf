#!/usr/bin/env python3
"""build_parallel_streams.py — emit genuine SECOND-speaker (stream B) events for
true simultaneous two-line subtitles, and produce the merged render layer.

The canonical-first speech_events_v2_adjudicated layer is stream A (the dominant
speaker per instant). In genuine crosstalk windows a second speaker talks at the
same time; the recall pass (plain-Whisper on overlap gaps) recovered those words,
but the builder either deduped them or folded them into the dominant speaker
because two diar speakers were active (ambiguous). This stage rescues them:

  for each overlap_gap window:
    survivors = recall words in window that are NOT a text+time duplicate of a
                canonical word (same time + same token = dup -> drop; same time +
                different token = keep, that IS the second speaker)
    group survivors into contiguous candidate events
    the second speaker's diar label = the active baseline label (gap.diar_speakers)
                that is NOT the dominant stream-A label
    VOICEPRINT GATE (the evidence, not an assumption): embed the candidate span
                with WeSpeaker, cosine to the dominant vs. the target centroid;
                keep ONLY if it matches the target speaker AND beats the dominant
                by a margin. Otherwise it is just more dominant-speaker speech (or
                ambiguous) and is discarded — a false second line is worse than none.

Outputs (atomic, MEDIA_ID-scoped, explicit empty outcomes):
  catalog/parallel_streams/{mid}.jsonl        stream-B events only
  catalog/render_events_v2/{mid}.jsonl        stream A (adjudicated) + confirmed B, merged
  catalog/parallel_streams_qc/{mid}.json      per-media QC counters

Run with the diarize venv (WeSpeaker):
  venvs/diarize/bin/python bin/build_parallel_streams.py --all --device cuda
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
PROX = ROOT / "media" / "proxies"
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root
CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam); unset = ROOT/catalog
try:
    from erf import outcomes
except Exception:
    outcomes = None

PAD_S = 0.25          # voiceprint crop padding around a candidate event
WIN_ABS = 0.30        # target cosine must reach this
WIN_MARGIN = 0.06     # ...and beat the dominant speaker by this
SAME_VOICE_THR = 0.50  # label centroids this similar = same person, no true 2nd speaker
                       # (WeSpeaker same-speaker cos typically >=0.5). Local same-model
                       # comparison only (diar_embeddings.json centroids, unrelated to the
                       # gallery-matching embedder in erf.embeddings / identity_resolve_v1 --
                       # gallery THR is 0.42 for redimnet_b6, not comparable to this value.)
GROUP_GAP_MS = 700    # group survivor words into one event within this gap
MIN_STREAMB_WORDS = 2 # need at least this many surviving words to bother
MIN_WORD_PROB = 0.50  # second-stream words are quieter/riskier -> stricter bar
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm_tok(s: str) -> str:
    return "".join(_WORD_RE.findall((s or "").lower()))


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


def _resolve_identity(label, idmap):
    if label is None:
        return None, None
    info = idmap.get(label)
    if not info:
        return None, None
    if info.get("source") == "gallery" and info.get("person_id"):
        return info["person_id"], None
    return None, info.get("cluster_id") or info.get("person_id")


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


def _overlap_frac(a0, a1, b0, b1) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1, min(a1 - a0, b1 - b0))


def _dominant_label(streamA: list[dict], w0: int, w1: int) -> str | None:
    """The stream-A diar label covering the most of [w0,w1)."""
    by_lbl: dict[str, int] = {}
    for e in streamA:
        lbl = e.get("diar_label")
        if not lbl:
            continue
        lo, hi = max(e["start_ms"], w0), min(e["end_ms"], w1)
        if hi > lo:
            by_lbl[lbl] = by_lbl.get(lbl, 0) + (hi - lo)
    if not by_lbl:
        return None
    return max(by_lbl, key=by_lbl.get)


def _group(words: list[dict]) -> list[list[dict]]:
    words = sorted(words, key=lambda w: (w["start_ms"], w["end_ms"]))
    groups, cur = [], []
    for w in words:
        if cur and w["start_ms"] - cur[-1]["end_ms"] > GROUP_GAP_MS:
            groups.append(cur); cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_media(mid, inf, Segment, tmp_dir) -> dict:
    qc = {"media_id": mid, "n_overlap_windows": 0, "n_candidate_events": 0,
          "n_confirmed": 0, "n_streamB_words": 0, "rej_no_second_label": 0,
          "rej_no_centroid": 0, "rej_too_few_words": 0, "rej_voiceprint": 0,
          "rej_no_audio": 0}

    streamA = _load_jsonl(CAT / "speech_events_v2_adjudicated" / f"{mid}.jsonl")
    out_par = CAT / "parallel_streams" / f"{mid}.jsonl"
    out_ren = CAT / "render_events_v2" / f"{mid}.jsonl"
    out_par.parent.mkdir(parents=True, exist_ok=True)
    out_ren.parent.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, rows: list[dict]):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.replace(tmp, path)

    gaps = [g for g in _load_jsonl(CAT / "asr_recall_gaps" / f"{mid}.jsonl")
            if g.get("gap_type") == "overlap_gap"]
    recall = _load_jsonl(CAT / "asr_recall_pass" / f"{mid}.jsonl")
    if not gaps or not recall:
        _write(out_par, [])
        _write(out_ren, streamA)  # explicit: render layer == adjudicated when no B
        return qc

    canon = _load_jsonl(CAT / "asr" / f"{mid}.words.jsonl")
    canon_spans = [(w["start_ms"], w["end_ms"], _norm_tok(w.get("text") or "")) for w in canon]
    # NOTE (2026-07-03): do NOT dedup against the A events' carried word
    # arrays — every recall word is in some A event by design (recall-first);
    # a confirmed B event is the re-attributed clean version of those words
    # and _superseded() drops the A-side fragments. Echo suppression is the
    # same-voice target guard below, not word dedup.
    centroids = _load_centroids(mid)
    idmap = _load_identity(mid)

    wav = tmp_dir / f"{mid}.wav"
    have_wav = _extract_wav(mid, wav)

    streamB: list[dict] = []
    ev_seq = 0
    for g in gaps:
        qc["n_overlap_windows"] += 1
        w0, w1 = int(g["start_ms"]), int(g["end_ms"])
        active = [l for l in (g.get("diar_speakers") or []) if l]
        # survivors: recall words in window, not a text+time dup of canonical
        surv = []
        for w in recall:
            if not (w["start_ms"] < w1 and w["end_ms"] > w0):
                continue
            if w.get("prob", 1.0) < MIN_WORD_PROB:
                continue
            rt = _norm_tok(w.get("text") or "")
            if not rt:
                continue
            dup = any(_overlap_frac(w["start_ms"], w["end_ms"], ca, cb) >= 0.5 and rt == ct
                      for ca, cb, ct in canon_spans)
            if dup:
                continue
            surv.append(w)
        if len(surv) < MIN_STREAMB_WORDS:
            continue

        dom = _dominant_label(streamA, w0, w1)
        # target label = an active baseline label != dominant, with a centroid.
        # Same-voice guard (2026-07-03): when two diar labels are the same
        # person (over-segmentation / demoted over-merge), a recall echo of the
        # dominant voice matches the "other" label's centroid and passes the
        # margin gate — there is no true second speaker between same-voice
        # labels, so exclude them from targets.
        targets = [l for l in active if l != dom and l in centroids
                   and not (dom in centroids
                            and _cos(centroids[l], centroids[dom]) >= SAME_VOICE_THR)]
        if not targets:
            # No genuine second speaker for this gap (same-voice guard excluded
            # every candidate, or none was ever active). The survivor words
            # still came in as an unresolved recall/unknown_overlap_speaker
            # fragment from SEV2/SEV2_ADJ though — left alone they stay
            # permanently person_id=None (D-Log-14: this was scoring as a real
            # attribution miss on golden truth that expected them folded into
            # the dominant speaker). Reuse the exact voiceprint check above
            # (WIN_ABS, same centroid/embedding machinery) against `dom`
            # itself: if the survivors' own audio matches the dominant
            # speaker, they're a same-voice echo, not a second speaker — fold
            # them into stream A by patching the matching source fragment(s)
            # in place, the same terminal state a genuine stream-B rejection
            # would otherwise leave for _superseded() to clean up.
            if have_wav and dom in centroids:
                fa, fb = surv[0]["start_ms"], surv[-1]["end_ms"]
                fa_s = max(0.0, fa / 1000.0 - PAD_S)
                fb_s = fb / 1000.0 + PAD_S
                try:
                    femb = np.asarray(inf.crop(str(wav), Segment(fa_s, fb_s))).reshape(-1)
                    fdom_cos = _cos(femb, centroids[dom])
                except Exception:
                    fdom_cos = -1.0
                if fdom_cos >= WIN_ABS:
                    pid, sid = _resolve_identity(dom, idmap)
                    for e in streamA:
                        frag = e.get("source") in ("recall", "recall_pass") or \
                            e.get("speaker_track_id") == "unknown_overlap_speaker"
                        if frag and e["start_ms"] < w1 and e["end_ms"] > w0:
                            e["speaker_track_id"] = f"{mid}:{dom}"
                            e["diar_label"] = dom
                            e["person_id"] = pid
                            e["stable_unknown_id"] = sid
                            e.setdefault("confidence", {})["identity"] = 1.0 if pid else 0.0
                            e.setdefault("evidence", {})["same_voice_fold"] = dom
                    qc["same_voice_folded"] = qc.get("same_voice_folded", 0) + 1
            qc["rej_no_second_label"] += 1
            continue

        for grp in _group(surv):
            if len(grp) < MIN_STREAMB_WORDS:
                continue
            qc["n_candidate_events"] += 1
            a_ms, b_ms = grp[0]["start_ms"], grp[-1]["end_ms"]
            if not have_wav:
                qc["rej_no_audio"] += 1
                continue
            a = max(0.0, a_ms / 1000.0 - PAD_S)
            b = b_ms / 1000.0 + PAD_S
            try:
                emb = np.asarray(inf.crop(str(wav), Segment(a, b))).reshape(-1)
            except Exception:
                qc["rej_no_audio"] += 1
                continue
            dom_cos = _cos(emb, centroids[dom]) if dom in centroids else -1.0
            scored = sorted(((l, _cos(emb, centroids[l])) for l in targets),
                            key=lambda x: x[1], reverse=True)
            tgt, tgt_cos = scored[0]
            # GATE: candidate must voiceprint-match the OTHER speaker and beat dominant
            if not (tgt_cos >= WIN_ABS and (tgt_cos - dom_cos) >= WIN_MARGIN):
                qc["rej_voiceprint"] += 1
                continue

            pid, sid = _resolve_identity(tgt, idmap)
            text = " ".join((w.get("text") or "").strip() for w in grp).strip()
            ev_seq += 1
            og = f"{mid}:og:{w0}"
            streamB.append({
                "event_id": f"{mid}:B{ev_seq:04d}",
                "media_id": mid, "source": "parallel_recall", "stream": "B",
                "start_ms": a_ms, "end_ms": b_ms, "text": text,
                "speaker_track_id": f"{mid}:{tgt}", "diar_label": tgt,
                "person_id": pid, "stable_unknown_id": sid,
                "overlap_group_id": og, "n_words": len(grp),
                "words": [{"text": (w.get("text") or "").strip(),
                           "start_ms": w["start_ms"], "end_ms": w["end_ms"],
                           "prob": round(float(w.get("prob", 0.0)), 3),
                           "source": "recall_pass"} for w in grp],
                "confidence": {"asr": round(float(np.mean([w.get("prob", 0.0) for w in grp])), 4),
                               "speaker": round(tgt_cos, 4), "identity": 1.0 if pid else 0.0},
                "evidence": {"voiceprint_target": tgt, "voiceprint_cos": round(tgt_cos, 4),
                             "voiceprint_dominant": dom, "voiceprint_dom_cos": round(dom_cos, 4),
                             "dominant_stream_label": dom, "diar_speakers": active},
            })
            qc["n_confirmed"] += 1
            qc["n_streamB_words"] += len(grp)
            # tag the concurrent stream-A event(s) with the same overlap_group_id
            for e in streamA:
                if e["start_ms"] < b_ms and e["end_ms"] > a_ms and e.get("diar_label") == dom:
                    e.setdefault("overlap_group_id", og)
                    e["has_parallel_stream"] = True

    _write(out_par, streamB)
    # A confirmed stream-B event is the clean, correctly-attributed version of the
    # same recall words that stream A carried as fragmented recall/ambiguous events.
    # Drop those superseded fragments from the render layer so the words appear once.
    # Canonical and mixed (canonical+recall) events are never dropped.
    b_spans = [(e["start_ms"], e["end_ms"]) for e in streamB]

    def _superseded(e) -> bool:
        src = e.get("source") or ""
        recall_frag = src in ("recall", "recall_pass") or \
            e.get("speaker_track_id") == "unknown_overlap_speaker"
        if not recall_frag:
            return False
        return any(_overlap_frac(e["start_ms"], e["end_ms"], a, b) >= 0.5 for a, b in b_spans)

    render_A = [e for e in streamA if not _superseded(e)]
    merged = sorted(render_A + streamB, key=lambda e: (e["start_ms"], e["end_ms"]))
    _write(out_ren, merged)
    qc["n_superseded_dropped"] = len(streamA) - len(render_A)
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    if a.only:
        mids = [a.only]
    else:
        mids = sorted(p.name[:-len(".jsonl")]
                      for p in (CAT / "speech_events_v2_adjudicated").glob("*.jsonl"))
    if a.limit:
        mids = mids[:a.limit]

    from pyannote.audio import Model, Inference
    from pyannote.core import Segment
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    import torch
    dev = torch.device(a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")
    inf = Inference(model, window="whole", device=dev)

    tmp_dir = Path(tempfile.mkdtemp(prefix="parstream_", dir="/tmp/claude-0"))
    tot = {"n_confirmed": 0, "n_streamB_words": 0, "n_candidate_events": 0,
           "rej_voiceprint": 0, "rej_no_second_label": 0, "media_with_B": 0}
    for mid in mids:
        try:
            qc = build_media(mid, inf, Segment, tmp_dir)
        except Exception as exc:
            print(f"ERR {mid}: {exc}", file=sys.stderr)
            if outcomes is not None:
                outcomes.record(mid, "PSTREAM", "error", reason=str(exc)[:200],
                                producer="build_parallel_streams.py")
            continue
        if outcomes is not None:
            outcomes.record(
                mid, "PSTREAM", "ok",
                counts={k: qc.get(k, 0) for k in
                        ("n_overlap_windows", "n_candidate_events", "n_confirmed",
                         "n_streamB_words", "rej_voiceprint", "n_superseded_dropped")},
                artifacts=[f"catalog/parallel_streams/{mid}.jsonl",
                           f"catalog/render_events_v2/{mid}.jsonl"],
                producer="build_parallel_streams.py")
        qcp = CAT / "parallel_streams_qc" / f"{mid}.json"
        qcp.parent.mkdir(parents=True, exist_ok=True)
        tmp = qcp.with_name(qcp.name + ".tmp")
        tmp.write_text(json.dumps(qc, indent=1))
        os.replace(tmp, qcp)
        for k in ("n_confirmed", "n_streamB_words", "n_candidate_events",
                  "rej_voiceprint", "rej_no_second_label"):
            tot[k] += qc.get(k, 0)
        if qc.get("n_confirmed", 0) > 0:
            tot["media_with_B"] += 1
        wp = tmp_dir / f"{mid}.wav"
        if wp.exists():
            wp.unlink()

    print("PARALLEL STREAMS BUILT")
    print(f"  media with stream B:      {tot['media_with_B']}")
    print(f"  candidate events:         {tot['n_candidate_events']}")
    print(f"  confirmed stream-B events:{tot['n_confirmed']}")
    print(f"  stream-B words:           {tot['n_streamB_words']}")
    print(f"  rejected (voiceprint):    {tot['rej_voiceprint']}")
    print(f"  rejected (no 2nd label):  {tot['rej_no_second_label']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build speech_events_v2: speaker-owned word spans = the canonical unit.

Merges canonical ASR words (catalog/asr/{mid}.words.jsonl) with recall-pass
recovered words (catalog/asr_recall_pass/{mid}.jsonl), attributes each word to a
speaker CANONICAL-FIRST (canonical speaker_label is the primary owner; exclusive
diarization is evidence only — see _attribute, verdict 2026-07-01), resolves the
speaker to a person via identity_map, then groups consecutive same-speaker words
into events. Each event carries its word-level detail in `words`.

Ambiguity is preserved, never forced: a word inside a multi-speaker overlap that
the exclusive track cannot cleanly own is tagged speaker_track_id
'unknown_overlap_speaker' rather than collapsed onto the dominant speaker.

Output: catalog/speech_events_v2/{mid}.jsonl  (one event per line)
Writes an explicit empty file when a media has no words (absence != signal).
Per-word voiceprint tie-break is a future enhancement (identity is already
voiceprint-derived via identity_map); see ATTRIB_TODO.

Phase 3 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §3.2): records
`pct_person_id` (events' person_id coverage) in every "ok" outcome's metrics,
and — only when this media's identity_map has >=1 gallery-anchored slot — a
`pct_person_id > 0` health predicate (stop letting a gallery voice resolve to
zero attributed person_id silently; rando-only media are legitimately 0 and
pass vacuously). Also carries `inputs={"identity_map": <fp>}`, the content
fingerprint of the identity_map consumed this run.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root, diar_source
CAT = catalog_root()  # honors ERF_CATALOG_ROOT (D-Log-35 seam); unset = ROOT/catalog
# Phase 4 (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md P4): erf.outcomes is a HARD
# dependency (no `if outcomes is not None` escape) — an unloggable SEV2 run must fail
# loudly, not silently degrade to never-logs-anything (matches CLUSTERS' Phase 3 pattern).
from erf import outcomes
try:
    from erf import stages as _stages_registry
except Exception:
    _stages_registry = None
GAP_MS = 700          # event break when same-speaker silence exceeds this
DEDUP_OVERLAP = 0.5   # recall word dropped only if >=50% overlapped by a canonical
                      # word AND the token matches (simultaneous *different* text is
                      # kept — that is the second speaker in genuine crosstalk).


def _norm_tok(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists() or p.stat().st_size == 0:
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _mid_of(w: dict) -> float:
    return (w["start_ms"] + w["end_ms"]) / 2.0


def _overlap_frac(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    dur = max(1, a1 - a0)
    return inter / dur


def _turn_at(turns: list[dict], t: float):
    """Return the speaker_label of the turn covering time t (ms), else None."""
    for tr in turns:
        if tr["start_ms"] <= t <= tr["end_ms"]:
            return tr["speaker_label"]
    return None


def _speakers_active(turns: list[dict], a: int, b: int) -> set[str]:
    return {tr["speaker_label"] for tr in turns
            if tr["start_ms"] < b and tr["end_ms"] > a}


def _load_identity(mid: str) -> dict:
    p = CAT / "identity_map" / f"{mid}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("speakers", {})
    except Exception:
        return {}


def _identity_map_fp(mid: str) -> dict:
    """`inputs=` fingerprint (Phase 3, D-Log-04 §4) for the identity_map this
    run consumed: first 16 hex chars of sha256(file content). Absent file ->
    omit the key entirely (not a placeholder null) — nothing was consumed."""
    p = CAT / "identity_map" / f"{mid}.json"
    if not p.exists():
        return {}
    return {"identity_map": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}


def _load_demoted_windows(mid: str) -> dict:
    """speaker_label -> [(start_ms, end_ms, person_id_or_None, score), ...] for
    this media's demoted_overmerge slots (D-Log-31: bin/build_demoted_window_identity.py
    scores each per-window voiceprint independently, rather than trusting the
    single per-slot anonymous ID applied uniformly across the whole label)."""
    p = CAT / "demoted_window_identity" / f"{mid}.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    by_label: dict[str, list] = {}
    for w in data.get("windows", []):
        by_label.setdefault(w["speaker_label"], []).append(
            (w["start_ms"], w["end_ms"], w.get("person_id"), w.get("score", 0.0)))
    return by_label


def _word_demoted_override(label: str | None, idmap: dict, word_start: int, word_end: int,
                            demoted_windows: dict) -> tuple[str | None, float]:
    """(person_id, score) for a SINGLE WORD's demoted_overmerge override.

    D-Log-32 fix: must be word-level, not event-level. A first attempt matched
    windows against a whole SEV2 event's aggregate (start_ms, end_ms) — but
    one continuous same-diar-label turn can genuinely contain several real
    speakers (the demoted-slot problem this whole mechanism targets), so an
    event-level match let one window's verdict bleed across words that should
    have stayed anonymous (17 words in a real corpus case, D-Log-32). Matching
    each word's own span keeps the correction scoped to exactly the words a
    window actually covers.
    """
    if label is None or not demoted_windows:
        return None, 0.0
    info = idmap.get(label)
    if not info or info.get("source") != "demoted_overmerge":
        return None, 0.0
    best_frac, best_pid, best_score = 0.0, None, 0.0
    for w_start, w_end, pid, score in demoted_windows.get(label, []):
        if pid is None:
            continue
        frac = _overlap_frac(word_start, word_end, w_start, w_end)
        if frac > best_frac:
            best_frac, best_pid, best_score = frac, pid, score
    if best_pid is not None and best_frac > 0.0:
        return best_pid, round(float(best_score), 4)
    return None, 0.0


def _resolve_identity(label: str | None, idmap: dict):
    """(person_id, stable_unknown_id, identity_conf) for a diar label —
    the uniform per-slot fallback, used when an event has no consistent
    word-level demoted override (see _word_demoted_override)."""
    if label is None:
        return None, None, 0.0
    info = idmap.get(label)
    if not info:
        return None, None, 0.0
    if info.get("source") == "gallery" and info.get("person_id"):
        return info["person_id"], None, round(float(info.get("gallery_score", 0.0)), 4)
    # unenrolled but voice-clustered: stable id from the cluster
    return None, info.get("cluster_id") or info.get("person_id"), 0.0


def _diar_paths(mid: str):
    d = CAT / "diar_hypotheses" / mid
    src = diar_source()  # ERF_DIAR_SOURCE (D-Log-35 seam); default community1
    excl = d / f"{src}_excl.jsonl"
    base = d / f"{src}.jsonl"
    if not base.exists():
        base = d / f"{src}_baseline.jsonl"
    return excl, base


def _overlap_groups(mid: str) -> list[dict]:
    return _load_jsonl(CAT / "asr" / f"{mid}.overlap.jsonl")


def _attribute(word_mid: float, word_a: int, word_b: int, canon_label,
               excl: list[dict], base: list[dict]):
    """Return (speaker_label|None|'__ambiguous__', speaker_conf, evidence)."""
    ev = {}
    ex_lab = _turn_at(excl, word_mid) if excl else None
    if ex_lab is not None:
        ev["community1_exclusive"] = ex_lab
    base_active = _speakers_active(base, word_a, word_b) if base else set()
    base_single = next(iter(base_active)) if len(base_active) == 1 else None
    if base_single:
        ev["community1_baseline"] = base_single
    if canon_label:
        ev["canonical_asr"] = canon_label

    # ATTRIBUTION PRIORITY (evidence-driven, 2026-07-01): corpus-wide voiceprint
    # adjudication resolved 217 conflicts toward canonical and 0 toward exclusive
    # diarization. Exclusive over-switches at the word level, so it is demoted to
    # EVIDENCE ONLY. Primary = canonical speaker_label (baseline-aligned, trusted);
    # for label-less recall words, baseline single-speaker; exclusive is a last
    # resort and flagged low-confidence.
    if canon_label:
        primary = canon_label
        conf = 0.9 if base_single == canon_label else 0.72
        return primary, conf, ev

    # recall word (no canonical label): attribute from baseline diar
    if base_single:
        return base_single, 0.75, ev
    if len(base_active) >= 2:
        # genuine overlap, no single owner -> defer to voiceprint adjudication
        return "__ambiguous__", 0.3, ev
    if ex_lab is not None:
        # only exclusive has an opinion, and it is unreliable
        return ex_lab, 0.4, ev
    return None, 0.0, ev


def _smooth_pingpong(words: list[dict]) -> int:
    """Conservative anti-ping-pong: fold an isolated SINGLE short word attributed
    to Y when both neighbours are the same speaker X (X != Y) and the Y word is
    SEQUENTIAL (not time-overlapping either neighbour). This removes fine
    exclusive-diar switching artifacts without collapsing genuine simultaneous
    overlap (different text at the same time is preserved) or multi-word turns.
    Returns the number of words folded.
    """
    folded = 0
    for i in range(1, len(words) - 1):
        w, prev, nxt = words[i], words[i - 1], words[i + 1]
        lab = w["attr_label"]
        if lab in ("__ambiguous__", None):
            continue
        pl, nl = prev["attr_label"], nxt["attr_label"]
        if pl != nl or pl == lab or pl in ("__ambiguous__", None):
            continue
        dur = w["end_ms"] - w["start_ms"]
        # sequential = the word sits after prev and before nxt with no real overlap
        seq = (w["start_ms"] >= prev["end_ms"] - 80 and
               nxt["start_ms"] >= w["end_ms"] - 80)
        if dur < 350 and seq:
            w["attr_label"] = pl
            w["evidence"] = {**w["evidence"], "smoothed": "pingpong",
                             "smoothed_from": lab}
            w["speaker_conf"] = min(w["speaker_conf"], 0.5)
            folded += 1
    return folded


def build_media(mid: str, out_dir: Path, write: bool = True) -> dict:
    canon = _load_jsonl(CAT / "asr" / f"{mid}.words.jsonl")
    recall = _load_jsonl(CAT / "asr_recall_pass" / f"{mid}.jsonl")
    excl_p, base_p = _diar_paths(mid)
    excl = _load_jsonl(excl_p)
    base = _load_jsonl(base_p)
    idmap = _load_identity(mid)
    demoted_windows = _load_demoted_windows(mid)  # D-Log-31
    # Phase 3 (D-Log-04 §4): conditional health-predicate gate — this media's
    # pct_person_id coverage is only REQUIRED to be > 0 when identity_map
    # actually anchored a slot to a known gallery person.
    has_gallery_slot = any(info.get("source") == "gallery" for info in idmap.values())
    overlaps = _overlap_groups(mid)

    # ---- merge words, dedup recall against canonical ----
    words = []
    for w in canon:
        words.append({"start_ms": w["start_ms"], "end_ms": w["end_ms"],
                      "text": (w.get("text") or "").strip(), "prob": w.get("prob", 0.0),
                      "source": "canonical", "canon_label": w.get("speaker_label")})
    canon_spans = [(w["start_ms"], w["end_ms"], _norm_tok(w.get("text") or "")) for w in canon]
    for w in recall:
        a, b = w["start_ms"], w["end_ms"]
        rt = _norm_tok(w.get("text") or "")
        # true duplicate = same token at the same time. Same time + DIFFERENT token
        # is a second speaker in overlap and must be kept.
        dup = any(_overlap_frac(a, b, ca, cb) >= DEDUP_OVERLAP and rt and rt == ct
                  for ca, cb, ct in canon_spans)
        if dup:
            continue
        words.append({"start_ms": a, "end_ms": b, "text": (w.get("text") or "").strip(),
                      "prob": w.get("prob", 0.0), "source": "recall_pass", "canon_label": None})
    words = [w for w in words if w["text"]]
    words.sort(key=lambda w: (w["start_ms"], w["end_ms"]))

    if not words:
        if write:
            _write(out_dir / f"{mid}.jsonl", [])
        return {"media_id": mid, "n_words": 0, "n_events": 0,
                "pct_person_id": 0.0, "has_gallery_slot": has_gallery_slot}

    # ---- attribute each word ----
    for w in words:
        m = _mid_of(w)
        lab, conf, ev = _attribute(m, w["start_ms"], w["end_ms"], w["canon_label"], excl, base)
        w["attr_label"] = lab
        w["speaker_conf"] = conf
        w["evidence"] = ev
        # overlap group
        w["ovl"] = None
        for i, o in enumerate(overlaps):
            if o["start_ms"] < w["end_ms"] and o["end_ms"] > w["start_ms"]:
                w["ovl"] = f"{mid}:ovl:{i:04d}"
                break

    n_pingpong = _smooth_pingpong(words)

    # D-Log-32: word-level demoted-slot override, computed AFTER pingpong
    # smoothing (which can rewrite attr_label) and before event grouping, so
    # a per-word identity change (e.g. two real speakers inside one
    # continuous diar turn) splits the event correctly and never uses a
    # stale pre-smoothing label.
    for w in words:
        w["demoted_pid"], w["demoted_score"] = _word_demoted_override(
            w["attr_label"], idmap, w["start_ms"], w["end_ms"], demoted_windows)

    # ---- group into events (same track + gap<=GAP_MS) ----
    events = []
    cur = None

    def track_id(lab):
        if lab == "__ambiguous__":
            return "unknown_overlap_speaker"
        if lab is None:
            return f"{mid}:spk:none"
        return f"{mid}:{lab}"

    def flush():
        nonlocal cur
        if cur is None:
            return
        lab = cur["attr_label"]
        wl = cur["words"]
        # D-Log-32: all words in a flushed event share the same demoted_pid by
        # construction (the grouping loop below splits on a demoted_pid
        # change), so wl[0] speaks for the whole event.
        if wl[0]["demoted_pid"] is not None:
            pid, suid = wl[0]["demoted_pid"], None
            iconf = max(w["demoted_score"] for w in wl)
        else:
            pid, suid, iconf = _resolve_identity(
                None if lab in ("__ambiguous__", None) else lab, idmap)
        srcs = set(w["source"] for w in cur["words"])
        source = "mixed" if len(srcs) > 1 else next(iter(srcs))
        events.append({
            "event_id": f"{mid}:evt:{len(events):04d}",
            "media_id": mid,
            "source": source,
            "start_ms": wl[0]["start_ms"], "end_ms": wl[-1]["end_ms"],
            "text": " ".join(w["text"] for w in wl),
            "speaker_track_id": cur["track"],
            "diar_label": None if lab in ("__ambiguous__", None) else lab,
            "person_id": pid,
            "stable_unknown_id": suid,
            "overlap_group_id": cur["ovl"],
            "n_words": len(wl),
            "words": [{"text": w["text"], "start_ms": w["start_ms"],
                       "end_ms": w["end_ms"], "prob": round(w["prob"], 3),
                       "source": w["source"]} for w in wl],
            "confidence": {
                "asr": round(sum(w["prob"] for w in wl) / len(wl), 3),
                "speaker": round(min(w["speaker_conf"] for w in wl), 3),
                "identity": iconf,
            },
            "evidence": wl[len(wl) // 2]["evidence"],
        })
        cur = None

    for w in words:
        tr = track_id(w["attr_label"])
        # D-Log-32: also split when the word-level demoted override changes,
        # even if track+gap would otherwise merge — a continuous same-label
        # turn can genuinely contain several real speakers.
        same_override = cur is not None and w["demoted_pid"] == cur["words"][-1]["demoted_pid"]
        if cur and tr == cur["track"] and same_override and w["start_ms"] - cur["words"][-1]["end_ms"] <= GAP_MS:
            cur["words"].append(w)
            if cur["ovl"] is None:
                cur["ovl"] = w["ovl"]
        else:
            flush()
            cur = {"track": tr, "attr_label": w["attr_label"], "ovl": w["ovl"], "words": [w]}
    flush()

    if write:
        _write(out_dir / f"{mid}.jsonl", events)

    # ---- QC metrics ----
    n_recall = sum(1 for w in words if w["source"] == "recall_pass")
    n_amb = sum(1 for e in events if e["speaker_track_id"] == "unknown_overlap_speaker")
    n_short = sum(1 for e in events if (e["end_ms"] - e["start_ms"]) < 350)
    n_singleword = sum(1 for e in events if e["n_words"] == 1)
    n_pp_residual = 0  # single-word events still sandwiched by same other speaker
    for i in range(1, len(events) - 1):
        if (events[i]["n_words"] == 1 and
                events[i - 1]["speaker_track_id"] == events[i + 1]["speaker_track_id"] and
                events[i]["speaker_track_id"] != events[i - 1]["speaker_track_id"]):
            n_pp_residual += 1
    n_id_conflict = sum(
        1 for e in events
        if e.get("evidence", {}).get("canonical_asr") and
        e.get("diar_label") and
        e["evidence"]["canonical_asr"] != e["diar_label"])
    n_recall_events = sum(1 for e in events if e["source"] in ("recall_pass", "mixed"))
    # Phase 3 (D-Log-04 §4): events' person_id coverage — the SEV2 equivalent
    # of T0's pct_person_id, over events (the canonical unit here) not words.
    n_with_person = sum(1 for e in events if e.get("person_id"))
    pct_person_id = round(100 * n_with_person / len(events), 1) if events else 0.0
    qc = {"media_id": mid, "n_words": len(words), "n_recall_words": n_recall,
          "n_events": len(events), "n_ambiguous_events": n_amb,
          "n_short_cues": n_short, "n_singleword_events": n_singleword,
          "n_pingpong_residual": n_pp_residual, "n_pingpong_folded": n_pingpong,
          "n_identity_conflict_events": n_id_conflict, "n_recall_events": n_recall_events,
          "pct_person_id": pct_person_id, "has_gallery_slot": has_gallery_slot}
    if write:
        qc_dir = CAT / "speech_events_v2_qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        (qc_dir / f"{mid}.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2))
    return qc


def _write(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    out_dir = CAT / "speech_events_v2"

    if a.only:
        mids = [a.only]
    else:
        mids = sorted({os.path.basename(p).split(".")[0]
                       for p in glob.glob(str(CAT / "asr" / "*.words.jsonl"))})
    if a.limit:
        mids = mids[:a.limit]

    agg = {k: 0 for k in ("n_words", "n_recall_words", "n_events", "n_ambiguous_events",
                          "n_short_cues", "n_singleword_events", "n_pingpong_residual",
                          "n_pingpong_folded", "n_identity_conflict_events", "n_recall_events")}
    for mid in mids:
        try:
            r = build_media(mid, out_dir, write=not a.report_only)
        except Exception as exc:
            # Per-media isolation: one bad media must not kill the batch, and the
            # failure must be an explicit error outcome (never silent absence).
            if not a.report_only:
                outcomes.record(mid, "SEV2", "error", reason=str(exc)[:200],
                                producer="build_speech_events_v2.py")
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue
        if not a.report_only:
            if r.get("n_words", 0) == 0:
                outcomes.record(mid, "SEV2", "empty", reason="no_words",
                                metrics={"pct_person_id": r.get("pct_person_id", 0.0)},
                                producer="build_speech_events_v2.py")
            else:
                # Phase 3 (D-Log-04 §4): pct_person_id always in metrics; the
                # health predicate only enters `values` (and so is only
                # evaluated) when this media's identity_map anchored a slot to
                # a gallery person — rando-only media pass vacuously.
                counts = {k: r.get(k, 0) for k in
                          ("n_words", "n_recall_words", "n_events",
                           "n_ambiguous_events", "n_identity_conflict_events")}
                counts["pct_person_id"] = r.get("pct_person_id", 0.0)
                values = {"pct_person_id": r["pct_person_id"]} if r.get("has_gallery_slot") else {}
                health = (_stages_registry.evaluate_health("SEV2", values)
                          if (_stages_registry is not None and values) else None)
                outcomes.record(
                    mid, "SEV2", "ok",
                    counts=counts, health=health, inputs=_identity_map_fp(mid),
                    artifacts=[f"catalog/speech_events_v2/{mid}.jsonl"],
                    producer="build_speech_events_v2.py")
        for k in agg:
            agg[k] += r.get(k, 0)
        if a.only or r["n_events"]:
            print(f"{mid}: {r['n_words']}w (+{r.get('n_recall_words',0)} recall) "
                  f"-> {r['n_events']} events, {r.get('n_ambiguous_events',0)} amb, "
                  f"pp_fold={r.get('n_pingpong_folded',0)} pp_resid={r.get('n_pingpong_residual',0)} "
                  f"idconf={r.get('n_identity_conflict_events',0)}")
    print(f"TOTAL media={len(mids)} words={agg['n_words']} recall_words={agg['n_recall_words']} "
          f"events={agg['n_events']} ambiguous={agg['n_ambiguous_events']}")
    print(f"QC: short_cues={agg['n_short_cues']} singleword_events={agg['n_singleword_events']} "
          f"pingpong_folded={agg['n_pingpong_folded']} pingpong_residual={agg['n_pingpong_residual']} "
          f"identity_conflict_events={agg['n_identity_conflict_events']} recall_events={agg['n_recall_events']}")


if __name__ == "__main__":
    main()

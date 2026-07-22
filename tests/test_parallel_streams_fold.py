"""GAMEPLAN.md 2026-07-12 handover, Sonnet-implementation worklist item 1
(D-Log-17 finding): `bin/build_parallel_streams.py`'s same-voice fold branch
(lines ~240-277) had ZERO tests before this file.

`build_media(mid, inf, Segment, tmp_dir)` takes its heavy deps (pyannote
`Inference` and `Segment`) as parameters, so it is testable without a GPU or
real audio: pass a fake `inf` whose `.crop()` returns a deterministic vector,
and a trivial stand-in `Segment` class (only ever used as an opaque arg
threaded through to `inf.crop`, never introspected by the real code).

`CAT` is a module global (`bin/build_parallel_streams.py:38`) monkeypatched to
tmp_path per the repo convention (see tests/test_semantic_health.py,
tests/test_degraded_proof.py, tests/test_se_adapter.py — all monkeypatch a
bin/-module's `CAT`, never touch the real catalog).

One additional seam beyond CAT: `_extract_wav()` shells out to ffmpeg against
`PROX / f"{mid}.mp4"` (a SEPARATE module global, `media/proxies/`, not
under CAT) to materialize the wav `inf.crop()` reads. There is no real proxy
video in the test env, so `_extract_wav` is monkeypatched to return True
without touching disk -- the fake `inf.crop()` never opens the wav path
either, so nothing downstream needs it to exist. This is the same
no-GPU/no-audio seam the GAMEPLAN spec already calls for (fake `inf`), just
extended to the one other real-I/O dependency in the function; no change to
build_parallel_streams.py itself.

The fold branch (bin/build_parallel_streams.py:240-277) triggers when a gap
window has NO usable second-speaker target -- either because only one label
was ever active, or (the interesting case, and what these tests exercise)
because a second active label's centroid is judged the SAME PERSON as the
dominant label (`SAME_VOICE_THR`, line 49/239: over-segmented diarization).
In that case the code does NOT give up: it re-embeds the recall-survivor
fragment's own audio via the fake `inf.crop()` and checks it against the
DOMINANT speaker's centroid (`WIN_ABS`, line 47/263). If that matches, the
fragment is a same-voice echo, not a genuine 2nd speaker -- it gets patched
in place into stream A (`same_voice_folded` QC counter) instead of being left
as an orphaned `unknown_overlap_speaker` fragment.

Three behaviors proven here, matching the GAMEPLAN spec exactly:
  1. test_fold_fires_on_same_voice_match       -- fold fires, patches fragment
  2. test_fold_does_not_fire_on_mismatch       -- fold does NOT fire
  3. test_genuine_second_speaker_untouched_by_fold -- real 2nd voice: stream-B
     event created via the normal voiceprint gate, fold logic never entered
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ps = _load_bin_module("ps_fold_test", "build_parallel_streams.py")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _vec(idx: int, dim: int = 256) -> list[float]:
    """Deterministic unit basis vector so cosine similarity is exact: two
    vectors built from the same index are cos==1.0 (same voice); vectors from
    different indices are orthogonal, cos==0.0 (different voice)."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


class _FakeSegment:
    """Stand-in for pyannote.core.Segment -- build_media only ever threads
    this straight through to inf.crop() as an opaque arg."""

    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeInference:
    """Stand-in for pyannote.audio.Inference -- .crop() is the only method
    build_media calls, and it always returns this fixed vector regardless of
    which (wav_path, segment) it's asked to embed. That's sufficient: each
    test controls what "the recorded audio actually sounds like" purely via
    this return vector vs. the centroids fixture."""

    def __init__(self, vector: list[float]):
        self.vector = vector
        self.calls: list[tuple] = []

    def crop(self, wav_path, segment):
        self.calls.append((wav_path, segment))
        return self.vector


def _base_fixture(tmp_path, mid: str, dom_label: str, dom_centroid, second_label: str | None,
                   second_centroid, include_fragment_in_streamA: bool = True):
    """Common scaffolding shared by all three tests: one overlap gap
    (1000-3000ms), a dominant stream-A event covering it, and two
    non-duplicate recall-pass survivor words inside it. Canonical
    asr/{mid}.words.jsonl is deliberately left absent (empty canon_spans ->
    no recall word can ever dedup-match, per build_media's own comment at
    line 195-199 that dedup is not the same-voice guard's job)."""
    streamA = [
        {
            "event_id": f"{mid}:A0001",
            "media_id": mid,
            "source": "canonical",
            "stream": "A",
            "start_ms": 1000,
            "end_ms": 3000,
            "text": "dominant speaker talking",
            "diar_label": dom_label,
            "speaker_track_id": f"{mid}:{dom_label}",
        }
    ]
    if include_fragment_in_streamA:
        streamA.append(
            {
                "event_id": f"{mid}:AFRAG",
                "media_id": mid,
                "source": "recall",
                "stream": "A",
                "start_ms": 1200,
                "end_ms": 1800,
                "text": "hey there",
                "diar_label": None,
                "speaker_track_id": "unknown_overlap_speaker",
            }
        )
    _write_jsonl(tmp_path / "speech_events_v2_adjudicated" / f"{mid}.jsonl", streamA)

    active = [dom_label] + ([second_label] if second_label else [])
    _write_jsonl(
        tmp_path / "asr_recall_gaps" / f"{mid}.jsonl",
        [{"gap_type": "overlap_gap", "start_ms": 1000, "end_ms": 3000, "diar_speakers": active}],
    )

    _write_jsonl(
        tmp_path / "asr_recall_pass" / f"{mid}.jsonl",
        [
            {"start_ms": 1200, "end_ms": 1400, "text": "hey", "prob": 0.9},
            {"start_ms": 1450, "end_ms": 1650, "text": "there", "prob": 0.9},
        ],
    )
    # canon words file intentionally absent -> _load_jsonl returns [] (missing path)

    centroids = {dom_label: dom_centroid}
    if second_label:
        centroids[second_label] = second_centroid
    _write_json(tmp_path / "asr" / f"{mid}.diar_embeddings.json", centroids)


def test_fold_fires_on_same_voice_match(tmp_path, monkeypatch):
    """Two diar labels active in the gap (SPEAKER_00 dominant, SPEAKER_01
    second) but their centroids are IDENTICAL (cos=1.0 >= SAME_VOICE_THR=0.50)
    -> SPEAKER_01 excluded from `targets` at line 237-239 (over-segmentation,
    not a real 2nd speaker) -> targets=[] -> fold-confirmation path (line
    254-276): fake inf.crop() returns a vector matching the DOMINANT centroid
    exactly (cos=1.0 >= WIN_ABS=0.30) -> fold FIRES. Proves: the recall
    fragment gets patched into stream A in place and
    qc['same_voice_folded'] increments (line 275)."""
    mid = "m_fold_fires"
    monkeypatch.setattr(ps, "CAT", tmp_path)
    monkeypatch.setattr(ps, "_extract_wav", lambda mid_, dst: True)

    _base_fixture(
        tmp_path, mid,
        dom_label="SPEAKER_00", dom_centroid=_vec(0),
        second_label="SPEAKER_01", second_centroid=_vec(0),  # identical -> same voice
    )

    fake_inf = _FakeInference(vector=_vec(0))  # audio matches dominant exactly
    qc = ps.build_media(mid, fake_inf, _FakeSegment, tmp_path)

    assert qc["same_voice_folded"] == 1
    assert qc["rej_no_second_label"] == 1
    assert qc["n_confirmed"] == 0
    assert qc["n_candidate_events"] == 0

    # stream B is empty -- no genuine second-speaker event was created
    par_rows = [json.loads(l) for l in (tmp_path / "parallel_streams" / f"{mid}.jsonl").read_text().splitlines() if l.strip()]
    assert par_rows == []

    # the render layer carries the PATCHED fragment: diar_label/speaker_track_id
    # rewritten to the dominant label, evidence.same_voice_fold recorded
    ren_rows = [json.loads(l) for l in (tmp_path / "render_events_v2" / f"{mid}.jsonl").read_text().splitlines() if l.strip()]
    frag = next(e for e in ren_rows if e["event_id"] == f"{mid}:AFRAG")
    assert frag["diar_label"] == "SPEAKER_00"
    assert frag["speaker_track_id"] == f"{mid}:SPEAKER_00"
    assert frag["evidence"]["same_voice_fold"] == "SPEAKER_00"
    assert frag["confidence"]["identity"] == 0.0  # no identity_map -> pid=None


def test_fold_does_not_fire_on_mismatch(tmp_path, monkeypatch):
    """Same same-voice-excluded setup as test 1 (targets=[]), but this time
    the fake inf.crop() returns a vector ORTHOGONAL to the dominant centroid
    (cos=0.0 < WIN_ABS=0.30) -> the fold-confirmation check at line 263
    fails -> fold does NOT fire. Proves: qc['same_voice_folded'] is never
    incremented (key absent, since it's not in build_media's initial qc dict
    literal), the fragment is left untouched, and rej_no_second_label still
    counts the window (that part of the branch is unconditional)."""
    mid = "m_fold_mismatch"
    monkeypatch.setattr(ps, "CAT", tmp_path)
    monkeypatch.setattr(ps, "_extract_wav", lambda mid_, dst: True)

    _base_fixture(
        tmp_path, mid,
        dom_label="SPEAKER_00", dom_centroid=_vec(0),
        second_label="SPEAKER_01", second_centroid=_vec(0),  # identical -> same voice, targets=[]
    )

    fake_inf = _FakeInference(vector=_vec(1))  # orthogonal to dominant centroid
    qc = ps.build_media(mid, fake_inf, _FakeSegment, tmp_path)

    assert "same_voice_folded" not in qc
    assert qc["rej_no_second_label"] == 1
    assert qc["n_confirmed"] == 0

    par_rows = [json.loads(l) for l in (tmp_path / "parallel_streams" / f"{mid}.jsonl").read_text().splitlines() if l.strip()]
    assert par_rows == []

    ren_rows = [json.loads(l) for l in (tmp_path / "render_events_v2" / f"{mid}.jsonl").read_text().splitlines() if l.strip()]
    frag = next(e for e in ren_rows if e["event_id"] == f"{mid}:AFRAG")
    assert frag["diar_label"] is None  # unchanged
    assert frag["speaker_track_id"] == "unknown_overlap_speaker"  # unchanged
    assert "evidence" not in frag or "same_voice_fold" not in frag.get("evidence", {})


def test_genuine_second_speaker_untouched_by_fold(tmp_path, monkeypatch):
    """SPEAKER_01's centroid is ORTHOGONAL to SPEAKER_00's (cos=0.0 <
    SAME_VOICE_THR=0.50) -> NOT excluded -> targets=['SPEAKER_01'] is
    non-empty -> the entire same-voice-fold branch (`if not targets:`,
    line 240-277) is skipped outright and control goes straight to the
    normal per-group voiceprint gate (line 279-330). The fake inf.crop()
    returns a vector matching SPEAKER_01 exactly (tgt_cos=1.0) and mismatching
    SPEAKER_00 (dom_cos=0.0), clearing both WIN_ABS and WIN_MARGIN -> a real
    stream-B event is confirmed. Proves the fold logic is completely inert on
    the genuine-second-voice path: same_voice_folded absent/0,
    rej_no_second_label==0, n_confirmed==1."""
    mid = "m_genuine_target"
    monkeypatch.setattr(ps, "CAT", tmp_path)
    monkeypatch.setattr(ps, "_extract_wav", lambda mid_, dst: True)

    _base_fixture(
        tmp_path, mid,
        dom_label="SPEAKER_00", dom_centroid=_vec(0),
        second_label="SPEAKER_01", second_centroid=_vec(1),  # orthogonal -> genuine 2nd speaker
        include_fragment_in_streamA=False,
    )

    fake_inf = _FakeInference(vector=_vec(1))  # audio matches the genuine target
    qc = ps.build_media(mid, fake_inf, _FakeSegment, tmp_path)

    assert qc.get("same_voice_folded", 0) == 0
    assert qc["rej_no_second_label"] == 0
    assert qc["n_confirmed"] == 1
    assert qc["n_streamB_words"] == 2
    assert qc["rej_voiceprint"] == 0

    par_rows = [json.loads(l) for l in (tmp_path / "parallel_streams" / f"{mid}.jsonl").read_text().splitlines() if l.strip()]
    assert len(par_rows) == 1
    b_event = par_rows[0]
    assert b_event["diar_label"] == "SPEAKER_01"
    assert b_event["stream"] == "B"
    assert b_event["speaker_track_id"] == f"{mid}:SPEAKER_01"
    assert "same_voice_fold" not in b_event.get("evidence", {})

    ren_rows = [json.loads(l) for l in (tmp_path / "render_events_v2" / f"{mid}.jsonl").read_text().splitlines() if l.strip()]
    labels = {e.get("diar_label") for e in ren_rows}
    assert labels == {"SPEAKER_00", "SPEAKER_01"}
    dom_event = next(e for e in ren_rows if e["event_id"] == f"{mid}:A0001")
    assert dom_event.get("has_parallel_stream") is True  # tagged by the genuine-B path (line 330)

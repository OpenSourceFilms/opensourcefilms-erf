"""Stage A — Ingest + ASR + Diarization.

ffprobe → media_manifest
faster-whisper large-v3 + VAD → words + segments JSONL
pyannote diarization → turns JSONL
Speaker reconciliation by time overlap.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
import uuid
from pathlib import Path

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass  # ffmpeg must be on PATH already

from .atomicio import jsonl_append, write_json, write_tmp_rename
from .ids import media_id as make_media_id, segment_id, word_id
from .journal import emit_start, reap_manifest
from .manifest import StageManifest

_DEFAULTS = {
    "erf_root": "/workspace/erf",
    "catalog": "/workspace/erf/catalog",
    "state": "/workspace/erf/state",
    "asr_model_dir": "/workspace/erf/weights/faster-whisper-large-v3",
    # Diarization defaults
    "diar_min_speakers": None,
    "diar_max_speakers": None,
    "diar_merge_gap_ms": 500,
    "diar_min_turn_ms": 200,
    "diar_fallback_ms": 3000,
    "diar_dedup_threshold": 0.80,
}

# pix_fmt → bit depth heuristic
_PIX_FMT_DEPTH: dict[str, int] = {}
for _fmt in ("yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p",
             "nv12", "nv21", "rgb24", "bgr24"):
    _PIX_FMT_DEPTH[_fmt] = 8
for _fmt in ("yuv420p10le", "yuv420p10be", "yuv422p10le", "yuv422p10be",
             "yuv444p10le", "yuv444p10be", "p010le", "p010be"):
    _PIX_FMT_DEPTH[_fmt] = 10
for _fmt in ("yuv420p12le", "yuv420p12be", "yuv422p12le", "yuv444p12le"):
    _PIX_FMT_DEPTH[_fmt] = 12


def _load_config(config_path: Path) -> dict:
    cfg = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        cfg["erf_root"] = raw.get("paths", {}).get("erf_root", cfg["erf_root"])
        cfg["catalog"] = raw.get("paths", {}).get("catalog", cfg["catalog"])
        cfg["state"] = raw.get("paths", {}).get("state", cfg["state"])
        cfg["asr_model_dir"] = raw.get("models", {}).get("asr_model_dir", cfg["asr_model_dir"])
        diar = raw.get("asr", {}).get("diarization", {})
        if "min_speakers" in diar:
            cfg["diar_min_speakers"] = int(diar["min_speakers"])
        if "max_speakers" in diar:
            cfg["diar_max_speakers"] = int(diar["max_speakers"])
        if "merge_gap_ms" in diar:
            cfg["diar_merge_gap_ms"] = int(diar["merge_gap_ms"])
        if "min_turn_ms" in diar:
            cfg["diar_min_turn_ms"] = int(diar["min_turn_ms"])
        if "fallback_ms" in diar:
            cfg["diar_fallback_ms"] = int(diar["fallback_ms"])
        if "dedup_threshold" in diar:
            cfg["diar_dedup_threshold"] = float(diar["dedup_threshold"])
    return cfg


def _content_blake2b(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _parse_rational(r: str) -> float | None:
    """Parse 'num/den' → float, or return None on failure."""
    try:
        parts = r.split("/")
        if len(parts) == 2:
            n, d = int(parts[0]), int(parts[1])
            return n / d if d else None
    except Exception:
        pass
    return None


def _build_media_manifest(path: Path, probe: dict, relpath: str, mid: str) -> dict:
    streams = probe.get("streams", [])
    fmt = probe.get("format", {})

    duration_s = float(fmt.get("duration", 0))
    duration_ms = int(round(duration_s * 1000))
    container = fmt.get("format_name", "").split(",")[0]

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    video: dict | None = None
    fps: float | None = None
    fps_rational: str | None = None

    if video_stream:
        # fps: prefer r_frame_rate for the exact rational
        r_fr = video_stream.get("r_frame_rate", "")
        a_fr = video_stream.get("avg_frame_rate", "")
        fps_rational = r_fr if "/" in r_fr else None
        fps = _parse_rational(r_fr) or _parse_rational(a_fr) or 0.0

        pix_fmt = video_stream.get("pix_fmt", "")
        bit_depth = _PIX_FMT_DEPTH.get(pix_fmt, 8)
        # fallback: check bits_per_raw_sample
        if "bits_per_raw_sample" in video_stream:
            try:
                bit_depth = int(video_stream["bits_per_raw_sample"]) or bit_depth
            except (ValueError, TypeError):
                pass

        video = {
            "codec": video_stream.get("codec_name", "unknown"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": round(fps, 6) if fps else 0.0,
            "fps_rational": fps_rational,
            "bit_depth": bit_depth,
            "color_primaries": video_stream.get("color_primaries") or None,
            "transfer": video_stream.get("color_transfer") or None,
        }

    audio: dict | None = None
    if audio_stream:
        audio = {
            "codec": audio_stream.get("codec_name", "unknown"),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
        }

    # media_class heuristic for source_note
    source_note_parts = []
    if not video_stream:
        source_note_parts.append("media_class:audio")
    elif fps and fps < 2:
        source_note_parts.append("media_class:image")

    manifest = {
        "schema_version": "1.1.0",
        "media_id": mid,
        "relpath": relpath,
        "filesize_bytes": path.stat().st_size,
        "duration_ms": duration_ms,
        "content_blake2b": _content_blake2b(path),
        "container": container,
        "video": video,
        "audio": audio,
        "ingested_at_utc": StageManifest.now(),
        "source_note": "; ".join(source_note_parts) if source_note_parts else None,
        "kind": "raw_footage",
    }
    return manifest, duration_ms


def _overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _assign_speaker(
    start_ms: int, end_ms: int, turns: list[dict], fallback_ms: int = 0
) -> str | None:
    best_label = None
    best_overlap = 0
    for t in turns:
        ov = _overlap_ms(start_ms, end_ms, t["start_ms"], t["end_ms"])
        if ov > best_overlap:
            best_overlap = ov
            best_label = t["speaker_label"]
    if best_overlap > 0:
        return best_label
    if fallback_ms > 0 and turns:
        # Nearest-turn fallback: assign the speaker of the closest turn endpoint
        mid_ms = (start_ms + end_ms) // 2
        best_dist = fallback_ms + 1
        for t in turns:
            dist = min(abs(mid_ms - t["start_ms"]), abs(mid_ms - t["end_ms"]))
            if dist < best_dist:
                best_dist = dist
                best_label = t["speaker_label"]
        return best_label if best_dist <= fallback_ms else None
    return None


def _dedup_speakers(
    turns: list[dict],
    embeddings: dict[str, list[float]],
    threshold: float = 0.80,
) -> list[dict]:
    """Merge speaker clusters whose centroid embeddings are above cosine similarity threshold.

    Phantom over-segmentation splits one speaker into 2+ clusters with similarity ~0.85-0.99.
    Genuine different speakers sit at ~0.2-0.5. Threshold 0.80 is a safe cut.
    """
    import numpy as _np

    labels = list(embeddings.keys())
    if len(labels) < 2 or not turns:
        return turns

    from collections import Counter
    label_count = Counter(t["speaker_label"] for t in turns)
    merges: dict[str, str] = {}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            ea = _np.array(embeddings[a], dtype=_np.float32)
            eb = _np.array(embeddings[b], dtype=_np.float32)
            na, nb = _np.linalg.norm(ea), _np.linalg.norm(eb)
            if na < 1e-9 or nb < 1e-9:
                continue
            sim = float(_np.dot(ea, eb) / (na * nb))
            if sim >= threshold:
                # merge whichever has fewer turns into the other
                keep, drop = (a, b) if label_count.get(a, 0) >= label_count.get(b, 0) else (b, a)
                merges[drop] = keep

    if not merges:
        return turns

    def resolve(label: str) -> str:
        visited: set[str] = set()
        while label in merges and label not in visited:
            visited.add(label)
            label = merges[label]
        return label

    for t in turns:
        t["speaker_label"] = resolve(t["speaker_label"])
    for i, t in enumerate(turns):
        t["turn_index"] = i
    return turns


def _merge_turns(turns: list[dict], merge_gap_ms: int, min_turn_ms: int) -> list[dict]:
    """Post-process diarization turns:
    1. Absorb turns shorter than min_turn_ms into the nearest same-speaker turn.
    2. Merge consecutive same-speaker turns separated by gap < merge_gap_ms.
    """
    if not turns:
        return turns
    turns = sorted(turns, key=lambda t: t["start_ms"])

    # Pass 1: absorb tiny turns into adjacent same-speaker turn
    result: list[dict] = []
    for t in turns:
        dur = t["end_ms"] - t["start_ms"]
        if dur < min_turn_ms and result:
            prev = result[-1]
            if prev["speaker_label"] == t["speaker_label"]:
                prev["end_ms"] = max(prev["end_ms"], t["end_ms"])
                continue
        result.append(dict(t))

    # Pass 2: merge consecutive same-speaker turns with small gap
    merged: list[dict] = []
    for t in result:
        if (
            merged
            and merged[-1]["speaker_label"] == t["speaker_label"]
            and t["start_ms"] - merged[-1]["end_ms"] <= merge_gap_ms
        ):
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], t["end_ms"])
        else:
            merged.append(dict(t))

    for i, t in enumerate(merged):
        t["turn_index"] = i
    return merged


def _load_hf_token() -> str | None:
    token_path = Path("/root/.cache/huggingface/token")
    if token_path.exists():
        return token_path.read_text().strip() or None
    return None


_GENERIC_PARENTS = {".", "media", "erf", "workspace", "BTS", "video", "videos", "footage", ""}
# Matches collection-level date-range folders like "aug25-sept25", "oct24_jan25"
_DATE_RANGE_FOLDER_RE = re.compile(r'^\w+\d{2}[-_]\w+\d{2}$')


def _derive_source_note(media_path: Path) -> str | None:
    """Auto-derive a human-readable source_note from the file's parent directory and filename date.

    Walks up to 2 non-generic ancestor directories so that nested folders like
    Subject_BTS/Pre-Production/ produce "Subject BTS Pre Production" rather than just "Pre Production".
    Collection date-range folders (aug25-sept25) are treated as generic and stop the walk.
    """
    parts: list[str] = []
    for ancestor in media_path.parents:
        name = ancestor.name
        if not name or name in _GENERIC_PARENTS or _DATE_RANGE_FOLDER_RE.match(name):
            break
        parts.append(name.replace("_", " ").replace("-", " "))
        if len(parts) >= 2:
            break

    if not parts:
        return None

    parts.reverse()  # outermost useful folder first
    label = " ".join(parts)

    # Extract YYYYMMDD from filename — first 8-digit run that looks like a valid date
    date_str: str | None = None
    for m in re.finditer(r"(\d{8})", media_path.name):
        candidate = m.group(1)
        year, month, day = int(candidate[:4]), int(candidate[4:6]), int(candidate[6:])
        if 2000 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            date_str = f"{year}-{month:02d}-{day:02d}"
            break

    return f"{label} — {date_str}" if date_str else label


# ── Process-level model caches ────────────────────────────────────────────────
# Stage A's batch driver calls run() once per media in a single long-lived process.
# The whisper model (~2.9 GB) and the pyannote pipeline (slow to instantiate from
# the MooseFS-mounted weights/cache) are deterministic, so we load each ONCE and
# reuse across all run() calls in the same process. Without this, a 310-file batch
# reloads both models 310× — hours of pure model loading. VRAM is freed when the
# subprocess exits (before the embedding phases load their 8B models).
_WHISPER_MODEL = None
_WHISPER_MODEL_DIR: str | None = None
_DIAR_PIPELINE = None
_DIAR_LOAD_FAILED = False


def _get_whisper_model(asr_model_dir: str):
    global _WHISPER_MODEL, _WHISPER_MODEL_DIR
    if _WHISPER_MODEL is None or _WHISPER_MODEL_DIR != asr_model_dir:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel(asr_model_dir, device="cuda", compute_type="float16")
        _WHISPER_MODEL_DIR = asr_model_dir
    return _WHISPER_MODEL


def _get_diar_pipeline(hf_token):
    """Load the diarization pipeline once per process; None if unavailable."""
    global _DIAR_PIPELINE, _DIAR_LOAD_FAILED
    if _DIAR_PIPELINE is not None or _DIAR_LOAD_FAILED:
        return _DIAR_PIPELINE
    from pyannote.audio import Pipeline as PyannPipeline
    for pipeline_name in ["pyannote/speaker-diarization-community-1",
                          "pyannote/speaker-diarization-3.1"]:
        try:
            _DIAR_PIPELINE = PyannPipeline.from_pretrained(pipeline_name, token=hf_token)
            break
        except Exception:
            continue
    if _DIAR_PIPELINE is None:
        _DIAR_LOAD_FAILED = True
    return _DIAR_PIPELINE


def _emit_a_outcome(media_id: str, status: str, journal_db, run_id: str, *,
                    reason: str | None = None, counts: dict | None = None,
                    has_speech: bool | None = None, n_speakers: int | None = None,
                    has_diar: bool | None = None, lang: str | None = None) -> None:
    """Stage A's explicit outcome + cross-stage flags (Outcome Contract). Never raises."""
    try:
        from . import outcomes
        outcomes.record(media_id, "A", status, reason=reason, counts=counts or {},
                        producer=run_id, db=journal_db)
        if has_speech is not None:
            outcomes.set_flag(media_id, "has_speech", has_speech,
                              reason=(reason if not has_speech else None))
        if n_speakers is not None:
            outcomes.set_flag(media_id, "n_speakers", n_speakers)
        if has_diar is not None:
            outcomes.set_flag(media_id, "has_diar", has_diar)
        if lang:
            outcomes.set_flag(media_id, "dominant_lang", lang)
    except Exception:
        pass


def run(
    media_path: str | Path,
    *,
    config_path: str | Path = "state/config.toml",
    source_note: str | None = None,
) -> StageManifest:
    media_path = Path(media_path).resolve()
    config_path = Path(config_path)
    cfg = _load_config(config_path)

    catalog = Path(cfg["catalog"])
    state = Path(cfg["state"])
    asr_model_dir = cfg["asr_model_dir"]

    run_id = f"A-{uuid.uuid4().hex[:8]}"
    at_utc = StageManifest.now()
    journal_db = state / "journal.db"

    error: str | None = None
    word_count = 0
    seg_count = 0
    turn_count = 0
    speaker_count = 0
    duration_ms = 0
    language: str | None = None
    outputs: list[str] = []
    tool_version: str | None = None
    mid: str | None = None

    try:
        # 1. ffprobe
        probe = _ffprobe(media_path)
        fmt = probe.get("format", {})
        duration_s = float(fmt.get("duration", 0))
        duration_ms = int(round(duration_s * 1000))
        filesize = media_path.stat().st_size
        relpath = media_path.name  # filename only

        mid = make_media_id(relpath, filesize, duration_ms)

        emit_start(journal_db, mid, "A", run_id)

        # 2. Build and write media_manifest
        manifest_record, duration_ms = _build_media_manifest(media_path, probe, relpath, mid)
        if source_note:
            existing = manifest_record.get("source_note") or ""
            manifest_record["source_note"] = (existing + "; " + source_note).strip("; ") if existing else source_note
        # Auto-derive from parent directory + filename date when caller provides no note
        if manifest_record.get("source_note") is None:
            _auto = _derive_source_note(media_path)
            if _auto:
                manifest_record["source_note"] = _auto

        mm_path = catalog / "media_manifest" / f"{mid}.json"
        write_json(mm_path, manifest_record, indent=2)
        outputs.append(str(mm_path))

        # 2b. No-audio guard: skip Whisper entirely (PyAV crashes on streams with no audio)
        streams = probe.get("streams", [])
        _has_audio = any(s.get("codec_type") == "audio" for s in streams)
        if not _has_audio:
            import sqlite3 as _sq3_na
            _conn_na = _sq3_na.connect(str(journal_db))
            try:
                _conn_na.execute(
                    # Phase 1 (D-Log-04): sub-event trace, not A's own disposition
                    # (A's real outcome is recorded via reap_manifest) -> events_trace.
                    "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                    "VALUES (?, ?, ?, 'A', 'skip', ?, ?)",
                    (StageManifest.now(), f"{mid}:A", mid, run_id,
                     json.dumps({"reason": "no_audio_stream"})),
                )
                _conn_na.commit()
            finally:
                _conn_na.close()
            asr_dir = catalog / "asr"
            asr_dir.mkdir(parents=True, exist_ok=True)
            for _p in (asr_dir / f"{mid}.words.jsonl", asr_dir / f"{mid}.segments.jsonl", asr_dir / f"{mid}.diar.jsonl"):
                if not _p.exists():
                    write_tmp_rename(_p, b"\n")
                outputs.append(str(_p))
            manifest = StageManifest(
                media_id=mid, stage="A", run_id=run_id, ok=True, at_utc=at_utc,
                tool_version=None,
                counts={"words": 0, "segments": 0, "turns": 0},
                error=None, outputs=outputs,
            )
            status_dir = catalog / "status"
            reap_manifest(journal_db, manifest, status_dir)
            _emit_a_outcome(mid, "empty", journal_db, run_id, reason="no_audio_stream",
                            counts={"words": 0, "segments": 0, "turns": 0},
                            has_speech=False, n_speakers=0, has_diar=False)
            return manifest

        # 3. ASR with faster-whisper
        import faster_whisper as _fw
        tool_version = _fw.__version__

        model = _get_whisper_model(asr_model_dir)  # cached per process

        segments_iter, info = model.transcribe(
            str(media_path),
            word_timestamps=True,
            language=None,
            vad_filter=True,
            # Stronger decoding to catch tricky words on clean audio, and
            # condition_on_previous_text=False to curb Whisper's word-repetition
            # ("stutter") artifacts.
            beam_size=10,
            best_of=5,
            condition_on_previous_text=False,
        )
        # Materialise the lazy iterator
        segments_list = list(segments_iter)
        language = getattr(info, "language", None)  # for the dominant_lang outcome flag

        # Collect words across all segments
        all_words: list[tuple[int, dict]] = []  # (global_index, word_record)
        seg_records: list[dict] = []

        word_idx = 0
        for seg_idx, seg in enumerate(segments_list):
            seg_id = segment_id(mid, seg_idx)
            seg_start_ms = int(round(seg.start * 1000))
            seg_end_ms = int(round(seg.end * 1000))

            seg_rec = {
                "schema_version": "1.0.0",
                "segment_id": seg_id,
                "media_id": mid,
                "index": seg_idx,
                "text": seg.text.strip(),
                "start_ms": seg_start_ms,
                "end_ms": seg_end_ms,
                "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
                "no_speech_prob": float(seg.no_speech_prob) if seg.no_speech_prob is not None else None,
                "speaker_label": None,
                "speaker_person_id": None,
            }
            seg_records.append(seg_rec)

            if seg.words:
                for w in seg.words:
                    w_rec = {
                        "schema_version": "1.0.0",
                        "word_id": word_id(mid, word_idx),
                        "media_id": mid,
                        "segment_id": seg_id,
                        "index": word_idx,
                        "text": w.word,
                        "start_ms": int(round(w.start * 1000)),
                        "end_ms": int(round(w.end * 1000)),
                        "prob": float(w.probability) if w.probability is not None else 0.0,
                        "speaker_label": None,
                        "speaker_person_id": None,
                    }
                    all_words.append(w_rec)
                    word_idx += 1

        # No-speech: A ran ASR and found no transcribable speech. This is an explicit
        # 'empty' outcome with has_speech=False (Outcome Contract) — downstream reads
        # the flag, never infers from an empty file.
        if not all_words:
            # Write 1-byte JSONL files — zero-byte files are prohibited (MASTER Invariant 2);
            # read_jsonl strips blank lines and returns [] for these, which is correct.
            asr_dir = catalog / "asr"
            asr_dir.mkdir(parents=True, exist_ok=True)
            _words_path = asr_dir / f"{mid}.words.jsonl"
            _segs_path = asr_dir / f"{mid}.segments.jsonl"
            _diar_path = asr_dir / f"{mid}.diar.jsonl"
            for _p in (_words_path, _segs_path, _diar_path):
                if not _p.exists():
                    write_tmp_rename(_p, b"\n")
            outputs.extend([str(_words_path), str(_segs_path), str(_diar_path)])

            manifest = StageManifest(
                media_id=mid, stage="A", run_id=run_id, ok=True, at_utc=at_utc,
                tool_version=tool_version,
                counts={"words": 0, "segments": 0, "turns": 0},
                error=None, outputs=outputs,
            )
            status_dir = catalog / "status"
            reap_manifest(journal_db, manifest, status_dir)
            _emit_a_outcome(mid, "empty", journal_db, run_id, reason="no_speech",
                            counts={"words": 0, "segments": 0, "turns": 0},
                            has_speech=False, n_speakers=0, has_diar=False)  # no lang: silent audio
            return manifest

        # 4. pyannote diarization
        hf_token = _load_hf_token()
        turns: list[dict] = []
        diar_min_speakers = cfg.get("diar_min_speakers")
        diar_max_speakers = cfg.get("diar_max_speakers")
        diar_merge_gap_ms = cfg.get("diar_merge_gap_ms", 500)
        diar_min_turn_ms = cfg.get("diar_min_turn_ms", 200)
        diar_fallback_ms = cfg.get("diar_fallback_ms", 3000)
        diar_dedup_threshold = cfg.get("diar_dedup_threshold", 0.80)
        speaker_embeddings: dict[str, list[float]] = {}
        try:
            pipeline = _get_diar_pipeline(hf_token)  # cached per process

            if pipeline is not None:
                import torch as _torch
                import tempfile as _tmp, subprocess as _sp, shutil as _sh
                import numpy as _np
                import scipy.io.wavfile as _wavfile
                import static_ffmpeg as _sff
                _sff.add_paths()
                _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
                pipeline.to(_device)

                # pyannote 4.0.4 + torchcodec: passing a file path crashes because
                # torchcodec needs libavutil which is not present on this pod.
                # Workaround: extract audio via static_ffmpeg → scipy → waveform dict.
                _ffmpeg = _sh.which("ffmpeg")
                with _tmp.NamedTemporaryFile(suffix=".wav", delete=False) as _tf:
                    _wav_path = _tf.name
                _sp.run(
                    [_ffmpeg, "-y", "-i", str(media_path), "-ar", "16000", "-ac", "1", _wav_path],
                    capture_output=True, check=True,
                )
                _sr, _wav_data = _wavfile.read(_wav_path)
                _waveform = _torch.from_numpy(_wav_data.astype(_np.float32) / 32768.0).unsqueeze(0)

                diar_kwargs: dict = {}
                if diar_min_speakers is not None:
                    diar_kwargs["min_speakers"] = diar_min_speakers
                if diar_max_speakers is not None:
                    diar_kwargs["max_speakers"] = diar_max_speakers

                diarization = pipeline(
                    {"waveform": _waveform, "sample_rate": _sr},
                    **diar_kwargs,
                )

                # pyannote 4.0.4 may return DiarizeOutput instead of Annotation
                _annotation = getattr(diarization, "speaker_diarization", diarization)
                # Extract per-speaker centroid embeddings (DiarizeOutput only)
                _emb_raw = getattr(diarization, "speaker_embeddings", None)
                if _emb_raw is not None and hasattr(_annotation, "labels"):
                    for _i, _label in enumerate(_annotation.labels()):
                        if _i < len(_emb_raw):
                            speaker_embeddings[_label] = _emb_raw[_i].tolist()
                raw_turns = []
                for t_idx, (turn, _, speaker) in enumerate(_annotation.itertracks(yield_label=True)):
                    raw_turns.append({
                        "turn_index": t_idx,
                        "speaker_label": speaker,
                        "start_ms": int(round(turn.start * 1000)),
                        "end_ms": int(round(turn.end * 1000)),
                    })
                # Dedup phantom speakers before merge
                if speaker_embeddings:
                    raw_turns = _dedup_speakers(raw_turns, speaker_embeddings, diar_dedup_threshold)
                turns = _merge_turns(raw_turns, diar_merge_gap_ms, diar_min_turn_ms)
        except Exception as diar_exc:
            # Diarization failure is non-fatal — emit skip event then proceed without speaker labels
            import sqlite3 as _sq3
            _conn = _sq3.connect(str(journal_db))
            try:
                _conn.execute(
                    # Phase 1 (D-Log-04): sub-event trace, not A's own disposition
                    # (A's real outcome is recorded via reap_manifest) -> events_trace.
                    "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
                    "VALUES (?, ?, ?, 'A', 'skip', ?, ?)",
                    (
                        StageManifest.now(), f"{mid}:A:diar", mid, run_id,
                        json.dumps({"reason": "diarization_unavailable", "error": str(diar_exc)[:300]}),
                    ),
                )
                _conn.commit()
            finally:
                _conn.close()

        turn_count = len(turns)
        speaker_count = len({t["speaker_label"] for t in turns})

        # 5. Speaker reconciliation
        if turns:
            for w_rec in all_words:
                w_rec["speaker_label"] = _assign_speaker(
                    w_rec["start_ms"], w_rec["end_ms"], turns, diar_fallback_ms
                )
            for s_rec in seg_records:
                s_rec["speaker_label"] = _assign_speaker(
                    s_rec["start_ms"], s_rec["end_ms"], turns, diar_fallback_ms
                )

        # 6. Write outputs
        asr_dir = catalog / "asr"
        asr_dir.mkdir(parents=True, exist_ok=True)

        words_path = asr_dir / f"{mid}.words.jsonl"
        segs_path = asr_dir / f"{mid}.segments.jsonl"
        diar_path = asr_dir / f"{mid}.diar.jsonl"

        # Idempotent: remove existing files before re-writing
        for p in (words_path, segs_path, diar_path):
            if p.exists():
                p.unlink()

        for w_rec in all_words:
            jsonl_append(words_path, w_rec)
        word_count = len(all_words)

        for s_rec in seg_records:
            jsonl_append(segs_path, s_rec)
        seg_count = len(seg_records)

        for t in turns:
            jsonl_append(diar_path, t)

        emb_path = asr_dir / f"{mid}.diar_embeddings.json"
        if speaker_embeddings:
            write_tmp_rename(emb_path, json.dumps(speaker_embeddings, separators=(",", ":")).encode())
        elif emb_path.exists():
            emb_path.unlink()

        outputs.extend([str(words_path), str(segs_path), str(diar_path)])

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if mid is None:
            mid = f"m_unknown_{run_id}"
        # Ensure emit_start was called
        try:
            emit_start(journal_db, mid, "A", run_id)
        except Exception:
            pass

    ok = error is None
    manifest = StageManifest(
        media_id=mid or f"m_unknown_{run_id}",
        stage="A",
        run_id=run_id,
        ok=ok,
        at_utc=at_utc,
        tool_version=tool_version,
        counts={"words": word_count, "segments": seg_count, "turns": turn_count, "speakers": speaker_count},
        error=error,
        outputs=outputs,
    )
    status_dir = catalog / "status"
    reap_manifest(journal_db, manifest, status_dir)
    _final_mid = mid or f"m_unknown_{run_id}"
    if ok:
        _emit_a_outcome(_final_mid, "ok", journal_db, run_id,
                        counts={"words": word_count, "segments": seg_count,
                                "turns": turn_count, "speakers": speaker_count,
                                "duration_ms": duration_ms},
                        has_speech=word_count > 0, n_speakers=speaker_count,
                        has_diar=turn_count > 0, lang=language)
    else:
        _emit_a_outcome(_final_mid, "error", journal_db, run_id, reason=error or "unknown")
    return manifest

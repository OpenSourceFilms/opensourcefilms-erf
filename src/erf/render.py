"""Render — beat_plan to MP4 (Strategy A: face-centroid reframe).

DEC-7: render is a downstream assembly operation, not a media analysis stage.
Journalling is the caller's responsibility.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

_log = logging.getLogger("erf.render")

_VERTICAL_PLATFORMS = {"tiktok", "reels", "youtube_shorts"}
_CROP_W = 304          # 540 * 9/16 rounded
_CROP_H = 540
_SRC_W = 960
_SRC_H = 540
_MAX_CROP_X = _SRC_W - _CROP_W   # 656
_CENTRE_CROP_X = (_SRC_W - _CROP_W) // 2  # 328


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def _add_ffmpeg() -> None:
    import static_ffmpeg
    static_ffmpeg.add_paths()


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    _log.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _log.error("ffmpeg stderr:\n%s", result.stderr[-2000:])
    return result


def _probe_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _probe_proxy_dims(proxy: Path) -> tuple[int, int]:
    """Return (width, height) of proxy video. Falls back to (960, 540) on error."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(proxy),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        w_s, h_s = out.split("x")
        return int(w_s), int(h_s)
    except Exception:
        return 960, 540


# ── Face-centroid reframe ──────────────────────────────────────────────────────

def _load_detections(catalog: Path, media_id: str) -> list[dict]:
    p = catalog / "detections" / f"{media_id}.face.jsonl"
    if not p.exists():
        return []
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _face_crop_x(
    detections: list[dict],
    in_ms: int,
    out_ms: int,
    src_w: int = _SRC_W,
    crop_w: int = _CROP_W,
) -> int:
    """EMA (α=0.15) + ±8% dead-zone crop-centre from sparse detections."""
    centre = (src_w - crop_w) // 2
    window = sorted(
        (d for d in detections if in_ms <= d["time_ms"] < out_ms),
        key=lambda d: d["time_ms"],
    )
    if not window:
        return centre

    alpha = 0.15
    b = window[0]["bbox"]
    smooth = (b[0] + b[2]) / 2.0 / src_w  # normalised [0,1]
    crop_centre = smooth

    for det in window[1:]:
        b = det["bbox"]
        cx = (b[0] + b[2]) / 2.0 / src_w
        smooth = alpha * cx + (1 - alpha) * smooth
        if abs(smooth - crop_centre) > 0.08:
            crop_centre = smooth

    face_px = int(crop_centre * src_w)
    crop_x = max(0, min(src_w - crop_w, face_px - crop_w // 2))
    # Clamp: face centroid must stay within centre 60% of crop_w (20% margins each side)
    margin = crop_w // 5
    face_in_crop = face_px - crop_x
    if face_in_crop < margin:
        crop_x = max(0, face_px - margin)
    elif face_in_crop > crop_w - margin:
        crop_x = min(src_w - crop_w, face_px - (crop_w - margin))
    return crop_x


# ── Active-speaker reframe ─────────────────────────────────────────────────────

def _load_active_speaker(catalog: Path, media_id: str) -> Optional[dict]:
    p = catalog / "active_speaker" / f"{media_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _asd_crop_x(
    asd_data: dict,
    in_ms: int,
    out_ms: int,
    threshold: float,
    src_w: int = _SRC_W,
    crop_w: int = _CROP_W,
) -> Optional[int]:
    """EMA (α=0.15) + ±8% dead-zone over ASD segment centroids. Returns None if no qualifying data."""
    centroids: list[tuple[int, float]] = []
    for seg in asd_data.get("segments", []):
        if seg["speaking_confidence"] < threshold:
            continue
        if seg["end_ms"] <= in_ms or seg["start_ms"] >= out_ms:
            continue
        for c in seg.get("centroids", []):
            if in_ms <= c["t_ms"] < out_ms:
                centroids.append((c["t_ms"], c["cx"]))
    if not centroids:
        return None
    centroids.sort(key=lambda x: x[0])

    alpha = 0.15
    smooth = centroids[0][1]  # already normalised [0,1]
    crop_centre = smooth
    for _, cx in centroids[1:]:
        smooth = alpha * cx + (1 - alpha) * smooth
        if abs(smooth - crop_centre) > 0.08:
            crop_centre = smooth

    face_px = int(crop_centre * src_w)
    crop_x = max(0, min(src_w - crop_w, face_px - crop_w // 2))
    # Clamp: face centroid must stay within centre 60% of crop_w (20% margins each side)
    margin = crop_w // 5
    face_in_crop = face_px - crop_x
    if face_in_crop < margin:
        crop_x = max(0, face_px - margin)
    elif face_in_crop > crop_w - margin:
        crop_x = min(src_w - crop_w, face_px - (crop_w - margin))
    return crop_x


# ── ASS caption generation ─────────────────────────────────────────────────────

def _ms_to_ass(ms: int) -> str:
    """Convert milliseconds to ASS timestamp H:MM:SS.cc"""
    cs = ms // 10           # centiseconds
    s = cs // 100
    cs_rem = cs % 100
    m, s_rem = divmod(s, 60)
    h, m_rem = divmod(m, 60)
    return f"{h}:{m_rem:02d}:{s_rem:02d}.{cs_rem:02d}"


def _load_words(catalog: Path, media_id: str) -> list[dict]:
    # Prefer the unified transcript (T0): WhisperX-accurate timing + speaker_label
    # + person_id, so captions are speaker-coloured. Fall back to raw wx/words.
    for suffix in (".transcript.jsonl", ".words.wx.jsonl", ".words.jsonl"):
        p = catalog / "asr" / f"{media_id}{suffix}"
        if not p.exists():
            continue
        words = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    words.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if words:
            return words
    return []


_vad_cache: dict[str, list[dict]] = {}


def _load_vad(catalog: Path, media_id: str) -> list[dict]:
    if media_id not in _vad_cache:
        p = catalog / "asr" / f"{media_id}.vad.jsonl"
        if not p.exists():
            _vad_cache[media_id] = []
        else:
            segs = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        segs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            _vad_cache[media_id] = segs
    return _vad_cache[media_id]


def _vad_snap(vad_segs: list[dict], t_ms: int, snap_ms: int = 80) -> int:
    """Snap t_ms to the nearest VAD silence/speech boundary within snap_ms.

    VAD boundaries are guaranteed to be actual audio energy transitions, so
    snapping to them avoids cutting mid-breath or mid-consonant.
    """
    if not vad_segs:
        return t_ms
    best_t, best_dist = t_ms, snap_ms + 1
    for seg in vad_segs:
        for b in (seg["start_ms"], seg["end_ms"]):
            d = abs(b - t_ms)
            if d < best_dist:
                best_dist, best_t = d, b
    return best_t


_ASS_ALIGNMENT = {"centre": 5, "lower_third": 2, "top": 8}
_ASS_MARGIN_V = {"centre": 0, "lower_third": 400, "top": 200}

# ASS inline color overrides keyed on person_id (BGR hex, no alpha prefix).
# speaker_label→color is resolved per-clip via identity_overlay so a given
# person is always the same color regardless of which SPEAKER_NN label they
# happen to carry in a given file. Empty by default — populate per-project
# with your own enrolled person_ids, e.g. {"p_alice": "00FFFF"}.
_PERSON_COLORS: dict[str, str] = {}


# Fallback palette (BGR hex) for speakers without a known person colour. Distinct
# from the _PERSON_COLORS above so identified vs unidentified speakers don't clash.
_FALLBACK_SPK_COLORS: list[str] = [
    "FF00FF",  # magenta
    "0000FF",  # red
    "E6D8AD",  # light blue
    "FFC0CB",  # pink
    "800080",  # purple
    "00D7FF",  # gold
]


def _load_speaker_colors(catalog: Path, media_id: str) -> dict[str, str]:
    """Return {speaker_label: BGR_hex} so every speaker gets a distinct caption colour.

    Resolution order per speaker:
    1. Known person (identity_overlay → person_id in _PERSON_COLORS): fixed colour,
       consistent for that person across every input video.
    2. Resolved but unknown person: a stable fallback colour keyed by person_id (still
       consistent across videos for that person).
    3. Unidentified speaker: a fallback colour by local speaker index, so different
       speakers within one edit are still visually distinguished.
    """
    # speaker_label -> person_id from identity_overlay (may be absent)
    spk_to_person: dict[str, str] = {}
    p = catalog / "identity_overlay" / f"{media_id}.json"
    if p.exists():
        try:
            overlay = json.loads(p.read_text())
            for a in overlay.get("speaker_assignments", []):
                lbl, pid = a.get("speaker_label"), a.get("person_id")
                if lbl:
                    spk_to_person[lbl] = pid or ""
        except Exception:
            pass

    # All speaker labels present in this media (diarization is the source of truth;
    # fall back to whatever the overlay listed). Also keep turn-level timing for
    # the smoothing pass below.
    labels: list[str] = []
    lbl_turns: dict[str, list[tuple[int, int]]] = {}  # label → [(start_ms, end_ms), ...]
    diar_p = catalog / "asr" / f"{media_id}.diar.jsonl"
    if diar_p.exists():
        seen: set[str] = set()
        try:
            for line in diar_p.read_text().splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                lbl = t.get("speaker_label")
                if lbl:
                    if lbl not in seen:
                        seen.add(lbl)
                        labels.append(lbl)
                    lbl_turns.setdefault(lbl, []).append(
                        (t.get("start_ms", 0), t.get("end_ms", 0))
                    )
        except Exception:
            pass
    for lbl in spk_to_person:
        if lbl not in labels:
            labels.append(lbl)

    result: dict[str, str] = {}
    used: set[str] = set()
    fallback_i = 0

    def _next_fallback() -> str:
        # next fallback colour not already taken by another label this edit
        nonlocal fallback_i
        for _ in range(len(_FALLBACK_SPK_COLORS)):
            c = _FALLBACK_SPK_COLORS[fallback_i % len(_FALLBACK_SPK_COLORS)]
            fallback_i += 1
            if c not in used:
                return c
        return _FALLBACK_SPK_COLORS[fallback_i % len(_FALLBACK_SPK_COLORS)]

    # Distinct colour per speaker_label (D-Identity-01), refined (D-Identity-02): two diarized
    # labels must never share a colour when they are DIFFERENT people or unknown — otherwise an
    # off-camera *different* speaker is visually indistinguishable from the on-camera one. BUT
    # two labels that resolve to the SAME confident person (diarization over-segmentation) MUST
    # share that person's one colour — they are literally the same speaker, so forcing them
    # apart paints one person in two hues (e.g. a speaker flickering yellow→magenta). Preferred
    # colour (known person → person colour; identified → stable per-person hue) is kept only if
    # still free; otherwise fall through to the next unused fallback.
    pid_color: dict[str, str] = {}  # person_id → colour, so an over-segmented person shares one hue
    for lbl in sorted(labels):
        pid = spk_to_person.get(lbl, "")
        if pid and pid in pid_color:  # same confident person already coloured → share it
            result[lbl] = pid_color[pid]
            continue
        pref = ""
        if pid and pid in _PERSON_COLORS:
            pref = _PERSON_COLORS[pid]
        elif pid:
            pref = _FALLBACK_SPK_COLORS[abs(hash(pid)) % len(_FALLBACK_SPK_COLORS)]
        elif lbl:
            # Unresolved speaker: stable hash keyed on (media_id, speaker_label) so the
            # same unknown speaker always gets the same colour across all renders that
            # include this clip — regardless of what other labels exist in the story.
            pref = _FALLBACK_SPK_COLORS[abs(hash(f"{media_id}:{lbl}")) % len(_FALLBACK_SPK_COLORS)]
        c = pref if (pref and pref not in used) else _next_fallback()
        result[lbl] = c
        used.add(c)
        if pid:
            pid_color[pid] = c

    # Smoothing pass: unresolved labels that are temporally nested inside a resolved
    # label's turns (>60% overlap) inherit that person's colour. This handles diarization
    # fragments that slip through the identity overlay — a brief SPEAKER_02 segment that
    # is actually the same person as the dominant SPEAKER_00 gets the same hue instead of
    # flickering to a fallback colour. Only applies when there is at least one resolved label.
    resolved_lbls = [l for l in labels if spk_to_person.get(l)]
    if resolved_lbls and lbl_turns:
        for lbl in labels:
            if spk_to_person.get(lbl):
                continue  # already resolved — skip
            my_turns = lbl_turns.get(lbl, [])
            if not my_turns:
                continue
            my_ms = sum(max(0, e - s) for s, e in my_turns)
            if my_ms <= 0:
                continue
            best_overlap, best_lbl = 0, ""
            for rl in resolved_lbls:
                ov = 0
                for rs, re in lbl_turns.get(rl, []):
                    for ms, me in my_turns:
                        ov += max(0, min(me, re) - max(ms, rs))
                frac = ov / my_ms
                if frac > best_overlap:
                    best_overlap, best_lbl = frac, rl
            if best_overlap >= 0.60 and best_lbl:
                result[lbl] = result[best_lbl]

    return result


# ffmpeg -vf correction for each non-upright V0-detected orientation. b-roll
# is the target: fix_orientation.py's face-detection auto-fixer has no faces
# to key off in generic cutaway footage, so mis-rotated b-roll was previously
# a silent defect at render time (W3(e), D-Audit-01 finding #4/D-Log-24 follow-on).
_ORIENTATION_FIX_VF = {
    "rotated_90_cw": "transpose=2",
    "rotated_90_ccw": "transpose=1",
    "upside_down": "transpose=1,transpose=1",
}


def _load_broll_orientation(catalog: Path, media_id: str) -> str:
    """V0 visual-grounding orientation verdict, defaulting to 'upright' when
    absent (older/unbackfilled records, or audio-only media)."""
    p = catalog / "visual_grounding" / f"{media_id}.json"
    if not p.exists():
        return "upright"
    try:
        orientation = json.loads(p.read_text()).get("orientation")
    except Exception:
        return "upright"
    return orientation if orientation in _ORIENTATION_FIX_VF else "upright"


_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,{alignment},{ml},{mr},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _build_ass(
    words_per_beat: list[list[dict]],
    clip_in_ms_per_beat: list[int],
    beat_offsets_ms: list[int],
    caption_style: str,
    speaker_colors: list[dict[str, str]] | None = None,
    cues_cfg: dict | None = None,
) -> str:
    """Build a full ASS document.

    Default 'phrase' mode groups words into short broadcast-style cues — a readable clause
    shown as a unit and held for its duration (D-Caption-01). This is the classy/documentary
    norm and, unlike the legacy one-word-at-a-time 'pop', a returning speaker's first words
    after tight crosstalk no longer flash for a few frames and read as missing — they live as
    part of a cue held >= min_cue_ms. 'word' mode (cues_cfg.mode='word') keeps the old per-word
    behaviour. Casing/punctuation come from the transcript (already sentence-cased)."""
    cues_cfg = cues_cfg or {}
    style = caption_style if caption_style in _ASS_ALIGNMENT else "centre"
    alignment = _ASS_ALIGNMENT[style]
    mv = _ASS_MARGIN_V[style]
    header = _ASS_HEADER.format(alignment=alignment, ml=10, mr=10, mv=mv)
    lines = [header]

    def _esc(t: str) -> str:
        return t.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    def _norm(t: str) -> str:
        return _esc(re.sub(r"\bi\b", "I", t))  # classy: standalone 'i' -> 'I' (ASR casing slip)

    def _capfirst(t: str) -> str:
        # Capitalise the first alphabetic char (sentence-start). ASR word streams are
        # often all-lowercase; the renderer guarantees a capital at each cue/sentence start.
        for i, ch in enumerate(t):
            if ch.isalpha():
                return t[:i] + ch.upper() + t[i + 1:]
            if ch.isdigit():
                break
        return t

    def _dim_color(c: str, f: float = 0.45) -> str:
        """Darken a BBGGRR hex colour (for unsung karaoke text)."""
        try:
            b, g, r = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return "%02X%02X%02X" % (int(b * f), int(g * f), int(r * f))
        except Exception:
            return c

    def _emit(start_ms: int, end_ms: int, text: str, color: str, fade_ms: int = 0, cap: bool = False) -> None:
        if cap:
            text = _capfirst(text)
        text = _norm(text)
        tags = ""
        if fade_ms > 0:
            tags += "{\\fad(%d,0)}" % fade_ms
        if color:
            tags += "{\\c&H" + color + "&}"
        lines.append(
            f"Dialogue: 0,{_ms_to_ass(start_ms)},{_ms_to_ass(end_ms)},Default,,0,0,0,,{tags}{text}"
        )

    def _emit_karaoke(start_ms: int, end_ms: int, words: list, color: str, fade_ms: int, cap: bool = False) -> None:
        """Emit one cue as a held line whose words light up (speaker colour) as spoken —
        broadcast karaoke: 'sung' words bright, upcoming words dimmed but readable."""
        bright = color or "FFFFFF"
        dim = _dim_color(bright)
        parts = []
        if fade_ms > 0:
            parts.append("{\\fad(%d,0)}" % fade_ms)
        parts.append("{\\1c&H%s&\\2c&H%s&}" % (bright, dim))
        for j, (ws, we, wt) in enumerate(words):
            nxt = words[j + 1][0] if j + 1 < len(words) else None
            dur = (nxt - ws) if nxt is not None else (we - ws)
            wt = _capfirst(wt) if (cap and j == 0) else wt
            parts.append("{\\kf%d}%s " % (max(1, round(dur / 10)), _norm(wt)))
        lines.append(
            f"Dialogue: 0,{_ms_to_ass(start_ms)},{_ms_to_ass(end_ms)},Default,,0,0,0,,{''.join(parts).rstrip()}"
        )

    if cues_cfg.get("mode", "phrase") == "word":
        for beat_idx, (words, clip_in_ms, beat_offset_ms) in enumerate(
            zip(words_per_beat, clip_in_ms_per_beat, beat_offsets_ms)
        ):
            _colors = (speaker_colors[beat_idx] if speaker_colors else {})
            for w in words:
                start_ms = beat_offset_ms + (w["start_ms"] - clip_in_ms)
                end_ms = beat_offset_ms + (w["end_ms"] - clip_in_ms)
                if end_ms <= start_ms or start_ms < 0:
                    continue
                text = (w.get("text") or "").strip()
                if not text:
                    continue
                _emit(start_ms, end_ms, text, _colors.get(w.get("speaker_label", ""), ""))
        return "\n".join(lines)

    # ── phrase mode ──────────────────────────────────────────────────────────
    max_words = int(cues_cfg.get("max_words", 5))
    max_chars = int(cues_cfg.get("max_chars", 32))
    gap_break = int(cues_cfg.get("gap_break_ms", 700))
    min_cue = int(cues_cfg.get("min_cue_ms", 600))
    hold_cap = int(cues_cfg.get("hold_cap_ms", 1200))
    highlight = cues_cfg.get("highlight", "karaoke")  # 'karaoke' = active-word fill | 'none'
    fade_ms = int(cues_cfg.get("fade_ms", 90))
    # Interjection quick-cut: a short turn flanked by a *different* speaker (a behind-camera
    # "yeah", a one-word retort, a rapid back-and-forth) snaps to ~its spoken length and
    # cuts straight to the next line — deliberately overriding the min_cue floor + hold
    # extension so exchanges stay snappy. Trades the D-Caption-01 readability hold for
    # responsiveness, per explicit request.
    interj_words = int(cues_cfg.get("interjection_max_words", 3))
    interj_max_ms = int(cues_cfg.get("interjection_max_ms", 1500))
    interj_min_ms = int(cues_cfg.get("interjection_min_ms", 250))
    _SENT_END = (".", "?", "!", "…", ".\"", "?\"", "!\"")

    def _even_chunks(gwords: list, mw: int, mc: int) -> list:
        """Split a hard-bounded run of words into EVEN chunks (<= mw words each), so no
        chunk is a lone word (a 6-word run -> 3+3, not 5+1). Bumps the chunk count if a
        chunk would exceed mc chars. A genuinely isolated 1-word run stays a 1-word cue."""
        n = len(gwords)
        if n <= 1:
            return [gwords]

        def _split(k: int) -> list:
            base, rem = divmod(n, k)
            out, idx = [], 0
            for i in range(k):
                sz = base + (1 if i < rem else 0)
                out.append(gwords[idx:idx + sz])
                idx += sz
            return out

        k = max(1, -(-n // mw))  # ceil(n / mw)
        chunks = _split(k)
        while k < n and not all(
            sum(len(w[2]) for w in c) + (len(c) - 1) <= mc for c in chunks
        ):
            k += 1
            chunks = _split(k)
        return chunks

    cues: list[dict] = []
    for beat_idx, (words, clip_in_ms, beat_offset_ms) in enumerate(
        zip(words_per_beat, clip_in_ms_per_beat, beat_offsets_ms)
    ):
        _colors = (speaker_colors[beat_idx] if speaker_colors else {})
        # 1. hard-bounded runs: break only on speaker change, a real pause, or AFTER a
        #    sentence-ending word — so a cue never ends mid-thought with one dangling word
        #    and never mixes the end of one sentence with the start of the next.
        runs: list[dict] = []
        cur: dict | None = None
        for w in words:
            start_ms = beat_offset_ms + (w["start_ms"] - clip_in_ms)
            end_ms = beat_offset_ms + (w["end_ms"] - clip_in_ms)
            if end_ms <= start_ms or start_ms < 0:
                continue
            text = (w.get("text") or "").strip()
            if not text:
                continue
            spk = w.get("speaker_label", "")
            si = w.get("sentence_index")
            brk = cur is not None and (
                spk != cur["spk"]
                or start_ms - cur["words"][-1][1] > gap_break
                or cur["words"][-1][2].endswith(_SENT_END)
                or (si is not None and cur["last_si"] is not None and si != cur["last_si"])
            )
            if cur is None or brk:
                if cur is not None:
                    runs.append(cur)
                cur = {"spk": spk, "color": _colors.get(spk, ""),
                       "words": [(start_ms, end_ms, text)], "last_si": si}
            else:
                cur["words"].append((start_ms, end_ms, text))
                cur["last_si"] = si
        if cur is not None:
            runs.append(cur)
        # 2. tag interjections: a short turn flanked by a different speaker. These quick-cut
        #    (see timing loop) even though it breaks the words-per-line / hold-on-speaker rules.
        for ri, run in enumerate(runs):
            prev_spk = runs[ri - 1]["spk"] if ri > 0 else None
            next_spk = runs[ri + 1]["spk"] if ri + 1 < len(runs) else None
            flanked = (prev_spk is not None and prev_spk != run["spk"]) or \
                      (next_spk is not None and next_spk != run["spk"])
            span_ms = run["words"][-1][1] - run["words"][0][0]
            run["interj"] = (
                len(run["words"]) <= interj_words and span_ms <= interj_max_ms and flanked
            )
        # 3. split each run into even chunks -> cues (interjections stay whole: already short).
        #    The first cue of a run is a sentence/clause start -> capitalise it.
        for run in runs:
            first = True
            for chunk in _even_chunks(run["words"], max_words, max_chars):
                if not chunk:
                    continue
                cues.append({
                    "start": chunk[0][0], "end": chunk[-1][1], "color": run["color"],
                    "words": chunk, "text": " ".join(w[2] for w in chunk),
                    "quick": run["interj"], "cap": first,
                })
                first = False

    cues.sort(key=lambda c: c["start"])
    for i, c in enumerate(cues):
        s, ne = c["start"], c["end"]
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else None
        if c.get("quick"):
            e = max(ne, s + interj_min_ms)            # snap to spoken length (tiny floor only)
        else:
            e = max(ne, s + min_cue)
            if nxt is not None:
                e = max(e, min(nxt, ne + hold_cap))   # hold until next cue, capped past natural end
        if nxt is not None:
            e = min(e, nxt)                           # never overlap the next cue
        e = max(e, s + 1)
        if highlight == "karaoke":
            _emit_karaoke(s, e, c["words"], c["color"], fade_ms, cap=c.get("cap", False))
        else:
            _emit(s, e, c["text"], c["color"], fade_ms, cap=c.get("cap", False))
    return "\n".join(lines)


# ── Beat-sync ─────────────────────────────────────────────────────────────────

def _beat_sync_clips(
    clips: list[dict],
    music_path: Path,
    tolerance_ms: int = 300,
) -> list[dict]:
    """Snap each clip's out_ms so the cumulative cut falls on a musical beat.

    Processes cuts independently (no cascading). Skips snaps where no beat falls
    within tolerance_ms. Clamps out_ms to at least in_ms + 100 ms.
    """
    try:
        import librosa
        y, sr = librosa.load(str(music_path), mono=True)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times_ms = [
            int(round(t * 1000))
            for t in librosa.frames_to_time(beat_frames, sr=sr)
        ]
        if not beat_times_ms:
            return clips

        _log.debug("beat_sync: %d beats from music, tolerance=%dms", len(beat_times_ms), tolerance_ms)

        clips = [dict(c) for c in clips]
        cumulative_ms = 0
        for i in range(len(clips) - 1):  # last clip's end needs no snap
            cumulative_ms += clips[i]["out_ms"] - clips[i]["in_ms"]
            nearest = min(beat_times_ms, key=lambda t: abs(t - cumulative_ms))
            delta = nearest - cumulative_ms
            if abs(delta) <= tolerance_ms:
                clips[i]["out_ms"] += delta
                clips[i]["out_ms"] = max(clips[i]["in_ms"] + 100, clips[i]["out_ms"])

        return clips

    except Exception as e:
        _log.warning("beat_sync_clips failed: %s", e)
        return clips


# ── Audio / trim helpers ───────────────────────────────────────────────────────

def _load_render_cfg(config_path: Optional[Path]) -> dict:
    """Reads [render] section from config.toml. Returns empty dict on error."""
    if config_path is None or not config_path.is_file():
        return {}
    try:
        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
        return data.get("render", {})
    except Exception:
        _log.warning("Could not read render config from %s", config_path)
        return {}


def _df3_denoise(wav_in: Path, wav_out: Path, audio_python: Path, helper: Path) -> bool:
    """Run df3_helper.py subprocess. Returns True on success, never raises."""
    cmd = [str(audio_python), str(helper), str(wav_in), str(wav_out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            _log.warning("df3_denoise non-zero exit: %s", result.stderr[-300:])
            return False
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            _log.warning("df3_denoise: empty stdout")
            return False
        return json.loads(lines[-1]).get("ok", False)
    except Exception as e:
        _log.warning("df3_denoise failed: %s", e)
        return False


def _audio_restore_cascade(
    wav_in: Path,
    wav_out: Path,
    audio_python: Path,
    script: Path,
    cfg: dict,
) -> tuple[bool, list[str], dict]:
    """Run audio_restore.py subprocess (EQ → DF3 → loudnorm).

    Returns (success, stages_run, payload).
    """
    config_json = json.dumps({
        k: cfg[k] for k in (
            "eq_enabled", "df3_enabled", "loudnorm_enabled",
            "freqsplit_enabled", "freqsplit_hp_hz",
            "highpass_hz", "low_mid_cut_hz", "low_mid_cut_q", "low_mid_cut_db",
            "df3_atten_lim_db",
            "speechnorm_enabled", "speechnorm_peak", "speechnorm_expansion",
            "speechnorm_compression", "speechnorm_threshold",
            "loudnorm_target_lufs", "loudnorm_lra", "loudnorm_true_peak",
        ) if k in cfg
    })
    cmd = [str(audio_python), str(script), str(wav_in), str(wav_out), "--config-json", config_json]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            _log.warning("audio_restore_cascade non-zero exit: %s", result.stderr[-300:])
            return False, [], {}
        lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
        if not lines:
            _log.warning("audio_restore_cascade: no JSON in stdout")
            return False, [], {}
        payload = json.loads(lines[-1])
        stages = payload.get("stages_run", [])
        _log.info("audio_restore_cascade: stages=%s in_lufs=%.1f out_lufs=%.1f tp=%.1f duration=%.1fs",
                  stages,
                  payload.get("input_lufs") or 0,
                  payload.get("output_lufs") or 0,
                  payload.get("true_peak") or 0,
                  payload.get("duration_s") or 0)
        return payload.get("ok", False), stages, payload
    except Exception as e:
        _log.warning("audio_restore_cascade failed: %s", e)
        return False, [], {}


def _ebur128_normalize(wav_in: Path, wav_out: Path, target_lufs: float = -16.0) -> bool:
    """Normalise to target LUFS using pyloudnorm. Returns True on success, never raises."""
    try:
        import soundfile as sf
        import pyloudnorm as pyln
    except ImportError:
        return False
    try:
        data, rate = sf.read(str(wav_in))
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        if not (abs(loudness) < float("inf")):
            # -inf or NaN: copy unchanged
            sf.write(str(wav_out), data, rate)
            return True
        norm_data = pyln.normalize.loudness(data, loudness, target_lufs)
        sf.write(str(wav_out), norm_data, rate)
        return True
    except Exception as e:
        _log.warning("ebur128_normalize failed: %s", e)
        return False


def _output_timeline_utterances(words_per_beat: list[list[dict]],
                                clip_in_ms_per_beat: list[int],
                                beat_offsets_ms: list[int],
                                max_gap_ms: int = 700) -> list[dict]:
    """Map per-beat (trimmed-timeline) words onto the concatenated output timeline and
    group into utterances — mirrors _build_ass's time mapping so the leveler aligns with
    the rendered audio. Returns [{speaker_label, start_ms, end_ms}] time-ordered."""
    ow: list[dict] = []
    for words, clip_in, offset in zip(words_per_beat, clip_in_ms_per_beat, beat_offsets_ms):
        for w in words:
            ow.append({"speaker_label": w.get("speaker_label"),
                       "start_ms": offset + (w["start_ms"] - clip_in),
                       "end_ms": offset + (w["end_ms"] - clip_in)})
    ow.sort(key=lambda x: x["start_ms"])
    utts: list[dict] = []
    cur: list[dict] = []
    for w in ow:
        if cur and (w["speaker_label"] != cur[-1]["speaker_label"]
                    or w["start_ms"] - cur[-1]["end_ms"] > max_gap_ms):
            utts.append({"speaker_label": cur[0]["speaker_label"],
                         "start_ms": cur[0]["start_ms"], "end_ms": cur[-1]["end_ms"]})
            cur = []
        cur.append(w)
    if cur:
        utts.append({"speaker_label": cur[0]["speaker_label"],
                     "start_ms": cur[0]["start_ms"], "end_ms": cur[-1]["end_ms"]})
    return utts


def _audio_qc_gate(cascade_payload: dict, cfg: Optional[dict] = None) -> dict:
    """Assemble an audio-QC summary + warn/fail verdict from the cascade payload.

    The journaled record that lets the pipeline observe audio quality without re-listening
    (D-Audio-Seg-02). Thresholds are config-driven ([render.audio_qc]); a 'fail' verdict
    aborts the render when gate_enabled (D-Audio-Seg-04). Defaults hard-fail only on the
    objective defect (true-peak/clipping); inter-speaker level disparity and low hero SNR are
    'warn' by default — they're data-quality facts of the source, not pipeline bugs — but can
    be promoted to 'fail' via config (fail_inter_speaker_delta_db / fail_hero_snr_db) once the
    corpus is clean enough to enforce them."""
    cfg = cfg or {}
    seg = cascade_payload.get("segment_level") or {}
    applied = seg.get("applied") or []
    nf = seg.get("noise_floor_db")
    gains = [a.get("gain_db", 0.0) for a in applied if a.get("status") == "leveled"]
    snrs = [a["snr_db"] for a in applied if a.get("snr_db") is not None]
    tp = cascade_payload.get("true_peak")

    # inter-speaker delta: per-speaker mean post-gain level (≈ noise_floor + snr + gain).
    spk_levels: dict[str, list[float]] = {}
    for a in applied:
        s = a.get("snr_db")
        if s is None or nf is None:
            continue
        spk_levels.setdefault(a.get("speaker_label") or "?", []).append(nf + s + a.get("gain_db", 0.0))
    spk_means = {k: sum(v) / len(v) for k, v in spk_levels.items() if v}
    inter_delta = (round(max(spk_means.values()) - min(spk_means.values()), 1)
                   if len(spk_means) >= 2 else None)
    hero_snr = round(min(snrs), 1) if snrs else None

    fail_tp = cfg.get("fail_true_peak_dbtp", -1.0)
    warn_delta = cfg.get("warn_inter_speaker_delta_db", 4.0)
    fail_delta = cfg.get("fail_inter_speaker_delta_db", 0.0)   # 0/absent = disabled
    warn_snr = cfg.get("warn_hero_snr_db", 6.0)
    fail_snr = cfg.get("fail_hero_snr_db", 0.0)                # 0/absent = disabled

    issues: list[dict] = []
    if tp is not None and fail_tp is not None and tp > fail_tp:
        issues.append({"level": "fail", "what": f"true_peak {tp} > {fail_tp} dBTP (clipping)"})
    if seg.get("n_unusable", 0) > 0:
        issues.append({"level": "warn", "what": f"{seg['n_unusable']} utterance(s) below SNR floor"})
    if inter_delta is not None:
        if fail_delta and inter_delta > fail_delta:
            issues.append({"level": "fail", "what": f"inter-speaker delta {inter_delta} > {fail_delta} dB"})
        elif warn_delta and inter_delta > warn_delta:
            issues.append({"level": "warn", "what": f"inter-speaker delta {inter_delta} > {warn_delta} dB"})
    if hero_snr is not None:
        if fail_snr and hero_snr < fail_snr:
            issues.append({"level": "fail", "what": f"hero SNR {hero_snr} < {fail_snr} dB"})
        elif warn_snr and hero_snr < warn_snr:
            issues.append({"level": "warn", "what": f"hero SNR {hero_snr} < {warn_snr} dB"})

    verdict = "fail" if any(i["level"] == "fail" for i in issues) else ("warn" if issues else "ok")
    return {
        "verdict": verdict,
        "issues": issues,
        "integrated_lufs": cascade_payload.get("output_lufs"),
        "true_peak_dbtp": tp,
        "noise_floor_db": nf,
        "n_utterances": seg.get("n_utterances"),
        "n_unusable": seg.get("n_unusable"),
        "inter_speaker_delta_db": inter_delta,
        "hero_snr_db": hero_snr,
        "max_gain_db": round(max(gains), 1) if gains else None,
        "median_gain_db": round(float(sorted(gains)[len(gains) // 2]), 1) if gains else None,
    }


def _audio_cleanup(
    list_file: Path,
    tmp_dir: Path,
    export_id: str,
    cfg: dict,
    audio_python: Path,
    restore_script: Path,
    utterances: Optional[list[dict]] = None,
    qc_out: Optional[Path] = None,
) -> Optional[tuple[Path, list[str], dict]]:
    """Extract and restore audio. Returns (WAV path, stages_run, cascade_payload) or None on failure."""
    try:
        raw_wav = tmp_dir / f"{export_id}_audio_raw.wav"
        r = _run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-vn", "-c:a", "pcm_s16le", str(raw_wav),
        ])
        if r.returncode != 0:
            _log.warning("Failed to extract raw audio for cleanup")
            return None

        current_wav = raw_wav
        stages_run: list[str] = []
        cascade_payload: dict = {}

        restored_wav = tmp_dir / f"{export_id}_audio_restored.wav"
        if cfg.get("segment_level_enabled") and utterances:
            # Dialogue-aware path: DF3 → segment_level → loudnorm (D-Audio-Seg-01).
            df3_wav = tmp_dir / f"{export_id}_audio_df3.wav"
            ok1, st1, _ = _audio_restore_cascade(
                current_wav, df3_wav, audio_python, restore_script,
                {**cfg, "loudnorm_enabled": False})
            lvl_in = df3_wav if ok1 else current_wav
            _seg_stats: Optional[dict] = None
            try:
                from .segment_level import apply_to_wav
                leveled_wav = tmp_dir / f"{export_id}_audio_leveled.wav"
                _seg_stats = apply_to_wav(str(lvl_in), str(leveled_wav), utterances,
                             target_dbfs=float(cfg.get("segment_target_dbfs", -23.0)),
                             cap_clean=float(cfg.get("segment_cap_clean", 15.0)),
                             stats_out=(str(qc_out) if qc_out else None))
                ln_in = leveled_wav
                st1 = list(st1) + ["segment_level"]
            except Exception as e:
                _log.warning("segment_level failed (%s) — skipping leveling", e)
                ln_in = lvl_in
            ok2, st2, cascade_payload = _audio_restore_cascade(
                Path(ln_in), restored_wav, audio_python, restore_script,
                {**cfg, "eq_enabled": False, "df3_enabled": False,
                 "freqsplit_enabled": False, "loudnorm_enabled": True})
            if ok2:
                current_wav = restored_wav
                stages_run = list(st1) + list(st2)
            else:
                _log.warning("loudnorm pass failed — using leveled/df3 audio")
                current_wav = Path(ln_in)
                stages_run = list(st1)
            if _seg_stats is not None:
                cascade_payload["segment_level"] = _seg_stats
        else:
            ok, stages, cascade_payload = _audio_restore_cascade(
                current_wav, restored_wav, audio_python, restore_script, cfg
            )
            if ok:
                current_wav = restored_wav
                stages_run = stages
            else:
                _log.warning("Audio restoration cascade failed — using raw audio")

        return current_wav, stages_run, cascade_payload
    except Exception as e:
        _log.warning("audio_cleanup error: %s", e)
        return None


def _compute_keep_intervals(
    words: list[dict],
    in_ms: int,
    out_ms: int,
    cfg: dict,
) -> list[tuple[int, int]]:
    """Return (start_ms, end_ms) intervals to KEEP from [in_ms, out_ms]."""
    if not words:
        return [(in_ms, out_ms)]

    aggressiveness = cfg.get("aggressiveness", "standard")
    filler_base = {"um", "uh", "hmm", "mm"}
    if aggressiveness == "conservative":
        drop_set = filler_base
        # 600->1500 (D-Render-10): removing sub-1.5s pauses was clipping quiet trailing
        # words whose ASR end-time is mistimed early (a quiet speaker's word tail sits in
        # the "pause" after the ASR-timed word, indistinguishable from silence by energy).
        # Conservative mode now keeps natural turn-taking pauses; only longer dead air is
        # trimmed. The real fix is ASR word-timing/VAD quality (D0v2).
        gap_ms = 1500
    elif aggressiveness == "aggressive":
        drop_set = filler_base | {"like", "well", "so", "right"}
        gap_ms = 200
    else:  # standard
        drop_set = filler_base | {"like"}
        gap_ms = 400

    beat_words = [w for w in words if in_ms <= w["start_ms"] < out_ms]
    if not beat_words:
        return [(in_ms, out_ms)]

    # Mark and skip fillers and false starts
    keep: list[dict] = []
    i = 0
    while i < len(beat_words):
        w = beat_words[i]
        text_lower = w.get("text", "").lower().strip()
        if text_lower in drop_set:
            i += 1
            continue
        # False start: same text as immediately following word
        if i + 1 < len(beat_words):
            next_text = beat_words[i + 1].get("text", "").lower().strip()
            if text_lower and text_lower == next_text:
                i += 1
                continue
        keep.append(w)
        i += 1

    if not keep:
        _log.warning("All words dropped in beat [%d, %d] — using full range", in_ms, out_ms)
        return [(in_ms, out_ms)]

    # Build intervals by merging adjacent words with gap <= gap_ms
    intervals: list[tuple[int, int]] = []
    cur_start = max(keep[0]["start_ms"], in_ms)
    cur_end = min(max(keep[0]["end_ms"], keep[0]["start_ms"] + 1), out_ms)
    for w in keep[1:]:
        ws = max(w["start_ms"], in_ms)
        we = min(max(w["end_ms"], w["start_ms"] + 1), out_ms)
        if ws - cur_end <= gap_ms:
            cur_end = max(cur_end, we)
        else:
            if cur_end > cur_start:
                intervals.append((cur_start, cur_end))
            cur_start = ws
            cur_end = we
    if cur_end > cur_start:
        intervals.append((cur_start, cur_end))

    if not intervals:
        return [(in_ms, out_ms)]

    # Expand each interval by pad_ms on both ends, clamp to [in_ms, out_ms], re-merge overlaps
    pad_ms = int(cfg.get("pad_ms", 0))
    if pad_ms > 0:
        padded = [(max(in_ms, s - pad_ms), min(out_ms, e + pad_ms)) for s, e in intervals]
        merged: list[tuple[int, int]] = [padded[0]]
        for s, e in padded[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        intervals = merged

    return intervals


def _remap_words_to_trim(
    words: list[dict],
    keep_intervals: list[tuple[int, int]],
) -> tuple[list[dict], int]:
    """Remap word timestamps to trimmed output timeline. Returns (words, new_clip_in_ms=0)."""
    remapped: list[dict] = []
    offset = 0
    for s, e in keep_intervals:
        for w in words:
            if s <= w["start_ms"] < e:
                nw = dict(w)
                nw["start_ms"] = offset + (w["start_ms"] - s)
                nw["end_ms"] = offset + min(max(0, w["end_ms"] - s), e - s)
                remapped.append(nw)
        offset += e - s
    return remapped, 0


# ── Render ────────────────────────────────────────────────────────────────────

def render(
    beat_plan: dict,
    clips: list[dict],
    export_id: str,
    catalog: Path,
    proxy_dir: Path,
    output_dir: Path,
    *,
    reframe_strategy: str = "face_centroid",
    music_bed_path: Optional[str] = None,
    caption_style: Optional[str] = None,
    journal_db: Optional[Path] = None,
    config_path: Optional[Path] = None,
    speaker_colors_override: Optional[list[dict[str, str]]] = None,
) -> dict:
    """Render beat_plan to MP4. Returns {"output_mp4_path": str, "duration_s": float}."""
    _add_ffmpeg()

    def _journal(event: str, extra: dict = {}) -> None:
        if journal_db is None:
            return
        try:
            import sqlite3, datetime
            # +00:00 offset (not "Z") — every other at_utc writer uses
            # datetime.now(timezone.utc).isoformat(), and pipeline_status's
            # _latest_status() compares at_utc as TEXT across events/events_trace;
            # a "Z" suffix sorts after "+00:00" within the same second and can
            # misorder the comparison (W2 checkpoint review finding, D-Log-12).
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with sqlite3.connect(str(journal_db)) as con:
                con.execute(
                    # Phase 1 (D-Log-04): export/render trace lives in events_trace,
                    # not the outcome-only `events` table (§3.1) — this call site
                    # emits arbitrary namespaced sub-events (e.g. "audio_restore/ok"),
                    # not a (media,stage) disposition.
                    "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) VALUES (?,?,?,?,?,?,?)",
                    (ts, f"erf:render:{export_id}", "erf", "render", event, export_id, json.dumps(extra))
                )
        except Exception as e:
            _log.warning("journal write failed: %s", e)

    beats = beat_plan.get("beats", [])
    if not beats:
        raise ValueError("beat_plan has no beats")
    if len(clips) != len(beats):
        raise ValueError(f"clips length {len(clips)} != beats length {len(beats)}")

    format_fit = beat_plan.get("format_fit", "youtube_shorts")
    is_vertical = format_fit in _VERTICAL_PLATFORMS
    video_bitrate = "8000k" if is_vertical else "12000k"

    effective_caption_style = (caption_style or beat_plan.get("caption_style") or "centre").lower()
    use_captions = effective_caption_style != "none"

    music_path: Optional[Path] = None
    if music_bed_path:
        p = Path(music_bed_path)
        if p.exists():
            music_path = p
        else:
            _log.warning("music_bed_path %s not found — skipping", music_bed_path)

    cfg_all = _load_render_cfg(config_path)
    audio_cfg = cfg_all.get("audio_cleanup", {})
    trim_cfg = cfg_all.get("disfluency_trim", {})
    asd_cfg = cfg_all.get("active_speaker", {})
    beat_sync_cfg = cfg_all.get("beat_sync", {})
    audio_cleanup_enabled = audio_cfg.get("enabled", True)
    trim_enabled = trim_cfg.get("enabled", True)
    asd_confidence_threshold = float(asd_cfg.get("asd_confidence_threshold", 0.5))

    _erf_root = Path(__file__).parent.parent.parent
    _audio_python = _erf_root / "venvs" / "audio" / "bin" / "python"
    _df3_helper = _erf_root / "bin" / "df3_helper.py"
    _audio_restore_script = _erf_root / "bin" / "audio_restore.py"

    # Beat-sync: snap clip endpoints to musical beats when music_bed is present
    if beat_sync_cfg.get("enabled", True) and music_path is not None:
        clips = _beat_sync_clips(clips, music_path, tolerance_ms=300)

    tmp_dir = Path(output_dir) / f"tmp_{export_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if trim_enabled:
        _journal("disfluency_trim/start", {
            "n_beats": len(beats),
            "aggressiveness": trim_cfg.get("aggressiveness", "standard"),
        })

    try:
        # ── Step 1: per-beat segments ─────────────────────────────────────
        seg_paths: list[Path] = []
        beat_audio_paths: list[Path] = []  # per-beat normalized PCM audio (sync-safe)
        words_per_beat: list[list[dict]] = []
        clip_in_ms_per_beat: list[int] = []
        beat_offsets_ms: list[int] = []
        total_ms = 0
        det_cache: dict[str, list[dict]] = {}
        asd_cache: dict[str, Optional[dict]] = {}
        dim_cache: dict[str, tuple[int, int]] = {}
        dur_cache: dict[str, int] = {}
        # track (in_ms, out_ms, keep_intervals) per processed beat — in/out are the
        # pad-expanded, proxy-capped window values (used only for the n_trimmed count)
        beat_trim_info: list[tuple[int, int, list[tuple[int, int]]]] = []
        beat_spk_colors: list[dict[str, str]] = []
        # (media_id, in_ms, out_ms) windows already rendered. Narrative beat plans
        # can repeat or overlap source windows across beats (atom-pool starvation or
        # adjacent atoms sharing a boundary); render each distinct clip once.
        _rendered_windows: list[tuple[str, int, int]] = []

        # pad_ms expands the outer clip window so _compute_keep_intervals can
        # add lead/tail around snapped word boundaries (which are otherwise flush
        # against in_ms/out_ms, making the inner interval padding a no-op).
        clip_pad_ms = int(trim_cfg.get("pad_ms", 0)) if trim_enabled else 0

        for i, (beat, clip) in enumerate(zip(beats, clips)):
            _is_broll = bool(clip.get("_is_broll"))
            media_id = clip["media_id"]
            proxy = proxy_dir / f"{media_id}.mp4"
            if not proxy.exists():
                raise FileNotFoundError(f"Proxy not found: {proxy}")

            # Cap the padded window at the proxy's real duration: caption offsets
            # assume requested segment length == actual, so a beat ending within
            # pad_ms of the clip end would otherwise drift later captions.
            if media_id not in dur_cache:
                dur_cache[media_id] = int(_probe_duration(proxy) * 1000)
            proxy_dur_ms = dur_cache[media_id]

            _pad = 0 if _is_broll else clip_pad_ms
            in_ms = max(0, int(clip["in_ms"]) - _pad)
            _out_pad = 0 if clip.get("_crosstalk_trimmed") else _pad
            out_ms = int(clip["out_ms"]) + _out_pad
            # Clamp expanded window at adjacent same-media beat boundaries.
            # Overlapping source windows produce audio duplication (stutter); split
            # the overlap at its midpoint so each side gets equal slack. Guard each
            # clamp with an ascending-order + actual-overlap test: narrative threads
            # reorder clips out of source order, so a naive midpoint between a beat
            # and a non-chronological same-media neighbour can land before the beat's
            # own in-point and collapse the window to zero. Only clamp against a
            # neighbour whose window is chronologically adjacent AND overlapping.
            if _pad > 0:
                _orig_in = int(clip["in_ms"])
                _orig_out = int(clip["out_ms"])
                if i > 0 and clips[i - 1]["media_id"] == clip["media_id"]:
                    _prev_in = int(clips[i - 1]["in_ms"])
                    _prev_out = int(clips[i - 1]["out_ms"])
                    # Clamp if padded regions overlap (gap < 2*pad) OR source windows overlap.
                    if _prev_in < _orig_in and (_prev_out + _pad) > in_ms:
                        in_ms = max(in_ms, (_prev_out + _orig_in) // 2)
                if i < len(clips) - 1 and clips[i + 1]["media_id"] == clip["media_id"]:
                    _next_in = int(clips[i + 1]["in_ms"])
                    # Clamp if padded regions overlap (gap < 2*pad) OR source windows overlap.
                    if _orig_in < _next_in and out_ms > (_next_in - _pad):
                        out_ms = min(out_ms, (_orig_out + _next_in) // 2)
            if proxy_dur_ms > 0:
                out_ms = min(out_ms, proxy_dur_ms)
            # D0 v2: snap clip boundaries to nearest VAD silence/speech transition.
            # Prevents cuts landing mid-breath or mid-consonant when ASR timestamps
            # are imprecise. Only snaps within ±80 ms; no-ops when VAD not available.
            if not _is_broll:
                _vad = _load_vad(catalog, media_id)
                if _vad:
                    in_ms = max(0, _vad_snap(_vad, in_ms, snap_ms=80))
                    out_ms = _vad_snap(_vad, out_ms, snap_ms=80)
                    if proxy_dur_ms > 0:
                        out_ms = min(out_ms, proxy_dur_ms)
            dur_ms = out_ms - in_ms
            if dur_ms <= 0:
                _log.warning("Beat %d has zero duration — skipping", i)
                continue
            # Overlap-aware dedup: two beats whose source windows substantially
            # overlap show the same footage even when the (in,out) tuples differ
            # (e.g. two atoms sharing a segment-group end boundary). Skip a beat
            # that overlaps an already-rendered same-media window by >50% of the
            # shorter window.
            _dup = False
            for (_rm, _ri, _ro) in _rendered_windows:
                if _rm != media_id:
                    continue
                _ov = min(out_ms, _ro) - max(in_ms, _ri)
                if _ov > 0 and _ov > 0.5 * min(out_ms - in_ms, _ro - _ri):
                    _dup = True
                    break
            if _dup:
                _log.warning("Beat %d overlaps an already-rendered window (%d-%d) — skipping", i, in_ms, out_ms)
                continue
            _rendered_windows.append((media_id, in_ms, out_ms))

            all_words = _load_words(catalog, media_id)
            beat_words = [w for w in all_words if in_ms <= w["start_ms"] < out_ms]
            if _is_broll:
                beat_words = []  # suppress captions for b-roll overlay

            if trim_enabled and not _is_broll:
                keep_intervals = _compute_keep_intervals(beat_words, in_ms, out_ms, trim_cfg)
                trim_dur_ms = sum(e - s for s, e in keep_intervals)
                # Sparse-speech guard: if trim collapses clip to <30% of original,
                # the beat window is mostly silence/other-speaker — use full window.
                if dur_ms > 0 and trim_dur_ms < 0.30 * dur_ms:
                    keep_intervals = [(in_ms, out_ms)]
                    trim_dur_ms = dur_ms
                caption_words, caption_clip_in_ms = _remap_words_to_trim(beat_words, keep_intervals)
            else:
                keep_intervals = [(in_ms, out_ms)]
                caption_words = beat_words
                caption_clip_in_ms = in_ms
                trim_dur_ms = dur_ms

            beat_trim_info.append((in_ms, out_ms, keep_intervals))

            # Probe proxy dimensions once per media_id (Bug 1: portrait proxies must not be cropped)
            if media_id not in dim_cache:
                dim_cache[media_id] = _probe_proxy_dims(proxy)
            pw, ph = dim_cache[media_id]

            # b-roll orientation flag (W3(e), D-Log-25): V0's VLM pass catches
            # mis-rotated cutaway footage fix_orientation.py's face-detector
            # can't (no faces to key off in generic b-roll) — but it is a single
            # categorical call with no confidence score. A 3-media spot-check
            # during this build found only 1/3 flagged clips were genuinely
            # rotated; the other 2 were upright footage the VLM misread from
            # motion blur / an off-centre low-angle composition. Auto-rotating
            # pixels on that hit rate would trade a silent under-detection
            # defect for a worse silent over-correction defect, so this stays
            # flag-only: journal the verdict for editorial review, do not
            # transform the frame. Promote to auto-correct only behind a
            # confidence signal (e.g. cross-checked against fix_orientation.py's
            # face-based verdict when both apply).
            if _is_broll:
                _orientation = _load_broll_orientation(catalog, media_id)
                if _orientation in _ORIENTATION_FIX_VF:
                    _journal("render/broll_orientation/flagged",
                             {"media_id": media_id, "beat": i, "orientation": _orientation})

            is_portrait_proxy = ph >= pw  # 540×960, 1080×1920, etc.

            if is_vertical:
                if is_portrait_proxy:
                    # Proxy is already portrait — scale only, no crop needed
                    vf = "scale=1080:1920:flags=lanczos"
                else:
                    # Landscape proxy: 9:16 crop at full proxy height
                    crop_w_px = int(ph * 9 / 16)
                    centre_x = (pw - crop_w_px) // 2

                    if reframe_strategy == "active_speaker":
                        if media_id not in asd_cache:
                            asd_cache[media_id] = _load_active_speaker(catalog, media_id)
                        asd_data = asd_cache[media_id]
                        crop_x = None
                        if asd_data is not None:
                            crop_x = _asd_crop_x(asd_data, in_ms, out_ms, asd_confidence_threshold, src_w=pw, crop_w=crop_w_px)
                        if crop_x is None:
                            if media_id not in det_cache:
                                det_cache[media_id] = _load_detections(catalog, media_id)
                            crop_x = _face_crop_x(det_cache[media_id], in_ms, out_ms, src_w=pw, crop_w=crop_w_px)
                    elif reframe_strategy == "face_centroid":
                        if media_id not in det_cache:
                            det_cache[media_id] = _load_detections(catalog, media_id)
                        crop_x = _face_crop_x(det_cache[media_id], in_ms, out_ms, src_w=pw, crop_w=crop_w_px)
                    elif reframe_strategy == "groundingdino":
                        from .reframe_gdino import get_crop_centres_gdino
                        centres = get_crop_centres_gdino(
                            media_id=media_id,
                            start_ms=in_ms,
                            end_ms=out_ms,
                            proxy_path=proxy,
                        )
                        crop_x = max(0, min(pw - crop_w_px, centres[-1][1])) if centres else centre_x
                    else:
                        crop_x = centre_x

                    vf = f"crop={crop_w_px}:{ph}:{crop_x}:0,scale=1080:1920:flags=lanczos"
            else:
                vf = "scale=1920:1080:flags=lanczos"

            seg = tmp_dir / f"beat_{i:03d}.mp4"
            _broll_audio = ["-af", "volume=0"] if _is_broll else []

            if keep_intervals == [(in_ms, out_ms)]:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{in_ms / 1000:.3f}",
                    "-t", f"{dur_ms / 1000:.3f}",
                    "-i", str(proxy),
                    "-vf", vf,
                    "-r", "30",
                    "-c:v", "libx264", "-preset", "fast", "-b:v", video_bitrate,
                    *_broll_audio,
                    "-c:a", "aac", "-b:a", "192k",
                    str(seg),
                ]
                r = _run(cmd)
                if r.returncode != 0:
                    raise RuntimeError(f"Segment encode failed for beat {i}")
            else:
                sub_segs: list[Path] = []
                for j, (s, e) in enumerate(keep_intervals):
                    sub_dur = e - s
                    if sub_dur <= 0:
                        continue
                    sub_seg = tmp_dir / f"beat_{i:03d}_sub_{j:03d}.mp4"
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{s / 1000:.3f}",
                        "-t", f"{sub_dur / 1000:.3f}",
                        "-i", str(proxy),
                        "-vf", vf,
                        "-r", "30",
                        "-c:v", "libx264", "-preset", "fast", "-b:v", video_bitrate,
                        *_broll_audio,
                        "-c:a", "aac", "-b:a", "192k",
                        str(sub_seg),
                    ]
                    r = _run(cmd)
                    if r.returncode != 0:
                        raise RuntimeError(f"Sub-segment encode failed beat {i} sub {j}")
                    sub_segs.append(sub_seg)

                if not sub_segs:
                    raise RuntimeError(f"No sub-segments for beat {i} after trim")

                sub_list = tmp_dir / f"sub_concat_{i:03d}.txt"
                sub_list.write_text(
                    "".join(f"file '{p.resolve()}'\n" for p in sub_segs), encoding="utf-8"
                )
                r = _run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(sub_list), "-c", "copy", str(seg),
                ])
                if r.returncode != 0:
                    raise RuntimeError(f"Sub-segment concat failed for beat {i}")

            # ── Per-beat audio: PCM straight from the proxy at the same keep
            # intervals, normalised per-clip, then locked to THIS segment's video
            # duration. Building audio independently of the AAC video segments
            # avoids the encoder-priming + frame-quantisation drift that desynced
            # the old "extract from concatenated segments" path; per-beat duration
            # locking keeps audio and video frame-aligned across the whole render.
            _seg_v_dur = _probe_duration(seg)
            _per_clip_lufs = float(audio_cfg.get("per_clip_lufs", -20.0))
            beat_wav = tmp_dir / f"beat_{i:03d}.wav"

            def _silence(out: Path, dur_s: float) -> None:
                _run([
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-t", f"{max(dur_s, 0.01):.3f}", "-c:a", "pcm_s16le", str(out),
                ])

            if _is_broll or _seg_v_dur <= 0:
                _silence(beat_wav, _seg_v_dur)
            else:
                _a_subs: list[Path] = []
                for j, (s, e) in enumerate(keep_intervals):
                    if e - s <= 0:
                        continue
                    _asub = tmp_dir / f"beat_{i:03d}_a{j:03d}.wav"
                    ra = _run([
                        "ffmpeg", "-y",
                        "-ss", f"{s / 1000:.3f}", "-t", f"{(e - s) / 1000:.3f}",
                        "-i", str(proxy), "-vn",
                        "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(_asub),
                    ])
                    if ra.returncode == 0:
                        _a_subs.append(_asub)
                if not _a_subs:
                    _silence(beat_wav, _seg_v_dur)
                else:
                    if len(_a_subs) == 1:
                        _araw = _a_subs[0]
                    else:
                        _araw = tmp_dir / f"beat_{i:03d}_araw.wav"
                        _alist = tmp_dir / f"beat_{i:03d}_alist.txt"
                        _alist.write_text(
                            "".join(f"file '{p.resolve()}'\n" for p in _a_subs), encoding="utf-8"
                        )
                        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                              "-i", str(_alist), "-c", "copy", str(_araw)])
                    _anorm = tmp_dir / f"beat_{i:03d}_anorm.wav"
                    if not _ebur128_normalize(_araw, _anorm, target_lufs=_per_clip_lufs):
                        _anorm = _araw
                    # Lock to the segment's video duration (pad if short, trim if long).
                    _run([
                        "ffmpeg", "-y", "-i", str(_anorm),
                        "-af", "apad", "-t", f"{_seg_v_dur:.3f}",
                        "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(beat_wav),
                    ])
            beat_audio_paths.append(beat_wav)

            seg_paths.append(seg)
            beat_offsets_ms.append(total_ms)
            # Caption offsets must track the ACTUAL rendered segment duration, not the
            # intended trim duration — otherwise small per-beat differences (30fps
            # quantisation, audio lock) accumulate and captions drift out of sync
            # toward the end of the video.
            total_ms += int(round(_seg_v_dur * 1000)) if _seg_v_dur > 0 else trim_dur_ms
            clip_in_ms_per_beat.append(caption_clip_in_ms)
            words_per_beat.append(caption_words)
            if speaker_colors_override and i < len(speaker_colors_override):
                beat_spk_colors.append(speaker_colors_override[i])
            else:
                beat_spk_colors.append(_load_speaker_colors(catalog, media_id))

        if not seg_paths:
            raise RuntimeError("No segments produced")

        if trim_enabled:
            n_trimmed = sum(
                1 for orig_in, orig_out, ki in beat_trim_info
                if ki != [(orig_in, orig_out)]
            )
            _journal("disfluency_trim/ok", {"beats_trimmed": n_trimmed})

        # ── Step 2: concat list ────────────────────────────────────────────
        list_file = tmp_dir / "concat.txt"
        list_file.write_text(
            "".join(f"file '{seg.resolve()}'\n" for seg in seg_paths), encoding="utf-8"
        )

        # ── Step 2.5: audio cleanup ────────────────────────────────────────
        # Resolve speaker ref from identity_overlay for AnyEnhance timbre conditioning
        if audio_cfg.get("anyenhance_use_speaker_ref") and not audio_cfg.get("anyenhance_speaker_ref"):
            _media_ids = [c["media_id"] for c in clips if "media_id" in c]
            _person_counts: dict[str, int] = {}
            for _mid in _media_ids:
                _ov_path = catalog / "identity_overlay" / f"{_mid}.json"
                if _ov_path.exists():
                    try:
                        for _a in json.loads(_ov_path.read_text()).get("speaker_assignments", []):
                            _pid = _a.get("person_id", "")
                            if _pid:
                                _person_counts[_pid] = _person_counts.get(_pid, 0) + 1
                    except Exception:
                        pass
            if _person_counts:
                _dom_pid = max(_person_counts, key=_person_counts.__getitem__)
                _ref_path = _erf_root / "state" / "speaker_refs" / f"p_{_dom_pid}_ref.wav"
                if _ref_path.exists():
                    audio_cfg = dict(audio_cfg)
                    audio_cfg["anyenhance_speaker_ref"] = str(_ref_path)
                    _log.info("AnyEnhance speaker ref: %s (person=%s)", _ref_path.name, _dom_pid)

        # Audio source for the cascade: concat of the per-beat normalised PCM WAVs
        # (built in the segment loop). PCM-in/PCM-out concat avoids the AAC
        # encoder-priming drift of extracting from the video segments.
        audio_list_file = tmp_dir / "audio_concat.txt"
        audio_list_file.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in beat_audio_paths), encoding="utf-8"
        )

        clean_audio_path: Optional[Path] = None
        _audio_qc_cfg = cfg_all.get("audio_qc", {})
        _qc_gate_enabled = bool(_audio_qc_cfg.get("gate_enabled", True))
        _qc_fail_abort: Optional[dict] = None
        if audio_cleanup_enabled and beat_audio_paths:
            _journal("audio_clean/start", {})
            _journal("render/audio_restore/start", {})
            try:
                _seg_utts = (_output_timeline_utterances(
                    words_per_beat, clip_in_ms_per_beat, beat_offsets_ms)
                    if audio_cfg.get("segment_level_enabled") else None)
                _qc_out = catalog / "audio_qc" / f"{export_id}.json"
                _qc_out.parent.mkdir(parents=True, exist_ok=True)
                _result = _audio_cleanup(
                    audio_list_file, tmp_dir, export_id, audio_cfg,
                    _audio_python, _audio_restore_script,
                    utterances=_seg_utts, qc_out=_qc_out,
                )
                if _result:
                    clean_audio_path, _stages, _cascade_payload = _result
                    _journal("audio_clean/ok", {"wav": clean_audio_path.name})
                    _journal("render/audio_restore/ok", {
                        "stages": _stages,
                        "input_lufs": _cascade_payload.get("input_lufs"),
                        "output_lufs": _cascade_payload.get("output_lufs"),
                        "true_peak": _cascade_payload.get("true_peak"),
                    })
                    # ── audio QC (D-Audio-Seg-02): journal verdict + enrich sidecar ──
                    _qc = _audio_qc_gate(_cascade_payload, _audio_qc_cfg)
                    _journal("render/audio_qc/" + _qc["verdict"], _qc)
                    if _qc["verdict"] != "ok":
                        _log.warning("audio QC %s: %s", _qc["verdict"], _qc["issues"])
                    # Hard gate (D-Audio-Seg-04): capture fail for enforcement AFTER this
                    # try/except (a raise here would be swallowed by the audio handler).
                    if _qc["verdict"] == "fail" and _qc_gate_enabled:
                        _qc_fail_abort = _qc
                    try:
                        _d = json.loads(_qc_out.read_text()) if _qc_out.exists() else {}
                        _d.update({"export_id": export_id, "qc": _qc})
                        _qc_out.write_text(json.dumps(_d, ensure_ascii=False, indent=1))
                    except Exception:
                        pass
                else:
                    _journal("audio_clean/error", {"error": "cleanup returned None"})
                    _journal("render/audio_restore/skip", {"reason": "cleanup_failed"})
            except Exception as e:
                _log.warning("audio_cleanup raised: %s", e)
                _journal("audio_clean/error", {"error": str(e)[:300]})
                _journal("render/audio_restore/skip", {"reason": str(e)[:200]})
                clean_audio_path = None

        # Fallback: if the cascade was skipped/failed, still use the sync-safe
        # concatenated per-beat PCM audio (not the drifty segment AAC audio).
        if clean_audio_path is None and beat_audio_paths:
            _combined_raw = tmp_dir / f"{export_id}_audio_combined.wav"
            r = _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                      "-i", str(audio_list_file), "-c:a", "pcm_s16le", str(_combined_raw)])
            if r.returncode == 0:
                clean_audio_path = _combined_raw

        # ── Audio QC hard gate (D-Audio-Seg-04) ───────────────────────────
        # Aborts BEFORE the mux so no defective mp4 is produced. Raised here
        # (not in the audio try/except, which swallows exceptions) and only when
        # gate_enabled + a 'fail' verdict. render/audio_qc/fail is already
        # journaled, so the outcome is explicit even though we raise.
        if _qc_fail_abort is not None:
            _fails = "; ".join(i["what"] for i in _qc_fail_abort["issues"] if i["level"] == "fail")
            raise RuntimeError(f"audio QC gate FAILED ({export_id}): {_fails}")

        # ── Step 3: ASS captions ──────────────────────────────────────────
        ass_path: Optional[Path] = None
        if use_captions and any(words_per_beat):
            ass_path = tmp_dir / "captions.ass"
            ass_content = _build_ass(
                words_per_beat, clip_in_ms_per_beat, beat_offsets_ms,
                effective_caption_style, speaker_colors=beat_spk_colors,
                cues_cfg=cfg_all.get("captions", {}),
            )
            ass_path.write_text(ass_content, encoding="utf-8")
            # Sidecars (always-on, D-Log-24): captions + per-beat word→event
            # selection, the caption/colour audit trail for every render.
            try:
                import shutil as _sh
                _sh.copy(ass_path, Path(output_dir) / f"{export_id}.captions.ass")
                _dbg = []
                for _bi, (_ws, _cin, _boff) in enumerate(
                        zip(words_per_beat, clip_in_ms_per_beat, beat_offsets_ms)):
                    _evs = []
                    for _w in _ws:
                        _s = _boff + (_w["start_ms"] - _cin)
                        _e = _boff + (_w["end_ms"] - _cin)
                        _evs.append({"t": _w.get("text"), "spk": _w.get("speaker_label"),
                                     "out_start": _s, "out_end": _e,
                                     "dropped": _e <= _s or _s < 0 or not (_w.get("text") or "").strip()})
                    _ti = beat_trim_info[_bi] if _bi < len(beat_trim_info) else None
                    _dbg.append({"beat": _bi, "trim_info": _ti, "clip_in_ms": _cin,
                                 "offset_ms": _boff, "events": _evs})
                (Path(output_dir) / f"{export_id}.beats_debug.json").write_text(
                    json.dumps(_dbg, ensure_ascii=False, indent=1))
            except Exception as _e:
                _log.warning("render sidecar dump failed: %s", _e)

        # ── Step 4: final concat + captions + music ────────────────────────
        out_tmp = tmp_dir / f"{export_id}.tmp.mp4"
        final_out = Path(output_dir) / f"{export_id}.mp4"

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]

        clean_idx: Optional[int] = None
        music_idx: Optional[int] = None
        next_input = 1
        if clean_audio_path:
            cmd += ["-i", str(clean_audio_path)]
            clean_idx = next_input
            next_input += 1
        if music_path:
            cmd += ["-i", str(music_path)]
            music_idx = next_input
            next_input += 1

        audio_src_label = f"[{clean_idx}:a]" if clean_idx is not None else "[0:a]"

        vf_parts: list[str] = []
        af_part: Optional[str] = None

        if ass_path:
            vf_parts.append(f"subtitles='{ass_path}'")
        if music_idx is not None:
            af_part = f"{audio_src_label}[{music_idx}:a]amix=inputs=2:weights=1 0.25[ao]"

        filters: list[str] = []
        map_args: list[str] = []

        if vf_parts:
            filters.append(f"[0:v]{''.join(vf_parts)}[vo]")
            map_args += ["-map", "[vo]"]
        else:
            map_args += ["-map", "0:v"]

        if af_part:
            filters.append(af_part)
            map_args += ["-map", "[ao]"]
        elif clean_idx is not None:
            map_args += ["-map", f"{clean_idx}:a"]
        else:
            map_args += ["-map", "0:a"]

        if filters:
            cmd += ["-filter_complex", ";".join(filters)]
        cmd += map_args

        if vf_parts:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-b:v", video_bitrate]
        else:
            cmd += ["-c:v", "copy"]

        cmd += ["-c:a", "aac", "-b:a", "192k"]
        if is_vertical:
            # Platform duration cap. Made explicit: a silent -t truncation would
            # report the capped length back as duration_s and hide the loss
            # (violates explicit-outcomes). Warn + journal when it actually bites.
            cap_s = float(cfg_all.get("max_duration_s", 60.0))
            intended_s = total_ms / 1000.0
            if intended_s > cap_s + 0.05:
                _log.warning(
                    "Render intended %.1fs exceeds vertical cap %.1fs — TRUNCATING %.1fs of content",
                    intended_s, cap_s, intended_s - cap_s,
                )
                _journal("render/duration_truncated", {
                    "intended_s": round(intended_s, 3),
                    "cap_s": cap_s,
                    "lost_s": round(intended_s - cap_s, 3),
                })
            cmd += ["-t", f"{cap_s:.3f}"]

        # Lock output to the (master) video length — guards against any residual
        # audio-vs-video drift, incl. music amix running past the video.
        cmd += ["-shortest"]
        cmd.append(str(out_tmp))

        r = _run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"Final concat failed: {r.stderr[-500:]}")

        out_tmp.rename(final_out)
        duration_s = _probe_duration(final_out)

        return {"output_mp4_path": str(final_out), "duration_s": duration_s}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── G-RENDER gate test ────────────────────────────────────────────────────────

def render_gate_test(catalog: Path, proxy_dir: Path, output_dir: Path) -> dict:
    """Synthetic 10s test: first available proxy, vertical, no captions, no music.

    Returns {"passed": bool, "output_path": str, "duration_s": float|None, "error": str|None}
    """
    _add_ffmpeg()
    proxy_dir = Path(proxy_dir)
    catalog = Path(catalog)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proxies = sorted(proxy_dir.glob("*.mp4"))
    if not proxies:
        return {"passed": False, "output_path": "", "duration_s": None, "error": "no proxies found"}

    proxy = proxies[0]
    media_id = proxy.stem

    beat_plan = {
        "beats": [{"in_ms": 0, "out_ms": 10000}],
        "format_fit": "youtube_shorts",
    }
    clips = [{"media_id": media_id, "in_ms": 0, "out_ms": 10000}]

    try:
        result = render(
            beat_plan=beat_plan,
            clips=clips,
            export_id="g_render_gate",
            catalog=catalog,
            proxy_dir=proxy_dir,
            output_dir=output_dir,
            reframe_strategy="face_centroid",
            music_bed_path=None,
            caption_style="none",
        )
        out_path = result["output_mp4_path"]
        dur = result["duration_s"]
        if not Path(out_path).exists() or dur < 1.0:
            return {"passed": False, "output_path": out_path, "duration_s": dur,
                    "error": f"output invalid: dur={dur}"}
        return {"passed": True, "output_path": out_path, "duration_s": dur, "error": None}
    except Exception as exc:
        _log.exception("G-RENDER gate test failed")
        return {"passed": False, "output_path": "", "duration_s": None, "error": str(exc)}

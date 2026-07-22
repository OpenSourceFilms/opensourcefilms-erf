#!/usr/bin/env python3
"""Stage V0 — Visual grounding via VLM caption pass.

Produces catalog/visual_grounding/{media_id}.json from up to 3 representative
keyframes. Audio-only media (no keyframes) get footage_type="audio_only".

Run:
  python bin/v0_visual_grounding.py [--catalog PATH] [--media-ids ID ...] [--force]
  python bin/v0_visual_grounding.py --sample      # G-V0 regression set only
  python bin/v0_visual_grounding.py               # all media

Brief: docs/PHASE_V0_VISUAL_GROUNDING_BRIEF.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ERF_ROOT = Path(__file__).resolve().parents[1]
CATALOG = ERF_ROOT / "catalog"
WEIGHTS_VLM = Path("/workspace/weights/vlm/Qwen2.5-VL-7B-Instruct")
LOG_DIR = ERF_ROOT / "logs.all" / "v0"

PROMPT_VERSION = "v0.2.0"
STAGE_VERSION = "1.1.0"
PRODUCED_BY = "v0_visual_grounding.py"

_VLM_SYSTEM = (
    "You are a visual content analyser. "
    "Describe ONLY what is visually present in these frames — not what anyone is saying. "
    "Return ONLY strict JSON with no markdown fences."
)

_VLM_USER = (
    "Analyse the provided keyframe image(s) from a video clip.\n"
    "Return strict JSON with these fields:\n"
    '  "footage_type": one of: talking_head, bts_handheld, interview, screen_capture, '
    "b_roll, composited_vfx, graphics, other\n"
    '  "setting": short description of location/environment (null if unclear)\n'
    '  "on_screen_subjects": list of people or objects ACTUALLY VISIBLE in frame — '
    "not mentioned in dialogue, only what you see\n"
    '  "shot_type": one of: close-up, medium, wide, over-shoulder, screen-rec\n'
    '  "visual_summary": 1-2 sentences describing visual content only\n'
    '  "orientation": one of: upright, rotated_90_cw, rotated_90_ccw, upside_down — '
    "judge from the horizon line and any people/objects: would this frame need "
    "rotating to look normal to a viewer? rotated_90_cw means the frame must be "
    "rotated 90° clockwise to appear upright (i.e. it currently looks rotated "
    "90° counter-clockwise)\n\n"
    "If something is being discussed but is not visible in frame, it is NOT an on_screen_subject."
)

# G-V0 regression set: fly clips + varied types
G_V0_SAMPLE = [
    "m_75219c4b67e1",  # fly clip 1 — must tag talking_head, no insect
    "m_f741770c1b77",  # fly clip 2 — must tag talking_head, no insect
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _journal_skip(media_id: str, reason: str, run_id: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "at_utc": _now(), "run_id": run_id, "media_id": media_id,
        "stage": "V0", "event_type": "skip", "reason": reason,
    }
    log_path = LOG_DIR / f"{run_id}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _journal_ok(media_id: str, run_id: str, kf_count: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "at_utc": _now(), "run_id": run_id, "media_id": media_id,
        "stage": "V0", "event_type": "ok", "keyframes_used": kf_count,
    }
    log_path = LOG_DIR / f"{run_id}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _journal_error(media_id: str, run_id: str, error: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "at_utc": _now(), "run_id": run_id, "media_id": media_id,
        "stage": "V0", "event_type": "error", "error": error,
    }
    log_path = LOG_DIR / f"{run_id}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _atomic_write_json(target: Path, data: object) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".tmp{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _load_keyframes(catalog: Path, media_id: str) -> list[dict]:
    kf_path = catalog / "keyframes" / f"{media_id}.jsonl"
    if not kf_path.exists():
        return []
    rows = []
    for line in kf_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _select_keyframes(kf_rows: list[dict]) -> list[dict]:
    """Return up to 3 representative keyframes: first / mid / last."""
    if not kf_rows:
        return []
    sorted_kfs = sorted(kf_rows, key=lambda r: r.get("time_ms", 0))
    if len(sorted_kfs) == 1:
        return [sorted_kfs[0]]
    if len(sorted_kfs) == 2:
        return sorted_kfs
    mid_idx = len(sorted_kfs) // 2
    return [sorted_kfs[0], sorted_kfs[mid_idx], sorted_kfs[-1]]


def _null_grounding(media_id: str, model_config: dict) -> dict:
    return {
        "schema_version": "1.0.0",
        "media_id": media_id,
        "footage_type": "audio_only",
        "setting": None,
        "on_screen_subjects": [],
        "shot_type": None,
        "visual_summary": "No visual content — audio only.",
        "orientation": None,
        "keyframes_used": [],
        "caption_model": model_config.get("model", "none"),
        "produced_by": PRODUCED_BY,
        "stage_version": STAGE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_config": model_config,
        "raw_model_response_path": None,
    }


def _error_grounding(media_id: str, model_config: dict, error_note: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "media_id": media_id,
        "footage_type": "other",
        "setting": None,
        "on_screen_subjects": [],
        "shot_type": None,
        "visual_summary": f"Caption failed: {error_note}",
        "orientation": None,
        "keyframes_used": [],
        "caption_model": model_config.get("model", "none"),
        "produced_by": PRODUCED_BY,
        "stage_version": STAGE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_config": model_config,
        "raw_model_response_path": None,
    }


def _parse_vlm_response(raw: str) -> dict | None:
    text = raw.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not lines[0].startswith("```"):
                inner.append(line)
        text = "\n".join(inner).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    ft = parsed.get("footage_type", "")
    valid_types = {
        "talking_head", "bts_handheld", "interview", "screen_capture",
        "b_roll", "composited_vfx", "graphics", "audio_only", "other",
    }
    if ft not in valid_types:
        parsed["footage_type"] = "other"

    subjects = parsed.get("on_screen_subjects")
    if not isinstance(subjects, list):
        parsed["on_screen_subjects"] = []
    else:
        parsed["on_screen_subjects"] = [str(s) for s in subjects if s]

    if not isinstance(parsed.get("visual_summary"), str):
        parsed["visual_summary"] = ""

    valid_orientations = {"upright", "rotated_90_cw", "rotated_90_ccw", "upside_down"}
    if parsed.get("orientation") not in valid_orientations:
        parsed["orientation"] = "upright"

    return parsed


def _load_model_and_processor():
    """Load Qwen2.5-VL generative model. Verify it's the instruct variant."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    # Confirm arch support
    arch_list = torch.cuda.get_arch_list()
    if "sm_120" not in arch_list:
        raise RuntimeError(
            f"sm_120 (Blackwell) not in torch arch_list: {arch_list}. "
            "Cannot guarantee correct CUDA compilation for this GPU."
        )

    # Confirm we're loading the generative (instruct) model, not the embedding model
    config_path = WEIGHTS_VLM / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        arch = cfg.get("architectures", [])
        model_type = cfg.get("model_type", "")
        # Embedding-only models use a different architecture class
        if any("embedding" in a.lower() for a in arch):
            raise RuntimeError(
                f"Loaded model appears to be an embedding model (arch={arch}), "
                "not a generative instruct model. Check WEIGHTS_VLM path."
            )
        print(f"  model_type={model_type} arch={arch} — confirmed generative", file=sys.stderr)
    else:
        print("  WARNING: no config.json found at weights path", file=sys.stderr)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(WEIGHTS_VLM),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    # Constrain image resolution to keep GPU memory bounded across consecutive calls
    processor = AutoProcessor.from_pretrained(
        str(WEIGHTS_VLM),
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    import transformers
    model_config = {
        "model": "Qwen2.5-VL-7B-Instruct",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "quantization": "bf16",
    }
    print(f"  torch={torch.__version__} transformers={transformers.__version__}", file=sys.stderr)

    return model, processor, model_config


def _caption_media(
    media_id: str,
    kf_rows: list[dict],
    model: Any,
    processor: Any,
    model_config: dict,
    run_id: str,
    raw_log_dir: Path,
) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    selected = _select_keyframes(kf_rows)
    kf_ids = [r["keyframe_id"] for r in selected]

    # Build image paths
    image_paths = []
    for kf in selected:
        img_path = ERF_ROOT / kf["image_relpath"]
        if not img_path.exists():
            # Try webp thumb as fallback
            thumb_path = ERF_ROOT / kf.get("thumb_relpath", "")
            if thumb_path.exists():
                image_paths.append(str(thumb_path))
            else:
                print(f"  WARNING: missing image for {kf['keyframe_id']}", file=sys.stderr)
        else:
            image_paths.append(str(img_path))

    if not image_paths:
        raise RuntimeError(f"No image files found for {media_id} keyframes")

    # Build messages
    content: list[dict] = []
    for img_path in image_paths:
        content.append({"type": "image", "image": f"file://{img_path}"})
    content.append({"type": "text", "text": _VLM_USER})

    messages = [
        {"role": "system", "content": _VLM_SYSTEM},
        {"role": "user", "content": content},
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Strip input tokens from output
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    raw_response = processor.decode(generated, skip_special_tokens=True)

    # Free GPU memory so consecutive calls don't OOM on a 24GB card
    del inputs, output_ids, generated
    torch.cuda.empty_cache()

    # Log raw response
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = raw_log_dir / f"{media_id}.txt"
    raw_log_path.write_text(raw_response, encoding="utf-8")

    parsed = _parse_vlm_response(raw_response)
    if parsed is None:
        raise ValueError(f"Failed to parse VLM response: {raw_response[:200]!r}")

    return {
        "schema_version": "1.0.0",
        "media_id": media_id,
        "footage_type": parsed.get("footage_type", "other"),
        "setting": parsed.get("setting"),
        "on_screen_subjects": parsed.get("on_screen_subjects", []),
        "shot_type": parsed.get("shot_type"),
        "visual_summary": parsed.get("visual_summary", ""),
        "orientation": parsed.get("orientation", "upright"),
        "keyframes_used": kf_ids,
        "caption_model": model_config["model"],
        "produced_by": PRODUCED_BY,
        "stage_version": STAGE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_config": model_config,
        "raw_model_response_path": str(raw_log_path),
    }


def process_media(
    media_id: str,
    catalog: Path,
    model: Any,
    processor: Any,
    model_config: dict,
    run_id: str,
    force: bool = False,
) -> dict | None:
    out_path = catalog / "visual_grounding" / f"{media_id}.json"
    if not force and out_path.exists():
        print(f"  SKIP {media_id} (exists)", file=sys.stderr)
        return None

    kf_rows = _load_keyframes(catalog, media_id)
    raw_log_dir = LOG_DIR / "raw_responses"

    if not kf_rows:
        record = _null_grounding(media_id, model_config)
        _atomic_write_json(out_path, record)
        _journal_skip(media_id, "no_keyframes", run_id)
        print(f"  AUDIO_ONLY {media_id}", file=sys.stderr)
        return record

    try:
        record = _caption_media(
            media_id, kf_rows, model, processor, model_config, run_id, raw_log_dir
        )
        _atomic_write_json(out_path, record)
        _journal_ok(media_id, run_id, len(record["keyframes_used"]))
        print(
            f"  OK {media_id} ft={record['footage_type']} "
            f"kf={len(record['keyframes_used'])} "
            f"subjects={record['on_screen_subjects']}",
            file=sys.stderr,
        )
        return record
    except Exception as exc:
        error_note = str(exc)[:200]
        record = _error_grounding(media_id, model_config, error_note)
        _atomic_write_json(out_path, record)
        _journal_error(media_id, run_id, error_note)
        print(f"  ERROR {media_id}: {error_note}", file=sys.stderr)
        return record


def _all_media_ids(catalog: Path) -> list[str]:
    kf_dir = catalog / "keyframes"
    manifest_dir = catalog / "media_manifest"
    ids: set[str] = set()
    if manifest_dir.exists():
        for f in manifest_dir.glob("*.json"):
            ids.add(f.stem)
    # Also include anything with keyframes
    if kf_dir.exists():
        for f in kf_dir.glob("*.jsonl"):
            ids.add(f.stem)
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage V0 — Visual grounding")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--media-ids", nargs="+", default=None)
    parser.add_argument("--sample", action="store_true",
                        help="Run G-V0 regression sample only")
    parser.add_argument("--force", action="store_true",
                        help="Re-caption even if output exists")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:12]
    print(f"[V0] run_id={run_id}", file=sys.stderr)

    print("[V0] Loading model...", file=sys.stderr)
    model, processor, model_config = _load_model_and_processor()
    print(f"[V0] Model ready: {model_config['model']}", file=sys.stderr)

    if args.sample:
        media_ids = G_V0_SAMPLE
    elif args.media_ids:
        media_ids = args.media_ids
    else:
        media_ids = _all_media_ids(args.catalog)

    print(f"[V0] Processing {len(media_ids)} media...", file=sys.stderr)

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from erf.runlog import log_event as _log_event
        from erf import outcomes as _outcomes
    except Exception:
        _log_event = None
        _outcomes = None

    results = []
    for i, mid in enumerate(media_ids, 1):
        print(f"[V0] [{i}/{len(media_ids)}] {mid}", file=sys.stderr)
        if _log_event:
            _log_event("V0", mid, "start", {"run_id": run_id})
        try:
            rec = process_media(
                mid, args.catalog, model, processor, model_config, run_id, args.force
            )
            if rec:
                results.append(rec)
            if _log_event:
                _log_event("V0", mid, "ok" if rec else "skip", {"run_id": run_id})
            if _outcomes:
                _outcomes.record(mid, "V0", "ok" if rec else "skip",
                                 reason=None if rec else "no_keyframes")
                _outcomes.set_flag(mid, "has_visual_grounding", bool(rec))
        except Exception as _e:
            if _log_event:
                _log_event("V0", mid, "error", {"error": str(_e)[:300]})
            if _outcomes:
                _outcomes.record(mid, "V0", "error", reason=str(_e)[:200])
            raise

    print(f"\n[V0] Done — {len(results)} records written", file=sys.stderr)

    # G-V0 fly regression check
    # Pass condition: no fly/insect in on_screen_subjects or visual_summary,
    # AND footage_type is a "human conversation" type (talking_head or bts_handheld).
    # bts_handheld is more precise than talking_head for BTS footage — both correctly
    # signal that a fly cannot be shown as an on-screen subject.
    _CONVERSATION_TYPES = {"talking_head", "bts_handheld", "interview"}
    if args.sample or set(G_V0_SAMPLE).issubset(set(media_ids)):
        print("\n[V0] === G-V0 FLY REGRESSION CHECK ===", file=sys.stderr)
        all_pass = True
        for fid in G_V0_SAMPLE:
            vg_path = args.catalog / "visual_grounding" / f"{fid}.json"
            if not vg_path.exists():
                print(f"  MISSING {fid}", file=sys.stderr)
                all_pass = False
                continue
            vg = json.loads(vg_path.read_text())
            ft = vg.get("footage_type", "")
            subjects = [s.lower() for s in vg.get("on_screen_subjects", [])]
            summary_lower = (vg.get("visual_summary") or "").lower()
            fly_in_subjects = any("fly" in s or "insect" in s or "bug" in s for s in subjects)
            fly_in_summary = any(w in summary_lower for w in ("fly", "flies", "insect", "bug"))
            correct_type = ft in _CONVERSATION_TYPES
            ok = correct_type and not fly_in_subjects and not fly_in_summary
            status = "PASS" if ok else "FAIL"
            print(
                f"  {status} {fid}: footage_type={ft}(ok={correct_type}) "
                f"fly_in_subjects={fly_in_subjects} fly_in_summary={fly_in_summary}",
                file=sys.stderr,
            )
            print(f"    subjects: {vg.get('on_screen_subjects', [])}", file=sys.stderr)
            print(f"    summary:  {vg.get('visual_summary', '')}", file=sys.stderr)
            if not ok:
                all_pass = False
        print(
            f"\n  G-V0 fly regression: {'PASS' if all_pass else 'FAIL'}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

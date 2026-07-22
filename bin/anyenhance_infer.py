#!/usr/bin/env python3
"""AnyEnhance inference subprocess.

Runs inside venvs/anyenhance. Called from audio_restore.py.
Usage:
    python anyenhance_infer.py --input <wav> --output <wav>
        [--ref_audio <wav>]        # future use — baseline ignores this
        [--config <json5>]
        [--ckpt <model.pt>]
        [--timesteps 20]
        [--cond_scale 1.0]
        [--device cuda|cpu]

Prints exactly one JSON line to stdout on exit.
"""
import json
import os
import sys
import time
import types
from pathlib import Path

# Pre-inject torchvision stub before torch loads — avoids broken C extension
# registration from a mismatched torchvision install.
def _stub_torchvision():
    import importlib.machinery
    if "torchvision" not in sys.modules:
        _tv = types.ModuleType("torchvision")
        _tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None, origin="<stub>")
        _tv.transforms = types.ModuleType("torchvision.transforms")
        _tv.transforms.__spec__ = importlib.machinery.ModuleSpec(
            "torchvision.transforms", loader=None, origin="<stub>")
        sys.modules["torchvision"] = _tv
        sys.modules["torchvision.transforms"] = _tv.transforms

_stub_torchvision()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPO = "/workspace/weights/anyenhance/repo"
CONFIG = "/workspace/weights/anyenhance/config_abs.json"
CKPT = "/workspace/weights/anyenhance/checkpoint/epoch-83-step-200000-loss-4.4187/model.pt"


def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ref_audio", default=None)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--timesteps", type=int, default=20)
    p.add_argument("--cond_scale", type=float, default=1.0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _pick_device(requested: str | None) -> str:
    import torch
    if requested:
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    free_gb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated(0)) / 1024 ** 3
    return "cuda" if free_gb > 4.0 else "cpu"


def main():
    t0 = time.monotonic()
    args = _parse_args()

    # Put repo on path so `from anyenhance import ...` works
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    import torch
    import torchaudio
    import json5

    device = _pick_device(args.device)

    # ── Guard: weights exist ─────────────────────────────────────────────────
    for label, path in [("config", args.config), ("ckpt", args.ckpt)]:
        if not Path(path).exists():
            print(json.dumps({"ok": False, "error": f"{label} not found: {path}",
                              "duration_s": round(time.monotonic() - t0, 2)}))
            sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    cfg = json5.load(open(args.config))

    import dac
    from anyenhance import AnyEnhance_v1, MaskGitTransformer, AudioEncoder

    model_cfg = cfg["model"]
    dac_model = dac.DAC.load(model_cfg["dac_path"]).to(device)
    dac_model.eval().requires_grad_(False)

    transformer = MaskGitTransformer(**model_cfg["MaskGitTransformer"])
    audio_encoder = AudioEncoder(**model_cfg["AudioEncoder"])
    model = AnyEnhance_v1(
        vq_model=dac_model,
        transformer=transformer,
        audio_encoder=audio_encoder,
        **model_cfg["AnyEnhance_v1"],
    ).to(device)

    state_dict = torch.load(args.ckpt, map_location=device)
    if "module" in list(state_dict.keys())[0]:
        import torch.nn as nn
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict, strict=False)
        model = model.module
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ── Inference ────────────────────────────────────────────────────────────
    # torchaudio.load shim (same issue as audio venv — torchcodec absent)
    def _load_wav(path: str):
        try:
            return torchaudio.load(path)
        except Exception:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            import torch as _t
            return _t.from_numpy(data.T.copy()), sr

    def _save_wav(path: str, tensor, sr: int):
        try:
            torchaudio.save(path, tensor, sr)
        except Exception:
            import soundfile as sf
            arr = tensor.cpu().numpy()
            if arr.ndim == 2:
                arr = arr.T
            sf.write(path, arr, sr, format="WAV", subtype="PCM_16")

    signal, sr = _load_wav(args.input)
    signal = torch.mean(signal, dim=0, keepdim=True).to(device)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100).to(device)
    signal = resampler(signal)

    window_size = 512 * model.seq_len
    overlap = 1024

    # Windowed processing (inlined from infer.py process_single_audio)
    def _pad_or_trunc(x, length):
        if x.size(-1) < length:
            repeat_times = length // x.size(-1) + 1
            x = x.repeat(1, repeat_times)[..., :length]
        elif x.size(-1) > length:
            x = x[..., :length]
        return x

    def _smooth_transition(chunks, ov=1024):
        if not chunks:
            return torch.Tensor([])
        window = torch.hann_window(ov * 2, periodic=True).to(chunks[0].device)
        result = chunks[0]
        for i in range(1, len(chunks)):
            fl = min(ov, result.shape[-1])
            prev_end = result[:, -fl:].clone()
            cr_fl = min(ov, chunks[i].shape[-1])
            curr_start = chunks[i][:, :cr_fl].clone()
            prev_end *= window[ov:ov + fl] if fl == ov else window[-fl:]
            curr_start *= window[:cr_fl]
            mn = min(fl, cr_fl)
            result[:, -mn:] = prev_end[:, -mn:] + curr_start[:, :mn]
            result = torch.cat((result, chunks[i][:, cr_fl:]), dim=1)
        return result

    original_length = signal.shape[-1]
    hop_size = window_size - overlap
    enhanced_chunks = []
    start = 0

    with torch.no_grad():
        while start < signal.shape[-1]:
            is_last = (start + window_size >= signal.shape[-1])
            if is_last:
                seg = signal[:, -window_size:]
            else:
                seg = signal[:, start:start + window_size]

            is_padding = seg.shape[-1] < window_size
            valid_len = seg.shape[-1]
            if is_padding:
                seg = _pad_or_trunc(seg, window_size)

            _, out_seg = model.generate(seg.unsqueeze(0), timesteps=args.timesteps,
                                        cond_scale=args.cond_scale)
            out_seg = out_seg.squeeze(0)

            if is_last:
                if is_padding:
                    out_seg = out_seg[:, :valid_len]
                last_valid = signal.shape[-1] - start or window_size
                out_seg = out_seg[:, -last_valid:]

            enhanced_chunks.append(out_seg)
            start += hop_size

    enhanced = _smooth_transition(enhanced_chunks, ov=overlap)
    enhanced = enhanced[:, :original_length]

    # Write output atomically (44100 Hz)
    tmp = args.output + ".writing.wav"
    _save_wav(tmp, enhanced.detach().cpu(), 44100)
    os.replace(tmp, args.output)

    used_ref = bool(args.ref_audio and Path(args.ref_audio).exists())
    print(json.dumps({
        "ok": True,
        "device": device,
        "ref_audio_used": used_ref,
        "ref_audio_path": args.ref_audio if used_ref else None,
        "duration_s": round(time.monotonic() - t0, 2),
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(json.dumps({
            "ok": False,
            "error": str(exc)[:400],
            "traceback": traceback.format_exc()[-600:],
        }))
        sys.exit(1)

#!/usr/bin/env python3
"""Thin subprocess wrapper for DeepFilterNet3 denoising.

Usage:
    python df3_helper.py <wav_in_path> <wav_out_path>

Exits 0 on success, 1 on failure.
Prints exactly one JSON line to stdout.
"""
import json
import sys
from pathlib import Path

from df.enhance import enhance, init_df, load_audio, save_audio


def main() -> None:
    if len(sys.argv) != 3:
        print(json.dumps({"ok": False, "error": "usage: df3_helper.py <in> <out>"}))
        sys.exit(1)

    wav_in = Path(sys.argv[1])
    wav_out = Path(sys.argv[2])

    try:
        model, df_state, _ = init_df()
        audio, _ = load_audio(str(wav_in), sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(str(wav_out), enhanced, sr=df_state.sr())
        print(json.dumps({"ok": True}))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)[:300]}))
        sys.exit(1)


if __name__ == "__main__":
    main()

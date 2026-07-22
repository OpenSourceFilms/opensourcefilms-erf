"""Subprocess helper: CNN14 audio classification via panns_inference.

Called by quality.py as an isolated subprocess. Prints one JSON line to stdout.
Usage: python audio_classify_helper.py <wav_path> [<panns_data_dir>]
"""
import json
import os
import sys


def main():
    try:
        wav_path = sys.argv[1] if len(sys.argv) > 1 else None
        panns_data_arg = sys.argv[2] if len(sys.argv) > 2 else None

        if not wav_path:
            raise ValueError("wav_path required")

        # Resolve checkpoint: explicit arg > /workspace/erf/weights/panns > ~/panns_data > auto-download
        ckpt_path = None
        candidates = []
        if panns_data_arg:
            candidates.append(os.path.join(panns_data_arg, "Cnn14_mAP=0.431.pth"))
        candidates += [
            "/workspace/erf/weights/panns/Cnn14_mAP=0.431.pth",
            os.path.join(os.path.expanduser("~"), "panns_data", "Cnn14_mAP=0.431.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                ckpt_path = c
                break
        # else: leave None so panns_inference auto-downloads to ~/panns_data/

        import numpy as np
        import librosa
        from panns_inference import AudioTagging

        # Load audio at CNN14's expected sample rate
        audio, _ = librosa.load(wav_path, sr=32000, mono=True, duration=30.0)
        audio = audio[np.newaxis, :]  # (1, samples)

        # Suppress stdout from panns_inference model loading
        devnull = open(os.devnull, "w")
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            at = AudioTagging(checkpoint_path=ckpt_path, device="cpu")
        finally:
            sys.stdout = old_stdout
            devnull.close()

        clipwise_output, _ = at.inference(audio)  # (1, 527)
        probs = clipwise_output[0]

        music_prob = float(probs[137])   # AudioSet index 137 = "Music"
        speech_prob = float(probs[0])    # AudioSet index 0 = "Speech"
        top_idx = int(probs.argmax())
        top_class = at.labels[top_idx] if hasattr(at, "labels") and at.labels else str(top_idx)

        print(json.dumps({
            "audio_music_prob": round(music_prob, 6),
            "audio_speech_prob": round(speech_prob, 6),
            "audio_top_class": top_class,
            "audio_classifier_model": "panns-cnn14",
        }))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}))


if __name__ == "__main__":
    main()

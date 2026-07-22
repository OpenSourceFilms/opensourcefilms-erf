# Third-party licenses

ERF's own code (everything in this repository) is MIT-licensed — see `LICENSE`. This repository does
**not** vendor any third-party source code or model weights; it depends on the packages and models
below at runtime, installed separately by you. Their licenses are their own and are **not** overridden
by ERF's MIT license. This document distinguishes a library's **code license** from any **pretrained
weights license**, since for several ML libraries below these differ.

**Commercial use of the full running system requires every dependency below to be commercially
compatible for your use case.** Where a component's license is marked `UNVERIFIED`, do not assume it
is safe for commercial use — verify it yourself before deploying commercially.

## Verified

| Component | Code license | Weights license | Commercial use | Verified |
|---|---|---|---|---|
| [InsightFace](https://github.com/deepinsight/insightface) | MIT | Common pretrained packs (`buffalo_l`, `antelopev2`, etc.) are **research/non-commercial only** — a commercial license must be obtained directly from InsightFace | Code: yes. Pretrained weights: **no**, without a separate commercial license | 2026-07-22, insightface.ai licensing page |
| [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) | MIT (includes bundled pyannote-audio, MIT; dscore, BSD-2-Clause) | Its pretrained diarization checkpoints (e.g. `diarizen-wavlm-large-s80-md`) are **CC-BY-NC-4.0** | Code: yes. Pretrained weights: **no** | Prior audit (2026-07-22 session, this org) |
| ECAPA2 speaker-embedding weights | — | **CC-BY-NC-4.0** | **No**, without a separate commercial license | Prior audit (2026-07-22 session, this org) |
| ReDimNet (speaker embedding) | MIT | MIT | Yes | Prior audit (2026-07-22 session, this org) |
| ReDimNet2 | States MIT in its own README; **no formal LICENSE file** in the repo at time of check | Same caveat | Likely yes, but unformalized — re-verify before relying on it commercially | Prior audit (2026-07-22 session, this org) |
| [pyannote-audio](https://github.com/pyannote/pyannote-audio) | MIT | Pretrained pipelines are gated on HuggingFace and carry their own per-model terms (not audited here — check whichever specific pipeline you load) | Code: yes. Pretrained pipelines: check per-model | Prior audit (2026-07-22 session, this org) |
| [TransNetV2](https://github.com/soCzech/TransNetV2) | MIT | MIT | Yes | Prior audit (2026-07-22 session, this org) |
| PANNs CNN14 (`panns_inference`) | Unspecified/unlicensed as distributed (pip-installed, weights cached locally, no clear license accompanying them) | Same | **Unknown — do not assume commercially safe** | Prior audit (2026-07-22 session, this org) |

## Not yet verified

The following are used by ERF's pipeline code (see the `bin/`/`src/` import surface) but a reliable
license lookup was not completed as part of this repository's release — an automated research pass
returned no usable source for these. **Do not treat these as cleared for commercial use.** Verify
directly (upstream repo's `LICENSE` file, model card, or PyPI page) before any commercial deployment:

- [IDEA-Research GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — object detection
- NVIDIA NeMo toolkit (`nemo_toolkit`) — the toolkit and any pretrained models you load via it may be
  licensed separately
- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) — active speaker detection
- `transnetv2_pytorch` (PyPI package specifically, distinct from the upstream TransNetV2 repo above)
- `qwen_vl_utils` (Qwen-VL utilities)
- Descript Audio Codec (`dac`)
- `whisperx`
- `nemo` (see NeMo toolkit above)
- Any other package imported by this codebase not listed above — this list reflects a name-level scan
  of the `bin/`/`src/` import surface, not an exhaustive audit

## Standard permissive libraries (not individually audited)

Widely-used, unambiguously permissively-licensed general-purpose libraries — `numpy`, `torch`,
`torchaudio`, `scipy`, `scikit-learn`, `pandas`-adjacent tooling, `Pillow`, `opencv-python` (`cv2`),
`faster-whisper`, `soundfile`, `librosa`, `fastapi`, `pydantic`, `jsonschema` — are not individually
re-audited here. These are well-established BSD/MIT/Apache-2.0 projects with no known non-commercial
restriction, consistent with common industry knowledge; if in doubt, check the specific version you
install.

## How to update this file

If you add a new model dependency, especially anything with pretrained weights, add a row above with:
the upstream URL, code license, weights license (if different), a commercial-use verdict, and the date
you checked it. Prefer checking the actual `LICENSE`/model-card file over a summary elsewhere — license
pages and blog posts can lag the actual repository state.

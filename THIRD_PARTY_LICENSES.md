# Third-party licenses

ERF's own code (everything in this repository) is MIT-licensed; see `LICENSE`. This repository does
**not** vendor any third-party source code or model weights; it depends on the packages and models
below at runtime, installed separately by you. Their licenses are their own and are **not** overridden
by ERF's MIT license. This document distinguishes a library's **code license** from any **pretrained
weights license**, since for several ML libraries below these differ.

**Commercial use of the full running system requires every dependency below to be commercially
compatible for your use case.** Where a component's license is marked `UNVERIFIED`, do not assume it
is safe for commercial use: verify it yourself before deploying commercially.

## Verified

| Component | Code license | Weights license | Commercial use | Verified |
|---|---|---|---|---|
| [InsightFace](https://github.com/deepinsight/insightface) | MIT | Common pretrained packs (`buffalo_l`, `antelopev2`, etc.) are **research/non-commercial only**: a commercial license must be obtained directly from InsightFace | Code: yes. Pretrained weights: **no**, without a separate commercial license | 2026-07-22, insightface.ai licensing page |
| [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) | MIT (includes bundled pyannote-audio, MIT; dscore, BSD-2-Clause) | Its pretrained diarization checkpoints (e.g. `diarizen-wavlm-large-s80-md`) are **CC-BY-NC-4.0** | Code: yes. Pretrained weights: **no** | Prior audit (2026-07-22 session, this org) |
| ECAPA2 speaker-embedding weights | N/A (standalone weights) | **CC-BY-NC-4.0** | **No**, without a separate commercial license | Prior audit (2026-07-22 session, this org) |
| ReDimNet (speaker embedding) | MIT | MIT | Yes | Prior audit (2026-07-22 session, this org) |
| ReDimNet2 | States MIT in its own README; **no formal LICENSE file** in the repo at time of check: re-verify before relying on it commercially | Same caveat | Likely yes, but unformalized; re-verify before relying on it commercially | Prior audit (2026-07-22 session, this org) |
| [pyannote-audio](https://github.com/pyannote/pyannote-audio) | MIT | Pretrained pipelines are gated on HuggingFace and carry their own per-model terms (not audited here; check whichever specific pipeline you load) | Code: yes. Pretrained pipelines: check per-model | Prior audit (2026-07-22 session, this org) |
| [TransNetV2](https://github.com/soCzech/TransNetV2) | MIT | MIT | Yes | Prior audit (2026-07-22 session, this org) |
| PANNs CNN14 (`panns_inference`) | Unspecified/unlicensed as distributed (pip-installed, weights cached locally, no clear license accompanying them) | Same | **Unknown: do not assume commercially safe** | Prior audit (2026-07-22 session, this org) |

## Additional external-library source licences verified 26 October 2026

The following source licences were checked against upstream LICENSE files
or the package publisher's official metadata. **Only the library source
licence is verified below.** The exact installed version, binary wheel,
container, model weights, model card, bundled datasets and transitive
dependencies are separate and have **not** thereby been cleared for
commercial use. These packages and weights are not redistributed in this
public code-only repository.

| ERF dependency | Verified source/publisher evidence | Scope limitation |
|---|---|---|
| GroundingDINO | [IDEA-Research/GroundingDINO LICENSE](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE): Apache-2.0 | Model checkpoints/datasets and any independently installed extension need their own review. |
| Light-ASD | [Junhua-Liao/Light-ASD LICENSE](https://github.com/Junhua-Liao/Light-ASD/blob/main/LICENSE): MIT, copyright Liao Junhua | Separate trained weights and face/audio datasets not reviewed. |
| whisperx | [m-bain/whisperX LICENSE](https://github.com/m-bain/whisperX/blob/main/LICENSE): BSD-2-Clause, copyright Max Bain | Whisper, alignment, diarization models and any downloaded weights retain their own terms. |
| Descript Audio Codec (`dac`) | [descriptinc/descript-audio-codec LICENSE](https://github.com/descriptinc/descript-audio-codec/blob/main/LICENSE): MIT | Code licence, not a check of any separately distributed audio codec weights. |
| qwen-vl-utils | [Qwen team's published PyPI metadata](https://pypi.org/project/qwen-vl-utils/) and [official package pyproject](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/pyproject.toml): Apache-2.0 | Utilities only; separate Qwen vision-language models may have different weight licences. |
| NVIDIA NeMo (`nemo_toolkit`) | [NVIDIA's NeMo Framework guide](https://docs.nvidia.com/nemo-framework/user-guide/26.02/overview.html) identifies the GitHub source as Apache-2.0 | NVIDIA explicitly distinguishes the **NeMo Framework container** under its AI Product Agreement, and possible bundled Llama/model terms. Verify the actual container, environment and checkpoints. |

These findings **do not** reclassify noncommercial pretrained InsightFace,
DiariZen or ECAPA2 weights above as commercially compatible.

## Still unverified for this public release

- `transnetv2_pytorch` on PyPI, **as installed**, is distinct from the
  MIT-licensed upstream TransNetV2 repository. Its package metadata,
  distributed weights and bundled contents have not been checked.
- The exact source/version and dependency set for each `nemo` import should
  be resolved against the historical runtime before treating it as the
  NeMo Framework distribution reviewed above.
- The real installed model/checkpoint revisions for the external libraries
  above; a permissive library licence does not automatically govern weights.
- Any other imported package not listed. The original list was a
  name-level scan, not a complete software bill of materials.

## Standard permissive libraries (not individually audited)

Widely-used, unambiguously permissively-licensed general-purpose libraries; `numpy`, `torch`,
`torchaudio`, `scipy`, `scikit-learn`, `pandas`-adjacent tooling, `Pillow`, `opencv-python` (`cv2`),
`faster-whisper`, `soundfile`, `librosa`, `fastapi`, `pydantic`, `jsonschema`; are not individually
re-audited here. These are well-established BSD/MIT/Apache-2.0 projects with no known non-commercial
restriction, consistent with common industry knowledge; if in doubt, check the specific version you
install.

## How to update this file

If you add a new model dependency, especially anything with pretrained weights, add a row above with:
the upstream URL, code license, weights license (if different), a commercial-use verdict, and the date
you checked it. Prefer checking the actual `LICENSE`/model-card file over a summary elsewhere; license
pages and blog posts can lag the actual repository state.

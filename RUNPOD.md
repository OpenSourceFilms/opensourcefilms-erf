# Running ERF on Runpod

ERF was designed, built, and operated on Runpod GPU infrastructure. This document describes what
is actually known about that deployment from the project's own specification (`MASTER.md`), states
a genuinely tested reproducible path for this public release, and is explicit about what could not
be exercised in this pass.

This is not a guide to standing up the full production system (that requires production model
weights, a MooseFS-backed network volume, and identity data this repo deliberately does not ship).
It is a guide to cloning this repo on a Runpod pod, installing it cleanly, and running the one path
that has actually been validated: the test suite.

## Recommended pod

Facts below are carried over from the live production deployment's own environment notes
(`MASTER.md` §0.2, §17-18), not re-measured in this pass:

- **Base image**: Runpod's own CUDA 12.8 + PyTorch base image, then activate a per-stage venv on
  top. This is the deployment's own documented default (`MASTER.md` §10, dispatch strategy "B");
  it needs no custom container image, only a boot script that activates the right venv.
- **GPU class / VRAM**: ERF splits GPU work into two roles, each with its own floor:
  - **audio** worker (faster-whisper + pyannote diarization): **≥ 16 GB VRAM**.
  - **embed** worker (Qwen3-Embedding-8B text + Qwen3-VL-Embedding-8B visual): **≥ 24 GB VRAM**,
    **≥ 40 GB preferred** for batch throughput.
  - The production dev pod is an RTX PRO 4000 Blackwell (sm_120), 12 vCPU, 31 GB. Blackwell means
    the `embed` venv specifically needs **torch ≥ 2.8, cu128 build**; an older torch 2.5.1 build on
    disk in the same deployment does not run on that card. If you're on an older (non-Blackwell)
    GPU, an earlier cu12x torch build will work; the cu128 requirement is a Blackwell-specific
    constraint, not a general ERF requirement.
- **Datacenter / storage**: production pins to Runpod's `EU-RO-1` datacenter with a persistent
  network volume mounted at `/workspace` (MooseFS-backed, not NFS/block storage, which is why the
  codebase is strict about single-writer-per-file and atomic-rename). None of that is required to
  clone and test this public repo; it matters if you're reconstructing the full multi-pod
  dispatch setup described in `MASTER.md` §10.

## What this repo actually ships vs. what production also has

This is pipeline code and architecture documentation only (see README "What's not included"). It
does **not** include: model weights/checkpoints, the MooseFS network volume, the RunPod dispatch
control plane's live state, or any identity/enrollment data. Rebuilding the full production
deployment on your own Runpod account means, at minimum, separately obtaining every checkpoint
listed below and standing up your own storage and dispatch layer; this document does not attempt
to script that.

## Per-stage virtual environments

ERF deliberately does **not** use one shared `requirements.txt`: its ML stages have overlapping
but not identical, sometimes conflicting, dependency sets, so production keeps them in isolated
venvs and there is no single lockfile that would honestly represent "the ERF environment." Per
`MASTER.md` §18.1:

| venv | role | key packages (as pinned in production) |
|---|---|---|
| `audio` | GPU, Stage A (ASR + diarization) | `faster-whisper==1.2.*`, `ctranslate2==4.6.*`, `pyannote.audio==4.0.4`, `torchaudio==2.11.0+cu128`, `torch` (cu128) |
| `embed` | GPU, Stages B1/B4 (text + visual embedding) | `torch>=2.8` (cu128), `torchvision` (cu128), `transformers`, `sentence-transformers`, `accelerate`, `lancedb`, `transnetv2-pytorch` |
| `control` | CPU, orchestration, API server, everything else in this repo | no ML deps; this is what `pip install -e .` gives you |

On a single dev pod, `control` and `embed` are typically merged into one venv (their packages don't
conflict); `audio` stays isolated because of `pyannote`/`torchaudio` pinning. When dispatching to
separate GPU pods per role, keep the three-way split; see `MASTER.md` §18.1 for the full rationale.

**torchaudio note** (from production, worth repeating here): install cu128 torch/torchaudio from
`--index-url https://download.pytorch.org/whl/cu128` specifically. The default PyPI build targets
a newer CUDA ABI and fails against CUDA 12.x drivers.

## Environment variables and secrets (names only)

Set these in your own pod environment; never commit values. This repo's `.gitignore` explicitly
excludes anything that could carry a plaintext key.

| Variable | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | Required for any LLM-backed stage (`src/erf/llm.py`): DeepSeek (`deepseek-v4-pro`) is ERF's sole LLM backend for editorial/story stages. Not needed for Stage A/B/C or the test suite. |
| `RUNPOD_API_KEY` | Only needed if you're reconstructing the GPU-pod dispatch layer (`MASTER.md` §10): a **restricted** key (create/delete/read pods only), not needed to run this repo's code directly. |
| `ERF_CATALOG_ROOT` | Overrides the default `<repo>/catalog` path (`src/erf/paths.py`); unset behaves exactly like the hardcoded default. |
| `ERF_DIAR_SOURCE` | Selects which diarization-hypothesis source to read (`"community1"` default, or `"diarizen"`). |
| HuggingFace auth (`huggingface-cli login`, or the standard `HF_TOKEN` env var) | Needed to pull gated pyannote diarization pipeline weights on first load. Not needed for the test suite. |

## Model / checkpoint requirements

None of these are shipped in this repo (see README "What's not included"); you supply your own:

- **ASR**: `faster-whisper-large-v3` (or another CTranslate2-compatible Whisper checkpoint).
- **Diarization**: `pyannote/speaker-diarization-community-1` (gated on HuggingFace, needs the HF
  auth above).
- **Embedding**: `Qwen3-Embedding-8B` (text) and `Qwen3-VL-Embedding-8B` (visual/cross-modal).
- **Faces** (Stage F0, opt-in): InsightFace `buffalo_l` (or equivalent) ONNX model pack. Its
  code is MIT but the pretrained pack itself is research/non-commercial only, see
  `THIRD_PARTY_LICENSES.md`.

## Running the safe representative test

The pipeline's actual GPU inference calls are hardcoded to require a real CUDA device (for example
`src/erf/asr.py`'s Whisper loader has no CPU fallback) and depend on multi-gigabyte checkpoints not
distributed here, so this pass could not exercise a live ASR/embedding/face run end-to-end without
a real GPU pod and your own model weights. What genuinely was exercised, in a fresh clone, fresh
venv, on this task's sandbox (no GPU attached):

```bash
git clone https://github.com/OpenSourceFilms/opensourcefilms-erf.git
cd opensourcefilms-erf
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest jsonschema scikit-learn
pytest tests/
```

This is the `control`-tier surface: contract/schema validation, ID generation, the status/outcome
journal, path-seam behaviour, the scheduler, and the render/export logic: the parts of ERF that
don't require GPU inference or production data to exercise honestly.

## Which parts require external APIs

- **DeepSeek** (`deepseek-v4-pro`, via `DEEPSEEK_API_KEY`) is the only external network dependency
  in the pipeline itself, used for LLM-backed editorial stages (labeling, beat-plan authoring,
  narrative assembly critique). ASR, diarization, embedding, and face stages all run local models,
  no external API calls.
- **RunPod's own REST/GraphQL API** (`RUNPOD_API_KEY`) is used only by the optional GPU-pod
  dispatch layer (`src/erf/dispatch.py`, `MASTER.md` §10), not by the pipeline stages themselves.

## Validation (this pass, 2026-08-19)

| Check | Result |
|---|---|
| Fresh venv creation (Python 3.11) | Pass |
| `pip install -e .` from a clean clone | Pass. Package has zero hard-pinned runtime deps declared in `pyproject.toml` (a real, pre-existing gap; each ML stage's deps live in its own venv, not here) |
| Full test suite, fresh environment | **233 passed, 2 failed, 2 skipped**, matching what the README already claimed; independently re-verified this pass rather than trusted at face value |
| The 2 failures | Both pre-existing and already understood, not caused by this pass: one needs real pipeline-generated catalog fixture data not shipped in a code-only export; one is a live demonstration of the documented hardcoded-root limitation in `src/erf/outcomes.py` (see README "Known limitations") |
| Real GPU (CUDA) inference, ASR, diarization, embedding, face | **Not exercised this pass**: no GPU in the validation sandbox, and the ASR loader specifically has no CPU fallback (`device="cuda"` is hardcoded, `src/erf/asr.py`) |

**Honest gap, stated plainly**: this validates that the public code installs cleanly and its
non-GPU logic is correct, on the day of this pass. It does not validate that a fresh Runpod GPU pod
can actually run a live ASR/embedding/face job end-to-end; that requires a real GPU, the model
checkpoints above, and hasn't been re-tested from a clean pod in this session. Treat this repo as
"reproducible for the code layer, documented but not re-verified for the GPU-inference layer"
until someone runs that check on real Runpod hardware.

## Licensing note: read before any commercial deployment

ERF's own code is MIT. Several of the pretrained model weights it depends on for the identity
stack are **not** permissively licensed for commercial use, most notably InsightFace's common
face-recognition packs and some diarization speaker-embedding checkpoints (CC-BY-NC-4.0). This is
not blurred here: see `THIRD_PARTY_LICENSES.md` for the full, honestly-caveated breakdown of what's
been verified versus what hasn't.

## Acknowledgments

ERF was developed and runs on GPU infrastructure provided in collaboration with Runpod.

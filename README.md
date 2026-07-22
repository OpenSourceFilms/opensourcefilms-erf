# ERF — Life® Editorial Retrieval Fabric

ERF is a self-hosted media intelligence pipeline built by Open Source Films to support the production of *Life®*, an independent feature film. It ingests raw production footage, processes it through audio, visual, and identity analysis, and exposes a queryable semantic index. Editors query in plain English and get back timestamped clips, thumbnails, and metadata that they pull into DaVinci Resolve. ERF is built for a handful of trusted in-house users — it is not a multi‑tenant product.

## Why

Behind‑the‑scenes footage from *Life®* is itself part of the final release. A small editorial team needed to navigate hundreds of hours of dailies quickly — finding specific moments by what is said, who is on screen, what is happening, or even by the emotional tone of a scene — without spending weeks logging footage manually. ERF automates that entire process, turning the production's raw media into a searchable memory that editors can query in natural language.

## Architecture

Every piece of data ERF produces falls into one of two categories: **canonical** or **derived**. Canonical data lives as explicit JSON/JSONL files under a catalog directory — that directory is the single source of truth. A vector database (LanceDB) is used for fast semantic search but is kept fully disposable: it can be rebuilt from the canonical JSON at any time with a single `reindex` command. Nothing ever lives only in the vector index.

**Core invariants enforced across all stages:**

- **Single writer per file** — concurrent writes to the same file are never safe on this filesystem, so every artifact type has exactly one writer, and every write uses write‑to‑tempfile‑then‑atomic‑rename.
- **Explicit status records** — every pipeline stage writes a status record (success or failure) before exit. Absence of a status record is an error; no silent failures.
- **Idempotent and resumable** — re‑running a completed stage on the same input always produces identical output. No cleanup is required before a re‑run.
- **Schema versions are a contract** — once data exists under a schema version, field meanings never change silently. New additive fields trigger a minor version bump; anything else forces a major version bump and a migration script.
- **Identity/speaker recognition is opt‑in per project** — it is enabled for the *Life®* production itself, but architecturally off by default.

## Pipeline stages

A high‑level overview — for every stage's input/output contract and behaviour, see `MASTER.md`.

- **Stage A** — ingest and automatic speech recognition (ASR).
- **Enrichment stages B\*** — keyframe extraction, motion analysis, scene detection.
- **Stage C** — vector materialization (derived acceleration layer; rebuildable from canonical data).
- **Stages E\* and F\*** — embeddings and face/quality analysis.
- **Speaker‑diarization and identity‑resolution spine** — takes raw ASR words through diarization, resegmentation, parallel‑stream reconciliation, adjudication, and finally produces a labelled transcript.
- **SHORT/STORY assembly** — sits on top of everything else and uses an LLM‑driven beat‑plan and critic‑review loop to generate short‑form edits and longer narrative story assemblies directly from the indexed footage.

## Tech stack

- **Language**: Python 3.11+
- **Per‑stage isolated virtual environments** — pipeline stages have different (sometimes conflicting) dependencies; each stage gets its own `venv`.
- **Core storage**: SQLite for the event/status journal; LanceDB for vector search.
- **ML dependencies**:
  - `faster-whisper` for ASR
  - `pyannote-audio` for speaker diarization
  - `Qwen3-VL-Embedding` for cross‑modal (text/image/video) semantic embeddings in a single shared vector space
  - `InsightFace` for face recognition
  - `GroundingDINO` for open‑vocabulary object detection
  - DeepSeek's API as the LLM/query‑parsing backend
- **Filesystem**: built against a distributed POSIX‑ish filesystem (MooseFS), not block storage or NFS — hence the single‑writer and atomic‑rename rules.

## Install

Python 3.11+ is required. Clone the repo and install from the root:

```
pip install -e .
```

The build backend is `hatchling` (configured in `pyproject.toml`). There is no pinned dependency lock file; the runtime dependencies are the ML libraries listed above, installed ad‑hoc inside each pipeline stage's own virtual environment. This is a known gap — no `requirements.txt` or equivalent is supplied.

You must obtain your own model checkpoints and weights; they are not distributed with this repository. If you enable identity features, you must also enrol your own identities.

## What's not included

This public release contains only the pipeline code and its architecture/schema documentation. Deliberately excluded:

- Actual production media or footage
- Biometric enrolment data (face embeddings, voiceprints, or any data tied to real enrolled people)
- Downloaded model checkpoints and weights
- Internal decision logs, roadmaps, and day‑to‑day project management documents

## Tests

`pip install -e .` then `pytest tests/` (also needs `jsonschema` and `scikit-learn`, which aren't
declared anywhere yet — see the install gap above). 233 of 237 tests pass out of the box. The 2
failures are expected consequences of this being a code-only export: one requires real
pipeline-generated catalog data that isn't shipped here, and one is a live demonstration of the
hardcoded-root limitation described below (`src/erf/outcomes.py` doesn't yet use the
`ERF_CATALOG_ROOT` seam that `src/erf/paths.py` provides, so it resolves against the original
deployment path rather than wherever you cloned this repo). 2 tests are skipped (need optional
extras not installed here).

## Known limitations

This is production pipeline code shared for its architecture and techniques, not a polished, turn‑key product. Expect to adapt paths and configuration for your own deployment.

- Many scripts under `bin/` still assume the pipeline's original deployment root path as a hardcoded default. The `src/erf/paths.py` module provides a partial seam: the catalog root can be overridden via the `ERF_CATALOG_ROOT` environment variable, but not every script has been migrated onto it yet.

## License

Open Source Films' own code in this repository is licensed under the MIT License.

ERF does **not** vendor third‑party model source code or model weights. At runtime it depends on several third‑party libraries and models whose pretrained weights come with their own licenses — some of which restrict use to non‑commercial/research only, even though the library source code is separately MIT/Apache licensed. Specifically:

- **InsightFace**: its code is MIT, but the common pretrained face‑recognition model packs (e.g. `buffalo_l`) are research/non‑commercial only.
- A diarization‑related dependency: its code is MIT, but some pretrained speaker‑embedding weights are licensed CC‑BY‑NC‑4.0.

Full details are in `THIRD_PARTY_LICENSES.md`. You must independently verify the license terms of every component you use before any commercial deployment — ERF's MIT license covers only the ERF code itself, not the runtime dependencies.

## Where to go next

- **`MASTER.md`** is the full authoritative technical specification: filesystem layout, schema contracts for every artifact type, ID conventions, the complete stage registry, and more. This README is a short orientation; `MASTER.md` is the real spec.
- **`docs/`** contains supplementary design documents (observability/logging design, narrative story‑engine design, an evaluation‑harness design, and research briefs comparing model choices).

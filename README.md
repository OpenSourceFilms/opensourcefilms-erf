# ERF: Editorial Retrieval Fabric

ERF is a self-hosted media intelligence pipeline built by Open Source Films to support an independent feature film production. It ingests raw production footage, processes it through audio, visual, and identity analysis, and exposes a queryable semantic index. Editors query in plain English and get back timestamped clips, thumbnails, and metadata that they pull into DaVinci Resolve. ERF is built for a handful of trusted in-house users; it is not a multi‑tenant product.

## Why

Behind‑the‑scenes footage from the production is itself part of the final release. A small editorial team needed to navigate hundreds of hours of dailies quickly (finding specific moments by what is said, who is on screen, what is happening, or even by the emotional tone of a scene) without spending weeks logging footage manually. ERF automates that entire process, turning the production's raw media into a searchable memory that editors can query in natural language.

## Architecture

Every piece of data ERF produces falls into one of two categories: **canonical** or **derived**. Canonical data lives as explicit JSON/JSONL files under a catalog directory; that directory is the single source of truth. A vector database (LanceDB) is used for fast semantic search but is kept fully disposable: it can be rebuilt from the canonical JSON at any time with a single `reindex` command. Nothing ever lives only in the vector index.

**Core invariants enforced across all stages:**

- **Single writer per file**: concurrent writes to the same file are never safe on this filesystem, so every artifact type has exactly one writer, and every write uses write‑to‑tempfile‑then‑atomic‑rename.
- **Explicit status records**: every pipeline stage writes a status record (success or failure) before exit. Absence of a status record is an error; no silent failures.
- **Idempotent and resumable**: re‑running a completed stage on the same input always produces identical output. No cleanup is required before a re‑run.
- **Schema versions are a contract**: once data exists under a schema version, field meanings never change silently. New additive fields trigger a minor version bump; anything else forces a major version bump and a migration script.
- **Identity/speaker recognition is opt‑in per project**: it is enabled for this production itself, but architecturally off by default.

## Pipeline stages

A high‑level overview: for every stage's input/output contract and behaviour, see `MASTER.md`.

- **Stage A**: ingest and automatic speech recognition (ASR).
- **Enrichment stages B\***: keyframe extraction, motion analysis, scene detection.
- **Stage C**: vector materialization (derived acceleration layer; rebuildable from canonical data).
- **Stages E\* and F\***: embeddings and face/quality analysis.
- **Speaker‑diarization and identity‑resolution spine**: takes raw ASR words through diarization, resegmentation, parallel‑stream reconciliation, adjudication, and finally produces a labelled transcript.
- **SHORT/STORY assembly**: sits on top of everything else and uses an LLM‑driven beat‑plan and critic‑review loop to generate short‑form edits and longer narrative story assemblies directly from the indexed footage.

## Tech stack

- **Language**: Python 3.11+
- **Per‑stage isolated virtual environments**: pipeline stages have different (sometimes conflicting) dependencies; each stage gets its own `venv`.
- **Core storage**: SQLite for the event/status journal; LanceDB for vector search.
- **ML dependencies**:
  - `faster-whisper` for ASR
  - `pyannote-audio` for speaker diarization
  - `Qwen3-VL-Embedding` for cross‑modal (text/image/video) semantic embeddings in a single shared vector space
  - `InsightFace` for face recognition
  - `GroundingDINO` for open‑vocabulary object detection
  - DeepSeek's API as the LLM/query‑parsing backend
- **Filesystem**: built against a distributed POSIX‑ish filesystem (MooseFS), not block storage or NFS; hence the single‑writer and atomic‑rename rules.

## Install

Python 3.11+ is required. Clone the repo and install from the root:

```
pip install -e .
```

The build backend is `hatchling` (configured in `pyproject.toml`). There is no pinned dependency lock file; the runtime dependencies are the ML libraries listed above, installed ad‑hoc inside each pipeline stage's own virtual environment. This is a known gap; no `requirements.txt` or equivalent is supplied.

You must obtain your own model checkpoints and weights; they are not distributed with this repository. If you enable identity features, you must also enrol your own identities.

## Run on Runpod

ERF was developed and runs in production on Runpod GPU pods. `RUNPOD.md` documents the recommended
GPU class/VRAM tiers per stage role, the per-stage venv split (ERF deliberately has no single
`requirements.txt`, since its ML stages have conflicting dependency sets), required environment
variables (names only, never values), model/checkpoint requirements, and which parts need external
APIs (DeepSeek for LLM-backed stages). It also states plainly what was and wasn't re-validated in
its most recent pass: the code-only test suite was, live GPU inference wasn't (no GPU was available
to validate against).

## What's not included

This public release contains only the pipeline code and its architecture/schema documentation. Deliberately excluded:

- Actual production media or footage
- Biometric enrolment data (face embeddings, voiceprints, or any data tied to real enrolled people)
- Downloaded model checkpoints and weights
- Internal decision logs, roadmaps, and day‑to‑day project management documents

## Tests

`pip install -e .` then `pytest tests/` (also needs `jsonschema` and `scikit-learn`, which aren't
declared anywhere yet; see the install gap above). 233 of 237 tests pass out of the box. The 2
failures in the recorded public-release test run were associated with missing real
pipeline-generated catalog data and incomplete deployment-root portability. Those are
historical results, not a fresh regression-test verdict. `src/erf/outcomes.py` now resolves its default catalog/journal under the
checked-out repository and supports `ERF_CATALOG_ROOT` / `ERF_JOURNAL_DB` for
explicit redirection. Other modules and scripts may still assume the original
filesystem layout. 2 tests were skipped in that recorded run (optional extras unavailable).

## Known limitations

This is production pipeline code shared for its architecture and techniques, not a polished, turn‑key product. Expect to adapt paths and configuration for your own deployment.

- Many scripts under `bin/` still assume the pipeline's original deployment root path as a hardcoded default. The `src/erf/paths.py` module provides a partial seam: the catalog root can be overridden via the `ERF_CATALOG_ROOT` environment variable, but not every script has been migrated onto it yet.
- For isolated outcome-journal experiments, set both `ERF_CATALOG_ROOT` and `ERF_JOURNAL_DB` **before importing** `erf.outcomes`. The module binds its defaults at import time. An overridden catalog alone does not redirect the SQLite journal. Use disposable paths and never point public smoke tests at a working production journal.

Per-media outcome files now reject `media_id` values containing path
separators, control characters or empty/dot-only names, and refuse an already
symlinked status file. The validation runs **before** the authoritative
journal insert, so an invalid filename cannot leave a successful new outcome
event without its status label. Normal deterministic media IDs are
unchanged. These are **local filename boundary checks**, not a guarantee
against a hostile process concurrently replacing files or directories, and
not a security assessment of the deployed ERF web/search service.

A separate standard-library test can verify safe outcome/journal paths without
installing model dependencies:

```bash
python3 -m unittest discover -s tests -p 'test_outcomes_portability_unittest.py' -v
```

This test writes only to temporary paths and does not require the production
footage catalog or GPUs.

To explore the event journal without any models, try the
[disposable outcome-journal example](docs/try-outcome-journal.md). It records
synthetic `ok` and `empty` outcomes, confirms projection behavior and removes
its temporary database automatically. GitHub Actions runs this example and
the isolated portability tests on relevant changes.

## Search example with timestamped results and synthetic thumbnails

Try the [runnable footage-search demonstration](docs/try-footage-search.md).
It invokes ERF's **actual SQLite FTS5 indexing and lexical search code** over
three invented transcript chunks, returning evidence-grounded snippets,
media IDs and in/out timecodes. A generated standalone HTML report shows
timestamped matches beside **clearly synthetic SVG illustrations**.

Run `python3 -B examples/fts_search_demo.py` to see the default
two-result `steam` query, or specify `--html /tmp/erf-fts-synthetic.html`
to create a self-contained browser preview. No GPU, model checkpoints, real
filming media, installed third-party Python packages or live services are
needed. This is **lexical retrieval only**, not the private multi-modal
semantic-search interface or a genuine video-thumbnail/clip extract.

## License

Open Source Films' own code in this repository is licensed under the MIT License.

ERF does **not** vendor third‑party model source code or model weights. At runtime it depends on several third‑party libraries and models whose pretrained weights come with their own licenses; some of which restrict use to non‑commercial/research only, even though the library source code is separately MIT/Apache licensed. Specifically:

- **InsightFace**: its code is MIT, but the common pretrained face‑recognition model packs (e.g. `buffalo_l`) are research/non‑commercial only.
- A diarization‑related dependency: its code is MIT, but some pretrained speaker‑embedding weights are licensed CC‑BY‑NC‑4.0.

Full details are in `THIRD_PARTY_LICENSES.md`. You must independently verify the license terms of every component you use before any commercial deployment; ERF's MIT license covers only the ERF code itself, not the runtime dependencies.

## Acknowledgments

ERF runs on GPU infrastructure provided in collaboration with Runpod. Their on-demand pricing is
what makes it practical to run a pipeline this GPU-heavy for a small independent production.

## Where to go next

- **`MASTER.md`** is the full authoritative technical specification: filesystem layout, schema contracts for every artifact type, ID conventions, the complete stage registry, and more. This README is a short orientation; `MASTER.md` is the real spec.
- **`docs/`** contains supplementary design documents (observability/logging design, narrative story‑engine design, an evaluation‑harness design, and research briefs comparing model choices).

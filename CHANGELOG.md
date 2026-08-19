# Changelog

## [Unreleased] - 2026-08-19

Deployment-presence pass: a tested Runpod setup path for the public repo's code-only surface. No
functional or architectural changes to the pipeline itself.

### Added

- `RUNPOD.md`: recommended GPU class/VRAM per stage role, the per-stage venv split (audio/embed/
  control), required environment variables by name only, model/checkpoint requirements, which
  parts require external APIs, and an honest validation record.
- "Run on Runpod" section in `README.md` pointing to it.

### Notes

- The full test suite was independently re-run in a fresh venv against this repo as-is: 233
  passed, 2 failed, 2 skipped, matching the count the README already stated, now verified rather
  than asserted.
- Live GPU inference (ASR, diarization, embedding, face) was not exercised this pass: no GPU was
  available in the validation environment, and `src/erf/asr.py`'s Whisper loader has no CPU
  fallback (`device="cuda"` is hardcoded). Stated explicitly in `RUNPOD.md` rather than left
  implicit.

## [0.1.0] - 2026-07-22

Initial release.

### Added

- Pipeline code (`bin/`, `src/erf/`): ingest/ASR, visual and audio enrichment, speaker
  diarization and identity resolution, vector materialization, and SHORT/STORY assembly stages.
- `schemas/`: JSON Schema contracts for every canonical artifact type.
- `tests/`: unit and integration test suite.
- `MASTER.md`: the authoritative technical specification (filesystem layout, schema contracts,
  ID conventions, full stage registry).
- `docs/`: supplementary design documents: observability/logging redesign, narrative
  story-engine design, evaluation-harness design, and model-choice research briefs.
- `LICENSE` (MIT) and `THIRD_PARTY_LICENSES.md`.

### Curated for public release

- Excluded all biometric enrollment data, real production media, model weights/checkpoints, and
  internal decision-log/roadmap documents (none of these were ever tracked in this project's git
  history, verified before release).
- Removed a script whose entire purpose was mining enrollment candidates for named individuals,
  and a supplemental-labels filename that referenced two people by name.
- Removed a hardcoded per-person subtitle color palette (functional code, not just a comment): the pipeline already falls back to a deterministic palette for any person without an explicit
  color, so this is a no-op change in behavior for anyone adopting the code fresh.
- Generalized internal decision-attribution references ("Daniel signs off") across `MASTER.md` and
  several `docs/` files to generic role language, without altering any technical content.

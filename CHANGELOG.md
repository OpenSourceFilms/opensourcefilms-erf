# Changelog

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

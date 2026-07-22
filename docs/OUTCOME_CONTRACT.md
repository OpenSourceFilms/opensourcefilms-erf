# OUTCOME_CONTRACT.md — what every stage must declare

Status: **v0.2, LARGELY IMPLEMENTED** (updated 2026-07-10 from code audit; was "DRAFT v0.1 (2026-06-23)").
Retrofit reality: **15 of the 24 table stages record fully** (explicit `outcomes.record` on every exit
path), **8 record partially** (only the automatic binary ok/error via `journal.reap_manifest` — B2, B3,
B4, P, Q0, tags, F0, F1 — missing the finer skip/empty reason vocab), **1 (RETRANSCRIBE) never records
under its own name** (its inner `asr.run()` records as "A"). Nine live stages were missing from the
table entirely and are now listed (§4b). Ledger record: D-Log-05.
Companion to MASTER.md §5.2.1 and `src/erf/outcomes.py`. **Successor:** this doc folds into the
`src/erf/stages.py` registry when `docs/OBSERVABILITY_REDESIGN.md` (D-Log-04) Phase 0 lands.

## 1. Principle — designed backward from consumers

Each stage's log is designed by asking **"what do the stages after me currently re-read
from raw artifacts, or infer from file presence, that I should hand them explicitly?"**
Not "what is convenient to emit here." A field is REQUIRED only when a *named* downstream
consumer (another stage, the query/render layer, or the audit) reads it. `counts` stays an
open dict for genuinely future needs — we do not spec speculative fields.

Every stage, on **every exit path**, calls exactly:
```
outcomes.record(media_id, stage, status, *, reason, counts, artifacts, producer)
outcomes.set_flag(media_id, name, value, *, reason)   # for media-level facts
```

## 2. Standard envelope (every outcome carries)

| field | meaning |
|-------|---------|
| `status` | `ok` \| `skip` \| `empty` \| `error` \| `partial` |
| `reason` | controlled string; **required** for skip/empty/error |
| `counts` | stage-specific quantities (open dict) |
| `artifacts` | paths written (so consumers/audit don't guess) |
| `producer` | code identity (sha/run_id) — ties to `artifact_fingerprint` |
| `at_utc`, `run_id` | set by the spine |

`status` semantics: **ok** = produced its normal output; **empty/skip** = ran to a definite
conclusion that legitimately produced nothing, with `reason` (e.g. `no_speech`); **partial** =
some media-units done, some failed (carry counts); **error** = failed, `reason` = cause.
`ok/skip/empty` ⇒ `status_projection.ok=1` (reached a conclusion). Zero-byte artifacts are
never written — an empty result is a `skip`/`empty` outcome.

## 3. Global reason vocabulary (extend per stage)

`no_speech`, `no_faces`, `no_scenes`, `no_chunks`, `no_words`, `no_events`, `no_keyframes`,
`no_transcript`, `no_evidence`, `below_threshold`, `missing_dep:<stage|artifact>`,
`unsupported_media`, `corrupt_media`, `encode_failed`, `model_unavailable`.

## 4. Per-stage contract (the table)

Legend: **flags** = media-level facts set via `set_flag` (cross-stage truth); **consumed-by**
lists who reads the emitted facts. `[v]` = consumer relationship to verify against code during retrofit.

| stage | status vocab | reason vocab | required counts | emitted flags | consumed-by |
|-------|-------------|-------------|-----------------|---------------|-------------|
| **A** (ingest+ASR+diar) | ok, empty, error | no_audio_stream, asr_empty, corrupt_media | words, segments, turns, duration_ms | `has_speech`, `n_speakers`, `has_diar`, `dominant_lang` | B0, B2, P, F0, T0, P0, REDIARIZE, EV |
| **B0** (chunk) | ok, skip | no_speech, missing_dep:A | search_chunks, shortform_chunks | `has_chunks` | B1, E0, C, query |
| **B1** (text embed) | ok, skip | no_chunks | vectors, dim | `has_text_vec` | C, query, E2 |
| **B2** (scene) | ok, empty | (none / single_shot) | scenes | `has_scenes`, `is_single_shot` | B3, E0 |
| **B3** (keyframe) | ok, skip | no_scenes | keyframes | `has_keyframes` | B4, P, F0, V0 |
| **B4** (visual embed) | ok, skip | no_keyframes | vectors, dim | `has_visual_vec` | C, query |
| **C** (materialise) | ok, partial | missing_dep:B1\|B4 | text_rows, visual_rows | — | query (index) |
| **E0** (events) | ok, empty, skip | no_chunks, no_scenes | events, beats, broll | `has_events` | E1, CARD, EV |
| **P** (proxy) | ok, error | encode_failed, missing_dep:B3 | duration_ms, width, height | `proxy_ready` | Q0, F0, F2, render |
| **Q0** (quality) | ok | missing_dep:P | (qc metrics) | `audio_music_advisory`, `quality_band` | render, EV [v] |
| **tags** | ok, empty | (none) | tags | — | query [v] |
| **F0** (faces) | ok, empty | no_faces, missing_dep:B3 | detections, face_tracks | `has_faces`, `n_persons_visible` | F1, F2, V0 |
| **F1** (identity) | ok, empty | no_faces, no_diar | speakers_resolved, faces_resolved, persons | `has_identity` | EV, render, voice_enroll |
| **F2** (ASD) | ok, skip | no_face_tracks | speaking_segments | `has_asd` | render |
| **RETRANSCRIBE** | ok, empty, error | asr_empty, corrupt_media | words, segments, turns | `has_speech`, `n_speakers` | (supersedes A facts) |
| **REDIARIZE** | ok, skip | no_speech | turns, speakers | `has_diar_embeddings`, `n_speakers` | F1, voice_enroll, render |
| **D0** (wx+vad) | ok, skip | no_speech | vad_segments, aligned_words | `has_vad`, `has_wx_align` | T0, render |
| **P0** (sentences) | ok, skip | no_words | sentences | `has_sentences` | T0, EV, render |
| **T0** (transcript) | ok, skip | no_words | words, sentences, speakers | `has_transcript` | EV, render |
| **V0** (visual grounding) | ok, skip | no_keyframes | grounded_keyframes | `has_visual_grounding` | EV, CARD, query |
| **CARD** (media card) | ok, skip | no_events | (1) | `has_card` | EV, digest |
| **E1** (event signals) | ok, skip | no_events | signals, hooks, quotes | `has_event_signals` | EV, E2 |
| **EV** (evidence) | ok, skip | no_transcript, no_events | evidence_units | `has_evidence` | LABEL, story engine |
| **LABEL** (editorial) | ok, skip | no_evidence | sentences_labeled, story_forces, archetypes, tensions | `has_editorial` | story engine, query |

### 4b. Stages recording in code but absent from the original table (added 2026-07-10, D-Log-05)

Verified against code; vocab below is what the code actually emits today — extend deliberately, not speculatively.

| stage | file | recording status | notes |
|-------|------|------------------|-------|
| **W** (asr words) | `bin/build_asr_words.py` | full | who's-speaking spine head; skip on no-source (MASTER §5.1.1) |
| **RG** (recall gaps) | `bin/build_recall_gaps.py` | full | always-writes → ok |
| **RP** (recall pass) | `bin/recall_pass.py` | full | records inside `process_media` |
| **DX** (community-1 diar) | `bin/run_diar_community1.py` | full | error on extract-fail |
| **SEV2** (speech events v2) | `bin/build_speech_events_v2.py` | full | |
| **SEV2_ADJ** (adjudication) | `bin/adjudicate_speech_events.py` | full | |
| **PSTREAM** (parallel streams) | `bin/build_parallel_streams.py` | full | |
| **B5** (OCR) | `src/erf/ocr.py` | **defective** | ok-path auto-records; its two skip paths (privacy gate, feature gate) write raw journal events, bypassing `outcomes.record` — exactly the silent-outcome class §5 exists to catch. Fix folded into D-Log-04 Phase 1. |
| **C_vis** (visual materialise) | `src/erf/materialise.py` | partial | explicit skip for empty-embeddings; no explicit path for `has_visual_vec`-false short-circuit |

## 5. Validator (closes the silent-stage gap)

`bin/reconcile_status.py --audit-outcomes`:
- For every (media, expected-stage): assert an explicit outcome exists. Flag **silent** =
  artifact-on-disk-without-recorded-outcome, OR ran-without-record.
- Assert required `counts`/`flags` present for `ok` outcomes; flag schema gaps.
- Assert `reason` ∈ the stage's vocab for skip/empty/error.
- Cross-check flags against artifacts (e.g. `has_speech=False` ⇒ no chunks file; `has_faces=True`
  ⇒ face_tracks present) — contradictions are defects.

## 6. Retrofit checklist (per stage)

1. Identify every exit path in `run()` (success, each early-return/skip, each caught error).
2. Replace ad-hoc journaling with one `outcomes.record(...)` per exit, correct status+reason.
3. Set the emitted flags via `set_flag`.
4. Replace any upstream **inference** (reading a file to test presence) with
   `outcomes.is_complete(...)` / `outcomes.flag(...)`.
5. Add the stage's reason vocab to this doc if extended.
6. Validate with `--audit-outcomes` on a 2–3 media sample (incl. one no-speech, one no-faces).

## 7. Open questions — RESOLVED 2026-07-10 (code-verified, D-Log-05)

- **Q0/tags consumers `[v]`** — RESOLVED, split verdict. `tags → query`: **confirmed consumed**
  (`src/erf/query.py` joins tag scores by chunk_id). `Q0 → render/EV`: **NOT consumed** — zero
  references to `audio_music_advisory`/`quality_band` outside their writer; per this doc's own rule
  those flags are **optional**, not required. (`reconcile_status.py` has presence predicates for both,
  but `pipeline_status.py` surfaces neither — a read-contract gap for D-Log-04 to close.)
- **C/index stages** — unchanged: excluded from zero-byte/artifact-presence checks (already so in reconcile).
- **RETRANSCRIBE vs A** — RESOLVED by observed behavior: RETRANSCRIBE never records under its own
  name; its inner `asr.run()` records as "A". The fold-in has de facto happened. Formalized: RETRANSCRIBE
  is an *invocation mode of A*, not an outcomes stage — its row in §4 is retained for vocab reference only;
  the D-Log-04 registry must not list it as a distinct stage.

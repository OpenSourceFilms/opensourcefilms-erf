# Life® Editorial Retrieval Fabric — MASTER.md

**Status:** authoritative build contract. Supersedes the prior `/workspace/pipeline/` system and the earlier B00–B46 ERF brief sequence (now void).
**Version:** 1.5.0
**Date:** 2026-07-10
**Primary reader:** Claude Code (any model). **Secondary:** the project lead.
**Entry point:** read `README.md` (the master spine) first — it orients and routes; this document is the build contract it points to.
**Authority:** Every decision here is final until a new Opus brief supersedes it. Where this document conflicts with the commissioning brief, this document wins — see §0.1 for the deviations and why.

---

## 0. How to read this document

This is a contract, not a discussion. Implement exactly what is written. Where a value must be confirmed against on-disk reality before it hardens, the document says so explicitly and gates it behind a Phase verification step. Nothing in a schema contract was guessed: dimensions and paths below were read off `/workspace` on 2026-06-01.

### 0.1 Deviations from the commissioning brief (decided against verified disk reality)

| # | Brief said | MASTER says | Why |
|---|------------|-------------|-----|
| D1 | ASR = faster-whisper **large-v3-turbo** | faster-whisper **large-v3** | large-v3 is already on disk (CTranslate2, Systran). turbo is not present and is lower-accuracy. No reason to fetch a worse model. turbo is a documented speed lever (§19). |
| D2 | **WhisperX** word alignment in core | **Dropped from core.** faster-whisper native `word_timestamps` + pyannote standalone + overlap reconciliation | `pyannote-audio 4.0.4` is installed; WhisperX pins older pyannote and will fight it. Editorial clip-finding does not need phoneme-perfect boundaries. wav2vec2 forced alignment is a precision lever (§19), not core. |
| D3 | **OpenCLIP** as the cross-modal bridge (Phase 2) | **Dropped from core.** Qwen3-VL-Embedding-8B is the bridge | Disk confirms Qwen3-VL-Embedding-8B is a genuine multimodal embedding model: text, image, and video frames land in **one** 4096-d space. It does text→image natively and better than ViT-L/14. OpenCLIP (installed in `cpu_vision`) remains a CPU-only fallback lever (§19). *(Amendment A1: originally wrote 2B/2048; corrected to 8B/4096 as confirmed on disk.)* |
| D4 | **YOLOv12** object detection in Phase 2 | **Deferred out of core.** GroundingDINO (open-vocab, Apache-2.0) as on-demand enrichment | Cross-modal frame search (Qwen3-VL) covers "find shots with X" for editorial discovery. Fixed 80-class COCO tags add little. Bounding boxes matter only for Phase 5 auto-framing; `detection` record stays in schema as a hook, populated later. YOLOv12 weights are not on disk and are not fetched. |
| D5 | Query LLM = **claude-haiku-4-5** (Anthropic API) | **DeepSeek Flash API primary, no fallback** | Hard-constraint #1 forbids subscription/external APIs; the brief's own §6 then mandated one. Resolved toward cheap-metered. DeepSeek Flash is fast and effectively free at query-parse token volumes. Ollama removed 2026-06-22 — never operational, motion not progress. LLM backend is pluggable (§13). |

### 0.3 Amendment v1.4.0 — 2026-06-06

| # | Change |
|---|--------|
| A1 | **Visual model corrections:** all residual `Qwen3-VL-Embedding-2B` / `qwen3vl-2b` / `2048` references purged. On-disk model is `Qwen3-VL-Embedding-8B` at `/workspace/weights/qwen3/Qwen3-VL-Embedding-8B/`; G-VISUAL-EMBED confirmed dim = **4096**. All dim references are registry-driven (see `state/model_registry.json`). |
| A2 | **Quality schema:** current version is **1.2.0** (`music_ratio`/`music_risk` fields removed per D-Q0-04; 9/10 false positives on BTS corpus). Becomes 1.3.0 after PANNs audio classifier (Step 3, Phase 4). |
| A3 | **Phase 4/5-foundation brief active:** `docs/_archive/PHASE_4_BRIEF.md` (from Opus, 2026-06-06) governs all Phase 4/5 implementation. Supersedes items in NEXT_STEPS.md §2. *(both archived 2026-07-10, D-Doc-02)* |
| A4 | **Craft and compounding roadmap:** `docs/_archive/ROADMAP.md` — long-horizon vision document *(archived 2026-07-10; its one live gap, Wave 3a, is tracked in GAMEPLAN W3)*. |

### 0.3.1 Amendment v1.5.0 — 2026-07-10 (currency sweep; revives the amendment mechanism, unused since A4 despite live §5.1/§5.1.1/§5.2 edits through 2026-07-05)

| # | Change |
|---|--------|
| A5 | **Session-state surface is `GAMEPLAN.md`, not CONTEXT.md** (D-Doc-01 formalizes drift that began ~2026-06-12). §4/§15/§18 updated; CONTEXT.md carries a retirement banner and is a historical build log only. |
| A6 | **§5.2 observability contract redesign landed (D-Log-04, Phases 0–4, `D-Log-06`…`D-Log-09` + this Phase 4).** `docs/OBSERVABILITY_REDESIGN.md` is now the design record of how it landed, not a pending target; §5.2 is rewritten to the single-surface, registry-driven contract it describes. |
| A7 | **§11 enrollment workflow marked historical:** identity was built and rebuilt (D-Identity-01…06, D-Arch-01); the operative spec is §5.1.1 + the voiceprint_v2 resolver. Face-enrollment mechanics remain current. |
| A8 | **§13 LLM backend list corrected:** `deepseek-v4-pro` added (default for LABEL/EV/SURVEY/STORY per §5.1 — §13 previously contradicted §5.1 in the same document). |
| A9 | **§20 "Unresolved" closed out** — all four items resolved in practice; section retitled historical. |
| A10 | **§0.1 broken cross-refs fixed** (levers moved §16→§19 in the current numbering). Decision-ID collisions in `DECISIONS.md` resolved with a generated ID index + uniqueness guard (D-Meta-01). |

### 0.2 Verified environment (read 2026-06-01)

| Property | Value | Consequence |
|----------|-------|-------------|
| Datacenter | `EU-RO-1` | All pods pin here. GPU availability preflighted here only. |
| Network volume id | `cmkgi43hcx` | Attached to every pod at `/workspace`. |
| Filesystem | **MooseFS** (`mfs#euro-2.runpod.net:9421`), 2.1 PB, 75% used | Distributed FUSE POSIX-ish. **Not** NFS, **not** block storage. Drives the SQLite journal-mode and single-writer rules (§8). Storage is effectively unbounded — do not optimise for space. |
| Current pod | RTX PRO 4000 Blackwell (sm_120), 12 vCPU, 31 GB | Blackwell ⇒ embedding venv needs **torch ≥ 2.8 cu128**; the on-disk torch 2.5.1 venv will not run here. |
| `Qwen3-Embedding-8B` | `/workspace/weights/qwen3/Qwen3-Embedding-8B/`, `Qwen3ForCausalLM`, hidden 4096, ST pooling head present | Text collection dim = **4096**, cosine. |
| `Qwen3-VL-Embedding-8B` | `/workspace/weights/qwen3/Qwen3-VL-Embedding-8B/`, `Qwen3VLForConditionalGeneration` + ST pooling, hidden 4096, video preprocessor present | Visual collection dim = **4096**, cosine. Cross-modal. *(Amendment A1: 2B superseded; 8B confirmed on disk.)* |
| ASR weights | `faster-whisper-large-v3` cached under `/workspace/pipeline/models/whispercpp/` | No ASR fetch needed. |
| Diarization | `pyannote-audio 4.0.4` full suite in `/workspace/.pod-persist/venvs/pipeline-v2/` | Use standalone. Confirm pipeline weights cached at Phase-1 gate. |
| Face | InsightFace `buffalo_l` ONNX set on disk (`det_10g`, `w600k_r50`, etc.) | Face embed dim = 512 (Phase 3). |
| HF token | cached at `/root/.cache/huggingface/token` (not exported) | Gated pyannote pulls work; export to env in worker boot if a fetch complains. |
| Ollama | **removed** | Deleted 2026-06-22. Was never operational; DeepSeek Flash is the sole LLM backend. |

---

## 1. Project identity

**Life® Editorial Retrieval Fabric (ERF).** A self-hosted media intelligence pipeline on a RunPod MooseFS persistent volume (`cmkgi43hcx`, EU-RO-1). It ingests raw production media for the not-for-profit feature *Life®*, processes it through audio, visual, and (later) identity analysis, and produces a queryable semantic index. In-house editors query in plain English and receive timestamped clips, thumbnails, and metadata they pull into DaVinci Resolve. A handful of trusted users. Not a product. Must not be a hassle to operate.

---

## 2. Non-goals

This system will never:

- Be a commercial product, multi-tenant, or exposed to the public internet beyond the existing authenticated Copyparty/Cloudflare tunnel.
- Do real-time or live ingest. Batch only.
- Be a general-purpose MAM or asset-management replacement.
- Transcode or render deliverables. Resolve owns editing and rendering.
- Train or fine-tune models. Inference only.
- Auto-publish anything anywhere.
- Treat the vector index as a source of truth (it is disposable — §3).
- Use multi-writer access to any single file (MooseFS forbids it safely — §8).

---

## 3. Invariants (the prime directives)

1. **Catalog is canonical; the vector index is disposable.** Every derived artefact lives in `/workspace/erf/catalog/**` as JSON/JSONL. The LanceDB tables are rebuildable from catalog by one command (`erf-reindex`) at any time. Never store anything in LanceDB that is not reconstructable from catalog.
2. **Single writer per file, always.** MooseFS permits concurrent attach but not safe concurrent writes to one file. The control plane is the only writer of the relational DB and the LanceDB tables. GPU workers write only to their own per-worker append-only paths. Every write is `write-tmp + atomic rename`.
3. **Every stage writes a status record before exit** — success or failure. No silent failures. A crashed stage that wrote no status is treated as `error` on reap.
4. **Stages are idempotent and resumable.** Re-running a completed stage on the same input reproduces identical output (same IDs, same content) and is safe. Determinism is enforced by the ID conventions in §6.
5. **Schema version is a contract.** Once data exists under a schema version, fields are not silently changed. Additive nullable fields ⇒ MINOR bump. Any change to a field's type, meaning, required-set, an ID convention, a vector dim/metric, or a table name ⇒ MAJOR bump + migration script + a new Opus brief (§15).
6. **Identity is opt-in per project, on for Life®.** The architecture supports running with identity off; Phase 3 fields exist in schema from day one but stay null until enrolled.
7. **Source-type axis.** Every media_manifest carries a `kind` field (`raw_footage` | `published_output` | `audience_text`). Stage A defaults all new ingests to `raw_footage`. Non-footage sources (finished edits, audience text) ride the same ingest/journal spine — they do not require a parallel system.
8. **Overlay-facet pattern for human-authored annotations.** Any human-authored annotation is an overlay facet written to `catalog/overlays/{facet_type}/{media_id}.json`, never a rewrite of an existing stage artifact. `identity_overlay` is the reference implementation of this pattern. Atoms are not immutable-final, but any facet layered over them must be machine-distinguishable from the canonical stage output.

---

## 4. Filesystem layout

Complete tree under `/workspace/erf/`. Create exactly this in Phase 0.

```
/workspace/erf/
├── MASTER.md                      # this document (committed Phase 0)
├── docs/_archive/CONTEXT.md       # HISTORICAL build log (retired ~2026-06-12; session state lives in GAMEPLAN.md — A5/D-Doc-01; archived D-Doc-02)
├── DECISIONS.md                   # append-only decision log; updated every session
├── schemas/                       # JSON Schema (draft 2020-12), one file per record type
│   ├── media_manifest.schema.json
│   ├── asr_word.schema.json
│   ├── chunk.schema.json
│   ├── embedding.schema.json
│   ├── scene.schema.json
│   ├── keyframe.schema.json
│   ├── detection.schema.json
│   ├── face_track.schema.json
│   ├── person.schema.json
│   ├── event.schema.json
│   ├── entity_node.schema.json
│   ├── status.schema.json
│   └── export_manifest.schema.json
├── catalog/                       # CANONICAL derived artefacts (JSON/JSONL)
│   ├── media_manifest/            # {media_id}.json
│   ├── asr/                       # {media_id}.words.jsonl, {media_id}.segments.jsonl, {media_id}.diar.jsonl
│   ├── chunks/                    # {media_id}.jsonl
│   ├── embeddings/                # {media_id}.text.jsonl, {media_id}.visual.jsonl  (vectors inline as float arrays)
│   ├── scenes/                    # {media_id}.jsonl
│   ├── keyframes/                 # {media_id}.jsonl  (+ image files under media/keyframes — see below)
│   ├── detections/                # {media_id}.jsonl   (Phase 3+; empty until then)
│   ├── face_tracks/               # {media_id}.jsonl   (Phase 3+)
│   ├── persons/                   # {person_id}.json   (enrollment records; Phase 3)
│   ├── events/                    # {media_id}.jsonl
│   ├── entity_nodes/              # {entity_node_id}.json (Phase 3)
│   ├── status/                    # {media_id}.json  (per-media, per-stage status map)
│   └── exports/                   # {export_id}.json + emitted FCPXML/EDL/OTIO (Phase 4)
├── media/                         # binary derivatives (NOT canonical metadata, but not re-derived casually)
│   ├── keyframes/{media_id}/{keyframe_id}.jpg
│   ├── thumbs/{media_id}/{keyframe_id}.webp        # small grid thumbnails
│   └── proxies/{media_id}.mp4                       # editorial proxy (stage P)
├── index/                         # DISPOSABLE — LanceDB store; rebuildable from catalog
│   └── lance/                     # text_v1.lance, visual_v1.lance, persons_v1.lance
├── state/                         # control-plane private state
│   ├── erf.db                     # SQLite (control plane is sole writer)
│   ├── workers/{pod_id}/          # per-worker dirs: assignment.jsonl, heartbeat.json, done.jsonl
│   └── storage_probe.json         # Phase-0 MooseFS capability probe result
├── enroll/                        # identity enrollment inputs (Phase 3)
│   └── {person_id}/{faces,voice}/ # reference images / audio
├── weights/                       # symlinks to /workspace/weights/** actually used by ERF
├── venvs/                         # ERF-owned venvs (see §17)
│   ├── control/                   # CPU: fastapi, lancedb, pyscenedetect, faster-whisper(cpu)
│   ├── audio/                     # GPU: faster-whisper, pyannote-audio 4.0.4
│   └── embed/                     # GPU: torch>=2.8 cu128, sentence-transformers, transformers
├── src/erf/                       # the package (see §17 for module map)
├── bin/                           # entrypoints: erf-stage, erf-reindex, erf-query, erf-dispatch, erf-enroll
└── tests/                         # contract validators + synthetic-data tests
```

Rule: anything under `index/` and `media/` may be deleted and rebuilt. Anything under `catalog/`, `state/erf.db`, `enroll/`, and the `schemas/` is precious. Back nothing else up.

---

## 5. Stage registry

Stage letters are an ERF convention carried forward from the prior pipeline. The prior visual stages were never implemented, so the visual letters are defined cleanly here. Letters are stable identifiers, not a strict execution total-order; the **depends-on** column is the real DAG.

| Letter | Name | Tool (version) | Input | Output | Tier | Depends-on | Phase |
|--------|------|----------------|-------|--------|------|-----------|-------|
| **A** | Ingest + ASR + Diarize | ffprobe; faster-whisper 1.2.x (large-v3); pyannote-audio 4.0.4 | source media file | `media_manifest`, `asr_word`, `asr_segment`, diarization turns | GPU-preferred / CPU-trickle | — | 1 |
| **B0** | Chunk | erf (rules) | asr words+segments+diar | `chunk` | CPU | A | 1 |
| **B1** | Text embed | sentence-transformers + Qwen3-Embedding-8B | chunk text | `embedding`(text) | **GPU** | B0 | 1 |
| **B2** | Scene detect | PySceneDetect 0.6.4 (ContentDetector) | source media | `scene` | CPU | A | 2 |
| **B3** | Keyframe extract | ffmpeg | scenes + media | `keyframe` (+ jpg/webp) | CPU | B2 | 2 |
| **B4** | Visual embed | sentence-transformers + Qwen3-VL-Embedding-8B | keyframe images | `embedding`(visual) | **GPU** | B3 | 2 |
| **C** | Vector materialise | LanceDB | embeddings (text+visual) | LanceDB `text_v1`,`visual_v1` | CPU | B1, B4 | 1/2 |
| **E0** | Events / beats | erf (+ optional LLM) | chunks + scenes | `event` (beat, b_roll, quality hooks) | CPU | B0, B2 | 2 (skeleton), 5 (full) |
| **P** | Proxy + thumbs | ffmpeg | source media + keyframes | proxy mp4, webp thumbs | CPU | B3 | 2 |
| **F0** | Entity mentions | InsightFace; spaCy NER | media; transcript | `detection`, `face_track`, entity mentions | GPU (faces) / CPU (NER) | A, B3 | 3 |
| **F1** | Entity nodes | erf (resolution) | face_tracks, diar, mentions, galleries | `entity_node`, `person` links | CPU | F0, enroll | 3 |

Tier meanings: **GPU** = must run on a GPU pod (model size / throughput). **GPU-preferred / CPU-trickle** = runs on the control pod CPU slowly for low volume, batches on GPU when a pod is up. **CPU** = control pod only.

### 5.1 Creative / enrichment stages (redesign 2026-06-18 → 19)

Built after the original registry. These run on the control pod and journal to the
SAME spine (§5.2). DS Pro (`deepseek-v4-pro`, 1M ctx) is the grounded editor; it is
served a fused evidence package, not summaries — see `project_redesign_grounded_editor` memory.

| Code | Name | In → Out | Entrypoint |
|------|------|----------|-----------|
| RETRANSCRIBE | Re-ASR + re-diar + re-F1 at max quality | source audio → words/segments/diar + identity_overlay | `bin/retranscribe.py` (continuous; heartbeat) |
| D0 | WhisperX align + Silero VAD + overlap | proxy → `asr/{id}.words.wx/vad/overlap` | `bin/d0_preprocess.py` |
| P0 | Sentence / punctuation overlay | words → `asr/{id}.sentences` | `bin/punctuation_overlay.py` |
| V0 | Visual grounding (Qwen2.5-VL) | keyframes → `visual_grounding` | `bin/v0_visual_grounding.py` |
| CARD | Media card (Flash synopsis) | events → `media_cards` | `bin/build_digest.py --only-cards` |
| E1 | Event signals (Flash hook/quote) | events → `event_signals` | `bin/run_e1.py` |
| E2 | Corpus scan — hook-seeded thread discovery (corpus-scoped, not per-media) | event_signals+text embeddings → `event_embeddings`, `narrative_threads` | `src/erf/scan.py` (`docs/_archive/PHASE_NARRATIVE_CODE_BRIEF.md` §2 — archived, but §2 remains the E2 spec) |
| Q0 | Quality metrics | proxy → `catalog/quality/{id}.jsonl` | `src/erf/quality.py`, invoked from `bin/batch_ingest_bts.py` inside BASE (D-Q0-01/02/03) |
| F2 | Active-speaker detection | proxy + face_tracks → `active_speaker` | `bin/asd_worker.py` |
| W | Speaker-agnostic WhisperX words | proxy audio → `asr_words/` | `bin/build_asr_words.py` |
| RG | Recall-gap detection | words+vad+overlap → `asr_recall_gaps/` | `bin/build_recall_gaps.py` |
| RP | Recall pass (plain whisper on gap windows; candidate words only) | gaps → `asr_recall_pass/` | `bin/recall_pass.py` |
| DX | Exclusive diarization (pyannote community-1; evidence only, labels permuted vs baseline) | proxy → `diar_hypotheses/{id}/community1.jsonl` + `community1_excl.jsonl` | `bin/run_diar_community1.py` |
| SEV2 | Speech events v2 — canonical-first word attribution; events carry `words` | words+recall+diar+identity_map → `speech_events_v2/` | `bin/build_speech_events_v2.py` |
| SEV2_ADJ | Voiceprint adjudication of conflict/flip/low-conf subset | SEV2+diar_embeddings → `speech_events_v2_adjudicated/` | `bin/adjudicate_speech_events.py` |
| PSTREAM | Parallel streams: voiceprint-confirmed 2nd-speaker events + render merge | adjudicated+recall → `parallel_streams/`, `render_events_v2/` | `bin/build_parallel_streams.py` |
| T0 | Unified transcript = **projection of the adjudicated spine** (render_events_v2 preferred, adjudicated fallback, legacy merge for pre-spine media) | spine → `asr/{id}.transcript.jsonl` (word rows + stable_unknown_id/source/event_id/stream) | `bin/build_transcript.py` |
| EV | Evidence package (DS-facing contract) | all signals → `catalog/evidence/{id}.json` | `bin/build_evidence.py` |
| LABEL | Editorial-function labelling (Flash) | evidence → `catalog/editorial/{id}.json` (per-sentence functions/tensions/chain-affordances/quote-completeness + clip story_forces/archetypes) | `src/erf/editorial.py` + `bin/label_editorial.py` |
| SURVEY | Corpus survey (DS Pro) | evidence → `catalog/survey/latest.json` | `bin/corpus_survey.py` |
| SHORT | Single-clip short | evidence → grounded edit → render → mp4 (`catalog/shorts/`) | `bin/make_short.py` |
| STORY | Cross-media narrative short | evidence(many) → grounded multi-clip edit → render (`catalog/stories/`) | `bin/make_story.py` |
| ORIENT | Content-based proxy orientation fix | proxy → re-encoded proxy | `bin/fix_orientation.py` — **UNRELIABLE corpus-wide** (keys on incidental faces); targeted use only |

Notes: `erf.llm.complete` takes `model=` (`deepseek-v4-flash` for cheap labelling/copy; default `-pro`) and `temperature=`. Vertical duration cap is `[render] max_duration_s` (now **120s**, was 60). LABEL is layer 1 of the storytelling methodology (labels → async adversarial population loop → graph-later); the population loop is NOT yet built — see `project_storytelling_methodology` memory. `batch_ingest` now emits a heartbeat via its `log()` helper.

### 5.1.1 The speech-events v2 spine (promoted 2026-07-03)

The canonical who-said-what surface. Governing verdicts (evidence-locked 2026-07-01, D-Arch-01 lineage):

- **Canonical-first attribution.** The canonical ASR `speaker_label` (baseline-aligned) is the
  PRIMARY word owner. Exclusive diarization is EVIDENCE ONLY (corpus-wide voiceprint adjudication
  preferred it exactly 0 times). Voiceprint adjudication touches only the conflict/flip/low-confidence
  subset — never corpus-wide churn.
- **Text-aware dedup.** A recall word is a duplicate only if it time-overlaps (≥50%) a canonical word
  AND carries the same normalized token. Same time + DIFFERENT token = second speaker in crosstalk — kept.
- **Two-line invariant.** A second caption line exists ONLY when recall recovered real text, it overlaps
  a dominant stream, it is a different token stream, AND voiceprint confirms the non-dominant speaker
  with margin (`WIN_ABS`/`WIN_MARGIN` in `build_parallel_streams.py`). Default everywhere = stable
  single line. Two-line render is a standalone capability (`bin/render_multiline_probe.py`), NOT wired
  into the production render (decision, project lead 2026-07-01).
- **Word carry.** Every event carries its `words` array (text/start_ms/end_ms/prob/source) through
  build → adjudication → parallel-streams, so downstream projections are lossless.
- **T0 is a projection.** `asr/{id}.transcript.jsonl` is projected from `render_events_v2` (stream A
  minus superseded recall fragments ⊕ confirmed stream B). All consumers (EV, LABEL, story engine,
  render) inherit attribution from this single surface. Projected rows keep TRUE timings — stream-B
  words genuinely overlap stream A; word `start_ms` is not globally monotonic.
- **Explicit-empty convention.** Spine layers write an explicit EMPTY per-media file for no-speech
  media, paired with a recorded `empty`/`skip` outcome. This is the sanctioned exception to the
  zero-byte prohibition: the file's presence is the artifact signal (reconcile uses `exists()`),
  the outcome record is the semantic signal.
- **Outcome stages:** `SEV2`, `SEV2_ADJ`, `PSTREAM` (+ existing `T0`), all under the §5.2 outcome
  contract and tracked by `bin/reconcile_status.py`.
- **Separation stays demoted.** Blind source separation (ClearVoice/MossFormer2, `checkpoints/`) was
  evidence-rejected for production; kept as a documented reserve only.

### 5.2 Logging / observability spine — single-surface, registry-driven (D-Log-04, all Phases 0–5 landed; W2 complete + checkpoint-reviewed, D-Log-12)

**This is the redesigned contract.** The pre-2026-07-10 manual contract (three independently-asserted
status surfaces bound together by a human checklist) is retired; `docs/OBSERVABILITY_REDESIGN.md` is
the design record of *why* (root causes §1, principles §2) and *how it landed, phase by phase*
(DECISIONS.md `D-Log-06`…`D-Log-09`, plus this Phase 4). What follows is the *current* architecture,
not a pending target. The underlying "why" is unchanged from the original contract and still holds:
Invariant 3 — **explicit outcomes only; absence is never a signal** — applies uniformly to every stage,
and no stage infers another stage's result from a file's presence, absence, or size.

The redesign's core move: **collapse every place drift could happen into one append-only fact and one
declarative registry, and make everything else a recomputed view.** Drift is only possible between two
things independently asserted; with one asserted surface, the drift class is structurally impossible,
not merely policed.

- **The one asserted surface — `state/journal.db.events`.** Append-only, one row per (media_id, stage,
  attempt): `status, reason, run_id, producer_sha, config_fp, metrics, health, inputs, artifacts, at_utc`
  (`D-Log-07`). A sibling table, **`events_trace`**, holds best-effort operational trace (start/warn/
  namespaced sub-events) — liveness/trace is explicitly **not** an outcome and is never read for status
  or health. The pre-redesign `events` table (event_type/payload shape) is frozen verbatim as
  **`events_legacy`** (47,778 rows, never written again) — nothing was discarded, only superseded.
- **The one writer — `erf.outcomes.record()`.** Every stage, on every exit path, calls
  `outcomes.record(media_id, stage, status, *, reason=None, metrics=None, health=None, inputs=None,
  artifacts=None, producer=None, config_fp=None, run_id=None)`, `status ∈ {ok, skip, empty, error,
  partial, degraded, stale}`. `reason` is **required** for `skip/empty/error/degraded/stale`. `counts=`
  is kept as a backward-compatible alias for `metrics=` (~50 pre-redesign call sites unchanged).
  `erf.runlog.log_event`, `erf.journal.reap_manifest`/`emit_ok`/`emit_error`, and `e1_signals`'s journal
  helper all funnel into this one path now (`D-Log-07`); `journal.emit_start` and liveness heartbeats
  write `trace()`/heartbeat only, never an outcome. **No silent swallow (P4):** a write failure
  **raises** — an unloggable stage outcome is treated as a failed stage, not a silently-green one. Every
  stage-owning script that calls `record()` imports `erf.outcomes` as a **hard, top-level dependency**
  (no `try/except ImportError → outcomes = None` escape) — an import failure must crash the stage
  loudly. `CLUSTERS`, `T0`, and `SEV2` do this as of Phase 3/4; `RG`/`RP` still guard the import itself
  (unchanged this phase — see below) but their per-media loops can no longer let one media's exception
  silently swallow every outcome after it (Phase 4, this section's last bullet).
- **The one registry — `src/erf/stages.py`.** A frozen-dataclass `Stage` per stage name (33 as of Phase
  4's `GATE` addition), the union of the three pre-redesign hardcoded universes plus a fourth that
  turned up mid-migration (`bin/timeline_audit.py`'s own `READ_EDGES`/`SPEECH_SPINE`/`VISUAL`) —
  `depends`, `tier`, `phase`, `scope` (`media`|`corpus`), `artifact_path`/`predicate_kind` (+
  `also_requires` for compound AND predicates), `critical`, `speech_spine`, `scheduled`, `visual`,
  `health_predicates`, `fingerprint_inputs` (`D-Log-06`). **Every consumer iterates this registry
  instead of hardcoding its own stage set** (`D-Log-08`): `scheduler.py` derives its schedule from
  `scheduled=True`; `reconcile_status.py` derives its artifact predicates from
  `stages.artifact_predicates()`; `pipeline_status.py` derives `CRITICAL`/`SPEECH_SPINE` from
  `critical_stages()`/`speech_spine_stages()`; `timeline_audit.py` derives `READ_EDGES` from `depends`.
  Adding a stage is now **one registration**, cross-checked by `tests/test_stages_registry.py` (which
  also asserts, by grep, that no stage-universe literal exists anywhere outside this file) — this is
  what retires the old §5.2.3 manual checklist **by construction**: forgetting to add a stage to one of
  four independent hardcoded sets is no longer a possible failure mode, because there is only one set.
- **Semantic health — `degraded`/`stale` (the green-but-wrong killer, `D-Log-09`).** A stage that ran
  and produced an artifact but failed a declared health predicate is not "ok" — it is `degraded`, a
  disposition as visibly not-green as `error`. `record()` **derives** this: pass `status="ok"` plus a
  `health` dict (the `{predicate: {value, floor, verdict}}` shape from
  `stages.evaluate_health(stage_name, values)`); if any predicate's `verdict="fail"`, `record()` flips
  the row to `status="degraded"` and auto-sets `reason` to the failing predicate name(s). A stage may
  also call `record(..., status="degraded", reason=...)` explicitly, bypassing derivation.
  `evaluate_health()` is the single home for floor-comparison logic (`(name, op, floor)` tuples in the
  registry's `health_predicates`) — a predicate a caller omits from `values` is skipped, not defaulted,
  which is how a *conditional* predicate is expressed (e.g. SEV2/T0's `pct_person_id > 0` only applies
  when this media's `identity_map` anchored ≥1 slot to a gallery person; rando-only media pass
  vacuously). Three emitters carry predicates today: `CLUSTERS` (corpus `gallery_share ≥ 0.45`,
  `no_embedding_share ≤ 0.15`), `SEV2` and `T0` (per-media `pct_person_id > 0`, conditional as above).
  `stale` is a legal disposition and `inputs=` (upstream-artifact fingerprints, e.g.
  `{"identity_map": <sha256>}`) is populated by SEV2/T0 — groundwork for a later phase; **no
  staleness-marking logic reads `inputs` yet.**
- **Gate-as-stage (Phase 4, this section).** The attribution gate (GAMEPLAN "Quality gate",
  `bin/eval_attribution.py`) emits into the same log as any pipeline stage — corpus-scoped stage `GATE`
  in the registry (`scope="corpus"`, no artifact predicate: it is a read-only verdict *over* the corpus,
  not a stage that produces a reconcilable artifact, so it is never scheduled/critical/speech_spine and
  never appears in `reconcile_status.py`'s drift count). After computing its existing pass/fail check
  against the loaded baseline (`state/eval/baseline/ATTR.json`), it calls
  `outcomes.record("__corpus__", "GATE", "ok"|"degraded", reason=..., metrics={"word_acc", "window_pass",
  "n_windows", "per_stratum"}, producer="eval_attribution.py")` — `erf.outcomes` is a hard dependency
  here too. "Green" now means meaning-intact (no attribution regression), not merely "the script didn't
  crash" (P5).
- **Derived views — recomputed from `events`, never independently asserted (D-Log-07/08).**
  `status_projection` (latest disposition per media/stage) is folded by `outcomes.rebuild_projection()`;
  `catalog/status/{media_id}.json` is folded by `outcomes.rebuild_status_files()` (the old
  `pipeline_status.py --project` second writer is deleted — there is exactly one path to this file now);
  `logs.all/_latest/health.json` is recomputed registry-sourced. **Auditor:**
  `bin/reconcile_status.py` re-folds `status_projection` from `events` before comparing against disk —
  it can no longer assert projection rows from disk (that assertion *was* the drift root cause); a
  disagreement now means "re-run the stage," not "patch the projection." **Drift must be 0** for every
  registry-tracked stage; a pre-existing 308-row drift (B0/B1/E0/tags/F0/D0, dating to before the
  redesign) is a known, reproducible backlog item that clears with the W3a full-corpus rebuild, not a
  regression to chase per-session. **Liveness** stays separate: `erf.runlog.heartbeat(component, …)` →
  `logs.all/_latest/heartbeat/{component}.json`, never a `record()` outcome. **Canonical operational-log
  surface remains `logs.all/`** (not `/tmp`).
- **Per-media batch isolation (P4, Phase 4, this section).** A batch stage's `main()` loop wraps only
  the per-media body in try/except — argument parsing and directory setup still fail the whole run, by
  design. `RG` (`bin/build_recall_gaps.py`) and `RP` (`bin/recall_pass.py`) both do this now: one
  media's exception is caught, recorded as `record(mid, stage, "error", reason=f"exception: ...")`, and
  the loop continues — it no longer kills the batch with no outcome recorded for the failing media or
  any media scheduled after it. `RP`'s `process_media()` already recorded its own outcomes on every
  *handled* exit path (skip/ok/error) before this phase; the loop-level except only fires for a crash
  `process_media` itself didn't catch.

**Standing discipline (unchanged from the prior contract):** an outcome nobody reads is dead. "Who
reads this outcome, to what end?" must have an answer for every stage before the stage is considered
done. In the registry-driven world that answer is almost always "automatically, by construction" —
`reconcile_status.py`, `pipeline_status.py`, and `rebuild_all.py`'s post-run read-back all iterate
`src/erf/stages.py`, so a stage that registers itself correctly is read by all three without a second
edit anywhere.

### 5.2.1 Stage-change checklist — retired by construction (was a manual GATE; now enforced)

The pre-redesign checklist (DAG row, write call, auditor predicate, health-set membership, orchestrator
read-back, gate run, docs — each a separate hand-edited box in four independently hardcoded literals)
is **retired**, not merely automated: there is only one place left to edit. Adding, renaming, re-pathing,
or removing a stage is:

1. **Register it** — one `Stage(...)` entry in `src/erf/stages.py` (`depends`, `tier`, `phase`, `scope`,
   `artifact_path`/`predicate_kind`, `critical`/`speech_spine`/`scheduled` as applicable,
   `health_predicates`/`fingerprint_inputs` if it carries semantic health). `tests/test_stages_registry.py`
   fails the build if any consumer's derived universe disagrees or if a stage-universe literal reappears
   outside this file.
2. **Write it** — the stage calls `outcomes.record(...)` on every exit path, `reason` set for non-ok,
   `erf.outcomes` imported as a hard dependency (no silent-degrade escape).
3. **Run `bin/reconcile_status.py`** — drift must not regress for the new stage (backfill from disk first
   if the stage pre-exists the registration).
4. **Run the attribution gate** (GAMEPLAN "Quality gate") if the change can affect who's-speaking — the
   gate is itself stage `GATE` in the registry (see above) and records its own verdict; no regression vs
   baseline.
5. **Update this table** (§5/§5.1) with the correct `Depends-on` if the stage set or a canonical path
   changed.
6. **Run `erf-eval` before/after for any B1/B4 embedding-model flip** (D-Audit-01, revived + confirmed
   still executing D-Log-28/worklist item 4, 2026-07-15 — dormant since its 2026-06-10 founding
   baseline). Mirrors the attribution gate's role for identity: no text- or visual-embedding model swap
   merges without an `erf-eval run --stage B1|B4` before/after comparison — vibes are not a gate.

What is *not* a manual step any more: keeping `scheduler.STAGE_REGISTRY`, `reconcile_status.STAGE_ARTIFACTS`,
`pipeline_status.CRITICAL`/`SPEECH_SPINE`, and `timeline_audit.READ_EDGES` in sync with each other — those
four literals no longer exist as independent facts to forget. That was the whack-a-mole engine (§0's
"Endpoints-only is a KNOWN DEFECT" admission); it cannot recur structurally, not just by discipline.

---

## 6. ID conventions

Every ID is deterministic from content or coordinates — no random UUIDs, no autoincrement for catalog records. This is what makes stages idempotent (Invariant 4). The relational DB may use autoincrement only for internal rows that never appear in catalog (e.g. `audit.id`).

| Record | ID form | Example | Determinism source |
|--------|---------|---------|--------------------|
| media | `m_<blake2b-12hex>` of `relpath \| filesize_bytes \| duration_ms` | `m_9f3a1c7be002` | file identity, stable across re-ingest |
| asr_word | `{media_id}:w:{index:06d}` | `m_9f3a..:w:000482` | word order |
| asr_segment | `{media_id}:s:{index:05d}` | `m_9f3a..:s:00031` | segment order |
| chunk | `{media_id}:c:{start_ms}-{end_ms}` | `m_9f3a..:c:42120-49880` | time span |
| scene | `{media_id}:sc:{index:04d}` | `m_9f3a..:sc:0007` | scene order |
| keyframe | `{media_id}:kf:{time_ms}` | `m_9f3a..:kf:45003` | timestamp |
| embedding | `emb:{source_id}:{model_tag}` | `emb:m_9f3a..:c:42120-49880:qwen3-8b` | source + model |
| detection | `{keyframe_id}:det:{index:03d}` | `m_9f3a..:kf:45003:det:002` | per-keyframe order |
| face_track | `{media_id}:ft:{index:04d}` | `m_9f3a..:ft:0003` | track order |
| person | `p_<slug>` (human-assigned at enroll) | `p_jane_doe` | manual, unique |
| event | `{media_id}:ev:{index:04d}` | `m_9f3a..:ev:0012` | event order |
| entity_node | `n_<type>_<slug>` or a `person_id` | `n_loc_warehouse`, `p_jane_doe` | manual/derived |
| export | `exp_<YYYYMMDDTHHMMSSZ>_<6hex>` | `exp_20260602T031421Z_a1b2c3` | time + content hash |

`model_tag` registry (fixed strings; adding one is additive): `qwen3-8b` (text), `qwen3vl-8b` (visual), `insightface-buffalo_l` (face, Phase 3), `pyannote-4` (voice, Phase 3).

---

## 7. Schema contracts

All schemas are JSON Schema **draft 2020-12**. The `$id` carries the schema version; bump per Invariant 5. Every record carries `schema_version`. Vectors are stored in catalog as plain float arrays (JSONL) — the canonical copy. LanceDB holds a second materialised copy (disposable).

Validators live in `tests/` and run in CI and at stage entry. A stage MUST validate its output against the schema before writing.

### 7.1 `media_manifest`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/media_manifest/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","media_id","relpath","filesize_bytes","duration_ms","container","video","audio","ingested_at_utc","content_blake2b"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "media_id": {"type": "string", "pattern": "^m_[0-9a-f]{12}$"},
    "relpath": {"type": "string", "description": "path relative to /workspace/erf/incoming or configured source root"},
    "filesize_bytes": {"type": "integer", "minimum": 0},
    "duration_ms": {"type": "integer", "minimum": 0},
    "content_blake2b": {"type": "string", "pattern": "^[0-9a-f]{32}$", "description": "first 16 bytes hex of blake2b(file) — integrity, not identity"},
    "container": {"type": "string", "examples": ["mov","mp4","mxf"]},
    "video": {
      "type": "object", "additionalProperties": false,
      "required": ["codec","width","height","fps","bit_depth"],
      "properties": {
        "codec": {"type": "string", "examples": ["h264","hevc","prores"]},
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "fps": {"type": "number", "description": "exact frame rate, e.g. 23.976, 25, 50"},
        "fps_rational": {"type": "string", "description": "e.g. 24000/1001", "default": null},
        "bit_depth": {"type": "integer", "enum": [8,10,12]},
        "color_primaries": {"type": ["string","null"], "default": null},
        "transfer": {"type": ["string","null"], "default": null}
      }
    },
    "audio": {
      "type": "object", "additionalProperties": false,
      "required": ["codec","sample_rate","channels"],
      "properties": {
        "codec": {"type": "string"}, "sample_rate": {"type": "integer"}, "channels": {"type": "integer"}
      }
    },
    "ingested_at_utc": {"type": "string", "format": "date-time"},
    "source_note": {"type": ["string","null"], "default": null, "description": "free text: shoot day, camera, etc."}
  }
}
```

### 7.2 `asr_word`

JSONL, one object per line in `{media_id}.words.jsonl`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/asr_word/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","word_id","media_id","segment_id","index","text","start_ms","end_ms","prob"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "word_id": {"type": "string", "pattern": "^m_[0-9a-f]{12}:w:[0-9]{6}$"},
    "media_id": {"type": "string"},
    "segment_id": {"type": "string"},
    "index": {"type": "integer", "minimum": 0},
    "text": {"type": "string"},
    "start_ms": {"type": "integer", "minimum": 0},
    "end_ms": {"type": "integer", "minimum": 0},
    "prob": {"type": "number", "minimum": 0, "maximum": 1, "description": "word confidence from faster-whisper"},
    "speaker_label": {"type": ["string","null"], "default": null, "description": "raw pyannote label e.g. SPEAKER_00, assigned by reconciliation"},
    "speaker_person_id": {"type": ["string","null"], "default": null, "description": "resolved person_id; null until Phase 3 enrollment"}
  }
}
```

`asr_segment` (in `{media_id}.segments.jsonl`) is the same shape with `segment_id` (`:s:` form), `index`, `text` (full segment), `start_ms`, `end_ms`, `avg_logprob`, `no_speech_prob`, plus `speaker_label`/`speaker_person_id`. Diarization turns (in `{media_id}.diar.jsonl`) are `{turn_index, speaker_label, start_ms, end_ms}`.

### 7.3 `chunk`

The retrieval unit for text search. Chunking rule (§B0): split on speaker turn boundaries, then merge/split to target 12–35 s windows (configurable), never crossing a speaker turn. Carries the short-form hook fields from day one.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/chunk/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","chunk_id","media_id","start_ms","end_ms","text","word_ids","speaker_label"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "chunk_id": {"type": "string", "pattern": "^m_[0-9a-f]{12}:c:[0-9]+-[0-9]+$"},
    "media_id": {"type": "string"},
    "start_ms": {"type": "integer", "minimum": 0},
    "end_ms": {"type": "integer", "minimum": 0},
    "text": {"type": "string"},
    "word_ids": {"type": "array", "items": {"type": "string"}},
    "speaker_label": {"type": ["string","null"]},
    "speaker_person_id": {"type": ["string","null"], "default": null, "description": "Phase 3 hook"},
    "emotional_beat": {"type": ["string","null"], "default": null, "enum": ["setup","tension","release","comedic","emotional","action","reveal","exposition","transition", null], "description": "Phase 5 hook; nullable now"},
    "b_roll_flag": {"type": ["boolean","null"], "default": null, "description": "Phase 5 hook: usable as non-sync coverage"}
  }
}
```

### 7.4 `embedding`

Canonical vector copy. One JSONL per modality per media. `vector` is a plain float array; length MUST equal the collection dim.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/embedding/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","embedding_id","source_id","source_type","modality","model_tag","dim","vector","normalized"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "embedding_id": {"type": "string"},
    "source_id": {"type": "string", "description": "chunk_id or keyframe_id"},
    "source_type": {"type": "string", "enum": ["chunk","keyframe"]},
    "media_id": {"type": "string"},
    "modality": {"type": "string", "enum": ["text","visual"]},
    "model_tag": {"type": "string", "enum": ["qwen3-8b","qwen3vl-8b"]},
    "dim": {"type": "integer", "enum": [4096]},
    "vector": {"type": "array", "items": {"type": "number"}},
    "normalized": {"type": "boolean", "description": "true if L2-normalized at write time"}
  }
}
```

Contract: `modality=text ⇒ model_tag=qwen3-8b, dim=4096`; `modality=visual ⇒ model_tag=qwen3vl-8b, dim=4096`. Enforce in the validator. The exact invocation (instruction prefix, pooling, normalization) is confirmed at the Phase-1 / Phase-2 embedding gates (§14) before any real media is embedded — do not embed production media until the gate passes.

### 7.5 `scene`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/scene/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","scene_id","media_id","index","start_ms","end_ms","start_frame","end_frame"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "scene_id": {"type": "string", "pattern": "^m_[0-9a-f]{12}:sc:[0-9]{4}$"},
    "media_id": {"type": "string"},
    "index": {"type": "integer", "minimum": 0},
    "start_ms": {"type": "integer"}, "end_ms": {"type": "integer"},
    "start_frame": {"type": "integer"}, "end_frame": {"type": "integer"},
    "detector": {"type": "string", "default": "pyscenedetect-content", "description": "detector + params tag"}
  }
}
```

### 7.6 `keyframe`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/keyframe/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","keyframe_id","media_id","scene_id","time_ms","frame","image_relpath","width","height"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "keyframe_id": {"type": "string", "pattern": "^m_[0-9a-f]{12}:kf:[0-9]+$"},
    "media_id": {"type": "string"},
    "scene_id": {"type": ["string","null"]},
    "time_ms": {"type": "integer"},
    "frame": {"type": "integer"},
    "image_relpath": {"type": "string", "description": "under media/keyframes/{media_id}/"},
    "thumb_relpath": {"type": ["string","null"], "default": null},
    "width": {"type": "integer"}, "height": {"type": "integer"},
    "visual_quality_score": {"type": ["number","null"], "minimum": 0, "maximum": 1, "default": null, "description": "Phase 5 hook: sharpness/exposure/composition heuristic"},
    "person_ids": {"type": ["array","null"], "items": {"type": "string"}, "default": null, "description": "Phase 3 hook"},
    "b_roll_flag": {"type": ["boolean","null"], "default": null, "description": "Phase 5 hook"}
  }
}
```

### 7.7 `detection` (Phase 3+ hook — schema only now)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/detection/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","detection_id","keyframe_id","media_id","label","score","bbox","detector"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "detection_id": {"type": "string"},
    "keyframe_id": {"type": "string"},
    "media_id": {"type": "string"},
    "label": {"type": "string", "description": "open-vocab text label (GroundingDINO) or class name"},
    "score": {"type": "number", "minimum": 0, "maximum": 1},
    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x,y,w,h] normalized 0-1"},
    "detector": {"type": "string", "examples": ["groundingdino"]},
    "is_face": {"type": "boolean", "default": false},
    "face_track_id": {"type": ["string","null"], "default": null}
  }
}
```

### 7.8 `face_track` (Phase 3 hook)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/face_track/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","face_track_id","media_id","start_ms","end_ms","embedding","person_id","match_score"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "face_track_id": {"type": "string"},
    "media_id": {"type": "string"},
    "start_ms": {"type": "integer"}, "end_ms": {"type": "integer"},
    "keyframe_ids": {"type": "array", "items": {"type": "string"}},
    "embedding": {"type": "array", "items": {"type": "number"}, "description": "512-d buffalo_l mean track embedding"},
    "person_id": {"type": ["string","null"], "description": "resolved or null if unmatched"},
    "match_score": {"type": ["number","null"], "description": "cosine to matched gallery template"}
  }
}
```

### 7.9 `person` (Phase 3; enrollment record)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/person/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","person_id","display_name","created_at_utc","face_enrolled","voice_enrolled"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "person_id": {"type": "string", "pattern": "^p_[a-z0-9_]+$"},
    "display_name": {"type": "string"},
    "created_at_utc": {"type": "string", "format": "date-time"},
    "consent": {"type": "boolean", "default": true, "description": "facial/speaker recognition consent on record (true for Life® cast/crew)"},
    "face_enrolled": {"type": "boolean", "default": false},
    "voice_enrolled": {"type": "boolean", "default": false},
    "face_gallery_relpath": {"type": ["string","null"], "default": null},
    "voice_gallery_relpath": {"type": ["string","null"], "default": null}
  }
}
```

### 7.10 `event`

The beat/segment abstraction over chunks + scenes. In Phase 2 events are skeletal (one per scene with text rollup); Phase 5 populates beat structure for assembly.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/event/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","event_id","media_id","start_ms","end_ms","kind"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "event_id": {"type": "string", "pattern": "^m_[0-9a-f]{12}:ev:[0-9]{4}$"},
    "media_id": {"type": "string"},
    "start_ms": {"type": "integer"}, "end_ms": {"type": "integer"},
    "kind": {"type": "string", "enum": ["scene_event","dialogue_event","beat"]},
    "chunk_ids": {"type": "array", "items": {"type": "string"}, "default": []},
    "scene_ids": {"type": "array", "items": {"type": "string"}, "default": []},
    "summary": {"type": ["string","null"], "default": null},
    "emotional_beat": {"type": ["string","null"], "default": null},
    "person_ids": {"type": ["array","null"], "items": {"type": "string"}, "default": null},
    "b_roll_flag": {"type": ["boolean","null"], "default": null},
    "visual_quality_score": {"type": ["number","null"], "default": null}
  }
}
```

### 7.11 `entity_node` (Phase 3 hook)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/entity_node/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","entity_node_id","type","label"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "entity_node_id": {"type": "string"},
    "type": {"type": "string", "enum": ["person","location","object","org","other"]},
    "label": {"type": "string"},
    "person_id": {"type": ["string","null"], "default": null},
    "aliases": {"type": "array", "items": {"type": "string"}, "default": []},
    "mention_refs": {"type": "array", "items": {"type": "string"}, "default": [], "description": "chunk_id / face_track_id / detection_id refs"}
  }
}
```

### 7.12 `status`

One file per media at `catalog/status/{media_id}.json`. The map is the authority on what has run. The control plane reaps worker `done.jsonl` into this.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/status/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","media_id","stages"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "media_id": {"type": "string"},
    "stages": {
      "type": "object",
      "description": "key = stage letter",
      "additionalProperties": {
        "type": "object",
        "additionalProperties": false,
        "required": ["at_utc","ok","run_id"],
        "properties": {
          "at_utc": {"type": "string", "format": "date-time"},
          "ok": {"type": "boolean"},
          "run_id": {"type": "string"},
          "tool_version": {"type": ["string","null"], "default": null},
          "counts": {"type": ["object","null"], "default": null, "description": "e.g. {words: 1842, chunks: 63}"},
          "error": {"type": ["string","null"], "default": null}
        }
      }
    }
  }
}
```

### 7.13 `export_manifest` (Phase 4)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/export_manifest/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","export_id","created_at_utc","query","format","clips","output_relpath"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "export_id": {"type": "string"},
    "created_at_utc": {"type": "string", "format": "date-time"},
    "query": {"type": ["string","null"], "description": "originating query, if any"},
    "format": {"type": "string", "enum": ["fcpxml","edl","otio"]},
    "frame_rate": {"type": "number"},
    "clips": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["media_id","in_frame","out_frame"],
        "properties": {
          "media_id": {"type": "string"},
          "source_relpath": {"type": "string"},
          "in_frame": {"type": "integer"}, "out_frame": {"type": "integer"},
          "label": {"type": ["string","null"], "default": null}
        }
      }
    },
    "output_relpath": {"type": "string", "description": "emitted file under catalog/exports/"}
  }
}
```

---

## 8. Vector store layout (LanceDB)

Embedded LanceDB at `/workspace/erf/index/lance/`. **Control plane is the sole writer.** Workers never open the Lance store for writing; they emit `embedding` JSONL to catalog, and stage **C** (on the control pod) materialises. Rationale: MooseFS concurrent-write hazard (§0.2) + LanceDB's single-writer-friendly manifest model.

Why LanceDB and not the alternatives: native arbitrary-string PKs match our deterministic compound IDs with zero mapping layer; immutable-fragment + atomic-manifest writes are the most MooseFS-tolerant; no daemon to keep alive; Apache-2.0; tables are pure derived state, droppable and rebuildable from catalog. Qdrant (UUID/uint64 IDs + mmap-on-distributed-FS risk) and Milvus Lite (scale ceiling, eventual daemon) both lose here.

### 8.1 Tables

**`text_v1`**

| field | type | note |
|-------|------|------|
| `id` | string (PK) | = `chunk_id` |
| `media_id` | string | filter |
| `vector` | fixed_size_list<float32, 4096> | cosine |
| `text` | string | display + rerank |
| `start_ms` / `end_ms` | int64 | clip bounds |
| `speaker_label` | string | filter |
| `speaker_person_id` | string (nullable) | Phase 3 filter |
| `emotional_beat` | string (nullable) | Phase 5 filter |
| `b_roll_flag` | bool (nullable) | Phase 5 filter |

**`visual_v1`**

| field | type | note |
|-------|------|------|
| `id` | string (PK) | = `keyframe_id` |
| `media_id` | string | filter |
| `vector` | fixed_size_list<float32, 4096> | cosine |
| `scene_id` | string | filter |
| `time_ms` | int64 | clip anchor |
| `image_relpath` | string | thumbnail |
| `visual_quality_score` | float32 (nullable) | Phase 5 |
| `person_ids` | list<string> (nullable) | Phase 3 filter |
| `b_roll_flag` | bool (nullable) | Phase 5 |

**`persons_v1`** (Phase 3): `id`=`person_id`, `vector` fixed_size_list<float32,512> (buffalo_l face template), `display_name`. Used as the face-match gallery.

### 8.2 Index policy

Brute-force (no ANN index) while a table has < 50,000 rows — exact search, trivially fast at this scale, no training. At ≥ 50,000 rows, build `IVF_PQ`: `num_partitions = round(sqrt(N))`, `num_sub_vectors = dim/16` (256 for text, 128 for visual), metric cosine. `erf-reindex` decides and (re)builds automatically. Index lives only in `index/` and is never backed up.

**Watch point (D-Audit-01, 2026-07-15; annotated D-Log-27, worklist item 3(d)):** measured corpus scale — `text_v1` 1,833 rows / `visual_v1` 929 / `persons_v1` 107. Brute-force confirmed correct at this scale, still ~27x below the 50,000-row trigger. Flagged because `events`/`events_trace` row growth (the closest proxy for corpus-wide growth rate) is approaching ~2k — worth a periodic glance, not an action: no code change until a table actually nears 50,000.

### 8.3 Rebuild

`erf-reindex [--table text_v1|visual_v1|all]`: drop table → stream `catalog/embeddings/*.{text,visual}.jsonl` → create table → build index if over threshold. Must be safe to run any time; it is the disaster-recovery path for the entire index.

---

## 9. Relational schema (SQLite)

One file: `/workspace/erf/state/erf.db`. Opened by the **control plane only**, with:

```sql
PRAGMA journal_mode = TRUNCATE;   -- NOT WAL: WAL's -shm needs mmap, unreliable on MooseFS/FUSE
PRAGMA locking_mode = EXCLUSIVE;  -- single connection holds the lock; no per-txn NFS-style locking
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;
```

The Phase-0 storage probe (§14) tests whether MooseFS supports the mmap'd `-shm` reliably; only if it passes unambiguously may a future revision switch to WAL (default stays TRUNCATE). No other process opens this file for writing — ever.

```sql
CREATE TABLE schema_meta (component TEXT PRIMARY KEY, version TEXT NOT NULL);

CREATE TABLE media (
  media_id TEXT PRIMARY KEY,
  relpath TEXT NOT NULL,
  duration_ms INTEGER,
  fps REAL,
  ingested_at_utc TEXT,
  last_stage TEXT,
  state TEXT NOT NULL DEFAULT 'new'      -- new|processing|ready|error
);

CREATE TABLE jobs (
  job_id TEXT PRIMARY KEY,               -- {media_id}:{stage}
  media_id TEXT NOT NULL REFERENCES media(media_id),
  stage TEXT NOT NULL,                   -- stage letter
  tier TEXT NOT NULL,                    -- cpu|gpu
  state TEXT NOT NULL DEFAULT 'pending', -- pending|assigned|running|done|error
  priority INTEGER NOT NULL DEFAULT 100,
  assigned_pod TEXT,                     -- pod_id of GPU worker, or 'control'
  attempts INTEGER NOT NULL DEFAULT 0,
  assigned_at_utc TEXT,
  started_at_utc TEXT,
  finished_at_utc TEXT,
  error TEXT
);
CREATE INDEX ix_jobs_state ON jobs(state, tier, priority);

CREATE TABLE workers (
  pod_id TEXT PRIMARY KEY,
  role TEXT NOT NULL,                    -- gpu-audio|gpu-embed|gpu-all
  gpu_type TEXT,
  cost_per_hr REAL,
  created_at_utc TEXT,
  last_heartbeat_utc TEXT,
  state TEXT NOT NULL DEFAULT 'booting'  -- booting|idle|busy|terminating|gone
);

CREATE TABLE persons (                    -- Phase 3; mirror of catalog/persons for query joins
  person_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  face_enrolled INTEGER NOT NULL DEFAULT 0,
  voice_enrolled INTEGER NOT NULL DEFAULT 0,
  created_at_utc TEXT
);

CREATE TABLE speaker_voiceprints (        -- Phase 3
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id TEXT NOT NULL REFERENCES persons(person_id),
  vector_relpath TEXT NOT NULL,           -- pyannote embedding stored on disk
  source_note TEXT
);

CREATE TABLE audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at_utc TEXT NOT NULL,
  actor TEXT NOT NULL,                    -- 'control' | 'code' | pod_id
  action TEXT NOT NULL,
  target TEXT,
  detail TEXT                             -- JSON
);
```

The relational DB is rebuildable too, but more expensively (replay catalog status). It is backed up nightly by copying the file to `state/erf.db.bak` (atomic rename) while the control plane briefly checkpoints.

---

## 10. GPU pod dispatch protocol

> **Status: unbuilt by design (confirmed D-Audit-01, 2026-07-15; annotated D-Log-27, worklist item 3(d)).**
> Spec-only — no `erf-dispatch`, no `erf-worker`, no `state/workers/` machinery exists on disk.
> The project runs on a single pod today (this doc's own §0.2/§4 assume one MooseFS volume mount).
> Multi-pod dispatch stays a documented **future option**, not a current gap: nothing here has ever
> blocked real work, and the standing directionality (DECISIONS.md 2026-07-15, D-Audit-01) is to grow
> swappable component seams before scaling out compute topology. Kept below for when corpus volume or
> a hard single-GPU throughput ceiling actually forces the question — do not build against it now.

CPU control pod orchestrates; GPU pods are cattle. API: RunPod REST `https://rest.runpod.io/v1`, Bearer with a **restricted** key (delete + create + read pods only) in `RUNPOD_API_KEY`. GPU-type discovery uses GraphQL `https://api.runpod.io/graphql` (REST has no price-enriched enumeration).

### 10.1 Lifecycle (control pod, `erf-dispatch`)

1. **Need.** A batch of `gpu`-tier jobs in `jobs` exceeds the dispatch threshold (default: any pending gpu job older than `MAX_QUEUE_AGE`, or ≥ `BATCH_MIN` jobs).
2. **Preflight availability + price.** GraphQL `gpuTypes(input:{secureCloud:true, dataCenterId:"EU-RO-1", gpuCount:1})`. Pick cheapest type meeting the role's VRAM floor (audio ≥ 16 GB; embed: Qwen3-8B needs ≥ 24 GB, prefer ≥ 40 GB for batch). If none available in EU-RO-1, log `WARN gpu_unavailable` and leave jobs pending (do not fall back to another DC — the volume is pinned).
3. **Create.** `POST /pods` with `cloudType:"SECURE"`, `computeType:"GPU"`, `interruptible:false`, `dataCenterIds:["EU-RO-1"]`, `dataCenterPriority:"custom"`, `networkVolumeId:"cmkgi43hcx"`, `volumeMountPath:"/workspace"`, `gpuTypeIds:[<chosen>]`, `gpuTypePriority:"custom"`, ports `["22/tcp","8000/http"]`, env `{WORKER_ROLE, POD_ROLE_DIR}`, and a boot command that activates the role venv and runs `erf-worker --role <role>`. Insert a `workers` row.
4. **Assign.** Control writes the job set to `state/workers/{pod_id}/assignment.jsonl` (append-only) and sets those `jobs.state='assigned'`, `assigned_pod=pod_id`.
5. **Monitor.** Poll `GET /pods/{id}?includeMachine=true` until `desiredStatus=RUNNING` and `publicIp` + `portMappings["22"]` present and a TCP probe to the SSH port succeeds. Worker writes `state/workers/{pod_id}/heartbeat.json` every 30 s.
6. **Worker loop.** Worker reads its `assignment.jsonl`, processes each job, writes `embedding`/`asr`/etc. to catalog via write-tmp+rename, appends `{job_id, ok, counts, at_utc}` to its own `done.jsonl`, updates heartbeat (`state`, `current_job_id`). Worker **never** touches `erf.db` or the Lance store.
7. **Reap.** Control tails each worker's `done.jsonl`, validates the named catalog outputs exist + schema-valid, sets `jobs.state='done'` (or `error`, increment `attempts`), writes/updates `status/{media_id}.json`.
8. **Materialise.** When text or visual embeddings for a media complete, control enqueues a `C` job (control-tier) to upsert into LanceDB.
9. **Teardown.** When a worker's assignment is fully `done`/`error` AND heartbeat `state=idle` for ≥ `IDLE_SECONDS` (default 300), `DELETE /pods/{id}`. Mark `workers.state='gone'`. DELETE is idempotent: a 404 ⇒ mark gone locally. **Never** terminate solely on SSH failure (could be booting).

### 10.2 Worker image

Two clean options; pick **B** for first build, migrate to A if boot time hurts.

- **A (baked):** a private image with the role venv pre-installed. Faster boot. Requires `containerRegistryAuthId` + maintaining the image.
- **B (base + activate):** RunPod base CUDA 12.8 PyTorch image; boot script activates `/workspace/erf/venvs/{audio|embed}` (which live on the shared volume) and starts the worker. Zero image maintenance; slower first import. **Default.**

`gpu_pod_manager.py` from the prior pipeline is **replaced** by `erf-dispatch` implementing the above; do not port the old module.

### 10.3 Cost discipline

Per-second billing; on-demand requires ≥ 1 hr credit reserved at create. Budget shares the $1K/mo RunPod allowance with VFX — be frugal: batch gpu jobs, prefer the cheapest sufficient GPU in EU-RO-1, tear down on idle, never leave a GPU pod up overnight. Log every create/delete with `costPerHr` to `audit`. Quota preflight via GraphQL `myself { machineQuota currentSpendPerHr }` before create; backoff on 429/5xx.

---

## 11. Enrollment workflow (Phase 3 — spec now, build later)

> **HISTORICAL AS SPEC (A7, 2026-07-10).** Identity was built — and rebuilt — repeatedly after this
> was written (D-Identity-01…06). The face/voice-bootstrap + label-propagation design below (steps
> 5–7 especially) is superseded by the **D-Arch-01 word-level speech-events spine (§5.1.1)** and the
> **voiceprint_v2 resolver** (`bin/identity_resolve_v1.py` + `bin/identity_apply_resolver_v1.py`,
> WeSpeaker gallery per D-Identity-06). Still current: the face-enrollment mechanics (buffalo_l,
> `persons_v1`) and the last paragraph's precious-data/backup rule. Do not implement from this section.

1. `erf-enroll new --name "Jane Doe"` → allocates `person_id=p_jane_doe`, writes `catalog/persons/p_jane_doe.json` + `persons` row, creates `enroll/p_jane_doe/{faces,voice}/`.
2. Drop 5–20 clear reference face crops into `enroll/p_jane_doe/faces/`.
3. `erf-enroll face p_jane_doe` → buffalo_l (`det_10g`+`w600k_r50`) embeds each (512-d), L2-normalize, store per-image + mean template into `persons_v1` LanceDB; set `face_enrolled=1`.
4. (Optional) drop reference audio into `enroll/p_jane_doe/voice/`; `erf-enroll voice p_jane_doe` → pyannote speaker embedding → `speaker_voiceprints`; set `voice_enrolled=1`.
5. **Face linkage (F0/F1):** per media, detect+track faces → mean track embedding → cosine vs `persons_v1` gallery; assign `person_id` if ≥ `FACE_MATCH_THRESHOLD` (default 0.5 cosine on normalized buffalo_l), else null. Write `face_track`.
6. **Speaker linkage:** match diarization speaker embeddings vs voiceprint gallery; OR bootstrap by audio-visual co-occurrence (a face_track of person X overlapping a diarization turn ⇒ that turn is X) when voiceprints are absent. Resolve `speaker_label → speaker_person_id`.
7. **Propagation:** resolved `person_id` backfills `chunk.speaker_person_id`, `keyframe.person_ids`, `event.person_ids`. This is an additive update to existing catalog records (re-emit those JSONL with the field populated; IDs unchanged ⇒ idempotent).

Enrollment data is precious (lives under `enroll/` + `persons_v1` + `persons`/`speaker_voiceprints`). The face/voice galleries are the one part of the index that is **not** cheaply rebuildable, so `persons_v1` is exempt from "index is disposable" and is included in backup.

---

## 12. Query pipeline

Plain English in, ranked timestamped clips out. Served by the control pod FastAPI at `/query`.

### 12.1 Steps

1. **Parse.** LLM (DeepSeek Flash — §13) converts the query to a `QuerySpec`. System prompt instructs JSON-only, no prose.
2. **Embed query.** For `modality` ∈ requested: text → Qwen3-8B query embedding (with the query instruction prefix per the §14 gate); visual → Qwen3-VL-2B text-path embedding (cross-modal into the 2048 space).
3. **Search.** LanceDB cosine search per table, applying `QuerySpec.filters` as predicate pushdown (media_id, speaker, person_id, time_range, emotional_beat, b_roll_flag). `top_k` per table (default 50).
4. **Fuse.** If both modalities queried, combine with Reciprocal Rank Fusion (k=60). Optional later: cross-encoder rerank on text.
5. **Assemble clips.** Group adjacent hits (same media, gap < `MERGE_GAP_MS` default 1500) into clip spans; snap span bounds outward to the enclosing `scene` bounds when within `SNAP_MS`. Attach nearest keyframe thumbnail, in/out timecodes, dialogue snippet, score, provenance.
6. **Return** a `ResultSet`.

### 12.2 `QuerySpec` (LLM output contract)

```json
{
  "modality": ["text","visual"],
  "semantic_text": "warehouse argument at night",
  "filters": {
    "media_id": null,
    "speaker_person_id": null,
    "time_range_ms": null,
    "emotional_beat": null,
    "b_roll_flag": null
  },
  "top_k": 50
}
```

### 12.3 `Result` / `ResultSet` (returned to UI)

```json
{
  "query": "the warehouse argument",
  "results": [
    {
      "result_id": "m_9f3a1c7be002:c:42120-49880",
      "media_id": "m_9f3a1c7be002",
      "media_relpath": "day12/A007C031.mov",
      "in_ms": 42120, "out_ms": 49880,
      "in_tc": "00:00:42:03", "out_tc": "00:00:49:21",
      "fps": 23.976,
      "thumb_url": "/media/thumbs/m_9f3a1c7be002/m_9f3a1c7be002:kf:45003.webp",
      "snippet": "you said you'd never come back here",
      "speaker": "p_jane_doe | SPEAKER_01",
      "modality": "text",
      "score": 0.83
    }
  ]
}
```

Timecode is `HH:MM:SS:FF` computed from `ms` and the media `fps` (drop-frame only if `fps_rational` is 30000/1001 or 60000/1001 — otherwise non-drop).

### 12.4 UI (minimal)

Static page served by the control pod: search box → thumbnail grid. Each card: thumbnail, in/out TC, media name, snippet, speaker, score. Hover scrubs the proxy. Click → larger proxy preview + "add to selection". Selection bar → "Export" → choose FCPXML/EDL/OTIO → file written under `catalog/exports/` and offered for download (and visible via the existing Copyparty tunnel). No build tooling required beyond static HTML + fetch; do not introduce a SPA framework.

---

## 13. LLM backend (pluggable)

A single interface `erf.llm.complete(messages, *, json_only: bool) -> str` with selectable backends:

- **deepseek-flash**: DeepSeek API, current Flash tier. Key in `DEEPSEEK_API_KEY`. Cheap enough to treat as free at query-parse volume; fast. Used for query parsing, beat tagging, cheap triage.
- **deepseek-v4-pro** (A8): DeepSeek API, Pro tier — the default for the narrative/editorial stages (LABEL, EV, SURVEY, STORY — see §5.1). Slower and costlier than Flash but still cheap-metered; used where editorial judgment quality materially matters (D-Story-*).
- Backend is config, never hardcoded at call sites. Any future model swaps here only.

A paid frontier model (Claude, etc.) is permitted **only** where it is decisively the only path to a needed result; if Code ever wires one in, it must be behind this interface, gated by an env flag, off by default, with a `# JUSTIFY:` comment stating why local/cheap was insufficient.

---

## 14. Verification gates (block real-media processing until passed)

Nothing was guessed into a permanent contract, but two runtime usage details must be confirmed against the on-disk models before production embedding. These are gates, not research:

- **G-STORAGE (Phase 0):** `erf-probe-storage` writes `state/storage_probe.json` recording, on `/workspace/erf/state/`: fsync durability, atomic-rename visibility, `flock`/`fcntl` lock behaviour, and whether an mmap'd file behaves correctly. Output sets the SQLite journal mode (default TRUNCATE; WAL only if mmap unambiguously passes) and confirms LanceDB's rename-commit is safe. Hard-fail the build if atomic rename is not durable.
- **G-TEXT-EMBED (Phase 1):** load `Qwen3-Embedding-8B` via `sentence-transformers` from disk; confirm output dim = 4096, read the model's own README/`config_sentence_transformers.json` for the **query vs document instruction format** and whether outputs are pre-normalized; embed a 5-pair query/document sanity set and confirm relevant > irrelevant by cosine. Record the confirmed invocation in `state/embed_text_invocation.json`. Only then may B1 run on real media.
- **G-VISUAL-EMBED (Phase 2):** load `Qwen3-VL-Embedding-8B`; confirm dim = 4096; confirm text and image embeddings are co-spatial (a captioned image scores higher to its caption than to a random caption); record invocation in `state/embed_visual_invocation.json`. Only then may B4 run. *(Amendment A1: gate passed; dim=4096 confirmed.)*

---

## 15. Agent rules — Claude Code (any model)

**May do unilaterally:** implement any stage to its contract; write/refactor code under `src/erf/`; create + populate the ERF venvs; fetch model weights and tools listed in §17; write contract validators and tests; run stages, dispatch, reindex; fix bugs; add **nullable, additive** fields with a MINOR `schema_version` bump and a `DECISIONS.md` entry; tune non-contract parameters (thresholds, batch sizes, chunk windows) and record them; author GAMEPLAN.md / DECISIONS.md every session (A5/D-Doc-01 — GAMEPLAN.md replaced CONTEXT.md as the session-state surface).

**May NOT do without a new Opus brief (these are contracts):** change any field's type/meaning/required-set or remove a field (MAJOR); change an ID convention; change a vector dim, metric, or table name; change the relational schema; change stage letters or the depends-on DAG; swap a core model (Qwen3-Embedding-8B, Qwen3-VL-Embedding-8B, faster-whisper large-v3, pyannote 4.0.4, PySceneDetect, InsightFace buffalo_l); replace LanceDB or SQLite; relax the single-writer rule; change DC/volume pinning; introduce a new always-on service or daemon; weaken the "catalog canonical / index disposable" invariant; enable a paid frontier LLM as a default.

**Always:** validate output against schema before writing; write status before exit; write-tmp + atomic rename for every file; keep stages idempotent; use per-worker dirs; log `WARN`/`ERROR` explicitly; one substantive brief's worth of work per session, then update GAMEPLAN.md + DECISIONS.md **and commit in-session** (canon discipline, README §0). Read README.md (the spine) then GAMEPLAN.md before starting; never modify a schema table to make your code easier — the schema is the contract, your code conforms to it.

---

## 16. Editorial export spec (Phase 4 — architected now)

OpenTimelineIO (OTIO) is the canonical interchange. FCPXML and EDL are adapters off the OTIO timeline. Resolve imports FCPXML cleanly (timecode + clip names preserved); EDL is the lowest-common-denominator fallback.

- **Build:** from an `export_manifest`, construct an `otio.schema.Timeline` with one track; each clip → `otio.schema.Clip` with `media_reference = ExternalReference(target_url="file://"+abs(source_relpath))` and `source_range = TimeRange(start=RationalTime(in_frame, fps), duration=RationalTime(out_frame-in_frame, fps))`. Clip `name` = label or `media_id`.
- **FCPXML:** `otio.adapters.write_to_file(tl, path, adapter_name="fcpx_xml")`. Frame rate must match the source; mixed frame rates ⇒ one timeline per rate (do not silently conform).
- **EDL:** `adapter_name="cmx_3600"`. Note EDL is single-track, no clip names beyond reel — acceptable for rough pulls only.
- **Resolve workflow:** editor downloads the FCPXML (via UI or Copyparty tunnel), in Resolve `File → Import → Timeline`, relinks media to `/workspace/erf` originals or `media/proxies/`. Proxies are H.264 matching source TC for offline; conform to originals on finish.
- Every export writes an `export_manifest` to `catalog/exports/{export_id}.json` and the emitted file alongside.

## 17. Short-form EDL stub (Phase 5 — schema hooks present now)

Not built in this MASTER. The schema already carries everything the generator will need, so no migration is required when it lands:

- `chunk.emotional_beat`, `chunk.b_roll_flag`, `chunk.speaker_person_id`
- `keyframe.visual_quality_score`, `keyframe.b_roll_flag`, `keyframe.person_ids`
- `event` beat structure (`kind=beat`, `emotional_beat`, `person_ids`, quality/b-roll rollups)

The future generator: takes a template (beat sequence + target duration + platform) + a semantic query → selects events/chunks per beat by score and `emotional_beat` filter, prefers high `visual_quality_score` b-roll for cutaways, attributes dialogue by `speaker_person_id` → emits an OTIO timeline → §16 export. E0 (events) is the stage that will populate beats; it ships skeletal in Phase 2 (one event per scene) and is filled by an LLM tagging pass (DeepSeek Flash) in Phase 5.

Full autonomous engine design intent: `docs/_archive/INTENT_SHORTFORM.md` (archived; partially realized by SHORT/STORY stages)

---

## 18. Implementation order

Strict. No phase starts until the prior phase's contract tests pass on real data (Phase 0 on synthetic). One session ≈ one coherent step; update GAMEPLAN.md + DECISIONS.md at the end of each (A5).

### Phase 0 — Scaffold (no media)
1. Create the §4 tree. Symlink used weights into `weights/`.
2. Write all 13 schemas (§7) to `schemas/`. Write the validator (`src/erf/contracts.py`) and synthetic-data tests in `tests/`. All contract tests pass.
3. Write `src/erf/status.py` (status writer: validate → write-tmp → rename) and `src/erf/ids.py` (the §6 generators).
4. Build the **control** venv; init `erf.db` with §9 DDL + `schema_meta` rows.
5. `erf-probe-storage` → `state/storage_probe.json` (gate **G-STORAGE**). Hard-fail if atomic rename not durable.
6. Commit MASTER.md, GAMEPLAN.md, DECISIONS.md.
**Exit:** contract tests green; storage probe recorded; DB initialised.

### Phase 1 — Audio (speech search works)
1. Build the **audio** venv (use installed pyannote-audio 4.0.4 from `pipeline-v2` as reference; faster-whisper 1.2.x + ctranslate2). Build the **embed** venv on **torch ≥ 2.8 cu128** + sentence-transformers + transformers (the on-disk 2.5.1 venv will not run on Blackwell — populate the empty `qwen3_embed_clean` or a fresh `embed` venv).
2. Confirm pyannote pipeline weights are cached (HF token at `/root/.cache/huggingface/token`); pull `pyannote/speaker-diarization-community-1` (pyannote 4.x default) if absent.
3. Stage **A**: ffprobe → manifest; faster-whisper large-v3 `word_timestamps=True` → words+segments; pyannote diarization → turns; reconcile (assign each word/segment its overlapping turn's `speaker_label`). Status written.
4. Stage **B0**: chunk on speaker turns → 12–35 s windows.
5. Gate **G-TEXT-EMBED**. Then stage **B1**: Qwen3-Embedding-8B on GPU → text embeddings JSONL.
6. Stage **C**: materialise `text_v1`.
7. Minimal `/query` (text modality only) end-to-end on one media file.
**Exit:** one file fully A→B0→B1→C; plain-English speech search returns correct timestamped chunks.

### Phase 2 — Visual (scene + cross-modal search works)
1. Stage **B2**: PySceneDetect ContentDetector → scenes.
2. Stage **B3**: ffmpeg keyframes (scene-midpoint + max one per `KEYFRAME_INTERVAL_MS`) → jpg + webp thumbs.
3. Gate **G-VISUAL-EMBED**. Then stage **B4**: Qwen3-VL-Embedding-8B on GPU → visual embeddings.
4. Stage **C**: materialise `visual_v1`. Stage **P**: proxy + thumbs.
5. Stage **E0** skeletal: one `event` per scene with text rollup.
6. `/query` gains visual + cross-modal (text→frame via Qwen3-VL text path) + RRF fusion + clip assembly + the minimal UI (§12.4).
**Exit:** "find the shot of X" returns correct keyframes/clips with thumbnails; fused text+visual results assemble into clip spans.

### Phases 3–5 — schema present, not implemented
Phase 3 (identity): F0/F1, enrollment (§11), person propagation. Phase 4 (export): §16. Phase 5 (short-form): §17. Each requires its own Opus brief before build; the hooks they consume already exist in the Phase 0–2 schemas (flagged "Phase N hook" throughout §7).

### 18.1 ERF venvs

**Production (separate pods):**

| venv | pod | key packages |
|------|-----|--------------|
| `control` | CPU control pod | fastapi, uvicorn, lancedb, pyscenedetect, pydantic, jsonschema, opentimelineio, static-ffmpeg, requests |
| `audio` | GPU audio worker | faster-whisper==1.2.*, ctranslate2==4.6.*, pyannote.audio==4.0.4, torchaudio==2.11.0+cu128, torch (cu128) |
| `embed` | GPU embed worker | torch>=2.8 (cu128), torchvision (cu128), transformers, sentence-transformers, accelerate, lancedb, transnetv2-pytorch |

**Single-pod dev (current setup — this pod):**

`control` and `embed` are merged into `venvs/embed/`. All stages and the API server run from `venvs/embed/`. `venvs/audio/` remains isolated (pyannote + torchaudio pinning). `venvs/control/` exists but is no longer the primary venv.

**Why:** On a single dev pod the control/embed split is friction not protection — their packages are complementary with no version conflicts. The three-venv model applies when deploying separate control and GPU pods in production. Restore the split at that point by activating the appropriate venv per pod role.

**torchaudio note:** Always install from `--index-url https://download.pytorch.org/whl/cu128`. Default PyPI build has CUDA 13.x ABI and fails on CUDA 12.x (see D-P2-02).

### 18.2 Fetch list (only what's missing)

On disk already, do **not** refetch: faster-whisper large-v3, pyannote-audio suite, Qwen3-Embedding-8B, Qwen3-VL-Embedding-8B (the 2B model is superseded; 8B at `/workspace/weights/qwen3/Qwen3-VL-Embedding-8B/`), InsightFace buffalo_l, open_clip (lever). Missing:

- **pyannote diarization pipeline weights** if not cached — pulled by code via HF token at first load.
- **GroundingDINO** (Phase 3 lever; populate the existing `groundingdino` venv) — defer.
- **Nothing else.** No Ollama (removed 2026-06-22), no wav2vec2 (D2), no YOLOv12 (D4), no OpenCLIP fetch (D3).

```bash
# DeepSeek key — set in the control pod environment (persist in /workspace/.erf.env)
echo 'export DEEPSEEK_API_KEY=...' >> /workspace/.erf.env
```

### 18.3 Module map (`src/erf/`)

```
src/erf/
├── contracts.py      # schema load + validate
├── ids.py            # deterministic ID generators (§6)
├── status.py         # status writer (validate→tmp→rename)
├── atomicio.py       # write_tmp_rename, jsonl append, file locks
├── manifest.py       # ffprobe → media_manifest (stage A)
├── asr.py            # faster-whisper + pyannote + reconcile (stage A)
├── chunk.py          # B0
├── embed_text.py     # B1 (Qwen3-8B)
├── scenes.py         # B2 (PySceneDetect)
├── keyframes.py      # B3 (ffmpeg)
├── embed_visual.py   # B4 (Qwen3-VL-8B)
├── materialise.py    # C (LanceDB upsert)
├── events.py         # E0
├── proxy.py          # P (ffmpeg)
├── reindex.py        # erf-reindex
├── dispatch.py       # erf-dispatch (RunPod REST/GraphQL)
├── worker.py         # erf-worker (GPU pod loop)
├── llm.py            # pluggable backend (§13)
├── query.py          # /query pipeline (§12)
├── api.py            # FastAPI app + static UI
├── enroll.py         # Phase 3
└── export.py         # Phase 4 (OTIO/FCPXML/EDL)
```

---

## 19. Levers (reserved alternatives — switching any is a parameter/Opus-brief decision, not a default)

- **ASR speed:** fetch faster-whisper large-v3-turbo if CPU-trickle throughput is the bottleneck.
- **Word-boundary precision:** add WhisperX/wav2vec2 forced alignment as a post-pass on A if editorial needs tighter word cuts.
- **Gradual transitions:** TransNetV2 (installed) alongside PySceneDetect for dissolves/fades B2 misses.
- **Cross-modal CPU fallback:** OpenCLIP (`cpu_vision` venv) when no GPU is up and a quick visual query is needed.
- **Object boxes:** GroundingDINO open-vocab enrichment populating `detection` for Phase 5 auto-framing.
- **Embedding size:** Qwen3 MRL truncation (text→1024/2048) or fp16 storage if the index ever grows enough to matter (it won't soon — 2.1 PB volume).
- **Better local LLM:** qwen2.5:14b on the control pod for richer beat tagging if RAM allows.

---

## 20. Unresolved — requires project-lead input (ALL RESOLVED — historical, A9)

All four original items resolved in practice by 2026-07-10; kept for the record:

1. **RunPod API key** — resolved; GPU work runs on the current pod, no separate dispatch key ever became blocking.
2. **DeepSeek API key** — resolved; `DEEPSEEK_API_KEY` set via environment (`/workspace/.pod-persist/env.sh` / settings).
3. **Control pod provisioning** — resolved in practice: the single Blackwell pod serves as both dev host and runner; a separate always-on control pod was never needed at current scale.
4. **Source media root** — resolved: source media lives under `/workspace/BTS`; ingest is invoked per-batch (`bin/batch_ingest_bts.py`). Ingest automation (#15) remains deferred (GAMEPLAN "Deferred jobs").

New unresolved items go to `GAMEPLAN.md` (the living surface), not here.

---

*End MASTER.md v1.5.0.*

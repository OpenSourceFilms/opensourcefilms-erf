# Story Engine — Phase 0 Seam Contracts (DRAFT for Opus review)

Status: **DRAFT — Code's first pass, grounded in real schemas. Not frozen.**
Author: Code. Reviewer/ratifier: Opus → the project lead.
Companion to: `docs/_archive/STORY_ENGINE_BRIEF.md` (§5 Phase 0; v1 brief, archived 2026-07-10 — precedence chain unchanged).
Precedence on conflict: **MASTER.md → STORY_ENGINE_BRIEF.md (§8) → patch v3 → v2/v1.**

This document drafts the **three frozen seams** the engine serves (Brief §5). It does
**not** include the probe harness spec (Brief reserves that for Opus) and contains **no
funnel code** (Phase 0 is hard-gated on these seams + the probe). Per the project lead’s instruction,
Code drafts these against the *real* schemas so Opus reviews a concrete mapping rather than
inventing one blind.

> **Three facts that shaped this draft — read first:**
> 1. **`patch_v3.md` is now on disk** (`/workspace/erf/docs/patch_v3.md` — the consolidated/
>    corrected version, all §8 fixes + D2 baked in). It supplies the candidate schema (§2), the
>    beat schema + audio fields (§3), and lane definitions (§1). The earlier `‹PROVISIONAL›`
>    markers are therefore **resolved**: Seam 1 §1.2/§1.3 and the audio checks §1.5 are
>    reconciled against it; **D1 and D3 closed**.
> 2. **The real `beat_plan` schema is `1.3.0`, not 1.2.0** (MemPalace was stale). Verified
>    from `schemas/beat_plan.schema.json` `$id: erf://schemas/beat_plan/1.3.0`.
> 3. **Stage A already exists.** Per `DECISIONS.md` D-Story-01 (2026-06-19), editorial-function
>    labelling is implemented as **stage LABEL** (`src/erf/editorial.py` + `bin/label_editorial.py`
>    → `catalog/editorial/{media_id}.json`, Flash). Its taxonomies (26 FUNCTIONS / 8 TENSIONS /
>    6 ARCHETYPES) and output shape are **concrete in code** — so Seam 2 *formalizes an existing
>    artifact*, it does not invent one, and the label vocabulary is **not** blocked on patch v3.
>    (Brief Phase 1 is therefore largely *built*; what remains is versioning/staleness — i.e.
>    Seam 2 itself.) ⚠ The current artifact is a **flat, overwrite-on-regen file**, which appears
>    to violate Invariant 2 (immutability); Seam 2 fixes this.

## Grounding sources (verbatim, with refs)

| Fact | Source |
|---|---|
| beat_plan target schema 1.3.0 | `schemas/beat_plan.schema.json` |
| Schema validation entrypoint | `src/erf/contracts.py` → `validate(record, "beat_plan")` |
| beat_plan storage + id convention | `src/erf/author.py:1030,1056-1059` → `catalog/beat_plans/{bp_id}.jsonl`, `bp_<YYYYMMDDTHHMMSSZ>_<6hex>`, append-only (latest = last line) |
| beat_plan loader (latest revision) | `bin/batch_render_narrative.py:63` `_load_latest_beat_plan` |
| "select atom, never invent timecodes" pattern | `src/erf/author.py:388-408` (Pass 2 prompt) + `build_atom_pool:457` |
| quote-completion repair + flags | `src/erf/editorial.py` (`quote_completeness {complete_thought, requires_previous, requires_next}`, repair fns L66/82/92) |
| Downstream-only pacing (NOT called by adapter) | `bin/batch_render_narrative.py:85` `_resolve_beat_boundaries`, `:112` `_trim_crosstalk_out` |
| Journal writer + table shape | `src/erf/runlog.py` (`events`, `status_projection`) |
| Journal spine prose + invariants | `MASTER.md` §2 (L195-204), Invariants 2/3/5/8 (L75-82), Agent rules (L856-861) |
| **existing editorial labels (Stage LABEL)** | `src/erf/editorial.py` (`label_clip`, FUNCTIONS/TENSIONS/ARCHETYPES) + `bin/label_editorial.py` → `catalog/editorial/{media_id}.json`; DECISIONS D-Story-01 |
| editorial_signals provenance precedent | `schemas/event_signals.schema.json` (Stage E1 per-event Flash signals) |

---

## Seam 1 — `beat_plan` adapter contract (Stage I)

**Target is frozen and known.** `beat_plan` schema `1.3.0`. The adapter's only authorized
output is a beat_plan that **passes `contracts.validate(bp, "beat_plan")`**. If it does not
validate, it does not render (Brief §1, §4.6).

### 1.1 Target field set (beat_plan 1.3.0) — what the adapter MUST emit

Top-level required: `schema_version="1.3.0"`, `bp_id`, `created_at_utc`, `topic`,
`format_fit ∈ {tiktok,reels,youtube_shorts,longform}`, `duration_target_s` (int ≥1), `beats[]`
(≥1). Optional: `revision`(default 0), `supersedes_revision`, `total_duration_s`,
`assembler_note`, `template_id`.

Per-beat required: `beat_index`(int ≥0), `chunk_id`, `media_id`, `in_ms`, `out_ms`,
`edit_role ∈ {hook,body,b_roll,proof,quote,transition,call_to_action,exposition}`.
Per-beat optional: `note`, `hook_score`, `quote_score`, `proof_score`,
`suggested_padding_ms`, `duration_s`, `b_roll_candidates[]`, `b_roll_selected`,
`broll{enabled,keyframe_id,reason}`.

### 1.2 Adapter input requirement (the seam, stated as obligations on the candidate)

The candidate `beats[]` schema is now defined in `patch_v3.md` §2–§3. The adapter consumes a
candidate beat (`patch_v3.md` §3: `role`, `clip_id`, `sentence_range`, `evidence_ids`,
`audio_mode`, `payoff_type`, …). A candidate beat is adaptable **iff** it resolves,
deterministically and without inventing timecodes, to:

| Target field | Required candidate input | Resolution rule |
|---|---|---|
| `media_id` | beat `clip_id` (one clip per `media_id`; evidence at `catalog/evidence/{media_id}.json`) | passthrough; must exist in `catalog/media_manifest/` (Brief §3 registry truth) |
| `chunk_id` | derived from `sentence_range`/`evidence_ids` (`clip_id.sN`) | resolve the sentence span → time → the covering `chunk_id`; **must be a real id** (C2 already validated `evidence_ids`); adapter re-asserts, never fabricates |
| `in_ms`, `out_ms` | the beat's `sentence_range` | resolve to ms via `catalog/asr/<media>.segments.jsonl` and copy those exact values (`author.py:400,408` pattern — no timecode synthesis) |
| `edit_role` | beat `role` + `audio_mode` + `payoff_type` | mapped via §1.3 → the 8-value enum |
| `note` | candidate rationale (≤ N chars) | passthrough, optional |
| `hook/quote/proof_score` | candidate scores | passthrough if present, else null |
| `b_roll_*` / `broll` | candidate visual selection | passthrough if present, else null/absent |

`topic`, `format_fit`, `duration_target_s` come from the candidate's envelope. **Lane (D2,
resolved)** is *not* a candidate→beat_plan field: the adapter reads the candidate's claimed
`primary_format_lane` only to assert the resolved total duration falls in that lane's band,
then **drops it**. The emitted beat_plan is lane-free. `format_fit` is platform, unrelated to
lane.

### 1.3 `edit_role` mapping (from candidate beat `role` + `audio_mode` + `payoff_type`)

The adapter's input is the **candidate beat** (`patch_v3.md` §3), **not** the sentence-level
`editorial.py` FUNCTIONS — those label sentences and feed candidate *generation* upstream; the
adapter never sees them. Map the beat `role` (`HOOK|WANT|OBSTACLE|ATTEMPT|DECISION|TURN|RESULT|
PAYOFF|BUTTON`), refined by `audio_mode`/`payoff_type`, onto the 8-value `edit_role` enum.
Default map (Opus to confirm the judgement calls), applied in precedence order:

```
condition                                                       → edit_role
role == HOOK                                                    → hook
role == BUTTON                                                  → call_to_action
audio_mode == sync_dialogue  (spoken on-screen line)           → quote
role == PAYOFF and payoff_type ∈ {visual_proof,
  technical_reveal, before_after}                              → proof
audio_mode ∈ {broll_audio, music_only, silent_visual}         → b_roll
audio_mode == vo  (narration over visuals)                     → exposition
otherwise (WANT/OBSTACLE/ATTEMPT/DECISION/TURN/RESULT)         → body
```

`transition` has no candidate source (renderer-internal) and is never emitted by the adapter.
If a future candidate carries an explicit `edit_role`, the adapter validates it against the
enum and passes it through; the map is the fallback.

**Precedence consequence (Opus to confirm):** because `audio_mode==sync_dialogue` is checked
*before* `PAYOFF→proof`, a *spoken* payoff (`role=PAYOFF`, `audio_mode=sync_dialogue`) maps to
`quote`, not `proof` — `proof` is reserved for non-sync visual payoffs. Implemented + tested in
`src/erf/se_adapter._edit_role`. If a spoken PAYOFF should instead be `proof`, swap those two
lines in the precedence. (Likewise `role=HOOK` always wins → `hook`, even if spoken.)

### 1.4 Adapter procedure (the validator + builder)

1. **Re-assert IDs** (`media_id`, `chunk_id`) resolve against catalog. Fabricated/ambiguous
   ⇒ reject deterministically (Brief §4.6 — no invalid input proceeds; C2 is the upstream
   floor, this is the last-line re-check at the seam).
2. **Resolve span → atom → `in_ms/out_ms`** by copying the atom's exact values. No timecode
   synthesis.
3. **Quote-completion repair** for spoken-quote beats only, via the existing repair in
   `src/erf/editorial.py` using `quote_completeness {complete_thought, requires_previous,
   requires_next}`. A `requires_previous/next` quote extends to the adjacent atom so the
   thought is whole. (Brief §5 Seam 1 "runs quote-completion repair".)
4. **Map `edit_role`** (§1.3).
5. **Lane duration-band check + drop (D2)**: assert resolved total duration ∈ the candidate's
   claimed `primary_format_lane` band; on mismatch, journal and reject. Then drop lane — it is
   not written to the beat_plan.
6. **Assemble top-level**: assign `bp_id = bp_<UTC>_<6hex>`, `created_at_utc`, `revision=0`,
   `format_fit`, `duration_target_s`, `topic`. (No lane field — beat_plan stays 1.3.0.)
7. **Validate**: `contracts.validate(bp, "beat_plan")` **plus** the new cross-field checks in
   §1.5. Any failure ⇒ do **not** write; journal `error` with the validation messages;
   return rejection.
8. **Write atomically** to `catalog/beat_plans/{bp_id}.jsonl` (write-tmp + rename; append-only
   revision model — latest is last line). Update `catalog/beat_plans/_thread_index.json` only
   if the engine is thread-scoped (else skip; the engine is corpus-scoped — see D4).
9. **Journal** a model/deterministic-stage event per Seam 3 (`stage="SE:I"` — see D5). The
   journal payload **may** record lane for provenance (it's observability, not the seam); the
   beat_plan itself stays lane-free.

### 1.5 Audio contradiction checks live at Stage C2, not the adapter (D3, resolved)

`patch_v3.md` §3 defines the audio fields (`audio_mode`, `sync_audio_required`,
`dialogue_required`, `payoff_type`, `visual_support`) on the **candidate beat**, and §4 puts
the §8.5 contradiction checks in **Stage C2** (programmatic validation): rejects
`silent_visual + dialogue_required`, `visual_proof + no visual_support`,
`sync_audio_required + empty sentence_range`, etc. — *before* the adapter, on the candidate.
The **beat_plan** has no audio fields, so `contracts.py` keeps only its existing cross-field
checks (beat_plan 1.3.0 untouched) and the adapter trusts C2's guarantee.

**Deliverable when frozen:** this mapping doc (above) + `src/erf/se_adapter.py` implementing
§1.4 + tests asserting (a) valid candidate → schema-valid beat_plan, (b) every reject path
journals and never writes, (c) timecodes are byte-identical to the source atom.

---

## Seam 2 — `editorial_signals` schema + resolver (formalizes the existing LABEL stage)

**Not net-new.** Stage LABEL already produces `catalog/editorial/{media_id}.json` via
`src/erf/editorial.py` (`label_clip`) + `bin/label_editorial.py`, Flash. Seam 2's job is to
**freeze that artifact as a versioned, staleness-tracked contract** — the shape and label
vocabulary are already concrete in code; what's missing is `schema_version`, provenance,
immutable versioning, and a manifest resolver.

### 2.1 The existing shape (grounded — `editorial.py:92-109`)

Per `media_id` (= per clip; files are `m_<12hex>.json`):
```
{ media_id,
  clip_level: { story_forces: {want,obstacle,decision,turn,payoff},  // each str|null
                central_tension,                                     // str|null ∈ TENSIONS
                archetype_candidates: [str] },                       // ∈ ARCHETYPES
  sentences: [ { sentence_index: int,
                 functions: [str],          // ∈ FUNCTIONS (26)
                 tension_tags: [str],       // ∈ TENSIONS (8)
                 chain_affordances: {can_start_story, can_escalate_story, can_payoff_story, needs_followup},
                 quote_completeness: {complete_thought, requires_previous, requires_next} } ] }
```
Vocabularies are fixed in `editorial.py`: **FUNCTIONS** (HOOK, CONTEXT, DESIRE, OBSTACLE,
STAKES, FAILED_ATTEMPT, DISCOVERY, DECISION, ESCALATION, REVERSAL, JOKE, REVEAL,
CRAFT_INSIGHT, PAYOFF, BUTTON, B_ROLL_SUPPORT, SCRAPPY_SOLUTION, INDIE_CONSTRAINT,
VFX_PROBLEM, PRACTICAL_TEST, PERFORMANCE_DISCOVERY, GEAR_HACK, LOCATION_PROBLEM,
TEAM_DYNAMIC, DANGEROUS_IDEA_BUT_SAFE, OPEN_SOURCE_METHOD); **TENSIONS** (ambition_vs_budget,
safety_vs_spectacle, professional_standard_vs_homemade, beautiful_idea_vs_ugly_reality,
simple_plan_vs_unexpected_constraint, ai_vfx_promise_vs_practical_mess,
serious_film_vs_absurd_method, control_vs_chaos); **ARCHETYPES** (THIS_SHOULD_NOT_WORK,
THE_INVISIBLE_PROBLEM, THE_SAFETY_PARADOX, THE_PROFESSIONAL_VERSION_COSTS_50K,
THE_SHOT_GOT_BIGGER, THE_DUMB_LOOKING_THING_IS_THE_SERIOUS_THING).

These are the labels §8.2 dedup reasons over (`central_tension`, `archetype_candidates`) and
the affordances the candidate generator chains on. They are also the `FUNCTIONS` source for the
Seam 1 `edit_role` map (§1.3).

### 2.2 What Seam 2 adds (the freeze)

1. **`schema_version` + provenance envelope** (additive, MINOR — non-breaking): `schema_version`,
   `editorial_job_id`, `source_evidence_hash`, `produced_by`, `model`, `mode`, `stage_version`,
   `created_at_utc`. The current artifact carries none of these.
2. **Immutable versioning + manifest resolver** (fixes the ⚠ Invariant-2 gap; D-Story-01
   approved as a standalone fix 2026-06-20): replace the flat overwrite-on-regen
   `catalog/editorial/{media_id}.json` with `catalog/editorial/{media_id}/{editorial_job_id}.json`
   (immutable; **keyed by the editorial job's own id**, never a story-batch `run_id` — §8.3) plus
   `catalog/editorial/{media_id}/manifest.json` as the latest-version resolver. Consumers read
   labels **only through the manifest**, never by globbing.
   - **Reader migration (the other half of the fix).** A versioned writer + an old reader still
     opening the flat filename = a half-fix that silently reads stale labels. **Verified: there
     are currently *zero* readers** of `catalog/editorial/` in `src/erf` or `bin` (the only
     consumer — the candidate generator — isn't built yet). So the migration is writer-side
     today; the standing rule is **forward**: Phase 2's candidate generator must read labels
     **only via the manifest resolver** (`src/erf/editorial_signals.py`), never the flat path.
     A test asserts no module opens `catalog/editorial/*.json` directly.
3. **Freeze the taxonomies as enums** so drift is a schema violation, not silent.
4. **Staleness:** `source_evidence_hash` over the input evidence package (the `pkg` fed to
   `label_clip`); a label set is stale when its hash ≠ the current evidence hash. A Flash/parse
   failure writes a `parse_error` record (explicit outcome, never silent absence — Invariant 3).

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "erf://schemas/editorial_signals/1.0.0",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version","media_id","editorial_job_id","source_evidence_hash",
               "produced_by","model","mode","stage_version","created_at_utc",
               "clip_level","sentences"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "media_id": {"type": "string"},
    "editorial_job_id": {"type": "string", "description": "this LABEL job's own id (NOT a story-batch run_id)"},
    "source_evidence_hash": {"type": "string", "description": "hash of the evidence pkg fed to label_clip; staleness key"},
    "produced_by": {"type": "string"},
    "model": {"type": "string", "description": "deepseek-v4-flash"},
    "mode": {"type": "string", "enum": ["thinking","non-thinking"]},
    "stage_version": {"type": "string"},
    "created_at_utc": {"type": "string", "format": "date-time"},
    "parse_error": {"type": ["string","null"], "default": null},
    "clip_level": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "story_forces": {
          "type": "object", "additionalProperties": false,
          "properties": {
            "want": {"type": ["string","null"]}, "obstacle": {"type": ["string","null"]},
            "decision": {"type": ["string","null"]}, "turn": {"type": ["string","null"]},
            "payoff": {"type": ["string","null"]}
          }
        },
        "central_tension": {"type": ["string","null"], "enum": [/* TENSIONS + null */]},
        "archetype_candidates": {"type": "array", "items": {"type": "string", "enum": [/* ARCHETYPES */]}}
      }
    },
    "sentences": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["sentence_index","functions","tension_tags","chain_affordances","quote_completeness"],
        "properties": {
          "sentence_index": {"type": "integer", "minimum": 0},
          "functions": {"type": "array", "items": {"type": "string", "enum": [/* FUNCTIONS */]}},
          "tension_tags": {"type": "array", "items": {"type": "string", "enum": [/* TENSIONS */]}},
          "chain_affordances": {
            "type": "object", "additionalProperties": false,
            "required": ["can_start_story","can_escalate_story","can_payoff_story","needs_followup"],
            "properties": {
              "can_start_story": {"type": "boolean"}, "can_escalate_story": {"type": "boolean"},
              "can_payoff_story": {"type": "boolean"}, "needs_followup": {"type": "boolean"}
            }
          },
          "quote_completeness": {
            "type": "object", "additionalProperties": false,
            "required": ["complete_thought","requires_previous","requires_next"],
            "properties": {
              "complete_thought": {"type": "boolean"},
              "requires_previous": {"type": "boolean"}, "requires_next": {"type": "boolean"}
            }
          }
        }
      }
    }
  }
}
```

**Deliverable when frozen:** `schemas/editorial_signals.schema.json` (enums filled from
`editorial.py`) + a small migration in `editorial.py`/`label_editorial.py` to emit the
provenance envelope and write the versioned path + manifest + a resolver in
`src/erf/editorial_signals.py` (get-latest-by-media, staleness) + tests. Adding a schema and
nullable-additive fields is within Code's unilateral rights (MASTER Agent rules), **but this
is a frozen seam → ships only after Opus ratifies** (Brief §7), and the path migration is a
behaviour change worth a DECISIONS entry.

---

## Seam 3 — Journal event shape

**Fully grounded — no patch v3 dependency.** Reuses the existing spine; the engine adds
**no new tables** and **no new writer**.

### 3.1 The existing tables (do not change — Invariant 5, single-writer)

```
events(at_utc, job_id, media_id, stage, event_type, run_id, payload)   -- append-only, every stage
status_projection(media_id, stage, ok, run_id, at_utc, error)          -- latest per (media_id, stage)
```
- Written **only** via `erf.runlog.stage_run(...)` / `erf.runlog.log_event(...)`.
- `event_type ∈ {start, ok, error, skip}`; `ok|error|skip` project into `status_projection`.
- `payload` is a free-form JSON dict — **this is where the engine's structured provenance
  goes**. The two event classes below are *payload conventions*, not schema changes.
- Invariant 3 / D-Log-01: status-before-exit, no silent failures; consumers must inspect
  `ok`/`error` (a caller that only checks "didn't raise" reports false success).

### 3.2 Deterministic-stage event payload (B, C2, C3, I)

```json
{
  "inputs":   [{"artifact": "...", "path": "...", "hash": "..."}],
  "decisions": {"kept": 0, "rejected": 0, "...": "stage-specific counts"},
  "skips":     [{"id": "...", "reason": "..."}],
  "exclusions":[{"id": "...", "reason": "fabricated_id|ambiguous|dup"}],
  "outputs":  [{"artifact": "...", "path": "...", "hash": "..."}],
  "lane":      "<primary_format_lane or _multi>"
}
```

### 3.3 Model-stage event payload (any Pro/Flash call: A, C, critic, tournament, mutation)

```json
{
  "model": "deepseek-v4-pro",
  "mode": "thinking|non-thinking",
  "temperature": 0.0,
  "latency_ms": 0,
  "usage": {"prompt_tokens": 0, "completion_tokens": 0},
  "schema_validation": {"ok": true, "errors": []},
  "decision_summary": "<=200 char",
  "candidate_ids": ["..."],
  "lane": "<primary_format_lane>"
}
```

These satisfy Brief §5 Seam 3 exactly (model, mode, temperature, latency, usage,
schema-validation, decision_summary for model stages; inputs/decisions/skips/exclusions/
output-provenance for deterministic stages).

### 3.4 Two conventions that need ratification (see D4, D5)

- **`media_id` column for corpus-level stages.** The engine's stages are mostly
  **corpus-wide**, not per-media; the column is non-null in practice. Propose a sentinel
  `"_corpus"` (or the `story_batch_id`) so corpus-level events are queryable and don't
  masquerade as media `""`. → **D4**.
- **`stage` namespace.** Pipeline uses `A–F`/`T0`/`EV`; the engine's `A/B/C2/C3/I` would
  **collide** in journal queries (esp. Stage A ingest vs Stage A editorial). Propose prefix
  `SE:` (`SE:A`, `SE:B`, `SE:C2`, `SE:C3`, `SE:C`, `SE:I`, `SE:critic`, `SE:tourney`,
  `SE:mutate`). → **D5**.

---

## D. Open decisions requiring Opus/project lead ratification

Per Brief §7, any change to a frozen seam escalates. These block freeze:

| # | Decision | Why it matters | Code's recommendation |
|---|---|---|---|
| **D1** | ✅ **RESOLVED — `patch_v3.md` is on disk** (`docs/patch_v3.md`, consolidated/corrected): candidate schema §2, beat+audio §3, lanes §1. | Closes the candidate-side gap; Seam 1 §1.2/§1.3 reconciled against it. | — |
| **D2** | ✅ **RESOLVED (Opus, 2026-06-20): lane is an upstream selection concept only.** It lives on the candidate + `shortlist_by_lane.json`; the adapter uses it **once** for a final duration-band sanity check (candidate claimed 15–25s lane, resolved duration 19s ✓) and then **drops it**. It does **not** enter `beat_plan`. No contract change — beat_plan stays **1.3.0**. | Invariant 4's "end to end" overreached by one stage: lane runs **end-to-seam, not end-to-render**. Every consumer (candidate, shortlist, open-thread registry) sits before/beside the seam; nothing downstream needs it (duration is computable from the beats). | If a downstream render difference is ever needed, add an explicit `render_profile` field set at the adapter — **never overload lane**. |
| **D3** | ✅ **RESOLVED — audio checks live at Stage C2 on the candidate** (`patch_v3.md` §3 defines the fields, §4 C2 enforces). | beat_plan 1.3.0 has no audio fields and stays untouched; the adapter trusts C2. | — (see §1.5) |
| **D4** | **`media_id` value for corpus-level engine events.** | Journal queryability; avoid false media scoping. | Sentinel `"_corpus"`. |
| **D5** | **Story-engine `stage` namespace.** | Avoid collision with pipeline `A–F`. | Prefix `SE:`. |
| **D6** | **Ratify the editorial_signals path migration** from flat `catalog/editorial/{media_id}.json` (overwrite-on-regen) to versioned `catalog/editorial/{media_id}/{editorial_job_id}.json` + `manifest.json`. | Current flat overwrite **violates Invariant 2** (immutability); migration fixes it. Granularity = per `media_id` (confirmed from existing artifact); vocab grounded in `editorial.py` (no patch v3 needed). | Migrate + DECISIONS entry. |
| **D7** | **Confirm `editorial.py` quote-completion repair is the canonical one the adapter reuses** (vs a story-engine-specific repair). | Avoid a second divergent repair implementation. | Reuse `editorial.py`. |

---

## Not in this document (by design)

- **Probe harness spec** — Brief reserves for Opus; the probe also needs the
  `deepseek-v4-pro` thinking endpoint confirmed/extended in `ds_flash.sh` first (Brief §3).
- **Any funnel code** — Phase 0 is hard-gated on these seams + the probe (Brief §5).
- **Stage A/B/C implementation** — Phases 1–3.

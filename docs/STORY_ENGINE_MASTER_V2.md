# Story Engine — Master Plan v2 (Reformulation)

**Author:** Opus (4.8), 2026-06-26. **Ratifier:** the project lead. **Status:** ACTIVE (SE-0 through SE-2 complete, see §10; ratified via D-Story-03) — supersedes
`STORY_ENGINE_BRIEF.md` (v1, 2026-06-19) *on the project lead's sign-off*, not before.
**Relationship to existing docs:** v1 brief, `patch_v3.md`, and `STORY_ENGINE_PHASE0_SEAMS.md`
remain valid where this doc doesn't override them. This doc is the new top of the stack.

**Precedence on conflict:** `MASTER.md` (v1.3.0) → **this doc** → v1 brief → `PHASE0_SEAMS`
(on its three seam topics) → `patch_v3`.

This is a *reformulation*, not a restart. v1's thesis is correct and kept. What changed is
(a) the plan now starts from **measured build reality** instead of assumptions, (b) seven
design points that are no longer ideal are corrected, and (c) a concrete grand plan +
execution-oversight design replaces v1's single-paragraph loop.

---

## 0. The one idea (unchanged from v1)

The engine is **one new ideation producer that emits *validated* `beat_plan` artifacts** —
that is the entire integration. It mines the BTS corpus for genuinely compelling short-form
stories, **proves their mechanics from evidence, structurally kills the boring ones**, and
hands survivors to the existing renderer. It does **not** replace the pipeline and **does not
touch anything downstream of `beat_plan`** (resolve → pacing → gate → render all work and stay
shared). `beat_plan` (schema 1.3.0) is the frozen seam; nothing the engine produces is true
until it validates against it.

---

## 1. Build reality — the baseline this plan starts from (audited 2026-06-26)

| Layer | State | Evidence |
|---|---|---|
| Seam 1 — beat_plan adapter | **Built**, not formally frozen | `src/erf/se_adapter.py` (Stage I); honors §9 enum |
| Seam 2 — editorial_signals | **Built + FROZEN** | `editorial_signals.py` + schema; **D-Story-02** froze it + landed the immutability fix; versioned paths live |
| Seam 3 — journal events | **Built**, not formally frozen | `journal.py`; `se_chain` already emits model-stage events |
| SE-1 editorial labels (Stage A) | **DONE at corpus scale** | 514 versioned files in `catalog/editorial/` |
| SE-2 deterministic spine (B/B2/C2/C3) | **NOT built as specified** | no mining/dedup module; C2 exists only *embedded* in `se_chain` |
| SE-3 vertical slice (C→adapter→render) | **Built + exercised as a shortcut** | `bin/se_chain.py`; 8 runs in `runs/se_chain_*` |
| SE-4 quality (critic/tournament/mutation) | **NOT built** | no code |
| SE-5 open-thread registry | **Schema only** | `narrative_thread.schema.json` exists; no registry, no `open_threads/` |
| Phase-0 **PROBE** | **NEVER RUN** | no `measurements.json` anywhere |
| Pro/Flash client | **Ready** | `src/erf/llm.py`: `deepseek-v4-pro` (thinking) / `-flash`; 1800s timeout; retry/backoff |

**Reusable as-is:** editorial labels, `se_adapter`, `render`, `erf.llm`, `journal`, the
`beat_plan`/`narrative_thread`/`editorial_signals` schemas.
**Net-new to build:** the probe; the SE-2 deterministic spine (mining + dedup + standalone C2
floor); the SE-4 quality layer; the SE-5 registry. **To rewire:** SE-3's Pro stage (see C4).

---

## 2. What changed from v1, and why (the corrections)

**C1 — Rename the engine's phases; kill the "Phase 4" collision.** v1 §6 "Phase 4" (quality
enforcement) collides with the unrelated operational `PHASE_4_BRIEF.md` *and* with the
colloquial "Phase 4 = the whole storyteller effort." This doc uses **SE-0 … SE-5** for engine
phases and the ratified `SE:` journal-stage prefix (D5). "Phase 4" is retired as a term here.

**C2 — The thinking-mode prerequisite is already met.** v1 §3 says "confirm/extend
`ds_flash.sh` for `deepseek-v4-pro` thinking mode before Phase 0." Reality: **"thinking" is
just selecting the `deepseek-v4-pro` model** — there is no separate flag — and `src/erf/llm.py`
already does this with a 1800s timeout and retry/backoff. The engine doesn't use `ds_flash.sh`
at all. **The probe runs on `erf.llm`.** `ds_flash.sh` parity is optional and not on the
critical path.

**C3 — Seam 2 is done.** v1/`patch_v3` treat editorial versioning as to-build. It's **frozen
and migrated** (D-Story-02). Remove it from the worklist; it's a dependency, not a task.

**C4 — `se_chain`'s shortcut is scaffolding, not architecture. (The most important
correction.)** `se_chain` selects rich clips and has **DS Pro *invent* the candidates**
(structure *and* the `clip_id`/`sentence_range` references), validating (C2) only *afterward*.
This is a fine Phase-SE-3 "prove the pipe carries water" demo, but it must **not** become the
production path, for three independent reasons:
  1. **It violates Standing Rule 6.** Pro is the ≤30-min/call stage; letting it originate IDs
     means hallucinated/array-index IDs are paid for, then caught downstream. The floor must be
     *before* Pro.
  2. **Pro thinking risks diversity collapse** (v1's own stated concern). Ideation diversity
     belongs to Flash (temp 0.6–0.9, archetype prompts, concurrency 6) — cheap and parallel.
     Pro should *judge and assemble*, not *generate*.
  3. **The quality layer the project lead actually wants cannot run on the shortcut.** Critic bundle +
     tournament + mutation need a **population of competing candidates per lane**; a single Pro
     call yields ~one candidate. **SE-4 structurally requires the SE-2 funnel.** This is the
     clincher: the shortcut is a dead end *for the goal*.
  Decision: keep `se_chain` as a throwaway pipe-proof; production runs the SE-2 spine so Pro
  **chain-completes pre-validated, evidence-grounded ingredient sets** and **never originates
  IDs**. A regression guard enforces this (see §7).

**C5 — Probe latency is decisive and may reshape the funnel.** Pro thinking calls can take up
to 30 min (`_PRO_TIMEOUT=1800`) at `max_concurrent=1` to start. The `patch_v3` budgets (≤42 /
≤69 Pro calls, ~8h wall-clock) are **provisional assumptions**. §8 defines explicit
probe-outcome branches so we adapt the funnel shape to measured reality instead of discovering
it mid-run.

**C6 — Division of labour, hardened: Flash for diversity, Pro for judgment.** Generation /
ideation / mutation = Flash (cheap, diverse, parallel). Chain-completion judgment, critic
scoring, pairwise tournament = Pro. This is now a design rule, not a preference (see §4 Rule 6+).

**C7 — Ratify the open seam conventions (D4/D5/D7).** As Opus reviewer I recommend accepting
all three; they're plumbing with obvious right answers (see §9). The project lead signs off.

---

## 3. Frozen contracts (the seams)

The three seams are the engine's only sanctioned boundaries. **Seam 2 is frozen.** Seams 1 & 3
are *implemented and exercised* but await formal freeze on this review (§9).

- **Seam 1 — candidate → `beat_plan` adapter (`se_adapter.py`, Stage I).** Maps the engine's
  internal candidate (`patch_v3` §2/§3: `role`/`clip_id`/`sentence_range`/`evidence_ids`/
  `audio_mode`…) onto the real `beat_plan` 1.3.0 schema (`beat_index`/`chunk_id`/`media_id`/
  `in_ms`/`out_ms`/`edit_role` + optional scores/`broll`). Resolves ranges→in/out via the
  existing resolver, runs quote-completion repair, asserts the resolved total duration falls in
  the candidate's claimed lane band **then drops lane** (D2 — lane never enters `beat_plan`),
  validates against MASTER's schema. Fail ⇒ no write, journal, reject.
- **Seam 2 — editorial_signals (FROZEN, D-Story-02).** Immutable, versioned
  `catalog/editorial/{media_id}/{job_id}.json` + `manifest.json` resolver; `source_evidence_hash`
  for staleness. Consumers read labels only through the manifest. Done.
- **Seam 3 — journal event shapes.** Deterministic-stage event and model-stage event
  (`model`, `mode`, `temperature`, `latency`, `usage`, schema-validation, `decision_summary`),
  conformant to MASTER's journal schema. Every stage emits one. `SE:` stage prefix (D5).

---

## 4. Standing rules (MASTER invariants — carried, with Rule 6 elevated)

1. **Immutability.** Never mutate existing catalog artifacts. New info = new versioned artifact.
2. **Journal every stage** (deterministic ones included) to the central queryable audit journal.
3. **Prime modularity.** Stages talk only through immutable versioned artifacts.
4. **Lanes threaded end-to-seam.** Every candidate carries one `primary_format_lane`; every
   stage preserves it; no global pooling after mining; lane dies at the adapter (D2).
5. **Budget-aware failure.** On exhaustion: stop at a stage boundary, write partials, journal it,
   leave a resume manifest, discard nothing valid. Under Pro pressure drop **mutation** first.
6. **No invalid input to Pro — and Pro never originates evidence IDs.** Programmatic validation
   + dedup run *before* any Pro call; fabricated/ambiguous IDs die deterministically. **(Elevated
   per C4:** Pro chain-completes pre-mined, evidence-grounded ingredient sets; it does not invent
   `clip_id`/`sentence_range`. Generation diversity is Flash's job.**)**

The v1 "do nots" hold: no story graph yet; one compact Pro critic bundle (not seven critics);
per-lane seeded single-elim (not round-robin); no mutating evidence/editorial; candidate beats
are never render-ready (adapter mandatory); no lane past the seam; no hardcoding budgets pre-probe.

---

## 5. The funnel (production target — `patch_v3` §4, corrected shape)

Lane identity preserved through every stage. Caps per-lane unless noted. Budgets in §8 are
**provisional until the probe**.

```
SE:A  Editorial labels        Flash t=0      new/changed clips only            [DONE]
SE:B  Ingredient mining       Flash t=.6-.9  32/lane (12 pressure+12 q+8 payoff)  → grounded refs
SE:B2 Ingredient dedup/rank    deterministic  → 8 chain inputs/lane
SE:C  Chain completion         PRO thinking   ≤8/lane → complete|reject|partial_open_thread
SE:C2 Evidence validation      deterministic  HARD FLOOR — bad IDs/ranges/audio die here, pre-Pro-critic
SE:C3 Candidate dedup          deterministic  per-lane, 5-signal rule (§7 patch_v3) → ≤6 reps/lane
SE:D  Cheap filter             Flash t=0      → ≤4 survivors/lane
SE:E  Compact critic bundle    PRO thinking   ONE call/candidate, ≤3/lane (story_physics, causal,
                                              shuffle_ablation, cold_viewer, anti_generic, retention, payoff)
SE:F  Per-lane tournament      PRO thinking   seeded single-elim, ≤2 pairwise calls/lane
SE:G  Mutation                 Flash t=.6-.9  top parent/lane ×4, lineage tracked
SE:H  Mutation comparator      deterministic+Flash; Pro spot-check ≤1/lane (delta≥5 or repair)
SE:I  beat_plan adapter        deterministic  Seam 1 — finalists only
SE:J  Shortlist + reports      deterministic  ≤5/lane + shortfall(cause) + open-thread + footage-needed
```

Note the ordering correction vs `se_chain`: **mining (Flash) and the C2 floor sit *before* the
first Pro call (SE:C)**, so Pro receives only validated, grounded ingredient sets.

---

## 6. The grand plan — ordered, gated worklist from here

Build in risk order; **stop and look at every gate**; the project lead's eyeball is a pipeline stage at
every render-producing phase. Each item is 1–2 Code sessions and **none starts until the prior
gate passes**.

**SE-0 — Foundations (do first; unblocks everything).**
- **0a Ratify + freeze the seams.** Opus review of Seam 1 & 3 + D4/D5/D7 → the project lead signs →
  record in `DECISIONS.md` (D-Story-03). *Gate:* three seams frozen.
- **0b Eyeball the 8 existing `se_chain` runs.** Free, on disk. Tells us if the pipe already
  makes anything good (SE-3 acceptance evidence) and seeds taste calibration. *Gate:* the project lead
  verdict recorded.
- **0c Build + run the PROBE on `erf.llm`.** 10–20 real calls each to pro(thinking)/flash
  against real evidence → `runs/probe_{date}/measurements.json`: p50/p95 latency per model/mode;
  JSON-failure rate broken out (missing field / array-index-as-ID / fabricated ID / non-JSON);
  concurrency tolerance (1→2 for Pro); diversity-collapse under temp/prompt variation. *Gate:*
  measurements file exists; §8 branch chosen; budgets re-sized.

**SE-2 — Deterministic spine (the missing floor).** Build SE:B (Flash ingredient mining,
grounded refs) → SE:B2 (dedup/rank) → SE:C2 as a **standalone hard-floor validator** (promote
`se_chain`'s embedded checks; add fabricated-ID/array-index/range/audio-contradiction
rejection) → SE:C3 (5-signal per-lane dedup). All cheap/free; produces inspectable JSONL.
*Gate (acceptance):* are mined ingredients worth a Pro call, and does C2 actually reject bad IDs
on a planted-bad-ID test? (SE-1 labels already pass their bar at scale.)

**SE-3 — Rewired vertical slice.** Replace `se_chain`'s invent-mode with SE:C (Pro
chain-completes the SE-2 ingredient sets — **no ID origination**) → SE:I adapter → render 3–5.
*Gate:* can the engine produce one genuinely good `beat_plan` that renders clean end-to-end,
fed by the real spine? **The project lead's taste-gate.** Prove the pipe before the superstructure.

**SE-4 — Quality enforcement (the storyteller layer the project lead wants).** SE:D Flash filter → SE:E
Pro critic bundle → SE:F per-lane tournament → SE:G/H mutation+comparator. Size the Pro budget
from 0c. Decide 3-vs-5 render-ready/lane on measured latency. *Gate:* do shuffle/ablation + the
critic bundle **measurably kill** montage-grade candidates SE-3 would have passed? **The project lead's
taste-gate on the survivors.**

**SE-5 — Adaptive loop.** Open-thread registry (`narrative_thread` 1.0.0) + shortfall(cause) +
future-footage requests as first-class outputs. *Gate:* does a new ingest batch correctly
complete/upgrade an existing open thread?

Critical path: **0a → 0c → SE-2 → SE-3 → SE-4 → SE-5** (0b runs in parallel with 0a/0c).

---

## 7. Execution oversight design

**Roles.**
- **Opus** — specs each phase against the frozen seams; reviews diffs; ratifies seam changes;
  owns the probe-branch decision and budget re-sizing; writes/maintains this master.
- **Code (Sonnet)** — implements the deterministic stages, tests, journal wiring, configs, the
  probe harness. Owns correctness-checkable work outright (no wait-on-review). Mechanical
  generation subtasks may route to DeepSeek; payload-heavy reads to Haiku; per the volume's tool
  protocol.
- **The project lead** — taste gate on every render-producing phase (SE-3, SE-4 especially) and final
  ratifier of seams/budgets. Their eyeball is a stage, not a formality.

**Phase-gate protocol.** Each phase ends with: (1) the inspectable artifact it promised, (2) its
single acceptance question answered with evidence, (3) a `DECISIONS.md` entry, (4) journal events
for every stage run. **No phase starts until the prior gate is signed.** Stop-and-look is enforced
by the gate, not by intent.

**The journal is the audit spine.** Every stage — deterministic included — emits a Seam-3 event:
inputs, decisions, skips, exclusions, output provenance, and (model stages) model/mode/latency/
usage/validation. Budget exhaustion, rejects, and shortfall causes are journalled, never inferred.
This is also how we *see* whether the storyteller is improving across runs.

**Budget-from-probe rule.** No `patch_v3` §5–§6 number is treated as fact until 0c. SE-4 caps,
concurrency, and wall-clock are derived from `measurements.json`. If the probe contradicts an
assumption, the funnel shape changes (§8) — the measurement wins.

**Regression guard against the C4 shortcut.** Production candidates must carry an
`ingredient_id` provenance field linking each beat to a SE:B-mined, SE:C2-validated ingredient.
SE:C2 **rejects any candidate whose evidence IDs were not present in its input ingredient set**
(i.e., Pro-originated IDs fail the floor). A config flag (`engine.mode = shortcut|production`)
keeps `se_chain` available for experiments but bars it from the nightly path. This makes "Pro
invented an ID" a hard, journalled failure rather than a silent quality leak.

**Taste-calibration timing.** Per the project lead's standing preference (hold edit/narrative-taste tuning
until DS Pro is optimised): SE-0/SE-2 are tool/plumbing-level and safe to build now; the
*taste-heavy* parameters (critic rubric weights, what "good" means, 3-vs-5/lane) are set at
SE-3/SE-4 on what the outputs *show*, with the project lead driving — not guessed up front.

**Escalation.** Default when unsure: journal the decision + flag it; don't silently improvise a
contract. Seam changes, invariant tensions, and probe results that break budget assumptions
badly enough to change funnel shape all escalate the project lead → Opus.

---

## 8. Probe-outcome decision branches (new — what 0c decides)

The probe is not a formality; it selects the funnel shape. Pre-committed branches:

- **Pro latency high** (p95 × Pro-call-count × concurrency-1 > wall-clock target): in priority
  order — (a) raise Pro concurrency to 2 *iff* the probe shows it's rate-limit-safe; (b) push
  more kills to SE:D Flash filter so fewer candidates reach Pro; (c) **collapse SE:F** to "rank
  by SE:E critic score, no pairwise Pro calls" (tournament becomes deterministic on critic
  output); (d) accept multi-night batches via resume manifests (Rule 5). Drop order under
  pressure stays: mutation → tournament-pairwise → (never) C2/dedup.
- **Pro thinking collapses diversity** under temp/prompt variation: confirm Pro is used **only**
  for judgment/assembly/pairwise — never ideation or mutation (already C6); ensure SE:G mutation
  stays on Flash.
- **JSON failure rate material** (esp. array-index-as-ID / fabricated ID): strengthen SE:C2
  repair for *unambiguous* index refs (warn+repair) and reject ambiguous; add a bounded Pro
  "reformat-only" retry at SE:C; tighten the schema-in-prompt. The floor absorbs the failure
  mode the shortcut would have rendered.
- **Concurrency intolerant** (2 concurrent Pro calls fail): hard-set Pro `max_concurrent=1`;
  re-budget wall-clock accordingly; Flash stays at 6.

---

## 9. Open decisions — ratification

Opus recommendations (project lead: accept / amend). On acceptance these + Seam 1 & 3 freeze land as
`DECISIONS.md` D-Story-03.

| # | Decision | Opus call |
|---|---|---|
| D4 | corpus-level event `media_id` value | **`"_corpus"` sentinel** — accept |
| D5 | engine journal `stage` namespace | **`SE:` prefix** — accept |
| D6 | editorial path immutability | already landed (D-Story-02) — closed |
| D7 | quote-completion repair source | **reuse `editorial.py`**, don't fork — accept |
| Seam 1 | `se_adapter` mapping | freeze as-is pending one Opus read of the mapping fn |
| Seam 3 | journal event shapes | freeze as-is; confirm against MASTER journal schema |

---

## 10. Status — SE-2 COMPLETE (2026-06-26)

- **SE-0a done** (D-Story-03): D4/D5/D7 ratified; Seam 1 frozen; Seam 3 = adopted payload convention.
- **SE-0b done**: 8 `se_chain` runs audited. 06-22 mp4s wiped by the 06-24 rebuild — 4 candidate
  definitions survive in `runs/`. Taste read deferred to SE-3 spine output (correct path).
- **SE-0c done** (D-Story-04): probe `runs/probe_20260626T113011Z`. Pro p50 29.5s / p95 39.4s,
  Flash p50 17.9s / p95 22.4s; 8/8 reliable on grounded prompt; concurrency 2 tolerated; diversity
  collapses → Flash for ideation, Pro for judgment. Pro `max_concurrent=2`.

- **SE-2 done** (D-Story-05, 2026-06-26): deterministic spine built and gate PASSED.
  - `src/erf/se_floor.py` — SE:C2 hard-floor validator. 15/15 planted-bad-ID tests pass.
  - `src/erf/se_dedup.py` — SE:C3 5-signal candidate dedup (union-find, composite score rep).
  - `bin/se_mine.py` — SE:B+B2 miner. 0/27 Flash rejections; 33 ingredients/lane, 8 B2/lane.
    Run: `runs/se_mine_20260626T144752Z/`. High-hook ingredients specific and BTS-authentic.
  - `bin/se_complete.py` — SE:C Pro chain-completion. **Ready but NOT run** — SE-3 taste gate.
  - Gate: C2 rejects 15/15 bad IDs ✓. Ingredients worth a Pro call: YES ✓ (D-Story-05).

**Next: SE-3 — rewired vertical slice** (the project lead's taste gate). Run `bin/se_complete.py` against
`runs/se_mine_20260626T144752Z/`, adapt via `se_adapter`, render 3–5 samples. Gate: can the
engine produce one genuinely good `beat_plan` fed by the real spine? **The project lead's taste-gate.**

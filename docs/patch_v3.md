# DS Pro Editorial Story Engine — Patch v3 (Consolidated, Corrected)

This is the authoritative stage-level spec for the story engine funnel. It supersedes the
raw v3 patch, absorbs the Brief's §8 corrections and the D2 lane resolution, and removes
duplication with the Phase 0 seam contracts.

Read alongside:
- `STORY_ENGINE_BRIEF.md` — execution plan, phases, roles, invariants. Wins on conflict.
- `STORY_ENGINE_PHASE0_SEAMS.md` — the three frozen seams. Owns the beat_plan mapping,
  editorial-signal versioning, and journal event shapes. Wins on those topics.
- `MASTER.md` — system contract, the real `beat_plan` schema, journal schema, invariants.
  Wins over everything.

Precedence: MASTER.md → Brief → PHASE0_SEAMS (on its three topics) → this doc.

---

## 0. Status of this document — frozen vs provisional vs owned-elsewhere

**Frozen (build to these):** format lanes, candidate schema, beat schema + audio fields,
funnel shape and stage caps, dedup rule, mutation comparator, confidence tiers, shortfall
contract.

**Provisional — pending the Phase 0 probe (do NOT hardcode as fact):** every Pro-call
budget, concurrency ceiling, wall-clock limit, and per-stage timeout in §5–§6. These are
sized on *assumptions* about DeepSeek V4 latency and concurrency. The probe measures
reality; Phase 4 re-sizes them. Until then they are defaults to start from, not contract.

**Owned elsewhere — do not restate or fork:**
- Editorial-signal versioning, manifest resolver, immutable path → **Seam 2**
  (`STORY_ENGINE_PHASE0_SEAMS.md §2`). This doc does not specify the editorial artifact.
- Candidate→beat_plan field mapping, in/out resolution, quote-completion repair → **Seam 1**
  + the real `beat_plan` schema in MASTER.md.
- Journal event shapes → **Seam 3**.

---

## 1. Format lanes (mandatory; upstream-only per D2)

Every candidate carries exactly one `primary_format_lane`, optionally one or more
`secondary_format_lanes`. Every filter/dedup/score/tournament/mutation/rank stage preserves
lane identity. No global pooling after ingredient mining.

Lanes:
- `single_clip_15_25s` — usually one clip; one complete quote or a tiny self-contained
  visual story. Runtime 15–25s.
- `mini_cross_media_30_60s` — usually 2–4 clips; clear setup → obstacle → attempt/result →
  payoff. Runtime 30–60s.
- `long_cross_media_60_120s` — usually 3+ clips; can carry more technical/contextual
  material but must still have a viewer question, a turn, and a payoff. Runtime 60–120s.

**D2 (resolved): lane is an upstream selection concept only.** It lives on the candidate and
on `shortlist_by_lane.json`. The Seam 1 adapter reads the claimed lane exactly once — to
assert the resolved total duration falls in that lane's band — then **drops it**. Lane is
**not** a `beat_plan` field; the emitted beat_plan is lane-free and beat_plan stays 1.3.0.
If a downstream render difference is ever genuinely needed, add an explicit `render_profile`
field at the adapter — never overload lane and never leak it past the seam.

---

## 2. Candidate schema (engine-internal; the adapter maps this onto the real beat_plan)

```json
{
  "candidate_id": "c_0001",
  "source_type": "pressure|viewer_question|payoff",
  "source_id": "p_0001",
  "primary_format_lane": "single_clip_15_25s",
  "secondary_format_lanes": ["mini_cross_media_30_60s"],
  "verdict": "complete|reject|partial_open_thread",
  "title": "string",
  "central_question": "string",
  "specific_tension": "string",
  "archetype": "string",
  "beats": [],                          // see §3
  "missing_beats": [],
  "known_weakness": "string",
  "future_footage_needed": [],
  "confidence_tier": "flash_scored_only" // see §9; set/updated as it moves through the funnel
}
```

A candidate is render-eligible only via the adapter, only if `verdict == "complete"`, and
only at an auto-eligible confidence tier (§9).

---

## 3. Beat schema + audio fields (the §8.5 correction — sync audio is never forced)

```json
{
  "role": "HOOK|WANT|OBSTACLE|ATTEMPT|DECISION|TURN|RESULT|PAYOFF|BUTTON",
  "clip_id": "string",
  "sentence_range": [0, 2],
  "evidence_ids": ["clip_id.s0", "clip_id.s1"],
  "why_this_beat": "string",
  "audio_mode": "sync_dialogue|vo|broll_audio|music_only|silent_visual",
  "sync_audio_required": false,
  "dialogue_required": false,
  "visual_payoff_required": false,
  "visual_support": "string|null",
  "payoff_type": "none|visual_proof|funny_button|technical_reveal|human_reaction|failure_that_teaches|before_after|open_loop"
}
```

Defaults: `sync_audio_required` and `dialogue_required` default `false`. Only spoken-quote
beats raise them.

Rules:
- Spoken-quote beat → `audio_mode=sync_dialogue`, `sync_audio_required=true`,
  `dialogue_required=true`.
- B-roll support → `audio_mode` ∈ {`broll_audio`, `music_only`, `vo`},
  `sync_audio_required=false`.
- VO over visuals → `audio_mode=vo`, `sync_audio_required=false`.
- `PAYOFF` with `payoff_type=visual_proof` → `audio_mode` may be any of broll_audio /
  music_only / silent_visual / vo / sync_dialogue; `sync_audio_required` must **not** be
  forced true.

Programmatic validation (Stage C2) rejects contradictions, including:
- `audio_mode=silent_visual` with `dialogue_required=true`
- `payoff_type=visual_proof` with no `visual_support`
- `sync_audio_required=true` with empty `sentence_range`

---

## 4. Default bounded nightly funnel

Lane identity preserved through every stage. Caps are per-lane unless stated global.

```
A.  Editorial labels (Flash, non-thinking, t=0) — new/changed clips only.
    Artifact + versioning owned by Seam 2.

B.  Ingredient mining (Flash, non-thinking, t=0.6–0.9) — per lane:
      12 story_pressures + 12 viewer_questions + 8 payoffs = 32/lane, 96 global.
    Use archetype prompts for structural diversity.

B2. Ingredient dedup/ranking — per lane → 8 chain-completion inputs/lane, 24 global.

C.  Chain completion (Pro, thinking) — ≤8/lane. Output: complete | reject |
    partial_open_thread. Partials feed the open-thread registry.

C2. Programmatic evidence validation (NO model) — the hard floor. Reject invalid JSON,
    missing fields, fabricated/nonexistent IDs, array-index-as-ID, ranges disagreeing with
    evidence_ids, empty beats, missing required roles (unless partial_open_thread),
    duration outside lane band, and the §3 audio contradictions. Unambiguous index refs
    repaired + warned; ambiguous → reject. ≤8 valid/lane.

C3. Candidate dedup/clustering (NO Pro) — per lane, see §7. ≤6 representatives/lane.

D.  Flash cheap filter (Flash, non-thinking, t=0) — ≤4 survivors/lane. Kill: no clear
    viewer question, no payoff, generic BTS, high montage risk, weak A/V support.

E.  Pro compact critic bundle (Pro, thinking) — ONE call/candidate, ≤3/lane. Single bundle
    scoring story_physics, causal_dependency, shuffle_ablation, cold_viewer,
    anti_genericness, retention, payoff_strength + viewer_experience + ablation. ID
    validation is NOT in the bundle (done in C2).

F.  Per-lane seeded single-elimination tournament (Pro, thinking) — ≤3/lane, ≤2 pairwise
    calls/lane. 1 candidate → rank #1, no call; 2 → one call; 3 → two calls.

G.  Mutation (Flash, non-thinking, t=0.6–0.9) — top 1 parent/lane, 4 mutations each, 12
    global. Track lineage.

H.  Mutation validation + cheap comparator (§8) → optional Pro spot-check ≤1/lane.

I.  Candidate→beat_plan adapter (Seam 1) — finalists only. Maps to the real beat_plan
    schema, resolves in/out, runs quote-completion repair, lane duration-band check + drop,
    validates against MASTER.md schema. Failure ⇒ no write, journal, reject.

J.  Final shortlist — up to 5/lane (see §10 on the realistic render-ready ceiling) +
    shortfall report + open-thread + future-footage outputs.
```

---

## 5. Pro-call budget — PROVISIONAL, pending Phase 0 probe

Default bounded mode target:
- Chain completion ≤24
- Compact critic bundle ≤9
- Per-lane tournament ≤6
- Mutation spot-check ≤3
- **Total ≤42 Pro thinking calls.**

Strict-coverage mode (§11): ≤69. These totals assume the funnel shape above; if the probe
shows Pro latency or concurrency forces a different shape, re-derive before committing.

---

## 6. Concurrency + wall-clock — PROVISIONAL, pending Phase 0 probe

Starting defaults (to be confirmed/replaced by probe results):
- Pro: start `max_concurrent=1`; raise to 2 only if the probe shows the endpoint tolerates
  parallel thinking calls without rate-limit/timeout failures. Per-call timeout ~1800s.
- Flash: `max_concurrent=6`, per-call timeout ~600s.
- Run wall-clock target ~8h; no new model calls after the limit; in-flight calls finish to
  their per-call timeout; pending calls cancelled; partial outputs written.

Budget-aware failure: stop at the current stage boundary, write partial outputs, journal the
exhaustion, leave a resume manifest, discard nothing valid. Under Pro pressure, drop
mutation before dropping dedup or validation.

---

## 7. Candidate dedup — corroborated (corrected: lane is the gate, not a signal)

Five **content** signals:
1. `sentence_jaccard >= 0.5`
2. `same_clip_set_and_archetype`
3. `central_question_similarity >= threshold`
4. `same_payoff_evidence_and_setup`
5. `same_tension_type_and_top_two_clips`

Cluster rule: two candidates are duplicates only if they share the same
`primary_format_lane` (**the gate**) **AND** at least **two of the five content signals**
are true. `same_primary_format_lane` is the gate and **never counts toward the two**.

Single-signal exception: `sentence_jaccard >= 0.8` alone clusters.

Never dedup across lanes. One clip can legitimately anchor a 20s quote, a 60s process
story, and a 120s technical explanation; mark these `related`, do not collapse. Per cluster,
pick one representative by cheap scores (validity, chain completeness, has-payoff,
hook/quote potential, A/V quality, lowest inside-baseball risk).

---

## 8. Mutation comparator (replaces "appears to beat original")

Per mutation, compute a deterministic + Flash score vs its parent:

```
total =
  0.20*viewer_question + 0.15*obstacle_clarity + 0.20*payoff_strength
+ 0.15*specificity + 0.15*retention + 0.05*anti_genericness
+ 0.05*hook_potential + 0.03*quote_potential + 0.02*av_quality
- penalties
delta = total - parent_total
```

Run a Pro mutation spot-check only if: `validation_passed == true` AND no fatal evidence
warnings AND (`delta >= 5` OR `mutation_addresses_parent_repair == true`). Never Pro-check a
mutation that fails ID validation.

---

## 9. Confidence tiers (one canonical enum) + adapter eligibility

Canonical enum:
`pro_scored_tournament_ranked | pro_scored | mutation_unchecked | flash_scored_only`

Auto-eligible for the beat_plan adapter by default: `pro_scored_tournament_ranked` and
`pro_scored` only. `mutation_unchecked` and `flash_scored_only` may appear in the shortlist
but require additional validation before render.

---

## 10. Final deliverable + shortfall contract (corrected — be honest about the cause)

Output: up to 5 per lane, with a per-lane shortfall report and the open-thread /
future-footage outputs.

**Realistic ceiling in default mode:** with `pro_scoring` capped at 3/lane plus ≤1
Pro-checked mutation, default mode produces ~3–4 **render-ready** (pro-tier) candidates per
lane. Slots 4–5 can only be flash-tier, which are not auto-eligible for the adapter. So the
default deliverable is "up to ~3 render-ready per lane." Five *renderable* per lane requires
strict-coverage mode (§11). State this plainly in output.

**Shortfall reason must distinguish cause.** When a lane comes up short, the report must say
whether the limit was a **budget cap** (the funnel deliberately scored only N/lane) or
**evidence scarcity** (footage genuinely couldn't support more). Do not report a budget cap
as a footage problem — that conflation is the exact failure this correction exists to
prevent. `shortfall_reason` carries an explicit `cause: "budget_cap" | "evidence_scarcity"`.

---

## 11. Strict-coverage mode (opt-in; not default)

Raises per-lane caps to target 5 render-ready/lane where evidence allows: chain completion
≤12/lane (36 global), dedup ≤8/lane, flash filter ≤6/lane, pro scoring ≤5/lane,
per-lane tournament ≤5/lane, mutation 2 parents/lane. Pro budget ≤69 (provisional, same
probe caveat as §5).

---

## 12. The "do nots"

No story graph yet. No seven separate Pro critics — one compact bundle. No round-robin
tournament — per-lane seeded single-elim. No invalid/ambiguous IDs into any Pro stage. No
mutating immutable evidence or editorial artifacts. No treating candidate beats as
render-ready — the adapter is mandatory. No lane past the seam. No hardcoding the §5–§6
budgets as fact before the probe.
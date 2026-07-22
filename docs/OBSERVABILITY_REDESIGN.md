# ERF Observability & Tool-Composition Contract — Ground-Up Redesign

> **Status:** APPROVED, pre-implementation (2026-07-08). Authority document for the
> logging-schema rebuild. Supersedes the manual observability contract in `MASTER.md` §5.2
> (that section is rewritten by Phase 4 of this plan). Read alongside `GAMEPLAN.md`.
> Implementation branch: `logging-schema-redesign`. Decision ledger entry: **D-Log-04** (approval, `DECISIONS.md` 2026-07-08).

---

## 0. Why this exists

The ERF pipeline does not crash. Every stage runs green (`rc=0`, no tracebacks) while the
*meaning* silently breaks in the coupling between stages — **"green-but-wrong."** The entire
defect ledger is one bug wearing different clothes, and every instance is **correctness that
was never observable**:

| Incident | Reported | Truth | Root |
|---|---|---|---|
| D-Identity-04 | gate 0.953 | 0.712 (a third of identity resolution silently gone) | status keyed on file presence, not meaning |
| logging-spine drift | green | 1019 phantom/unrecorded rows | three surfaces asserted independently |
| phantom outcomes | green | 308 records with no artifact | ditto |
| T0 phantom-ok | green | 53 rows ok with null person_id | health computed then discarded |
| stale-artifact invisibility | green | downstream ran on stale inputs | no input fingerprinting |

The directive (the project lead, 2026-07-08): *"Every part of the pipeline functioning in itself as a
tool, and working seamlessly with all the other tools and the system as a whole. Redesign the
logging schema from the ground up. I don't want to be running whack-a-mole patches for the rest
of the year."* Accuracy/threshold tuning is explicitly **out of scope** ("niceness not
correctness"). This is a **mechanism** rebuild.

## 1. Root causes (verified against `src/erf/outcomes.py`, `MASTER.md` §5.2, static audit 2026-07-08)

1. **Status keys on artifact *presence*, not semantic *success*.** A stage writes a file → green.
   Whether that file attributes words to the right person → unmeasured. `build_transcript.py`
   even *computes* `pct_person_id` then discards it before recording.
2. **Three status surfaces, independently asserted** → drift is structurally possible:
   `journal.db.status_projection`, `catalog/outcomes/*.json`, `catalog/status/*.json`.
3. **Three writer libraries that collide:** `erf.outcomes`, `erf.runlog`, `erf.journal`.
   `status_projection` alone has **five** independent writers (`journal.emit_*`,
   `outcomes._journal`, `runlog.log_event`, `e1_signals._journal_event`, `reconcile_status`).
4. **Three hardcoded stage-universes** kept in sync by a *manual* checklist (MASTER §5.2.3):
   `scheduler.STAGE_REGISTRY` (14), `reconcile.STAGE_ARTIFACTS` (29),
   `pipeline_status.CRITICAL`/`SPEECH_SPINE` (10/9). Forgetting a box = a silent hole.
   The doc admits "Endpoints-only is a KNOWN DEFECT." **This is the whack-a-mole engine.**
5. **The most failure-prone tool is entirely off-spine.** `build_identity_clusters.py` records
   nothing on any path (silent exit 0), has no reconcile predicate, is not in CRITICAL, and is
   absent from the orchestrator. The tool that drove D-Identity-04 is invisible to write AND read.
6. **The write path is killable.** `outcomes._journal` swallows exceptions ("logging must never
   break a stage"); SEV2/SEV2_ADJ/PSTREAM guard all recording behind `if outcomes is not None`
   (import failure → silent no-op); RG/RP have no per-media try/except (one exception kills the
   batch with no `error` outcome).

## 2. Design principles

The redesign converts "green-but-wrong" from a *possible* state into an **impossible** one, by
making wrongness a **disposition with a threshold** and collapsing the surfaces that drift.

- **P1 — One append-only log, one writer, everything else DERIVED.** `state/journal.db.events`
  is the single asserted source of truth. `status_projection`, `catalog/status/*.json`, and
  `health.json` become **materialized views recomputed from events** — never independently
  written. Drift is only possible when two things are asserted independently; make one derived
  and drift is structurally impossible. Collapse `erf.outcomes` + `erf.runlog` + `erf.journal`
  into one `record()` path.
- **P2 — One declarative Stage Registry is the single source of truth for "what stages exist."**
  Each stage declares itself once; every consumer (auditor, health, orchestrator, scheduler)
  *iterates the registry* instead of hardcoding its own set. Adding a stage = one registration.
  This retires the manual §5.2.3 checklist **by construction** — being off-spine becomes impossible.
- **P3 — The outcome record carries semantic health, not just a disposition.** Two new
  dispositions: **`degraded`** (ran, produced an artifact, but a declared health predicate failed —
  e.g. coverage below floor; visibly NOT green — the green-but-wrong killer) and **`stale`** (an
  input fingerprint no longer matches — kills the partial-rebuild contamination class behind
  D-Identity-04).
- **P4 — The write path is unkillable.** `record()` failures surface (no silent swallow);
  `erf.outcomes` is a hard dependency (remove the `if outcomes is not None` escape); per-media
  batch work is wrapped so one media's exception emits `error` and continues.
- **P5 — Health rolls up to one verdict, gate included.** Corpus is GREEN **iff** every registered
  stage × media has disposition ∈ {ok, empty-by-design, skip-by-design}, zero degraded/stale/error,
  drift 0. The attribution gate emits into the same log as a stage, so "green" means meaning-intact,
  not merely file-exists.

## 3. Target schema (concrete)

### 3.1 `events` — the one asserted surface (`state/journal.db`)
Append-only. One row per (media_id, stage, attempt). Everything below is DERIVED from it.
```
events(
  id            INTEGER PK,
  media_id      TEXT,          -- or "__corpus__" for corpus-scoped stages
  stage         TEXT,          -- registry key
  status        TEXT,          -- ok | skip | empty | partial | degraded | stale | error
  reason        TEXT,          -- required for skip/empty/error/degraded/stale
  run_id        TEXT,
  producer_sha  TEXT,          -- sha256 of the producing script
  config_fp     TEXT,          -- fingerprint of stage config
  metrics       TEXT (json),   -- stage-declared counts, mandatory where the registry declares them
  health        TEXT (json),   -- {predicate: {value, floor, verdict}}, from registry health_predicates
  inputs        TEXT (json),   -- {upstream_artifact: fingerprint} consumed this run
  artifacts     TEXT (json),   -- produced artifact paths + fingerprints
  at_utc        TEXT
)
```
Dispositions: `COMPLETED = {ok, skip, empty}`; `NOT_GREEN = {partial, degraded, stale, error}`.
`_REASON_REQUIRED = {skip, empty, error, degraded, stale}`.

### 3.2 `record()` — the one writer (extends today's `outcomes.record` signature)
```
record(media_id, stage, status, *, reason=None, metrics=None, health=None,
       inputs=None, artifacts=None, producer=None, config_fp=None, run_id=None)
```
- Writes exactly one `events` row (atomic).
- `health`: computed by the stage against its registry `health_predicates`; if any predicate
  fails, the caller passes `status="degraded"` (or `record()` derives it — decided in Phase 3).
- `inputs`: fingerprints of consumed upstream artifacts; a mismatch against the registry's
  expected upstream fingerprint marks the row `stale`.
- **No silent swallow.** A write failure raises; the pipeline treats an unloggable stage as a
  failed stage (P4).
- `runlog.log_event`, `e1_signals._journal_event`, and `journal.emit_*`/`reap_manifest` all funnel
  into this. `runlog.heartbeat` stays separate (liveness ≠ outcome).

### 3.3 `src/erf/stages.py` — the one registry (seed: `scheduler.STAGE_REGISTRY`)
```
STAGES = {
  "<name>": Stage(
     name, depends=[...], tier="cpu|gpu-preferred|gpu", phase=int,
     scope="media|corpus",
     artifact_path="catalog/.../{mid}...",     # canonical output pattern
     predicate_kind="ne|exists",               # ne=exists+nonempty; exists=explicit-empty-allowed
     critical=bool, speech_spine=bool,
     consumes=[...], produces=[...],
     health_predicates=[HealthPredicate(name, fn, floor)],   # optional; e.g. resolution-rate
     fingerprint_inputs=[...],                 # upstream artifacts whose fp defines staleness
  ), ...
}
```
Union of all three current universes (A…LABEL incl. the full speech spine + `build_identity_clusters`
as a first-class stage). Resolves the **`F1` name collision** (scheduler face-DAG node vs reconcile
`identity_overlay` artifact) — they become distinct registry keys.

### 3.4 Derived views (recomputed from `events`, never asserted)
- `status_projection` — latest disposition per (media, stage). Rebuilt by folding `events`.
- `catalog/status/{mid}.json` — per-media rollup. (Retires the second writer:
  `pipeline_status --project`.)
- `logs.all/_latest/health.json` — corpus verdict (P5). Readers `rebuild_all.py` and
  `timeline_audit.py` keep their `critical_gaps`/`at_utc` contract.

## 4. Migration (incremental, no flag-day; pipeline runnable at every phase)

- **Phase 0 — `src/erf/stages.py` registry, no behavior change.** Union registry + a test
  (`tests/test_stages_registry.py`) proving the three existing hardcoded sets are consistent with
  it *before* any consumer depends on it. Resolve `F1`.
- **Phase 1 — Collapse to one log + one writer; everything else DERIVED (anti-drift core).**
  Rewrite `journal.reap_manifest` as the single fan-in to `events` (keep its signature — ~20 stage
  callers untouched). Point `outcomes.record`, `runlog.log_event`, `e1_signals._journal_event` at
  one `record()`. Turn `status_projection`/`catalog/status`/`health.json` into recompute-from-events
  views (delete the `pipeline_status --project` writer). Backward-compat shim preserves the ~50
  `erf.outcomes` call sites and the two `health.json` readers.
- **Phase 2 — Rewire consumers to iterate the registry; delete the hardcoded literals** in
  `reconcile_status.py`, `pipeline_status.py`, `rebuild_all.py`, `scheduler.py`. Adding a stage
  becomes one registration; §5.2.3 checklist retired by construction.
- **Phase 3 — Semantic health in the record (the green-but-wrong killer).** Add
  `metrics`/`health`/`inputs`/`config_fp`; add `degraded`/`stale`. First predicates: resolution-rate
  on `build_identity_clusters` (gallery vs no_embedding/cluster_unknown/demoted), null-`person_id`
  coverage on SEV2 and T0 (stop discarding the `pct_person_id` T0 already computes). **Put
  `build_identity_clusters.py` on the spine.** These are exactly the signals that would have caught
  D-Identity-04.
- **Phase 4 — Unkillable write + gate-as-stage + docs.** Remove the silent swallow and the
  `if outcomes is not None` escape (hard dependency); per-media try/except on RG (`generate_gaps`)
  and RP (`process_media`); fix T0's not-force exit to record. Emit the attribution gate result as a
  first-class stage. Rewrite `MASTER.md` §5.2 to the single-surface, registry-driven contract.
- **Phase 5 — Verification** (see §6).

## 5. File map

- **NEW:** `src/erf/stages.py`, `tests/test_stages_registry.py`.
- **Rewrite:** `src/erf/outcomes.py` (single-surface `record()` + derived views + new dispositions/fields).
- **Rewire:** `bin/reconcile_status.py`, `bin/pipeline_status.py`, `bin/rebuild_all.py`, `src/erf/scheduler.py`.
- **Health predicates + on-spine:** `bin/build_identity_clusters.py`, `bin/build_speech_events_v2.py`,
  `bin/build_transcript.py`; per-media guards in `bin/build_recall_gaps.py`, `bin/recall_pass.py`.
- **Docs:** `MASTER.md` §5.2 (full rewrite), `DECISIONS.md` (D-Log-04), `GAMEPLAN.md`.

### Reused, not reinvented
- `erf.outcomes.record()` signature (already carries `counts`/`artifacts`/`producer`) — extended.
- `scheduler.STAGE_REGISTRY` field shape — the registry seed.
- `reconcile_status` predicate lambdas — migrated into registry `artifact_path`/`predicate_kind`.
- `atomicio.write_tmp_rename` / `_atomic_write_json` — all derived-view writes.

## 6. Verification (end-to-end)

1. `rtk pytest` — full suite green incl. new `test_stages_registry.py`.
2. `venvs/control/bin/python bin/reconcile_status.py` → `total drift 0`; `--audit-outcomes` clean.
3. `venvs/control/bin/python bin/pipeline_status.py` → `critical_gaps` empty **for the right reason**
   (every registered stage covered, not endpoints-only).
4. **Degraded-proof (decisive):** null-out one media's `identity_map`, rerun SEV2→T0, assert the row
   disposition is `degraded` and the corpus verdict flips to not-green. Today it stays green — this
   test failing-to-flip is the whole bug; passing is the fix.
5. One-command rebuild of a small media subset settles with every registered stage recording an
   explicit outcome and zero drift.

## 7. Scope guard / non-goals

- **Mechanism only.** No accuracy work, no threshold tuning, no embedder change. **Embedder-agnostic**
  and must remain so — validated identically whichever embedder the corpus holds. (Bake-off 2026-07-08:
  WeSpeaker 0.954 > redimnet_v1 0.840 > redimnet2_v2 0.712; the WeSpeaker revert is **D-Identity-06**,
  a separate decision that does not touch this work.)
- **Out of scope, one-DB blast radius:** touches `state/journal.db` only. The other three SQLite DBs —
  `state/erf.db` (review/persons/jobs), `state/erf_fts.db` (search FTS),
  `state/narrative_pipeline.db` (narrative threads) — and the LanceDB vector index are untouched.
- **No flag-day.** The pipeline stays runnable at every phase; nothing is committed until the full
  suite + drift-0 + degraded-proof pass.

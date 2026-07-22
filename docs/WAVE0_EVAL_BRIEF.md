# Wave 0 — Eval Harness & Regression Gating
## Implementation Brief for Claude Code (Sonnet)

**From:** Opus. **Date:** 2026-06-10. **Authority:** ROADMAP.md Wave 0; MASTER.md §15.
**Status:** final. Implement exactly this. Every decision below is final until a superseding Opus brief.

---

## 0. Framing: what M4 is and why it doesn't exist yet

ROADMAP.md refers to "the M4 versioning mechanism" as if it exists. It does not. M4 is a forward reference — a name for a model-versioning system that must be built as part of this Wave 0 work. This brief defines it.

**M4 = a per-stage active-model pointer** stored in `state/eval/stage_versions.json`. It records, per stage, which model tag is "active" (i.e., what the corpus was last embedded with, and which eval run blessed it). A "stage_version flip" is any change to the active model tag or its key invocation params. The gate intercepts that change at the CLI level.

Nothing in this implementation changes catalog schemas, adds always-on services, or requires §15 authority.

---

## 1. Directory layout

```
state/eval/
├── stage_versions.json        # M4 pointer (control-plane-only writer, atomic writes)
├── queries.json               # golden query set (human-curated, version-controlled)
├── exemplars.json             # exemplar short library (human-curated)
├── baseline/
│   ├── B1.json                # copy of the latest gate-passing eval run for stage B1
│   └── B4.json                # same for B4
└── runs/
    └── {eval_run_id}.json     # per-run result (full per-query diff + aggregates)
```

`bin/erf-eval` — CLI entrypoint (Python script, embed venv).
`src/erf/eval.py` — core logic: query running, metric computation, comparison, gate flip.

No new catalog directories. No new SQLite tables. No LanceDB tables. Results in `state/` only.

---

## 2. File schemas

### 2.1 `state/eval/queries.json`

The golden query set. Human-curated. Checked into version control. Every entry is a query where the operator has manually confirmed which chunks belong in the top results.

```json
{
  "schema_version": "1.0.0",
  "updated_at_utc": "2026-06-10T...",
  "queries": [
    {
      "eval_query_id": "eq_<blake2b-8hex of query+sorted_expected_ids>",
      "query": "Alex talking about morning routines",
      "modality": "text",
      "expected_clips": [
        {
          "chunk_id": "m_xxxx:c:12000-24000",
          "media_id": "m_xxxx",
          "note": "segment where Alex describes morning prep"
        }
      ],
      "min_recall_k": 5,
      "stored_query_vector": [0.032, -0.011, ...],
      "stored_vector_model_tag": "qwen3-8b",
      "stored_vector_at_utc": "2026-06-10T...",
      "added_by": "reviewer_1",
      "added_at_utc": "2026-06-10T..."
    }
  ]
}
```

**`stored_query_vector`**: pre-embedded using the current active model at the time of adding the query. This is what makes VRAM-free gate checks possible — the vector is used directly for ANN search, no model inference needed. It is updated when a flip is confirmed (the new model's embedding replaces it).

`eval_query_id` is `eq_` + first 8 hex chars of `blake2b(query + sorted(chunk_ids joined by |))`. Deterministic; safe to recalculate.

**Modality values**: `"text"` (query LanceDB `text_v1`), `"visual"` (query `visual_v1` with the text side of the cross-modal model), `"both"` (query both, union results).

### 2.2 `state/eval/exemplars.json`

Exemplar shorts: references to approved review_queue items, used to verify the retrieval substrate hasn't degraded for assembly use cases.

```json
{
  "schema_version": "1.0.0",
  "updated_at_utc": "...",
  "exemplars": [
    {
      "exemplar_id": "ex_<blake2b-8hex of review_id>",
      "name": "Alex intro hook — approved 2026-06-09",
      "review_id": "rev_01KTQ8ZNM3M60X4XPHMBWD91HC",
      "beat_plan": { ... },
      "expected_beat_scores": [0.82, 0.74, 0.81, 0.79, 0.68, 0.77, 0.83],
      "added_by": "reviewer_1",
      "added_at_utc": "2026-06-10T...",
      "notes": "reference quality: clean hook, good energy"
    }
  ]
}
```

`expected_beat_scores`: the retrieval score (`score` field from `/shorts/score`) for each beat's selected chunk at the time the exemplar was added. These are the baseline scores; regression = significant drop.

`beat_plan` is stored inline (a copy, not a reference) so the exemplar is self-contained if the beat_plan is later overwritten.

### 2.3 `state/eval/stage_versions.json`

The M4 active-version pointer. Control-plane-only writer. Atomic write via temp-rename. Never edited manually by operators.

```json
{
  "schema_version": "1.0.0",
  "updated_at_utc": "2026-06-10T...",
  "stages": {
    "B1": {
      "model_tag": "qwen3-8b",
      "invocation_fingerprint": "<sha256 of embed_text_invocation.json content>",
      "active_since_utc": "2026-06-10T...",
      "active_eval_run_id": "eval_20260610T224521Z_ea4532",
      "eval_verdict": "pass"
    },
    "B4": {
      "model_tag": "qwen3vl-8b",
      "invocation_fingerprint": "<sha256 of embed_visual_invocation.json content>",
      "active_since_utc": "2026-06-10T...",
      "active_eval_run_id": "eval_20260610T224610Z_b1c2d3",
      "eval_verdict": "pass"
    }
  }
}
```

`invocation_fingerprint` = `sha256(json.dumps(invocation_file_contents, sort_keys=True))`. Detects any change to model path, dim, prompt params, or encode kwargs. When the fingerprint diverges from the current invocation file, a flip is pending.

### 2.4 `state/eval/runs/{eval_run_id}.json`

One file per eval run. Full per-query diffs plus aggregates.

```json
{
  "schema_version": "1.0.0",
  "eval_run_id": "eval_<YYYYMMDDTHHMMSSz>_<6hex>",
  "run_type": "baseline",
  "stage_tag": "B1",
  "model_tag": "qwen3-8b",
  "invocation_fingerprint": "...",
  "run_at_utc": "...",
  "query_results": [
    {
      "eval_query_id": "eq_...",
      "query": "...",
      "modality": "text",
      "top_k_retrieved": ["m_xxxx:c:12000-24000", "m_yyyy:c:5000-17000", ...],
      "expected_chunk_ids": ["m_xxxx:c:12000-24000"],
      "recall_at_k": 1.0,
      "reciprocal_rank": 1.0,
      "min_recall_k": 5
    }
  ],
  "exemplar_results": [
    {
      "exemplar_id": "ex_...",
      "beat_scores_retrieved": [0.82, 0.74, 0.81, 0.79, 0.68, 0.77, 0.83],
      "median_beat_score": 0.79,
      "beat_count": 7
    }
  ],
  "aggregate": {
    "mean_recall_at_k": 0.87,
    "mean_mrr": 0.74,
    "median_exemplar_beat_score": 0.79
  },
  "comparison": {
    "baseline_eval_run_id": null,
    "recall_at_k_delta_relative": null,
    "mrr_delta_relative": null,
    "exemplar_score_delta_relative": null,
    "regression_flags": [],
    "verdict": null
  }
}
```

For non-baseline runs, `comparison` is populated and `verdict` is `"pass"` or `"fail"`.

### 2.5 `state/eval/baseline/{stage}.json`

A copy (not a symlink — MooseFS symlink behaviour is not verified) of the most recent gate-passing run for that stage. Written atomically via temp-rename when a flip passes.

---

## 3. Metric definitions (all computable, no human review)

### 3.1 Query metrics

**Recall@K**: for each query, `len(expected_chunks ∩ top_K_retrieved) / len(expected_chunks)`. K = `min_recall_k` field per query (default 5).

**MRR (Mean Reciprocal Rank)**: for each query, `1 / rank_of_first_expected_chunk_in_results` if any expected chunk appears in top-20, else 0. Averaged across queries.

**Regression conditions** (any one triggers FAIL):
1. `mean_recall_at_k` drops by > 20% relative vs baseline (e.g., 0.85 → < 0.68)
2. Any individual query drops from Recall@K ≥ 0.5 to Recall@K = 0.0 (total collapse)
3. `mean_mrr` drops by > 20% relative vs baseline

Rationale for 20% threshold: the corpus is small (1,586 chunks) and retrieval is already strong; a 20% degradation is clearly bad and not attributable to random variance at this scale.

### 3.2 Exemplar metrics

For each beat in an exemplar's beat_plan: run `/shorts/score` internally with the beat's role + the beat's clip metadata to get the retrieval score for the expected clip. Record `median_beat_score` across all beats in the exemplar.

**What this measures**: can the retrieval substrate still surface clips of the right quality for the expected beat slots? It is NOT a quality assessment of the output video — that remains human-reviewed.

**Regression condition**: `median_exemplar_beat_score` drops by > 20% relative vs baseline.

**What it does NOT measure** (intentionally): visual quality, audio quality, editorial coherence of the assembled short. Those require human review and are gated by the review_queue, not the eval harness.

### 3.3 Scope of triggers

| Stage flip | Triggers eval? | Why |
|---|---|---|
| B1 model tag or param change | **Yes** | Directly changes text retrieval |
| B4 model tag or param change | **Yes** | Directly changes visual/cross-modal retrieval |
| B0 chunking rule or window change | **Yes** (manual run only) | Changes the retrievable corpus — chunk_ids in expected_clips may be invalidated |
| A (ASR model) | No — MASTER.md freezes faster-whisper large-v3 | If ever changed, requires §15 brief |
| E0 (event tagging) | No | Events are metadata, not the retrieval substrate |
| F1 (identity model) | No | Speaker attribution, not retrieval embedding |
| Q0 | No | Quality scores, not retrieval embedding |
| C / C_vis (materialisation) | No | Mechanical — if B1/B4 passed, C is deterministic |
| Reranker | Separate eval metric if desired, not in founding set | Reranker affects ordering but not corpus coverage |

**B0 note**: B0 changes invalidate `chunk_id`-based expected_clips in `queries.json` if chunks are regenerated. After any B0 flip, re-run `erf-eval add-query` to refresh expected_clips before capturing a new baseline.

---

## 4. Gate wiring

### 4.1 Where the gate lives

The gate is **CLI-enforced**, not code-enforced. There is no runtime guard inside `embed_text.py` or `materialise.py` that checks stage_versions.json before running. The convention is:

> **Never update `state/embed_text_invocation.json`, `state/embed_visual_invocation.json`, or `state/model_registry.json` directly. Always go through `erf-eval gate-flip`.** Editing them directly bypasses the gate — this is intentional friction. An operator who does it knowingly is making a deliberate choice.

This matches the existing discipline: G-TEXT-EMBED and G-VISUAL-EMBED are already CLI-gated (the invocation files are only written after the gates pass). Wave 0 extends that discipline to versioned flips.

### 4.2 Gate flip flow

```
erf-eval gate-flip --stage B1 --model-tag qwen3-8b-v2
```

1. Read `state/eval/queries.json` — fail if missing or empty.
2. Read `state/eval/baseline/B1.json` — fail if missing (run `erf-eval baseline --stage B1` first).
3. Compute `invocation_fingerprint` of the candidate invocation state.
4. If fingerprint == current active fingerprint: no flip needed, exit 0 with message.
5. **Embed eval queries** with the candidate model (VRAM required, ~seconds for 20 queries).
6. **Query LanceDB** `text_v1` (or `visual_v1` for B4) with the freshly embedded vectors to get top-K for each query.
7. **Compute metrics** (Recall@K, MRR, exemplar scores).
8. **Compare** to baseline aggregates using the threshold rules in §3.
9. Write `state/eval/runs/{eval_run_id}.json` with verdict.
10. **If FAIL**: emit `eval/error` journal event (§5). Write `state/eval/runs/{eval_run_id}.json`. Exit non-zero. Do NOT update any pointer.
11. **If PASS**: emit `eval/ok` journal event. Update `state/eval/baseline/B1.json` (copy of run). Update `state/eval/stage_versions.json` with new model tag + fingerprint + run ID. Update `state/embed_text_invocation.json` / `state/model_registry.json` to reflect new active model. Exit 0.

### 4.3 VRAM-free check (index integrity, not model comparison)

```
erf-eval run --stage B1 --vram-free
```

Uses `stored_query_vector` from `queries.json` (embedded with the previously active model). Queries current LanceDB. Compares to baseline. Detects: index corruption, accidental table truncation, reindex errors. Does NOT detect model-level regressions (query vectors are old). Use this for periodic health checks between flips.

### 4.4 Gate failure mode

**Hard abort** — non-zero exit, no pointer updated. Operator sees:
- Exit code 1
- Failure summary on stdout (which metrics failed, by how much)
- Full diff in `state/eval/runs/{eval_run_id}.json`

No `--force`, no `--skip-eval`. Manual override = deliberately editing `stage_versions.json` with a comment in DECISIONS.md. This makes override visible in git history and audit trail.

### 4.5 Automatic vs. manual

The gate does NOT run automatically on every B1/B4 stage invocation (that would require models to be loaded for every re-run, violating the no-VRAM constraint and being impractically slow). The gate runs:
- **Manually**: `erf-eval run` or `erf-eval gate-flip` — always operator-triggered
- **On pointer flip requests**: `erf-eval gate-flip` — always and only when a flip is attempted

There is no cron job or file-watcher. This is a single-operator system; human intent is the only trigger.

---

## 5. Journal event shapes

Journal DB columns: `at_utc, job_id, media_id, stage, event_type, run_id, payload`.

```json
// eval start
{
  "at_utc": "2026-06-10T22:30:00Z",
  "job_id": "erf:eval:B1:eval_20260610T223000Z_abc123",
  "media_id": "erf",
  "stage": "eval",
  "event_type": "start",
  "run_id": "eval_20260610T223000Z_abc123",
  "payload": {
    "stage_tag": "B1",
    "model_tag": "qwen3-8b",
    "run_type": "candidate",
    "n_queries": 18,
    "n_exemplars": 5,
    "vram_free": false
  }
}

// eval pass
{
  "stage": "eval",
  "event_type": "ok",
  "run_id": "eval_20260610T223000Z_abc123",
  "payload": {
    "eval_run_id": "eval_20260610T223000Z_abc123",
    "stage_tag": "B1",
    "model_tag": "qwen3-8b",
    "mean_recall_at_k": 0.87,
    "mean_mrr": 0.74,
    "median_exemplar_beat_score": 0.79,
    "baseline_recall_at_k": 0.85,
    "baseline_mrr": 0.72,
    "baseline_exemplar_score": 0.79,
    "verdict": "pass"
  }
}

// eval fail
{
  "stage": "eval",
  "event_type": "error",
  "run_id": "eval_20260610T223100Z_def456",
  "payload": {
    "eval_run_id": "eval_20260610T223100Z_def456",
    "stage_tag": "B1",
    "model_tag": "qwen3-new-candidate",
    "mean_recall_at_k": 0.60,
    "baseline_recall_at_k": 0.85,
    "recall_at_k_delta_relative": -0.294,
    "regression_flags": ["recall_at_k_dropped_29pct", "query_eq_003_collapsed_to_zero"],
    "verdict": "fail"
  }
}
```

`event_type` uses the established vocabulary: `"start"`, `"ok"`, `"error"`, `"skip"`. No new event_type values needed.

---

## 6. Results storage and retention

**Per run**: `state/eval/runs/{eval_run_id}.json`. Written atomically (temp-rename). Contains full per-query breakdown, not just aggregates — essential for diagnosing regressions.

**Baseline pointer**: `state/eval/baseline/{stage}.json`. Updated only on gate pass. Never deleted while the stage has an active version. This is the reference that candidate runs compare against.

**Retention**: keep the last 10 runs per stage in `state/eval/runs/`. `erf-eval prune` deletes older runs (oldest first), but NEVER deletes:
- The run referenced in `state/eval/stage_versions.json[stage].active_eval_run_id`
- The run at `state/eval/baseline/{stage}.json`
- Any run with `verdict: "fail"` from the last 30 days (regression record)

---

## 7. CLI invocation spec

### `bin/erf-eval` — all commands

```bash
# --- DATASET MANAGEMENT ---

# Add a golden query (interactive; embeds the query to store vector)
erf-eval add-query \
  --query "Alex talking about morning routines" \
  --modality text \
  --expected-chunk m_xxxx:c:12000-24000 \
  --expected-chunk m_yyyy:c:5000-17000 \
  --min-recall-k 5 \
  --note "Alex morning prep scene"

# Add an exemplar from an approved review
erf-eval add-exemplar \
  --review-id rev_xxxx \
  --name "Alex intro hook" \
  --notes "clean hook energy"

# List current eval dataset
erf-eval list

# --- BASELINE CAPTURE ---

# Capture the baseline for a stage (VRAM required — embeds all queries)
# Run once on initial setup, and after each successful gate flip
erf-eval baseline --stage B1
erf-eval baseline --stage B4

# --- GATE CHECKS ---

# Run eval on current index — no flip (VRAM-free if query vectors stored)
erf-eval run [--stage B1] [--vram-free]

# Attempt a stage flip — runs eval then flips pointer if passing
erf-eval gate-flip --stage B1 --model-tag <new_model_tag>

# Show status of M4 pointers
erf-eval status

# --- MAINTENANCE ---

# Prune old runs (respects retention rules)
erf-eval prune [--stage B1]

# Re-embed stored query vectors with current active model (after a confirmed flip)
erf-eval refresh-vectors --stage B1
```

### Module invocation (for scripting)
```bash
PYTHONPATH=src venvs/embed/bin/python -m erf.eval run --stage B1
PYTHONPATH=src venvs/embed/bin/python -m erf.eval gate-flip --stage B1 --model-tag qwen3-8b-v2
```

### Initialization sequence (one-time)
```bash
# 1. Create eval dataset (repeat add-query for each query; expect ~30 min total)
erf-eval add-query ...   # 15-18 queries total

# 2. Add exemplars from approved reviews (5 exemplars)
erf-eval add-exemplar --review-id rev_01KTQ8ZNM3M60X4XPHMBWD91HC --name "Alex hook 2026-06-09"

# 3. Capture baseline (VRAM required — text model loads once, embed all queries)
erf-eval baseline --stage B1
erf-eval baseline --stage B4

# 4. Confirm status
erf-eval status
```

---

## 8. Founding set specification

**Recommended founding size: 18 queries + 5 exemplars.**

Rationale: 220 files, 1,586 chunks, 3 enrolled persons. The corpus is small enough that 18 queries cover the major retrieval modes without requiring extensive curation. Each query takes <10ms for ANN search, so 18 queries + 5 exemplar-score computations runs in under 5 seconds (VRAM-free mode).

### Query coverage targets

| Category | Count | Examples |
|---|---|---|
| Speaker-specific (p_alex) | 3–4 | "Alex explaining [topic]", "Alex's take on [theme]" |
| Speaker-specific (p_jordan) | 2–3 | "Jordan describing [topic]" |
| Speaker-specific (p_sam) | 2 | "Sam talking about [topic]" |
| Thematic/topic | 3–4 | "morning routine", "travel prep", "on-camera advice" |
| Edit-role (hook content) | 2 | Queries that surface the 3 known hook-rated chunks |
| Visual/cross-modal | 3 | "outdoor daylight interview", "close-up face talking" |
| **Total** | **15–18** | |

Do NOT add "negative queries" (queries with no expected clips). Recall@K cannot be computed for them, and they complicate the metric cleanly. If you want to verify no-false-positives, that's a separate manual check, not an automated metric.

### Exemplar coverage targets

5 exemplars from approved review_queue items. If fewer approved items exist at init time, start with what's available and add more as the review queue accumulates.

Cover: at minimum one per enrolled speaker as a "primary subject" in the beat_plan, and one that uses a mix of b-roll + dialogue beats.

---

## 9. MASTER §15 items

**None required.** The eval harness is entirely additive:

- No new stage letter. `eval` is a harness utility, not a media analysis stage; it has no `stage` letter in the §5 registry.
- No new LanceDB table. Eval query vectors stored in `state/eval/queries.json` as plain float arrays. Not a new index table.
- No catalog schema changes. All results in `state/` only, as specified.
- No new always-on service or daemon.

**Flag for consideration** (not a blocker, Code decides):

> **Optional §15 future item**: adding a `stage_version_fingerprint` field to catalog embedding records (`.text.jsonl`, `.visual.jsonl`) would allow per-artifact provenance tracing — knowing which model produced each embedding. This is a MAJOR schema change (new field, meaning change per Invariant 5) and requires a §15 brief. **Not recommended for Wave 0.** The `state/eval/stage_versions.json` pointer is sufficient for regression gating. Artifact-level provenance is only needed if you ever need to partially re-embed (e.g., only files added after a model flip). Defer to a future brief if that need arises.

---

## 10. Implementation order (strict)

1. **`src/erf/eval.py`** — core module: `EvalRun`, `run_queries()`, `compare_to_baseline()`, metric computation. No CLI yet. Write a minimal test against existing state (mock LanceDB if needed — but the real LanceDB is on disk so prefer real queries).
2. **`bin/erf-eval`** — CLI wrapper: `add-query`, `add-exemplar`, `baseline`, `run`, `gate-flip`, `status`, `list`, `prune`, `refresh-vectors`. Use `argparse`. Python script using embed venv.
3. **Initialize M4 state**: `state/eval/stage_versions.json` with current B1/B4 invocation fingerprints (pre-flip state — no baseline captured yet, just pointer initialized). `stage_versions.json` always exists once the harness is initialized; gate-flip won't run without it.
4. **Curate founding set**: run `erf-eval add-query` 18 times (interactive, requires operator to verify expected clips). This is the one step automation can't do alone — it requires a human reviewer to confirm which clips are the correct expected results for each query.
5. **Capture baseline**: `erf-eval baseline --stage B1 && erf-eval baseline --stage B4`. Requires VRAM.
6. **Verify**: `erf-eval run --vram-free` against current index → should show 100% Recall@K (queries were added from current index results). Confirm journal events.

---

## 11. Decisions deferred to Code

These are within §15 unilateral authority:

- Exact arg parsing library (argparse recommended, no dependencies)
- Batch size for eval query embedding (small — 20 queries, batch_size=20 is fine)
- Whether `erf-eval` is a `__main__.py` module or a plain script (plain script preferred — consistent with existing bin/ entrypoints)
- Exact path for `erf-eval` (recommend `/workspace/erf/bin/erf-eval`, symlinked or on PATH)
- Whether the baseline capture writes both `state/eval/runs/` and `state/eval/baseline/` or just the latter (write both — consistent with the retention scheme)

---

*End of brief. All questions from the commissioning brief (§§1–5) are answered above. No open questions remain.*

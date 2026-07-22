"""Stage registry — the single source of truth for "what stages exist" (D-Log-04, Phase 0).

`docs/OBSERVABILITY_REDESIGN.md` §3.3/§4. Until now the pipeline kept three independently
hardcoded stage-universes that drift by construction:

  1. `erf.scheduler.STAGE_REGISTRY`        (14 stages: DAG shape — depends/tier/phase)
  2. `bin.reconcile_status.STAGE_ARTIFACTS` (29 stages: artifact-path exists/nonempty predicates)
  3. `bin.pipeline_status.CRITICAL` / `SPEECH_SPINE` (10 / 9 stages: health-gate membership)

`STAGES` below is the UNION of all three, seeded verbatim from each (see field-provenance
notes per stage in the dict). **Phase 0 makes no behavior change** — `scheduler.py`,
`reconcile_status.py`, and `pipeline_status.py` are untouched; their literals are deleted in
Phase 2 once consumers are rewired to iterate this registry instead.

## Name resolution: the "F1 collision"

§3.3 anticipated a collision between scheduler's `F1` (a DAG node: `depends=["F0"]`,
`tier="cpu"`, `phase=3`) and reconcile's `F1` (a predicate on `catalog/identity_overlay/{mid}.json`).
**Investigated and found NOT to be a real collision.** `src/erf/identity.py` — the actual stage
implementation — opens with the docstring 'Stage F1 -- identity resolution + overlay
(control, no GPU).', journals via `emit_start(journal_db, media_id, "F1", run_id)`, and
writes exactly `catalog/identity_overlay/{media_id}.json` (`identity.py:329`, `outputs=[...
identity_overlay ...]`). This is consistent with `MASTER.md` §5's F1 row ("Entity nodes" —
erf resolution — `face_tracks, diar, mentions, galleries` -> `entity_node, person` links) and
with `bin/timeline_audit.py`'s own `READ_EDGES["F1"] = ["F0"]`. scheduler-F1 and reconcile-F1
are the same real stage described from two angles (schedule position vs. artifact predicate),
not two different tools sharing a name. **Resolution: merged into one registry key `"F1"`**
carrying both the scheduler DAG fields and the reconcile artifact predicate — splitting it
into `F1`/`F1_overlay` would have fabricated a second stage that does not exist and worked
against the redesign's own P2 principle (one declarative registration per real stage). This is
flagged as a correction to the §3.3 premise, not a silent deviation — see the Phase 0 report.

## Deliberate exclusions / additions

- **`RETRANSCRIBE` is excluded.** Per D-Log-05 it is an invocation *mode* of stage `A`
  (`bin/retranscribe.py`: re-ASR + re-diar + re-F1 at max quality), not a distinct stage.
  `reconcile_status.py` already excludes it explicitly (its own comment: "RETRANSCRIBE
  overwrites A's words ... intentionally NOT listed"); scheduler and pipeline_status never
  had it either. No universe actually disagrees here.
- **`build_identity_clusters.py` (`CLUSTERS`) is added as a first-class stage**, per §4 Phase 0
  and §1.5 of the redesign doc (off-spine today: no journal call anywhere in the script, no
  reconcile predicate, absent from CRITICAL/scheduler — "the tool that drove D-Identity-04 is
  invisible to write AND read"). Its artifact paths (`catalog/identity_clusters/clusters.json`,
  `catalog/identity_map/{mid}.json`) are read directly from its own docstring/source
  (`bin/build_identity_clusters.py` lines 14-15, 259, 271), not invented.

## Field provenance for stages not in `scheduler.STAGE_REGISTRY`

`depends` for stages absent from scheduler were NOT invented: they are seeded from
`bin/timeline_audit.py`'s own `READ_EDGES` dict (a *fourth*, previously undocumented,
hardcoded DAG the redesign doc doesn't mention — see the Phase 0 report) where available, else
a documented best-effort inference (REDIARIZE, CLUSTERS — no existing source declares their
upstream deps).

`tier` for stages absent from scheduler was checked against source (grep for
`torch`/`cuda`/`device=`) rather than defaulted blind: D0, RP, DX, V0, F2, REDIARIZE
confirmed GPU-capable (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`
patterns); tags/P0/RG/CARD/SEV2/SEV2_ADJ/PSTREAM/T0/EV/LABEL/CLUSTERS confirmed CPU-only
(deterministic logic, DeepSeek Flash API calls, or sklearn — no torch import). W is inferred
gpu-preferred by pattern (WhisperX word extraction) though the script itself has no direct
torch/cuda hit (likely delegates) — documented as inference, not verified.

`phase` for stages absent from scheduler has no source anywhere (the MASTER §5.1 creative-stage
table has no Phase column) — assigned informally by dependency depth, purely illustrative, and
NOT asserted equal to anything by the Phase 0 test.

`health_predicates` and `fingerprint_inputs` are `()` for every stage in Phase 0 by design
(populated in Phase 3).

## Phase 3 addendum (D-Log-04 §4 Phase 3, docs/OBSERVABILITY_REDESIGN.md §3.2/§3.3)

`health_predicates` is now populated for CLUSTERS/SEV2/T0: a tuple of
`(name, op, floor)` where `op` is one of `(">=", "<=", ">")`. The floor-comparison
logic lives in exactly one place — `evaluate_health()` below — so a floor value
never has a second copy to drift out of sync (P2's anti-drift principle, applied
to health floors rather than stage existence). Stages compute their own raw
values and call `evaluate_health(stage_name, values)`; a predicate name absent
from `values` is skipped entirely, not defaulted — this is how a CONDITIONAL
predicate (e.g. SEV2/T0's "only applies if this media's identity_map has >=1
gallery-anchored slot") is expressed: the calling stage decides per-call whether
the predicate is in scope; `evaluate_health` only ever does the floor comparison.

`fingerprint_inputs` is populated for SEV2/T0 (`catalog/identity_map/{mid}.json`)
— the upstream artifact whose content-fingerprint they now carry in `inputs=`
(see `outcomes.record`). No staleness-marking logic reads it yet (that is a
read-side concern, tracked for a later phase) — this phase only makes the data
exist.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    depends: tuple[str, ...] = ()
    tier: str = "cpu"                  # "cpu" | "gpu-preferred" | "gpu"
    phase: int = 0
    scope: str = "media"               # "media" | "corpus" | "on_demand"
    artifact_path: str | None = None   # repo-root-relative, "{mid}" placeholder
    predicate_kind: str | None = None  # "ne" (exists+nonempty) | "exists" (explicit-empty-allowed)
    also_requires: tuple[tuple[str, str], ...] = ()  # extra (predicate_kind, artifact_path) branches
    # that must ALSO hold for the stage's artifact predicate — Phase 2 (D-Log-04):
    # reconcile_status.STAGE_ARTIFACTS had compound `A and B` predicates for A/D0/DX;
    # Phase 0 preserved the second branch only in a comment (see per-stage notes
    # below). This reproduces that AND exactly instead of silently dropping it.
    critical: bool = False
    speech_spine: bool = False
    scheduled: bool = False    # Phase 2: True for exactly the 14 stages scheduler.py schedules
    visual: bool = False       # Phase 2: True for the visual/face chain timeline_audit.VISUAL tracked
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    health_predicates: tuple = ()             # Phase 0: always empty; populated Phase 3
    fingerprint_inputs: tuple[str, ...] = ()  # Phase 0: always empty; populated Phase 3


def _s(name: str, *, depends=(), tier="cpu", phase=0, scope="media",
       artifact_path=None, predicate_kind=None, also_requires=(), critical=False,
       speech_spine=False, scheduled=False, visual=False,
       consumes=(), produces=(), health_predicates=(), fingerprint_inputs=()) -> Stage:
    return Stage(
        name=name, depends=tuple(depends), tier=tier, phase=phase, scope=scope,
        artifact_path=artifact_path, predicate_kind=predicate_kind,
        also_requires=tuple(also_requires),
        critical=critical, speech_spine=speech_spine,
        scheduled=scheduled, visual=visual,
        consumes=tuple(consumes), produces=tuple(produces),
        health_predicates=tuple(health_predicates),
        fingerprint_inputs=tuple(fingerprint_inputs),
    )


STAGES: dict[str, Stage] = {

    # ── Core registry (S: scheduler.STAGE_REGISTRY, R: reconcile.STAGE_ARTIFACTS,
    #    P: pipeline_status.CRITICAL/SPEECH_SPINE) ───────────────────────────────

    "A": _s(
        "A", depends=[], tier="gpu-preferred", phase=1,                 # S verbatim
        artifact_path="catalog/asr/{mid}.words.jsonl", predicate_kind="ne",  # R (primary of AND)
        # R's actual predicate is an AND: _ne(media_manifest/{mid}.json) and _ne(asr/{mid}.words.jsonl).
        # Phase 2: reproduced exactly via also_requires (was a comment-only note in Phase 0).
        also_requires=(("ne", "catalog/media_manifest/{mid}.json"),),
        critical=True, speech_spine=False, scheduled=True,              # P: in CRITICAL, not in SPEECH_SPINE
        consumes=["source media file"],
        produces=["media_manifest", "asr_word", "asr_segment", "diarization turns"],
    ),
    "B0": _s(
        "B0", depends=["A"], tier="cpu", phase=1,                       # S verbatim
        artifact_path="catalog/chunks/{mid}.jsonl", predicate_kind="ne",  # R
        scheduled=True,
        consumes=["asr words+segments+diar"], produces=["chunk"],
    ),
    "B1": _s(
        "B1", depends=["B0"], tier="gpu", phase=1,                      # S verbatim
        artifact_path="catalog/embeddings/{mid}.text.jsonl", predicate_kind="ne",  # R
        scheduled=True,
        consumes=["chunk text"], produces=["embedding(text)"],
    ),
    "B2": _s(
        "B2", depends=["A"], tier="cpu", phase=2,                       # S verbatim
        artifact_path="catalog/scenes/{mid}.jsonl", predicate_kind="ne",  # R
        scheduled=True, visual=True,
        consumes=["source media"], produces=["scene"],
    ),
    "B3": _s(
        "B3", depends=["B2"], tier="cpu", phase=2,                      # S verbatim
        artifact_path="catalog/keyframes/{mid}.jsonl", predicate_kind="ne",  # R
        scheduled=True, visual=True,
        consumes=["scenes", "media"], produces=["keyframe"],
    ),
    "B4": _s(
        "B4", depends=["B3"], tier="gpu", phase=2,                      # S verbatim
        artifact_path="catalog/embeddings/{mid}.visual.jsonl", predicate_kind="ne",  # R
        scheduled=True, visual=True,
        consumes=["keyframe images"], produces=["embedding(visual)"],
    ),
    "C": _s(
        "C", depends=["B1", "B4"], tier="cpu", phase=1,                 # S verbatim
        # R deliberately has NO predicate for C ("materialise to LanceDB ... no distinct
        # per-media artifact ... intentionally NOT listed" — reconcile_status.py comment).
        artifact_path=None, predicate_kind=None, scheduled=True,
        consumes=["embeddings(text+visual)"], produces=["LanceDB text_v1", "LanceDB visual_v1"],
    ),
    "E0": _s(
        "E0", depends=["B0", "B2"], tier="cpu", phase=2,                # S verbatim
        artifact_path="catalog/events/{mid}.jsonl", predicate_kind="ne",  # R
        scheduled=True,
        consumes=["chunks", "scenes"], produces=["event"],
    ),
    "P": _s(
        "P", depends=["B3"], tier="cpu", phase=2,                       # S verbatim
        artifact_path="media/proxies/{mid}.mp4", predicate_kind="ne",   # R (ROOT-relative, not catalog/)
        scheduled=True,
        consumes=["source media", "keyframes"], produces=["proxy mp4", "webp thumbs"],
    ),
    "Q0": _s(
        "Q0", depends=["P"], tier="cpu", phase=2,                       # S verbatim (consumes_hard=["proxy"])
        artifact_path="catalog/quality/{mid}.jsonl", predicate_kind="ne",  # R
        scheduled=True,
        consumes=["proxy [hard]"], produces=["quality"],
    ),
    "F0": _s(
        "F0", depends=["A", "B3"], tier="gpu", phase=3,                 # S verbatim (consumes_hard=["proxy"])
        artifact_path="catalog/face_tracks/{mid}.jsonl", predicate_kind="ne",  # R
        scheduled=True, visual=True,
        consumes=["media", "transcript", "proxy [hard]"],
        produces=["detection", "face_track", "entity mentions"],
    ),
    "F1": _s(
        "F1", depends=["F0"], tier="cpu", phase=3,                      # S verbatim
        # R's "F1" predicate (identity_overlay) merged in here — see module docstring: this is
        # NOT a collision, both universes describe the same real stage (src/erf/identity.py).
        artifact_path="catalog/identity_overlay/{mid}.json", predicate_kind="ne",  # R
        critical=False, speech_spine=False, scheduled=True, visual=True,
        consumes=["face_tracks", "diar", "mentions", "galleries"],
        produces=["entity_node", "person links", "identity_overlay"],
    ),
    "E1": _s(
        "E1", depends=["E0"], tier="cpu", phase=4,                      # S verbatim
        artifact_path="catalog/event_signals/{mid}.jsonl", predicate_kind="ne",  # R
        scheduled=True,
        consumes=["event@* [hard]", "identity_overlay@* [soft]"], produces=["event_signals"],
    ),
    "E2": _s(
        "E2", depends=["E1", "C"], tier="cpu", phase=5, scope="corpus",  # S verbatim
        # Not in R (corpus-scoped, no per-media artifact predicate).
        artifact_path=None, predicate_kind=None, scheduled=True,
        consumes=["event_signals@* [hard]", "embeddings@* [hard]", "identity_overlay@* [soft]"],
        produces=["event_embeddings", "narrative_threads"],
    ),

    # ── Reconcile-only additions (R; depends/tier seeded from bin/timeline_audit.py
    #    READ_EDGES + source grep where scheduler is silent — see docstring) ──────

    "F2": _s(
        "F2", depends=["F0", "P"], tier="gpu-preferred", phase=3,       # depends: timeline_audit READ_EDGES;
        # tier: bin/asd_worker.py:294 torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artifact_path="catalog/active_speaker/{mid}.json", predicate_kind="ne", visual=True,  # R
        consumes=["proxy", "face_tracks"], produces=["active_speaker"],
    ),
    "tags": _s(
        "tags", depends=["B0", "E0"], tier="cpu", phase=2,
        # depends: src/erf/tags.py reads catalog/chunks + catalog/events directly;
        # batch_ingest_bts.py runs stages in "Q0 -> E0 -> tags" order (phase=2, alongside Q0/E0).
        artifact_path="catalog/tags/{mid}.jsonl", predicate_kind="ne",  # R
        consumes=["chunks", "events"], produces=["tags"],
    ),
    "REDIARIZE": _s(
        "REDIARIZE", depends=["A"], tier="gpu-preferred", phase=3,
        # depends: bin/rediarize.py prefers media/proxies/{mid}.mp4, falls back to A's source_path
        # (soft dep on P). tier: rediarize.py:84 pipeline.to(torch.device("cuda" if ... else "cpu")).
        # NOTE: distinct from RETRANSCRIBE (excluded, D-Log-05) — REDIARIZE is a real recorded
        # stage (outcomes.record(mid, "REDIARIZE", ...), rediarize.py:196/200/208).
        artifact_path="catalog/asr/{mid}.diar_embeddings.json", predicate_kind="ne",  # R
        consumes=["source_path/proxy audio"], produces=["diar_embeddings"],
    ),
    "D0": _s(
        "D0", depends=["P"], tier="gpu-preferred", phase=3,             # depends: timeline_audit READ_EDGES
        # tier: bin/d0_preprocess.py imports torch/cuda (WhisperX align + Silero VAD)
        artifact_path="catalog/asr/{mid}.words.wx.jsonl", predicate_kind="ne",  # R (primary of AND)
        # R's actual predicate is an AND with catalog/asr/{mid}.vad.jsonl — reproduced via also_requires.
        also_requires=(("ne", "catalog/asr/{mid}.vad.jsonl"),),
        consumes=["proxy"], produces=["asr_words.wx", "vad", "overlap"],
    ),
    "P0": _s(
        "P0", depends=["A"], tier="cpu", phase=3,                       # depends: timeline_audit READ_EDGES
        artifact_path="catalog/asr/{mid}.sentences.jsonl", predicate_kind="ne",  # R
        consumes=["words"], produces=["sentences"],
    ),
    "W": _s(
        "W", depends=["D0", "A"], tier="gpu-preferred", phase=3,        # depends: timeline_audit READ_EDGES
        # tier: inferred GPU-preferred by pattern with D0/RP (WhisperX word extraction);
        # bin/build_asr_words.py itself has no direct torch/cuda hit — likely delegates. Documented
        # inference, not independently verified.
        artifact_path="catalog/asr_words/{mid}.jsonl", predicate_kind="ne",  # R
        critical=True, speech_spine=True,                               # P: in CRITICAL and SPEECH_SPINE
        consumes=["proxy audio"], produces=["asr_words"],
    ),
    "RG": _s(
        "RG", depends=["W", "D0"], tier="cpu", phase=4,                 # depends: timeline_audit READ_EDGES
        # tier: bin/build_recall_gaps.py has no torch/cuda import — pure detection logic.
        artifact_path="catalog/asr_recall_gaps/{mid}.jsonl", predicate_kind="exists",  # R (presence, not _ne)
        critical=True, speech_spine=True,                               # P
        consumes=["words", "vad", "overlap"], produces=["asr_recall_gaps"],
    ),
    "RP": _s(
        "RP", depends=["RG"], tier="gpu-preferred", phase=5,            # depends: timeline_audit READ_EDGES
        # tier: bin/recall_pass.py imports torch/cuda (plain whisper on gap windows)
        artifact_path="catalog/asr_recall_pass/{mid}.jsonl", predicate_kind="exists",  # R (presence)
        critical=True, speech_spine=True,                               # P
        consumes=["asr_recall_gaps"], produces=["asr_recall_pass"],
    ),
    "DX": _s(
        "DX", depends=["P"], tier="gpu-preferred", phase=3,             # depends: timeline_audit READ_EDGES
        # tier: bin/run_diar_community1.py imports torch/cuda (pyannote community-1)
        artifact_path="catalog/diar_hypotheses/{mid}/community1.jsonl", predicate_kind="exists",  # R primary of AND
        # R's actual predicate is an AND with .../community1_excl.jsonl — reproduced via also_requires.
        also_requires=(("exists", "catalog/diar_hypotheses/{mid}/community1_excl.jsonl"),),
        critical=True, speech_spine=True,                               # P
        consumes=["proxy"], produces=["diar_hypotheses/community1", "diar_hypotheses/community1_excl"],
    ),
    "SEV2": _s(
        "SEV2", depends=["W", "RP", "DX"], tier="cpu", phase=6,         # depends: timeline_audit READ_EDGES
        artifact_path="catalog/speech_events_v2/{mid}.jsonl", predicate_kind="exists",  # R (presence)
        critical=True, speech_spine=True,                               # P
        consumes=["words", "recall", "diar", "identity_map [soft]"], produces=["speech_events_v2"],
        # Phase 3 (D-Log-04 §4): conditional coverage predicate — see module
        # docstring. Caller includes "pct_person_id" in evaluate_health()'s
        # `values` only when this media's identity_map has >=1 gallery slot;
        # rando-only media omit the key -> predicate not evaluated -> vacuous pass.
        health_predicates=(("pct_person_id", ">", 0.0),),
        fingerprint_inputs=("catalog/identity_map/{mid}.json",),
    ),
    "SEV2_ADJ": _s(
        "SEV2_ADJ", depends=["SEV2"], tier="cpu", phase=7,              # depends: timeline_audit READ_EDGES
        artifact_path="catalog/speech_events_v2_adjudicated/{mid}.jsonl", predicate_kind="exists",  # R (presence)
        critical=True, speech_spine=True,                               # P
        consumes=["speech_events_v2", "diar_embeddings [soft]"], produces=["speech_events_v2_adjudicated"],
    ),
    "PSTREAM": _s(
        "PSTREAM", depends=["SEV2_ADJ", "RP"], tier="cpu", phase=8,     # depends: timeline_audit READ_EDGES
        artifact_path="catalog/render_events_v2/{mid}.jsonl", predicate_kind="exists",  # R (presence)
        critical=True, speech_spine=True,                               # P
        consumes=["speech_events_v2_adjudicated", "asr_recall_pass"],
        produces=["parallel_streams", "render_events_v2"],
    ),
    "T0": _s(
        "T0", depends=["PSTREAM", "SEV2_ADJ"], tier="cpu", phase=9,     # depends: timeline_audit READ_EDGES
        artifact_path="catalog/asr/{mid}.transcript.jsonl", predicate_kind="ne",  # R
        critical=True, speech_spine=True,                               # P
        consumes=["render_events_v2", "speech_events_v2_adjudicated [fallback]"],
        produces=["asr.transcript"],
        # Phase 3 (D-Log-04 §4): same conditional coverage predicate as SEV2 —
        # stop discarding the pct_person_id T0 already computed (§2 root cause #1).
        health_predicates=(("pct_person_id", ">", 0.0),),
        fingerprint_inputs=("catalog/identity_map/{mid}.json",),
    ),
    "V0": _s(
        "V0", depends=["B3"], tier="gpu-preferred", phase=3,            # depends: timeline_audit READ_EDGES
        # tier: bin/v0_visual_grounding.py imports torch/cuda (Qwen2.5-VL)
        artifact_path="catalog/visual_grounding/{mid}.json", predicate_kind="ne", visual=True,  # R
        consumes=["keyframes"], produces=["visual_grounding"],
    ),
    "CARD": _s(
        "CARD", depends=["E0"], tier="cpu", phase=4,                    # depends: timeline_audit READ_EDGES
        artifact_path="catalog/media_cards/{mid}.json", predicate_kind="ne",  # R
        consumes=["events"], produces=["media_cards"],
    ),
    "EV": _s(
        "EV", depends=["T0", "V0", "E1", "CARD", "F2", "F1"], tier="cpu", phase=10,
        # depends: timeline_audit READ_EDGES
        artifact_path="catalog/evidence/{mid}.json", predicate_kind="ne",  # R
        critical=True, speech_spine=True,                               # P
        consumes=["all signals (T0,V0,E1,CARD,F2,F1)"], produces=["evidence"],
    ),
    "LABEL": _s(
        "LABEL", depends=["EV"], tier="cpu", phase=11,                  # depends: timeline_audit READ_EDGES
        artifact_path="catalog/editorial/{mid}/manifest.json", predicate_kind="ne",  # R
        consumes=["evidence"], produces=["editorial"],
    ),

    # ── Added, not in any of the three universes (spec §4 Phase 0 + §1.5) ────────

    "CLUSTERS": _s(
        "CLUSTERS", depends=["REDIARIZE"], tier="cpu", phase=4, scope="corpus",
        # bin/build_identity_clusters.py: originally no journal/outcomes call anywhere (silent
        # exit 0 — off-spine by construction, the exact D-Identity-04 blind spot); on-spine as
        # of Phase 3, in RECONCILE_ORDER as of D-Log-16 (2026-07-12). depends/tier/artifact
        # paths read directly from source (lines 14-15 docstring, 259/271 writes), not invented.
        # Runs once across the whole corpus (not per-media) -> scope="corpus", though it also
        # writes a per-media catalog/identity_map/{mid}.json side-file (used as artifact_path
        # since that's what SEV2 consumes_soft depends on).
        artifact_path="catalog/identity_map/{mid}.json", predicate_kind="ne",
        consumes=["diar_embeddings", "voiceprint_v2/gallery_clean"],
        produces=["identity_clusters/clusters.json", "identity_map"],
        # Phase 3 (D-Log-04 §4): corpus-level resolution-rate predicates. Both
        # live on the "__corpus__" record (bin/build_identity_clusters.py) —
        # this stage clusters the WHOLE corpus in one pass, so there is no
        # meaningful per-media gallery_share. Healthy corpus values measured
        # 2026-07-10: gallery_share=0.585, no_embedding_share=0.053 (DECISIONS.md
        # D-Log-08 "Next").
        health_predicates=(
            ("gallery_share", ">=", 0.45),
            ("no_embedding_share", "<=", 0.15),
        ),
    ),

    "GATE": _s(
        "GATE", depends=[], tier="cpu", phase=99, scope="corpus",
        # bin/eval_attribution.py: a read-only verdict OVER the corpus (compares
        # this run's attribution metrics against state/eval/baseline/ATTR.json),
        # not a pipeline stage that produces an artifact — no artifact_path/
        # predicate_kind (never reconciled, never scheduled), not critical/
        # speech_spine (its own regression check IS its health signal, expressed
        # directly as record()'s status=ok|degraded rather than a floor-compare
        # health_predicates tuple — the baseline it compares against is loaded
        # from disk at runtime, not a fixed floor, so it doesn't map onto the
        # (name, op, floor) shape `evaluate_health()` expects). Phase 4
        # (D-Log-04 §4, docs/OBSERVABILITY_REDESIGN.md §4): "Emit the attribution
        # gate result as a first-class stage."
        consumes=["catalog/asr/{mid}.transcript.jsonl (all golden media)",
                  "state/eval/attribution_golden.jsonl", "state/eval/baseline/ATTR.json"],
        produces=["state/eval/runs/attr_{run_utc}.json"],
    ),

    # ── On-demand stages (D-Audit-01 finding #4, worklist item 3(c)/D-Log-26):
    #    SHORT/STORY/SURVEY/ORIENT each journal outcomes but were absent from
    #    every stage universe — green-but-untracked. Registered here with
    #    scope="on_demand": invoked manually per editorial/maintenance need,
    #    deliberately excluded from the rebuild DAG (scheduled stays False, the
    #    default — scheduler.py's STAGE_REGISTRY only includes `if s.scheduled`;
    #    no rebuild_all.py entry exists for any of the four either). No
    #    artifact_path/predicate_kind — same reasoning as GATE above: none of
    #    the four produce a reconcilable one-artifact-per-every-media output
    #    (SHORT/STORY are export_id-keyed and apply to a chosen subset; SURVEY
    #    is a single corpus-wide file; ORIENT's output path collides with P's
    #    own artifact), so they stay outside _tracked_predicate_stages/
    #    RECONCILE_ORDER below rather than fabricating a predicate that would
    #    either double-count or under-count real coverage. ───────────────────

    "SHORT": _s(
        "SHORT", depends=["E2"], tier="cpu", phase=0, scope="on_demand",
        # bin/make_short.py: grounded-editor short-form export for one media_id.
        # Journal stage name from stage_run("SHORT", mid) (make_short.py:70).
        consumes=["catalog/evidence/{mid}.json"],
        produces=["catalog/exports/{export_id}.mp4", "catalog/shorts/{export_id}.json"],
    ),
    "STORY": _s(
        "STORY", depends=["E2"], tier="cpu", phase=0, scope="on_demand",
        # bin/make_story.py: cross-media narrative short, DS-Pro-surveyed.
        # Journal stage name from stage_run("STORY", title) (make_story.py:96).
        consumes=["catalog/evidence/*.json (whole corpus)"],
        produces=["catalog/exports/{export_id}.mp4", "catalog/stories/{export_id}.json"],
    ),
    "SURVEY": _s(
        "SURVEY", depends=[], tier="cpu", phase=0, scope="on_demand",
        # bin/corpus_survey.py: DS-Pro corpus-wide marketing-moment picker.
        # log_event("SURVEY", "_corpus", ...) (corpus_survey.py:56) — corpus-wide
        # single-file output, not per-media, but scope="on_demand" (not
        # "corpus") since it also isn't part of the corpus rebuild DAG.
        consumes=["condensed corpus evidence"],
        produces=["catalog/survey/latest.json"],
    ),
    "ORIENT": _s(
        "ORIENT", depends=["P"], tier="cpu", phase=0, scope="on_demand",
        # bin/fix_orientation.py: content-based (face-detection) proxy rotation
        # fixer; re-encodes media/proxies/{mid}.mp4 in place when mis-rotated.
        # log_event("ORIENT", mid, ...) (fix_orientation.py:95) — the script's
        # own docstring header calls it "P1", but the journaled stage name is
        # ORIENT; registered under the name actually used at runtime. No-face
        # footage (e.g. generic b-roll) is left untouched and reported — V0's
        # orientation field (D-Log-25) now covers that gap for b-roll specifically.
        consumes=["media/proxies/{mid}.mp4", "face detections"],
        produces=["media/proxies/{mid}.mp4 (corrected in place)"],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Accessors
# ──────────────────────────────────────────────────────────────────────────────

def get(name: str) -> Stage:
    return STAGES[name]


def critical_stages() -> dict[str, Stage]:
    return {k: v for k, v in STAGES.items() if v.critical}


def speech_spine_stages() -> dict[str, Stage]:
    return {k: v for k, v in STAGES.items() if v.speech_spine}


def media_scoped() -> dict[str, Stage]:
    return {k: v for k, v in STAGES.items() if v.scope == "media"}


def corpus_scoped() -> dict[str, Stage]:
    return {k: v for k, v in STAGES.items() if v.scope == "corpus"}


def on_demand_scoped() -> dict[str, Stage]:
    return {k: v for k, v in STAGES.items() if v.scope == "on_demand"}


def scheduled_stages() -> dict[str, Stage]:
    return {k: v for k, v in STAGES.items() if v.scheduled}


def visual_stages() -> dict[str, Stage]:
    return {k: v for k, v in STAGES.items() if v.visual}


# ──────────────────────────────────────────────────────────────────────────────
# Health predicates (D-Log-04 §4 Phase 3, docs/OBSERVABILITY_REDESIGN.md §3.1/§3.2)
# ──────────────────────────────────────────────────────────────────────────────

_HEALTH_OPS = {
    ">=": lambda value, floor: value >= floor,
    "<=": lambda value, floor: value <= floor,
    ">":  lambda value, floor: value > floor,
}


def evaluate_health(stage_name: str, values: dict) -> dict:
    """Evaluate `stage_name`'s declared `health_predicates` against caller-
    supplied raw values. Returns the §3.1 health-dict shape:

        {predicate_name: {"value": v, "floor": f, "verdict": "pass"|"fail"}}

    one entry per predicate name PRESENT in `values` — a predicate the caller
    omits from `values` is skipped entirely (not evaluated, not defaulted).
    This is the single home for floor-comparison logic: stages compute their
    own raw values (gallery_share, pct_person_id, ...) and pass them in; only
    the floor and the pass/fail comparison live here, so they can never drift
    out of sync between stages that share the same predicate.

    A stage with no declared `health_predicates` (or an unknown `stage_name`)
    returns `{}` — no predicates means no possible `record()` degraded flip.
    """
    stage = STAGES.get(stage_name)
    if stage is None or not stage.health_predicates:
        return {}
    out: dict = {}
    for name, op, floor in stage.health_predicates:
        if name not in values:
            continue
        cmp_fn = _HEALTH_OPS.get(op)
        if cmp_fn is None:
            raise ValueError(f"{stage_name}: unknown health predicate op {op!r} for {name!r}")
        value = values[name]
        out[name] = {"value": value, "floor": floor,
                     "verdict": "pass" if cmp_fn(value, floor) else "fail"}
    return out


def artifact_predicates(stage: Stage) -> tuple[tuple[str, str], ...]:
    """(predicate_kind, artifact_path) branches this stage's disk predicate ANDs
    together — primary (`stage.artifact_path`/`stage.predicate_kind`) plus every
    `also_requires` branch, in that order. `()` if the stage has no predicate
    (e.g. C, E2 — corpus-materialised, no distinct per-media artifact).

    This is THE registry helper Phase 2 consumers (reconcile_status.py,
    timeline_audit.py) resolve against actual files on disk — the registry
    itself stays filesystem-free (repo-root-relative path *templates* only).
    """
    if stage.predicate_kind is None or stage.artifact_path is None:
        return ()
    return ((stage.predicate_kind, stage.artifact_path),) + stage.also_requires


# Canonical display/iteration order for the reconcile-style artifact-predicate view
# (pre-Phase-2 reconcile_status.STAGE_ARTIFACTS / timeline_audit ledger order).
# Purely cosmetic continuity with existing tool output — stage SET membership and
# predicate VALUES are 100% derived from STAGES above; this tuple only fixes
# display order. Self-validated immediately below so it can never silently drift
# out of sync with the registry (P2's anti-drift guarantee, applied to display
# order rather than existence): adding/removing a predicate-bearing stage without
# updating this tuple raises at import time instead of silently reordering.
#
# CLUSTERS was excluded through W2/early W3a: on-spine as of Phase 3
# (bin/build_identity_clusters.py calls erf.outcomes.record() on every exit
# path — see its module docstring), but the ~250 pre-existing identity_map
# files on disk predated that write path and had ZERO matching events rows,
# which would have inflated "unrecorded" drift by ~250 and broken the
# drift-308 invariant reconcile_status.py exists to preserve. Included as of
# D-Log-16 (2026-07-12): the W3a run 3 corpus rebuild re-ran CLUSTERS and
# every identity_map file now has a corresponding events row (confirmed via
# the raw events table, not the not-yet-reconciled catalog/status/*.json
# projection — that surface only reflects stages already in this tuple, so
# checking it for CLUSTERS pre-flip is circular) (docs/OBSERVABILITY_REDESIGN.md §4).
RECONCILE_ORDER: tuple[str, ...] = (
    "A", "B0", "B1", "B2", "B3", "B4", "P", "E0", "Q0", "tags", "F0", "F1", "F2",
    "REDIARIZE", "D0", "P0", "W", "RG", "RP", "DX", "SEV2", "SEV2_ADJ", "PSTREAM",
    "T0", "V0", "CARD", "E1", "EV", "LABEL", "CLUSTERS",
)
_tracked_predicate_stages = {
    name for name, s in STAGES.items() if s.artifact_path is not None
}
assert set(RECONCILE_ORDER) == _tracked_predicate_stages, (
    "RECONCILE_ORDER out of sync with registry predicate stages: "
    f"missing={_tracked_predicate_stages - set(RECONCILE_ORDER)!r} "
    f"extra={set(RECONCILE_ORDER) - _tracked_predicate_stages!r}"
)

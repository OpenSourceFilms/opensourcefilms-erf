"""Wave 0 eval harness — golden query set, metrics, and M4 gate."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomicio import write_json
from .query import embed_query, embed_visual_query, search_lance, search_visual_lance


# ── Config & paths ────────────────────────────────────────────────────────────

def _load_config(config_path: str = "state/config.toml") -> dict:
    p = Path(config_path)
    with p.open("rb") as f:
        return tomllib.load(f)


def _state(cfg: dict) -> Path:
    return Path(cfg["paths"]["state"])


def _eval_dir(state: Path) -> Path:
    d = state / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / "runs").mkdir(exist_ok=True)
    (d / "baseline").mkdir(exist_ok=True)
    return d


def _queries_path(state: Path) -> Path:
    return state / "eval" / "queries.json"


def _exemplars_path(state: Path) -> Path:
    return state / "eval" / "exemplars.json"


def _versions_path(state: Path) -> Path:
    return state / "eval" / "stage_versions.json"


def _baseline_path(state: Path, stage: str) -> Path:
    return state / "eval" / "baseline" / f"{stage}.json"


def _run_path(state: Path, run_id: str) -> Path:
    return state / "eval" / "runs" / f"{run_id}.json"


# ── IDs & timestamps ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval_{ts}_{uuid.uuid4().hex[:6]}"


def _make_query_id(query: str, chunk_ids: list[str]) -> str:
    payload = (query + "|" + "|".join(sorted(chunk_ids))).encode()
    return "eq_" + hashlib.blake2b(payload, digest_size=4).hexdigest()


def _make_exemplar_id(review_id: str) -> str:
    return "ex_" + hashlib.blake2b(review_id.encode(), digest_size=4).hexdigest()


def _fingerprint(invocation_path: Path) -> str:
    raw = json.dumps(json.loads(invocation_path.read_text(encoding="utf-8")), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Journal ───────────────────────────────────────────────────────────────────

def _journal(journal_db: Path, run_id: str, event_type: str, payload: dict | None = None, stage_tag: str = "") -> None:
    try:
        conn = sqlite3.connect(str(journal_db))
        conn.execute("PRAGMA journal_mode = TRUNCATE")
        job_id = f"erf:eval:{stage_tag}:{run_id}" if stage_tag else f"erf:eval:{run_id}"
        conn.execute(
            # Phase 1 (D-Log-04): eval-run trace -> events_trace (§3.1).
            "INSERT INTO events_trace (at_utc, job_id, media_id, stage, event_type, run_id, payload) "
            "VALUES (?, ?, 'erf', 'eval', ?, ?, ?)",
            (_now(), job_id, event_type, run_id, json.dumps(payload or {})),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ── File I/O ──────────────────────────────────────────────────────────────────

def load_queries(state: Path) -> dict:
    p = _queries_path(state)
    if not p.exists():
        return {"schema_version": "1.0.0", "updated_at_utc": _now(), "queries": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_queries(state: Path, data: dict) -> None:
    _eval_dir(state)
    data["updated_at_utc"] = _now()
    write_json(_queries_path(state), data, indent=2)


def load_exemplars(state: Path) -> dict:
    p = _exemplars_path(state)
    if not p.exists():
        return {"schema_version": "1.0.0", "updated_at_utc": _now(), "exemplars": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_exemplars(state: Path, data: dict) -> None:
    _eval_dir(state)
    data["updated_at_utc"] = _now()
    write_json(_exemplars_path(state), data, indent=2)


def load_stage_versions(state: Path) -> dict:
    p = _versions_path(state)
    if not p.exists():
        return {"schema_version": "1.0.0", "updated_at_utc": _now(), "stages": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_stage_versions(state: Path, data: dict) -> None:
    _eval_dir(state)
    data["updated_at_utc"] = _now()
    write_json(_versions_path(state), data, indent=2)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    if not expected:
        return 1.0
    hits = set(retrieved[:k]) & set(expected)
    return len(hits) / len(expected)


def _mrr(expected: list[str], retrieved: list[str], limit: int = 20) -> float:
    expected_set = set(expected)
    for rank, item in enumerate(retrieved[:limit], start=1):
        if item in expected_set:
            return 1.0 / rank
    return 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2.0 if n % 2 == 0 else s[mid]


def compute_aggregate(query_results: list[dict], exemplar_results: list[dict]) -> dict:
    recalls = [r["recall_at_k"] for r in query_results if "recall_at_k" in r]
    mrrs = [r["reciprocal_rank"] for r in query_results if "reciprocal_rank" in r]
    ex_scores = [r["median_beat_score"] for r in exemplar_results if "median_beat_score" in r]
    return {
        "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "mean_mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        "median_exemplar_beat_score": _median(ex_scores) if ex_scores else None,
    }


def compare_to_baseline(
    candidate_agg: dict,
    baseline_agg: dict,
    candidate_qresults: list[dict],
    baseline_qresults: list[dict],
) -> tuple[str, list[str]]:
    flags: list[str] = []
    THRESHOLD = 0.20  # 20% relative drop triggers fail

    def _relative_drop(cand: float | None, base: float | None) -> float | None:
        if base is None or cand is None or base == 0:
            return None
        return (base - cand) / base  # positive = degradation

    rd_recall = _relative_drop(candidate_agg.get("mean_recall_at_k"), baseline_agg.get("mean_recall_at_k"))
    if rd_recall is not None and rd_recall > THRESHOLD:
        flags.append(f"recall_at_k_dropped_{rd_recall*100:.0f}pct")

    rd_mrr = _relative_drop(candidate_agg.get("mean_mrr"), baseline_agg.get("mean_mrr"))
    if rd_mrr is not None and rd_mrr > THRESHOLD:
        flags.append(f"mrr_dropped_{rd_mrr*100:.0f}pct")

    rd_ex = _relative_drop(
        candidate_agg.get("median_exemplar_beat_score"),
        baseline_agg.get("median_exemplar_beat_score"),
    )
    if rd_ex is not None and rd_ex > THRESHOLD:
        flags.append(f"exemplar_score_dropped_{rd_ex*100:.0f}pct")

    # Per-query collapse: baseline recall ≥ 0.5 → candidate = 0.0
    base_by_id = {r["eval_query_id"]: r for r in baseline_qresults}
    for qr in candidate_qresults:
        qid = qr.get("eval_query_id")
        base_q = base_by_id.get(qid)
        if base_q and base_q.get("recall_at_k", 0) >= 0.5 and qr.get("recall_at_k", 1) == 0.0:
            flags.append(f"query_{qid}_collapsed_to_zero")

    verdict = "fail" if flags else "pass"
    return verdict, flags


# ── Query runner ──────────────────────────────────────────────────────────────

def run_queries_against_index(
    queries: list[dict],
    state: Path,
    lance_path: Path,
    *,
    vram_free: bool = False,
    stage_tag: str = "",
) -> list[dict]:
    if stage_tag == "B1":
        queries = [q for q in queries if q.get("modality", "text") == "text"]
    elif stage_tag == "B4":
        queries = [q for q in queries if q.get("modality", "text") == "visual"]
    results = []
    for q in queries:
        modality = q.get("modality", "text")
        expected = [c["chunk_id"] for c in q.get("expected_clips", [])]
        k = q.get("min_recall_k", 5)
        search_limit = max(k * 3, 20)

        if vram_free:
            vec = q.get("stored_query_vector")
            if not vec:
                continue
        elif modality == "visual":
            try:
                vec = embed_visual_query(q["query"], state)
            except Exception:
                continue
        else:
            try:
                vec = embed_query(q["query"], state)
            except Exception:
                continue

        if modality == "visual":
            rows = search_visual_lance(vec, lance_path, top_k=search_limit)
            retrieved = [r["keyframe_id"] for r in rows]
        else:
            rows = search_lance(vec, lance_path, top_k=search_limit)
            retrieved = [r["chunk_id"] for r in rows]

        results.append({
            "eval_query_id": q["eval_query_id"],
            "query": q["query"],
            "modality": modality,
            "top_k_retrieved": retrieved[:search_limit],
            "expected_chunk_ids": expected,
            "recall_at_k": _recall_at_k(expected, retrieved, k),
            "reciprocal_rank": _mrr(expected, retrieved),
            "min_recall_k": k,
        })
    return results


def run_exemplars_against_index(
    exemplars: list[dict],
    state: Path,
    lance_path: Path,
) -> list[dict]:
    results = []
    for ex in exemplars:
        bp = ex.get("beat_plan", {})
        topic = bp.get("topic", "")
        beats = bp.get("beats", [])
        if not topic or not beats:
            continue
        try:
            vec = embed_query(topic, state)
        except Exception:
            continue
        rows = search_lance(vec, lance_path, top_k=50)
        row_scores = {r["chunk_id"]: 1.0 - float(r.get("score", 1.0)) for r in rows}
        beat_scores = [row_scores.get(b.get("chunk_id", ""), 0.0) for b in beats]
        results.append({
            "exemplar_id": ex["exemplar_id"],
            "beat_scores_retrieved": beat_scores,
            "median_beat_score": _median(beat_scores),
            "beat_count": len(beats),
        })
    return results


# ── Add query / exemplar ──────────────────────────────────────────────────────

def add_query(
    query_text: str,
    modality: str,
    expected_chunk_ids: list[str],
    min_recall_k: int = 5,
    added_by: str = "operator",
    note: str = "",
    *,
    config_path: str = "state/config.toml",
) -> str:
    cfg = _load_config(config_path)
    state = _state(cfg)
    _eval_dir(state)

    qid = _make_query_id(query_text, expected_chunk_ids)

    if modality == "visual":
        vec = embed_visual_query(query_text, state)
        model_tag = json.loads((state / "embed_visual_invocation.json").read_text())["model_tag"]
    else:
        vec = embed_query(query_text, state)
        model_tag = json.loads((state / "embed_text_invocation.json").read_text())["model_tag"]

    data = load_queries(state)
    for existing in data["queries"]:
        if existing["eval_query_id"] == qid:
            raise ValueError(f"Query already exists: {qid}")

    entry = {
        "eval_query_id": qid,
        "query": query_text,
        "modality": modality,
        "expected_clips": [{"chunk_id": c, "note": note} for c in expected_chunk_ids],
        "min_recall_k": min_recall_k,
        "stored_query_vector": vec,
        "stored_vector_model_tag": model_tag,
        "stored_vector_at_utc": _now(),
        "added_by": added_by,
        "added_at_utc": _now(),
    }
    data["queries"].append(entry)
    save_queries(state, data)
    return qid


def add_exemplar(
    review_id: str,
    name: str,
    beat_plan: dict,
    added_by: str = "operator",
    notes: str = "",
    *,
    config_path: str = "state/config.toml",
) -> str:
    cfg = _load_config(config_path)
    state = _state(cfg)
    lance_path = Path(cfg["paths"]["index"])
    _eval_dir(state)

    ex_id = _make_exemplar_id(review_id)
    data = load_exemplars(state)
    for existing in data["exemplars"]:
        if existing["exemplar_id"] == ex_id:
            raise ValueError(f"Exemplar already exists: {ex_id}")

    topic = beat_plan.get("topic", "")
    beats = beat_plan.get("beats", [])
    expected_beat_scores: list[float] = []
    if topic and beats:
        try:
            vec = embed_query(topic, state)
            rows = search_lance(vec, lance_path, top_k=50)
            row_scores = {r["chunk_id"]: 1.0 - float(r.get("score", 1.0)) for r in rows}
            expected_beat_scores = [row_scores.get(b.get("chunk_id", ""), 0.0) for b in beats]
        except Exception:
            pass

    entry = {
        "exemplar_id": ex_id,
        "name": name,
        "review_id": review_id,
        "beat_plan": beat_plan,
        "expected_beat_scores": expected_beat_scores,
        "added_by": added_by,
        "added_at_utc": _now(),
        "notes": notes,
    }
    data["exemplars"].append(entry)
    save_exemplars(state, data)
    return ex_id


# ── Core eval run ─────────────────────────────────────────────────────────────

def run_eval(
    stage_tag: str,
    run_type: str = "candidate",
    *,
    vram_free: bool = False,
    config_path: str = "state/config.toml",
) -> dict:
    cfg = _load_config(config_path)
    state = _state(cfg)
    lance_path = Path(cfg["paths"]["index"])
    journal_db = Path(cfg["paths"]["journal_db"])
    _eval_dir(state)

    run_id = _make_run_id()
    queries_data = load_queries(state)
    exemplars_data = load_exemplars(state)
    queries = queries_data.get("queries", [])
    exemplars = exemplars_data.get("exemplars", [])

    _journal(journal_db, run_id, "start", {
        "stage_tag": stage_tag,
        "model_tag": _active_model_tag(state, stage_tag),
        "run_type": run_type,
        "n_queries": len(queries),
        "n_exemplars": len(exemplars),
        "vram_free": vram_free,
    }, stage_tag=stage_tag)

    q_results = run_queries_against_index(queries, state, lance_path, vram_free=vram_free, stage_tag=stage_tag)
    ex_results = [] if (vram_free or stage_tag != "B1") else run_exemplars_against_index(exemplars, state, lance_path)
    agg = compute_aggregate(q_results, ex_results)

    run = {
        "schema_version": "1.0.0",
        "eval_run_id": run_id,
        "run_type": run_type,
        "stage_tag": stage_tag,
        "model_tag": _active_model_tag(state, stage_tag),
        "invocation_fingerprint": _active_fingerprint(state, stage_tag),
        "run_at_utc": _now(),
        "query_results": q_results,
        "exemplar_results": ex_results,
        "aggregate": agg,
        "comparison": {
            "baseline_eval_run_id": None,
            "recall_at_k_delta_relative": None,
            "mrr_delta_relative": None,
            "exemplar_score_delta_relative": None,
            "regression_flags": [],
            "verdict": None,
        },
    }
    write_json(_run_path(state, run_id), run, indent=2)
    return run


# ── Baseline capture ──────────────────────────────────────────────────────────

def capture_baseline(stage_tag: str, *, vram_free: bool = False, config_path: str = "state/config.toml") -> dict:
    cfg = _load_config(config_path)
    state = _state(cfg)
    journal_db = Path(cfg["paths"]["journal_db"])

    run = run_eval(stage_tag, run_type="baseline", vram_free=vram_free, config_path=config_path)
    run_id = run["eval_run_id"]

    run["comparison"]["verdict"] = "pass"
    write_json(_run_path(state, run_id), run, indent=2)
    write_json(_baseline_path(state, stage_tag), run, indent=2)

    sv = load_stage_versions(state)
    sv["stages"].setdefault(stage_tag, {}).update({
        "active_eval_run_id": run_id,
        "invocation_fingerprint": _active_fingerprint(state, stage_tag),
        "model_tag": _active_model_tag(state, stage_tag),
        "active_since_utc": _now(),
        "eval_verdict": "pass",
    })
    save_stage_versions(state, sv)

    _journal(journal_db, run_id, "ok", {
        "stage_tag": stage_tag,
        "run_type": "baseline",
        "eval_run_id": run_id,
        **run["aggregate"],
        "verdict": "pass",
    }, stage_tag=stage_tag)
    return run


# ── Gate flip ─────────────────────────────────────────────────────────────────

def gate_flip(
    stage_tag: str,
    new_model_tag: str,
    *,
    config_path: str = "state/config.toml",
) -> dict:
    cfg = _load_config(config_path)
    state = _state(cfg)
    journal_db = Path(cfg["paths"]["journal_db"])

    baseline_path = _baseline_path(state, stage_tag)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"No baseline for stage {stage_tag}. Run: erf-eval baseline --stage {stage_tag}"
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    candidate_run = run_eval(stage_tag, run_type="candidate", vram_free=False, config_path=config_path)
    run_id = candidate_run["eval_run_id"]

    verdict, flags = compare_to_baseline(
        candidate_run["aggregate"],
        baseline["aggregate"],
        candidate_run["query_results"],
        baseline["query_results"],
    )

    b_agg = baseline["aggregate"]
    c_agg = candidate_run["aggregate"]

    def _rd(c, b):
        if b and b > 0 and c is not None:
            return round((b - c) / b, 4)
        return None

    candidate_run["comparison"].update({
        "baseline_eval_run_id": baseline["eval_run_id"],
        "recall_at_k_delta_relative": _rd(c_agg.get("mean_recall_at_k"), b_agg.get("mean_recall_at_k")),
        "mrr_delta_relative": _rd(c_agg.get("mean_mrr"), b_agg.get("mean_mrr")),
        "exemplar_score_delta_relative": _rd(c_agg.get("median_exemplar_beat_score"), b_agg.get("median_exemplar_beat_score")),
        "regression_flags": flags,
        "verdict": verdict,
    })
    write_json(_run_path(state, run_id), candidate_run, indent=2)

    if verdict == "fail":
        _journal(journal_db, run_id, "error", {
            "eval_run_id": run_id,
            "stage_tag": stage_tag,
            "model_tag": new_model_tag,
            **c_agg,
            "regression_flags": flags,
            "verdict": "fail",
        }, stage_tag=stage_tag)
        return {"verdict": "fail", "eval_run_id": run_id, "flags": flags}

    # Pass — advance M4 pointer
    write_json(_baseline_path(state, stage_tag), candidate_run, indent=2)

    sv = load_stage_versions(state)
    sv["stages"].setdefault(stage_tag, {}).update({
        "model_tag": new_model_tag,
        "invocation_fingerprint": _active_fingerprint(state, stage_tag),
        "active_since_utc": _now(),
        "active_eval_run_id": run_id,
        "eval_verdict": "pass",
    })
    save_stage_versions(state, sv)

    _journal(journal_db, run_id, "ok", {
        "eval_run_id": run_id,
        "stage_tag": stage_tag,
        "model_tag": new_model_tag,
        **c_agg,
        "baseline_recall_at_k": b_agg.get("mean_recall_at_k"),
        "baseline_mrr": b_agg.get("mean_mrr"),
        "verdict": "pass",
    }, stage_tag=stage_tag)
    return {"verdict": "pass", "eval_run_id": run_id}


# ── M4 init ───────────────────────────────────────────────────────────────────

def init_stage_versions(*, config_path: str = "state/config.toml") -> None:
    cfg = _load_config(config_path)
    state = _state(cfg)
    _eval_dir(state)

    if _versions_path(state).exists():
        return  # already initialized

    stages: dict[str, Any] = {}
    for stage, inv_name in [("B1", "embed_text_invocation.json"), ("B4", "embed_visual_invocation.json")]:
        inv_path = state / inv_name
        if inv_path.exists():
            inv = json.loads(inv_path.read_text())
            stages[stage] = {
                "model_tag": inv.get("model_tag", "unknown"),
                "invocation_fingerprint": _fingerprint(inv_path),
                "active_since_utc": _now(),
                "active_eval_run_id": None,
                "eval_verdict": None,
            }

    sv = {"schema_version": "1.0.0", "updated_at_utc": _now(), "stages": stages}
    save_stage_versions(state, sv)


# ── Maintenance ───────────────────────────────────────────────────────────────

def prune_runs(stage_tag: str | None = None, keep: int = 10, *, config_path: str = "state/config.toml") -> int:
    from datetime import timedelta
    cfg = _load_config(config_path)
    state = _state(cfg)
    runs_dir = state / "eval" / "runs"
    sv = load_stage_versions(state)

    protected_ids: set[str] = set()
    for st, info in sv.get("stages", {}).items():
        if stage_tag and st != stage_tag:
            continue
        if info.get("active_eval_run_id"):
            protected_ids.add(info["active_eval_run_id"])
    for st in (sv.get("stages") or {}):
        if stage_tag and st != stage_tag:
            continue
        bp = _baseline_path(state, st)
        if bp.exists():
            try:
                b = json.loads(bp.read_text())
                protected_ids.add(b["eval_run_id"])
            except Exception:
                pass

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=30)
    runs = sorted(runs_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    deleted = 0
    stage_runs: dict[str, list[Path]] = {}
    for p in runs:
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        st = r.get("stage_tag", "unknown")
        if stage_tag and st != stage_tag:
            continue
        stage_runs.setdefault(st, []).append(p)

    for st, plist in stage_runs.items():
        if len(plist) <= keep:
            continue
        for p in plist[: len(plist) - keep]:
            try:
                r = json.loads(p.read_text())
                rid = r.get("eval_run_id", "")
                if rid in protected_ids:
                    continue
                if r.get("comparison", {}).get("verdict") == "fail":
                    run_dt = datetime.fromisoformat(r.get("run_at_utc", "2000-01-01T00:00:00+00:00"))
                    if run_dt > cutoff_dt:
                        continue
                p.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted


def status(*, config_path: str = "state/config.toml") -> dict:
    cfg = _load_config(config_path)
    state = _state(cfg)
    queries = load_queries(state).get("queries", [])
    exemplars = load_exemplars(state).get("exemplars", [])
    sv = load_stage_versions(state)
    baselines: dict[str, str | None] = {}
    for st in sv.get("stages", {}):
        bp = _baseline_path(state, st)
        if bp.exists():
            try:
                b = json.loads(bp.read_text())
                baselines[st] = b.get("eval_run_id")
            except Exception:
                baselines[st] = None
        else:
            baselines[st] = None
    return {
        "stage_versions": sv.get("stages", {}),
        "n_queries": len(queries),
        "n_exemplars": len(exemplars),
        "baselines": baselines,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _active_model_tag(state: Path, stage_tag: str) -> str:
    inv_map = {"B1": "embed_text_invocation.json", "B4": "embed_visual_invocation.json"}
    inv_name = inv_map.get(stage_tag, "embed_text_invocation.json")
    p = state / inv_name
    if p.exists():
        return json.loads(p.read_text()).get("model_tag", "unknown")
    return "unknown"


def _active_fingerprint(state: Path, stage_tag: str) -> str | None:
    inv_map = {"B1": "embed_text_invocation.json", "B4": "embed_visual_invocation.json"}
    inv_name = inv_map.get(stage_tag, "embed_text_invocation.json")
    p = state / inv_name
    return _fingerprint(p) if p.exists() else None

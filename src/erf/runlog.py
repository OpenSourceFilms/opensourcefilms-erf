"""Central run-logging to state/journal.db for the redesigned creative pipeline.

Every new stage (transcript, evidence package, grounded editor, render) emits
explicit start/ok/error/skip events — so every part is aware of what every other
part has done, attempted, or skipped (and why).

Phase 1 (D-Log-04): `log_event` no longer writes the events/status_projection
tables directly. It funnels into the single outcome writer, `erf.outcomes`:
"start" -> `outcomes.trace()` (events_trace, best-effort, never raises — same
contract as before); "ok"/"error"/"skip" -> `outcomes.record()` (the ONE outcome
surface — P4: no silent swallow, a DB write failure now RAISES; an unloggable
stage outcome is a failed stage, not a silently-green one). This is a real
behavior change for terminal calls: callers relying on log_event never raising
must now handle exceptions from ok/error/skip calls (stage_run already does, via
its own except-reraise). `heartbeat` stays separate (liveness != outcome).

Usage:
    from erf.runlog import stage_run, log_event
    with stage_run("T0", media_id, {"n_words": 285}) as run_id:
        ...                      # emits start, then ok (or error on exception)
    log_event("T0", media_id, "skip", {"reason": "output exists"})
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path("/workspace/erf/state/journal.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(stage: str, media_id: str, event_type: str,
              payload: dict | None = None, run_id: str | None = None,
              db: Path = DEFAULT_DB) -> str:
    """Append one event. "start" is trace-only (best-effort, never raises);
    "ok"/"error"/"skip" are outcome dispositions routed through
    erf.outcomes.record() (status_projection updated inline; DB failure raises).
    """
    from . import outcomes
    if event_type == "start":
        return outcomes.trace(media_id, stage, "start", payload, run_id=run_id, db=db)
    if event_type in ("ok", "error", "skip"):
        payload = payload or {}
        reason = None
        if event_type in ("error", "skip"):
            reason = payload.get("reason") or payload.get("error") or event_type
        # Generate the run_id here (not inside record()) so the id we return is
        # the same one actually persisted to the row — record() would otherwise
        # mint its own on a None run_id and we'd hand back a different one.
        run_id = run_id or f"{stage}-{outcomes.uuid.uuid4().hex[:8]}"
        outcomes.record(media_id, stage, event_type, reason=reason, metrics=payload,
                         run_id=run_id, db=db)
        return run_id
    # Unrecognised event_type (none in current callers): fall back to trace-only
    # so an unexpected value can never crash a stage the way record() now would.
    return outcomes.trace(media_id, stage, event_type, payload, run_id=run_id, db=db)


@contextmanager
def stage_run(stage: str, media_id: str, payload: dict | None = None,
              db: Path = DEFAULT_DB):
    """Context manager: emit start, then ok on success / error on exception."""
    run_id = log_event(stage, media_id, "start", payload, db=db)
    try:
        yield run_id
        log_event(stage, media_id, "ok", payload, run_id=run_id, db=db)
    except Exception as e:
        log_event(stage, media_id, "error", {"error": str(e)[:1000]}, run_id=run_id, db=db)
        raise


_HEARTBEAT_DIR = Path("/workspace/erf/logs.all/_latest/heartbeat")


def heartbeat(component: str, extra: dict | None = None) -> None:
    """Liveness signal for a long-running/continuous loop. One file per component
    (no merge races): logs.all/_latest/heartbeat/{component}.json. Never raises."""
    try:
        _HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"component": component, "at_utc": _now(), **(extra or {})}
        path = _HEARTBEAT_DIR / f"{component}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass

"""StageManifest type — stage metadata container (Phase 0 type only; no stage logic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class StageManifest:
    """Records what a stage invocation produced and its outcome.

    Written by every stage before exit (Invariant 3: explicit outcomes only).
    The control plane reaps these from workers' done.jsonl into status/{media_id}.json.
    """
    media_id: str
    stage: str                          # stage letter e.g. "A", "B0", "B1"
    run_id: str                         # unique per invocation
    ok: bool                            # True = success
    at_utc: str                         # ISO 8601 UTC datetime string
    tool_version: Optional[str] = None
    counts: Optional[dict[str, int]] = None  # e.g. {"words": 1842, "chunks": 63}
    error: Optional[str] = None
    outputs: list[str] = field(default_factory=list)  # catalog paths written

    @classmethod
    def now(cls) -> str:
        return datetime.now(timezone.utc).isoformat()

    def to_status_entry(self) -> dict[str, Any]:
        """Serialise to the shape used in catalog/status/{media_id}.json stages map."""
        return {
            "at_utc": self.at_utc,
            "ok": self.ok,
            "run_id": self.run_id,
            "tool_version": self.tool_version,
            "counts": self.counts,
            "error": self.error,
        }

    def to_done_entry(self) -> dict[str, Any]:
        """Serialise to the shape appended to state/workers/{pod_id}/done.jsonl."""
        return {
            "media_id": self.media_id,
            "stage": self.stage,
            "run_id": self.run_id,
            "ok": self.ok,
            "at_utc": self.at_utc,
            "tool_version": self.tool_version,
            "counts": self.counts,
            "error": self.error,
            "outputs": self.outputs,
        }

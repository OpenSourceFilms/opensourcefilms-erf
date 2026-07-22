"""Path seam (D-Log-35, executes D-Log-34 §2): env-overridable catalog root
and diarizer hypothesis source.

Unset env ⇒ behavior is byte-identical to the previous hardcoded constants:
catalog_root() == <repo>/catalog, diar_source() == "community1".

ERF_CATALOG_ROOT — absolute path to an alternate catalog tree (shadow bake-off
arms live under state/bakeoff/...). Read at call time; chain scripts bind it
once at import, so export before launching the script.

ERF_DIAR_SOURCE — basename prefix for diar-hypothesis files under
diar_hypotheses/{mid}/ ("community1" | "diarizen"). Consumed by
build_speech_events_v2._diar_paths.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def catalog_root() -> Path:
    v = os.environ.get("ERF_CATALOG_ROOT")
    return Path(v) if v else ROOT / "catalog"


def diar_source() -> str:
    return os.environ.get("ERF_DIAR_SOURCE", "community1")

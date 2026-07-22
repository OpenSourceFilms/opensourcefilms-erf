"""Shared fixtures for tests that exercise erf.render.render() against real
corpus fixture media (reads: asr/identity_map/visual_grounding/etc.) but must
not leak writes into production catalog/. render()'s only catalog-relative
write is the audio QC sidecar (render.py: `catalog / "audio_qc" / f"{export_id}.json"`)
— everything else under `catalog` is read-only for render()."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REAL_CATALOG = ROOT / "catalog"


@pytest.fixture
def render_catalog(tmp_path):
    """Real catalog/ subdirs symlinked read-only, except audio_qc which gets a
    real tmp directory so render()'s audio QC write never touches production."""
    cat = tmp_path / "catalog"
    cat.mkdir()
    for entry in REAL_CATALOG.iterdir():
        if entry.name == "audio_qc":
            continue
        (cat / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (cat / "audio_qc").mkdir()
    return cat

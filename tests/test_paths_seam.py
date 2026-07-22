"""D-Log-35 seam: ERF_CATALOG_ROOT / ERF_DIAR_SOURCE / ERF_JOURNAL_DB overrides.

Unset env must be byte-identical to the old hardcoded constants; set env must
redirect. The env-set cases for import-time-bound modules (erf.outcomes) run in
a subprocess so this test never mutates the already-imported production module.
bin/ is not a package — loaded via importlib.util, matching
tests/test_speech_events_v2_demoted.py.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"

sys.path.insert(0, str(ROOT / "src"))
from erf.paths import catalog_root, diar_source  # noqa: E402


def _load_bin_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_catalog_root_default_is_production(monkeypatch):
    monkeypatch.delenv("ERF_CATALOG_ROOT", raising=False)
    assert catalog_root() == ROOT / "catalog"


def test_catalog_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ERF_CATALOG_ROOT", str(tmp_path))
    assert catalog_root() == tmp_path


def test_diar_source_default_and_override(monkeypatch):
    monkeypatch.delenv("ERF_DIAR_SOURCE", raising=False)
    assert diar_source() == "community1"
    monkeypatch.setenv("ERF_DIAR_SOURCE", "diarizen")
    assert diar_source() == "diarizen"


def test_sev2_diar_paths_honor_source(monkeypatch):
    sev2 = _load_bin_module("build_speech_events_v2_seamtest",
                            "build_speech_events_v2.py")
    monkeypatch.delenv("ERF_DIAR_SOURCE", raising=False)
    excl, base = sev2._diar_paths("m_deadbeef0000")
    assert excl.name == "community1_excl.jsonl"
    assert base.name.startswith("community1")
    monkeypatch.setenv("ERF_DIAR_SOURCE", "diarizen")
    excl, base = sev2._diar_paths("m_deadbeef0000")
    assert excl.name == "diarizen_excl.jsonl"
    assert base.name.startswith("diarizen")


def test_outcomes_env_redirect_subprocess(tmp_path):
    """erf.outcomes binds CAT/DB at import — verify env redirection in a fresh
    interpreter, and the unset default in another, without touching this one."""
    code = ("import sys; sys.path.insert(0, %r); import erf.outcomes as o; "
            "print(o.CAT); print(o.DB)" % str(ROOT / "src"))
    env_set = {**os.environ,
               "ERF_CATALOG_ROOT": str(tmp_path / "cat"),
               "ERF_JOURNAL_DB": str(tmp_path / "j.db")}
    out = subprocess.run([sys.executable, "-c", code], env=env_set,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == [str(tmp_path / "cat"), str(tmp_path / "j.db")]

    env_unset = {k: v for k, v in os.environ.items()
                 if k not in ("ERF_CATALOG_ROOT", "ERF_JOURNAL_DB")}
    out = subprocess.run([sys.executable, "-c", code], env=env_unset,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == [str(ROOT / "catalog"), str(ROOT / "state" / "journal.db")]

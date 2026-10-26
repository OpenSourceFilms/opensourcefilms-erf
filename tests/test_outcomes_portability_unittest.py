"""Isolated stdlib checks for ERF outcome/journal root portability.

No production state is read or written: subprocesses use a disposable checkout
and a temporary catalog/database. Run with Python 3.11+.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _probe(code: str, env: dict[str, str]) -> dict:
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env={**env, "PYTHONPATH": str(SRC)}, cwd=ROOT,
        capture_output=True, text=True, check=True, timeout=20,
    )
    return json.loads(result.stdout)


class OutcomesPortabilityTests(unittest.TestCase):
    def test_defaults_point_inside_cloned_repo(self):
        env = os.environ.copy()
        env.pop("ERF_CATALOG_ROOT", None)
        env.pop("ERF_JOURNAL_DB", None)
        got = _probe(
            "import json; from erf import outcomes as o; "
            "print(json.dumps({'root': str(o.ROOT), 'cat': str(o.CAT), 'db': str(o.DB)}))",
            env,
        )
        self.assertEqual(got, {
            "root": str(ROOT),
            "cat": str(ROOT / "catalog"),
            "db": str(ROOT / "state" / "journal.db"),
        })

    def test_overrides_write_only_to_disposable_paths(self):
        with tempfile.TemporaryDirectory(prefix="erf_outcomes_test_") as tmp:
            base = Path(tmp)
            env = {**os.environ,
                   "ERF_CATALOG_ROOT": str(base / "catalog"),
                   "ERF_JOURNAL_DB": str(base / "journal.db")}
            code = """
import json, sqlite3
from erf import outcomes as o
with sqlite3.connect(o.DB) as conn:
    conn.executescript(o._NEW_EVENTS_DDL)
o.record('m_synthetic_fixture', 'A', 'ok', db=o.DB)
status = o.STATUS_DIR / 'm_synthetic_fixture.json'
with sqlite3.connect(o.DB) as conn:
    count = conn.execute("select count(*) from events").fetchone()[0]
print(json.dumps({'cat': str(o.CAT), 'db': str(o.DB),
                  'status_exists': status.is_file(), 'event_count': count,
                  'status': json.loads(status.read_text())['stages']['A']['status']}))
"""
            got = _probe(code, env)
            self.assertEqual(got, {
                "cat": str(base / "catalog"),
                "db": str(base / "journal.db"),
                "status_exists": True,
                "event_count": 1,
                "status": "ok",
            })


    def test_malformed_media_ids_cannot_escape_label_directory_or_write_db(self):
        with tempfile.TemporaryDirectory(prefix="erf_media_id_test_") as tmp:
            base = Path(tmp)
            env = {**os.environ,
                   "ERF_CATALOG_ROOT": str(base / "catalog"),
                   "ERF_JOURNAL_DB": str(base / "journal.db")}
            code = """
import json, sqlite3
from erf import outcomes as o
o.migrate_phase1()
invalid = ["../escape", "../../other", "/tmp/escape", r"nested\\name",
           "nested/name", "", ".", "..", "line\\nbreak", None, 17]
tested = 0
for media in invalid:
    for action in (
        lambda: o.record(media, "A", "ok", db=o.DB),
        lambda: o.set_flag(media, "synthetic", True),
        lambda: o.for_media(media),
    ):
        try:
            action()
        except ValueError:
            tested += 1
        else:
            raise AssertionError("unsafe status media ID accepted: " + repr(media))
with sqlite3.connect(o.DB) as c:
    event_count = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
outside = (o.CAT / "escape.json").exists() or (o.CAT.parent / "other.json").exists()
o.record("m_111111111111", "A", "ok", db=o.DB)
o.set_flag("m_111111111111", "demo", True)
print(json.dumps({
    "tested": tested,
    "unsafe_event_count": event_count,
    "outside_written": outside,
    "valid_status": o.status_of("m_111111111111", "A"),
    "valid_flag": o.flag("m_111111111111", "demo")
}))
"""
            got = _probe(code, env)
            self.assertEqual(got, {
                "tested": 33,
                "unsafe_event_count": 0,
                "outside_written": False,
                "valid_status": "ok",
                "valid_flag": True,
            })

    def test_symlinked_status_file_cannot_redirect_reads_or_writes(self):
        with tempfile.TemporaryDirectory(prefix="erf_symlink_test_") as tmp:
            base = Path(tmp)
            env = {**os.environ,
                   "ERF_CATALOG_ROOT": str(base / "catalog"),
                   "ERF_JOURNAL_DB": str(base / "journal.db")}
            code = """
import json, sqlite3
from erf import outcomes as o
o.migrate_phase1()
o.STATUS_DIR.mkdir(parents=True, exist_ok=True)
outside = o.CAT.parent / "external.json"
outside.write_text('{"secret":"unrelated synthetic file"}', encoding="utf-8")
destination = o.STATUS_DIR / "m_symlink_demo.json"
destination.symlink_to(outside)
blocked = 0
for action in (
    lambda: o.record("m_symlink_demo", "A", "ok", db=o.DB),
    lambda: o.set_flag("m_symlink_demo", "demo", True),
    lambda: o.for_media("m_symlink_demo"),
):
    try:
        action()
    except ValueError:
        blocked += 1
    else:
        raise AssertionError("a symlinked media label was followed")
with sqlite3.connect(o.DB) as c:
    events = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
print(json.dumps({
    "blocked": blocked,
    "events": events,
    "outside_unchanged": outside.read_text() == '{"secret":"unrelated synthetic file"}',
    "link_preserved": destination.is_symlink()
}))
"""
            got = _probe(code, env)
            self.assertEqual(got, {
                "blocked": 3,
                "events": 0,
                "outside_unchanged": True,
                "link_preserved": True,
            })


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

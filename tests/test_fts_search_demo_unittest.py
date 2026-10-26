"""CPU-only acceptance for the publicly documented FTS search demonstration."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import fts_search_demo as demo  # noqa: E402


class SyntheticFtsDemoTests(unittest.TestCase):
    def test_real_fts_index_returns_two_timestamped_steam_hits(self):
        result = demo.run_demo("steam")
        self.assertTrue(result["demo_only"])
        self.assertEqual(result["fps"], 24)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(
            {r["media_id"] for r in result["results"]},
            {"m_111111111111", "m_222222222222"},
        )
        for row in result["results"]:
            self.assertLess(row["in_ms"], row["out_ms"])
            self.assertRegex(row["in_tc"], r"^\d\d:\d\d:\d\d:\d\d$")
            self.assertIn("synthetic SVG", row["thumbnail"])

    def test_other_term_returns_different_media_without_padding(self):
        result = demo.run_demo("water")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["media_id"], "m_333333333333")
        self.assertEqual(result["results"][0]["in_tc"], "00:00:32:00")
        self.assertEqual(demo.run_demo("unicorn")["results"], [])

    def test_search_is_reproducible(self):
        self.assertEqual(demo.run_demo("steam"), demo.run_demo("steam"))

    def test_demo_rejects_operator_and_phrase_injection(self):
        for value in ('steam" OR "water', "steam; DROP TABLE chunks_fts", "", "  ", "*"):
            with self.subTest(query=value):
                with self.assertRaises(ValueError):
                    demo.run_demo(value)

    def test_html_is_standalone_and_labels_thumbnails(self):
        text = demo.render_html(demo.run_demo("steam"))
        self.assertIn("SYNTHETIC ILLUSTRATION", text)
        self.assertIn("<svg", text)
        self.assertIn("not extracted footage", text)
        self.assertNotIn("<script", text)
        self.assertNotIn("<video", text)
        self.assertNotIn("https://", text)
        self.assertNotIn("/workspace", text)

    def test_html_escapes_attacker_controlled_query_and_labels(self):
        payload = demo.run_demo("steam")
        payload["query"] = '<script>alert("test")</script>'
        payload["results"][0]["media_label"] = '<img src=x onerror=alert(1)>'
        payload["results"][0]["snippet"] = '<iframe src=x></iframe>'
        html = demo.render_html(payload)
        self.assertNotIn("<script", html)
        self.assertNotIn("<iframe", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;iframe", html)

    def test_opt_in_export_never_overwrites_or_writes_to_checkout(self):
        with tempfile.TemporaryDirectory() as scratch:
            p = Path(scratch) / "synthetic.html"
            payload = demo.run_demo()
            demo.write_preview(p, payload)
            self.assertTrue(p.is_file())
            self.assertIn("<svg", p.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                demo.write_preview(p, payload)
        with self.assertRaises(ValueError):
            demo.write_preview(ROOT / "never-write-this-demo.html", demo.run_demo())

    def test_cli_reports_identical_json_without_output_writes(self):
        cmd = [sys.executable, "-B", str(ROOT / "examples" / "fts_search_demo.py")]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["query"], "steam")
        self.assertEqual(len(result["results"]), 2)
        self.assertNotIn("erf_public_fts_demo_", proc.stdout)


if __name__ == "__main__":
    unittest.main()

"""Offline FTS5 phrase-literal regression using an invented SQLite catalog.

The test exercises the real public erf.fts.search_exact helper; no production
catalog, media, remote service, GPU, model or account access is required.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from erf.fts import search_chunks, search_exact  # noqa: E402


class ExactPhraseIsolationTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="erf_fts_exact_synthetic_")
        self.addCleanup(scratch.cleanup)
        self.db = Path(scratch.name) / "fts.db"
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "chunk_id UNINDEXED, media_id UNINDEXED, "
                "start_ms UNINDEXED, end_ms UNINDEXED, "
                "text, speaker_label UNINDEXED)"
            )
            for idx, text in enumerate((
                "The valve releases steam.",
                "The tank contains water.",
                'A literal note says steam" OR "water on the label.',
                "The steam valve failed.",
                "A water pump is running.",
            ), 1):
                conn.execute(
                    "INSERT INTO chunks_fts "
                    "(chunk_id,media_id,start_ms,end_ms,text,speaker_label) "
                    "VALUES (?,?,?,?,?,?)",
                    (str(idx), f"m_synthetic_{idx}", idx * 1000,
                     idx * 1000 + 500, text, "Synthetic"),
                )

    def test_embedded_quotes_cannot_expand_exact_phrase_to_boolean_or(self):
        # This is intentionally a valid FTS Boolean expression when
        # incorrectly spliced into the old f'"{phrase}"' construction.
        phrase = 'steam" OR "water'
        broad = search_chunks(self.db, '"steam" OR "water"')
        self.assertEqual({row["chunk_id"] for row in broad}, {"1", "2", "3", "4", "5"})

        exact = search_exact(self.db, phrase)
        self.assertEqual([row["chunk_id"] for row in exact], ["3"])
        self.assertEqual(exact[0]["text"],
                         'A literal note says steam" OR "water on the label.')

    def test_unbalanced_quotes_do_not_throw_fts_syntax_error(self):
        # SQLite's FTS5 tokenizer treats quote punctuation as separators,
        # but it must never treat an unpaired quote as query syntax.
        results = search_exact(self.db, 'steam" valve')
        self.assertEqual([row["chunk_id"] for row in results], ["4"])

    def test_normal_multiword_phrase_and_unmatched_phrase(self):
        self.assertEqual([r["chunk_id"] for r in search_exact(self.db, "steam valve")],
                         ["4"])
        self.assertEqual(search_exact(self.db, "water valve"), [])

    def test_empty_and_nontext_phrase_refused_before_database_search(self):
        for phrase in ("", "   ", None, 5, [], False):
            with self.subTest(phrase=phrase), self.assertRaisesRegex(
                    ValueError, "nonempty string"):
                search_exact(self.db, phrase)

    def test_queries_do_not_modify_synthetic_index(self):
        with sqlite3.connect(self.db) as conn:
            before = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        search_exact(self.db, 'steam" OR "water')
        with sqlite3.connect(self.db) as conn:
            after = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

"""Offline ERF FTS media attribution and pre-mutation validation.

All catalog and FTS5 data is synthetic and stored under disposable paths.
No film footage, embeddings, production catalogs, GPUs or network access.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from erf import fts  # noqa: E402

A = "m_111111111111"
B = "m_222222222222"


def chunk(mid: str, value: str) -> dict:
    return {
        "chunk_id": f"{mid}:c:100-200",
        "media_id": mid,
        "start_ms": 100,
        "end_ms": 200,
        "text": value,
        "speaker_label": "Synthetic",
    }


def word(mid: str, value: str) -> dict:
    return {
        "word_id": f"{mid}:w:000001",
        "media_id": mid,
        "start_ms": 100,
        "end_ms": 200,
        "text": value,
    }


def ocr(mid: str, value: str) -> dict:
    return {
        "ocr_id": f"{mid}:ocr:01",
        "media_id": mid,
        "keyframe_id": f"{mid}:kf:100",
        "text": value,
    }


class FtsAttributionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="erf_fts_media_scope_")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.catalog = self.root / "catalog"
        self.db = self.root / "fts.db"
        for kind in ("chunks", "asr", "ocr"):
            (self.catalog / kind).mkdir(parents=True)
        with sqlite3.connect(self.db) as c:
            c.executescript(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "chunk_id UNINDEXED, media_id UNINDEXED, start_ms UNINDEXED, "
                "end_ms UNINDEXED, text, speaker_label UNINDEXED);"
                "CREATE VIRTUAL TABLE words_fts USING fts5("
                "word_id UNINDEXED, media_id UNINDEXED, start_ms UNINDEXED, "
                "end_ms UNINDEXED, text);"
                "CREATE VIRTUAL TABLE ocr_fts USING fts5("
                "ocr_id UNINDEXED, media_id UNINDEXED, keyframe_id UNINDEXED, text);"
            )
        for mid in (A, B):
            self.write("chunks", mid, [chunk(mid, f"original {mid}")])
            self.write("asr", mid, [word(mid, f"word {mid}")])
            self.write("ocr", mid, [ocr(mid, f"label {mid}")])
            self.assertEqual(fts.index_media(mid, self.catalog, self.db),
                             {"chunks": 1, "words": 1})
            self.assertEqual(fts.index_ocr(mid, self.catalog, self.db),
                             {"ocr": 1})

    def write(self, kind, mid, rows):
        suffix = ".words.jsonl" if kind == "asr" else ".jsonl"
        name = f"{mid}{suffix}"
        path = self.catalog / kind / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8")

    def snapshot(self):
        with sqlite3.connect(self.db) as conn:
            return tuple(
                tuple(tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid"))
                for table in ("chunks_fts", "words_fts", "ocr_fts")
            )

    def test_cross_media_chunk_cannot_poison_other_media_or_delete_previous(self):
        before = self.snapshot()
        self.write("chunks", A, [chunk(B, "FALSE ATTRIBUTION TO B")])
        with self.assertRaisesRegex(ValueError, "chunk record 0"):
            fts.index_media(A, self.catalog, self.db)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(fts.search_chunks(self.db, "FALSE"), [])

    def test_bad_word_row_cannot_partially_reindex_valid_chunks(self):
        before = self.snapshot()
        self.write("chunks", A, [chunk(A, "new text must never be indexed")])
        self.write("asr", A, [word(B, "wrong voice")])
        with self.assertRaisesRegex(ValueError, "word record 0"):
            fts.index_media(A, self.catalog, self.db)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(fts.search_chunks(self.db, "new"), [])

    def test_mixed_correct_and_foreign_ocr_rejected_before_ocr_delete(self):
        before = self.snapshot()
        self.write("ocr", A, [ocr(A, "valid row"), ocr(B, "wrong media OCR")])
        with self.assertRaisesRegex(ValueError, "OCR record 1"):
            fts.index_ocr(A, self.catalog, self.db)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(fts.search_ocr(self.db, "wrong"), [])

    def test_nondict_record_rejected_without_mutation(self):
        before = self.snapshot()
        self.write("chunks", A, [chunk(A, "valid"), ["not a chunk"]])
        with self.assertRaisesRegex(ValueError, "chunk record 1"):
            fts.index_media(A, self.catalog, self.db)
        self.assertEqual(self.snapshot(), before)
        self.write("ocr", A, ["not an OCR mapping"])
        with self.assertRaisesRegex(ValueError, "OCR record 0"):
            fts.index_ocr(A, self.catalog, self.db)
        self.assertEqual(self.snapshot(), before)

    def test_unsafe_media_ids_fail_before_catalog_access_or_database_write(self):
        before = self.snapshot()
        malformed = (None, 3, True, "", ".", "..", "../other", "nested/path",
                     "nested\\path", "space name", "bad\nnewline", "m_" + "X" * 200)
        for mid in malformed:
            with self.subTest(mid=repr(mid)):
                with patch.object(fts, "read_jsonl", side_effect=AssertionError("catalog accessed")):
                    with self.assertRaisesRegex(ValueError, "media_id"):
                        fts.index_media(mid, self.catalog, self.db)
                    with self.assertRaisesRegex(ValueError, "media_id"):
                        fts.index_ocr(mid, self.catalog, self.db)
                self.assertEqual(self.snapshot(), before)

    def test_valid_reindex_removes_stale_rows_only_for_target_media(self):
        baseline = self.snapshot()
        self.write("chunks", A, [chunk(A, "fresh distinct transcript")])
        self.write("asr", A, [word(A, "freshly")])
        self.write("ocr", A, [ocr(A, "fresh engraved label")])
        self.assertEqual(fts.index_media(A, self.catalog, self.db),
                         {"chunks": 1, "words": 1})
        self.assertEqual(fts.index_ocr(A, self.catalog, self.db), {"ocr": 1})
        self.assertEqual([r["media_id"] for r in fts.search_chunks(self.db, "fresh")], [A])
        self.assertEqual([r["media_id"] for r in fts.search_ocr(self.db, "engraved")], [A])
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM chunks_fts WHERE media_id=?", (B,))),
                (baseline[0][1],)
            )
            self.assertEqual(
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM words_fts WHERE media_id=?", (B,))),
                (baseline[1][1],)
            )
            self.assertEqual(
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM ocr_fts WHERE media_id=?", (B,))),
                (baseline[2][1],)
            )
        after = self.snapshot()
        self.assertEqual(fts.index_media(A, self.catalog, self.db),
                         {"chunks": 1, "words": 1})
        self.assertEqual(fts.index_ocr(A, self.catalog, self.db), {"ocr": 1})
        # FTS5 index rowid allocation may change on idempotent reindex;
        # payload rows and all cross-media results must remain equivalent.
        self.assertEqual(self.snapshot(), after)


if __name__ == "__main__":
    unittest.main()

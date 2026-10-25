# Try the explicit outcome journal without models or production data

ERF records a stage outcome **as a fact**, rather than inferring success from
the existence or absence of a media file. The following example runs in a
temporary catalogue and journal database using Python 3.11+ standard library.
It does not need a GPU, model checkpoints, network access or a media collection.

```bash
git clone https://github.com/OpenSourceFilms/opensourcefilms-erf.git
cd opensourcefilms-erf
python3 examples/explicit_outcomes_demo.py
```

The program creates a fresh SQLite `events` table and a derived
`status_projection` table, writes two synthetic outcomes and a media flag,
checks that `empty` is a completed stage while an unattempted stage is
unknown, and prints a short deterministic JSON summary. All state is
created inside `tempfile.TemporaryDirectory()` and removed on exit.

**Important:** `ERF_CATALOG_ROOT` and `ERF_JOURNAL_DB` are both set before
importing `erf.outcomes`. That module binds its destination paths at import
time. Setting only the catalogue override would leave the journal pointing
to its ordinary location. Do not use a production path for demonstrations.

For a fast regression check against a clean clone:

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests \
  -p 'test_outcomes_portability_unittest.py' -v
python3 examples/explicit_outcomes_demo.py
```

This demonstrates the **storage contract only**. It does not qualify ASR,
diarization, face recognition, editor retrieval, model licensing,
media processing, or the many environment-specific `bin/` scripts.

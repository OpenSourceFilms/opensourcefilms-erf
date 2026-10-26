# A real lexical ERF search over completely synthetic footage records

This is a **runnable, CPU-only demonstration of the actual repository
implementation** in src/erf/fts.py: index_media() and search_chunks().
It indexes three fabricated transcript chunks, executes a real SQLite FTS5
query, and returns results with media IDs, in/out timestamps and snippets.
The displayed timecodes are computed by ERF's ms_to_tc() helper.

It is intentionally **not** a sample of real production footage, Qwen
semantic retrieval, image matching, diarization, identity recognition,
LLM query interpretation or the private live search interface.

## Try it on a clean clone

Python 3.11+ with SQLite compiled with FTS5 is sufficient. No pip install,
GPU, models, internet access, B2 connection or film footage is required.

~~~bash
git clone https://github.com/OpenSourceFilms/opensourcefilms-erf.git
cd opensourcefilms-erf
python3 -B examples/fts_search_demo.py
python3 -B examples/fts_search_demo.py --query water
python3 -B -m unittest discover -s tests -p 'test_fts_search_demo_unittest.py' -v
~~~

The first command issues the lexical query **steam**, which matches **two**
fabricated pieces of footage. One result includes:

| Field | Example returned value |
|---|---|
| query | steam |
| media_id | m_111111111111 |
| in_ms | 14000 |
| out_ms | 19500 |
| in_tc | 00:00:14:00 |
| speaker | Operator |
| snippet | We release steam through the valve and check the pressure. |
| thumbnail | Synthetic illustration, not a film frame |

The example prints JSON to standard output. The second command looks up
**water** and returns a different synthetic record. An unmatched query returns
an empty results array rather than manufacturing a hit.

## See timestamped results and illustrated thumbnails

The HTML preview is generated from **the executed FTS query results**, not
a hard-coded mock search response. Its inline SVG diagrams represent
fabricated workshop scenes. They are **not** extracted keyframes or genuine
video clips, and there are no external assets or media fetches.

~~~bash
# Use a fresh path. Existing targets will not be overwritten.
python3 -B examples/fts_search_demo.py --html /tmp/erf-fts-synthetic.html
# Open /tmp/erf-fts-synthetic.html in your browser.
~~~

The HTML is a standalone file, works without a server, and has no
network-dependent resources. The script writes nothing into the repository
or production catalog. The scratch SQLite database and JSONL records
are created within a temporary directory and discarded at the end of each
search. HTML output is **opt-in** and must be a new path outside the checkout.

Queries accepted by this teaching example are deliberately restricted to
1-5 simple words; it is not a general-purpose public search endpoint.

## Media attribution integrity of the indexer

The real `erf.fts.index_media()` and `index_ocr()` functions now refuse
a catalog JSONL record whose `media_id` disagrees with the media being
reindexed. The check runs **before** the SQLite `DELETE`/insert
transaction. Invalid mixed-media chunks, words or OCR rows therefore
cannot silently cross-contaminate another recording's FTS results or
delete the target's previous search entries.

The requested `media_id` is also validated as one safe ASCII basename
before constructing a catalog path, preventing a path-like argument
from reading arbitrary JSONL files adjacent to the catalog. This
allows canonical `m_<12hex>` IDs and compatible simple legacy labels;
it does not infer that a record is correctly attributed merely because
its declared ID agrees. Upstream catalogue provenance and speaker
identity still need separate evidence checks.

Six new disposable FTS5 tests cover malicious/malformed media keys,
mixed-media transcript and OCR records, zero index mutations on
rejection, and successful idempotent reindexing that preserves
unrelated media. Run the tests without pip dependencies:

~~~bash
python3 -B -m unittest discover -s tests -p 'test_fts_media_attribution_unittest.py' -v
~~~

## What was actually verified

The repository regression suite exercises the real indexing/search code,
checks hit IDs and timestamps, negative queries, determinism, malformed
query rejection, HTML escaping and opt-in output safety. It runs in public
GitHub Actions and does not use GPU resources.

This demonstrates **lexical FTS5 evidence retrieval**, not whether the live
ERF search can find a requested person, match a frame visually, interpret
complex natural language or play an editorial clip. Those require the
separate production retrieval stack and different acceptance evidence.

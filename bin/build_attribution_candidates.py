#!/usr/bin/env python3
"""
bin/build_attribution_candidates.py
Generates stratified golden-window candidates for speaker attribution evaluation.
Run: python bin/build_attribution_candidates.py [--limit N]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path('/workspace/erf')
CAT = ROOT / 'catalog'
OUT = ROOT / 'state/eval/attribution_candidates.jsonl'

STRATUM_PRIORITY = {'A': 0, 'B': 1, 'D': 2, 'C': 3}


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_json(path: Path) -> Dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def write_jsonl_atomic(path: Path, lines: Sequence[Dict]) -> None:
    """Write list of dicts as newline-delimited JSON, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp' + os.extsep + os.urandom(4).hex())
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            for obj in lines:
                # compact JSON, ASCIIdumps, but keep key order as defined
                json.dump(obj, fh, ensure_ascii=False, separators=(',', ':'))
                fh.write('\n')
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ----------------------------------------------------------------------
# Media discovery
# ----------------------------------------------------------------------
def discover_media_ids() -> List[str]:
    manifest_dir = CAT / 'media_manifest'
    mids = []
    for manifest in sorted(manifest_dir.glob('m_*.json')):
        mid = manifest.stem  # ids include the m_ prefix
        if (CAT / 'asr' / f'{mid}.transcript.jsonl').exists():
            mids.append(mid)
    return mids


# ----------------------------------------------------------------------
# Stratum generators
# ----------------------------------------------------------------------
def generate_A(mid: str,
               transcript: List[Dict],
               adjud_events: List[Dict],
               asd: Dict,
               face_assignments: Dict) -> List[Dict]:
    """Triangulated candidates (stratum A)."""
    # build face_track → person_id for match_score >= 0.5
    face_map: Dict[str, str] = {}
    for fa in face_assignments.get('face_assignments', []):
        if fa.get('match_score', 0) >= 0.5:
            face_map[fa['face_track_id']] = fa['person_id']

    candidates = []
    for event in adjud_events:
        conf_speaker = event.get('confidence', {}).get('speaker', 0)
        pid = event.get('person_id')
        if conf_speaker < 0.85 or pid is None:
            continue
        dur = event['end_ms'] - event['start_ms']
        if not (1500 <= dur <= 8000):
            continue
        if event.get('n_words', 0) < 3:
            continue

        # compute face coverage
        total_overlap = 0
        max_asd_conf = 0.0
        for seg in asd.get('segments', []):
            if seg.get('speaking_confidence', 0) < 0.5:
                continue
            ftid = seg.get('face_track_id')
            if ftid not in face_map or face_map[ftid] != pid:
                continue
            ol = max(0.0, min(seg['end_ms'], event['end_ms']) -
                     max(seg['start_ms'], event['start_ms']))
            if ol > 0:
                total_overlap += ol
                if seg['speaking_confidence'] > max_asd_conf:
                    max_asd_conf = seg['speaking_confidence']
        coverage = total_overlap / dur if dur > 0 else 0.0
        if coverage < 0.6:
            continue

        cand = {
            'media_id': mid,
            'start_ms': event['start_ms'],
            'end_ms': event['end_ms'],
            'stratum': 'A',
            'expected': [{'person_id': pid}],
            'verified': True,
            'text_preview': event.get('text', '')[:120],
            'provenance': {
                'conf_speaker': conf_speaker,
                'face_coverage': coverage,
                'asd_conf': max_asd_conf,
                'source_event_id': event.get('event_id'),
            },
            '_rank': coverage * conf_speaker,
            'person_id': pid,
        }
        candidates.append(cand)
    return candidates


def generate_B(mid: str,
               transcript: List[Dict],
               render_events: List[Dict]) -> List[Dict]:
    """Crosstalk candidates (stratum B)."""
    # stream A words: every word where stream is not 'B'
    words_A = [w for w in transcript if w.get('stream', 'A') != 'B']
    candidates = []
    for event in render_events:
        if event.get('stream', 'A') != 'B':
            continue
        start = event['start_ms']
        end = event['end_ms']
        # majority person_id among overlapping stream-A words
        overlap_pids = []
        for w in words_A:
            if w['end_ms'] > start and w['start_ms'] < end:
                pid = w.get('person_id')
                if pid is not None:
                    overlap_pids.append(pid)
        if not overlap_pids:
            continue
        majority_pid = Counter(overlap_pids).most_common(1)[0][0]

        expected = []
        # stream B entry
        if event.get('person_id') is not None:
            expected.append({'stream': 'B', 'person_id': event['person_id']})
        else:
            expected.append({'stream': 'B', 'anonymous': True})
        expected.append({'stream': 'A', 'person_id': majority_pid})

        cand = {
            'media_id': mid,
            'start_ms': start,
            'end_ms': end,
            'stratum': 'B',
            'expected': expected,
            'verified': False,
            'text_preview': event.get('text', '')[:120],
            'provenance': {'source_event_id': event.get('event_id')},
        }
        candidates.append(cand)
    return candidates


def generate_C(mid: str,
               adjud_events: List[Dict]) -> List[Dict]:
    """Conflict candidates (stratum C)."""
    candidates = []
    for event in adjud_events:
        cond1 = (event.get('speaker_track_id', '')
                 .endswith('unknown_overlap_speaker'))
        cond2 = event.get('confidence', {}).get('speaker', 0) <= 0.4
        ev = event.get('evidence', {})
        cond3 = ('canonical_asr' in ev and 'community1_baseline' in ev and
                 ev['canonical_asr'] != ev['community1_baseline'])
        if not (cond1 or cond2 or cond3):
            continue
        dur = event['end_ms'] - event['start_ms']
        if dur < 800:
            continue

        pid = event.get('person_id')
        expected = [{'person_id': pid}] if pid is not None else [{'anonymous': True}]
        cand = {
            'media_id': mid,
            'start_ms': event['start_ms'],
            'end_ms': event['end_ms'],
            'stratum': 'C',
            'expected': expected,
            'verified': False,
            'text_preview': event.get('text', '')[:120],
            'provenance': {
                'conf_speaker': event.get('confidence', {}).get('speaker', 0),
                'source_event_id': event.get('event_id'),
            },
            'duration': dur,
        }
        candidates.append(cand)
    return candidates


def generate_D(mid: str,
               transcript: List[Dict],
               identity_map: Dict) -> List[Dict]:
    """Demoted-overmerge candidates (stratum D)."""
    speakers = identity_map.get('speakers', {})
    demoted_entries = [(lbl, entry) for lbl, entry in speakers.items()
                       if entry.get('source') == 'demoted_overmerge']
    if not demoted_entries:
        return []

    label_set = {label for label, _ in demoted_entries}
    label_to_entry = dict(demoted_entries)
    words_by_label: Dict[str, List[Dict]] = defaultdict(list)
    for w in transcript:
        sl = w.get('speaker_label')
        if sl in label_set:
            words_by_label[sl].append(w)

    candidates = []
    for label, words in words_by_label.items():
        if not words:
            continue
        words.sort(key=lambda w: w['start_ms'])
        entry = label_to_entry[label]
        demoted_from = entry.get('demoted_from')
        keeper = entry.get('person_id')

        # merge gaps ≤ 1000 ms
        spans: List[List[Dict]] = []
        current = [words[0]]
        for w in words[1:]:
            if w['start_ms'] - current[-1]['end_ms'] <= 1000:
                current.append(w)
            else:
                spans.append(current)
                current = [w]
        spans.append(current)

        for span in spans:
            start = span[0]['start_ms']
            end = span[-1]['end_ms']
            text_joined = ' '.join(w['text'] for w in span)

            # scoped to the demoted diar label: unscoped anonymous would fail
            # on keeper words interleaved inside merged gaps. speaker_label
            # scope is rebuild-fragile (diar label permutation) — regenerate
            # stratum D after any re-diarization.
            expected = [
                {'speaker_label': label, 'anonymous': True},
            ]
            cand = {
                'media_id': mid,
                'start_ms': start,
                'end_ms': end,
                'stratum': 'D',
                'expected': expected,
                'verified': False,
                'text_preview': text_joined[:120],
                'provenance': {
                    'demoted_label': label,
                    'demoted_from': demoted_from,
                    'keeper_label': entry.get('keeper_label'),
                    'demoted_unknown_id': keeper,
                },
            }
            candidates.append(cand)
    return candidates


# ----------------------------------------------------------------------
# Deduplication & golden ID
# ----------------------------------------------------------------------
def deduplicate(candidates: List[Dict]) -> List[Dict]:
    """Remove lower-priority candidates that overlap >50% with a higher-priority one
    in the same media."""
    by_media: Dict[str, List[Dict]] = defaultdict(list)
    for c in candidates:
        by_media[c['media_id']].append(c)

    kept = []
    for media_id, cands in by_media.items():
        # sort by priority then by start_ms for determinism
        cands.sort(key=lambda x: (STRATUM_PRIORITY[x['stratum']], x['start_ms']))
        media_kept: List[Dict] = []
        for c in cands:
            dur = c['end_ms'] - c['start_ms']
            if dur <= 0:
                continue
            overlap = False
            for k in media_kept:
                inter = max(0.0, min(c['end_ms'], k['end_ms']) -
                            max(c['start_ms'], k['start_ms']))
                if inter / dur > 0.5:
                    overlap = True
                    break
            if not overlap:
                media_kept.append(c)
        kept.extend(media_kept)
    return kept


def assign_golden_ids(candidates: List[Dict]) -> List[Dict]:
    """Add deterministic golden_id to each candidate."""
    for c in candidates:
        token = f"{c['media_id']}|{c['start_ms']}|{c['end_ms']}|{json.dumps(c['expected'], sort_keys=True)}"
        c['golden_id'] = 'ag_' + hashlib.blake2b(token.encode(), digest_size=4).hexdigest()
    return candidates


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Build attribution golden candidates')
    parser.add_argument('--limit', type=int, default=None, help='Max media to process')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger('build_attribution_candidates')

    mids = discover_media_ids()
    if args.limit is not None:
        mids = mids[:args.limit]
    logger.info('Processing %d media', len(mids))

    all_A: List[Dict] = []
    all_B: List[Dict] = []
    all_C: List[Dict] = []
    all_D: List[Dict] = []

    for mid in mids:
        try:
            transcript = load_jsonl(CAT / 'asr' / f'{mid}.transcript.jsonl')
            adjud_events = load_jsonl(CAT / 'speech_events_v2_adjudicated' / f'{mid}.jsonl')
            render_events = load_jsonl(CAT / 'render_events_v2' / f'{mid}.jsonl')
            # face/ASD/identity are optional signals (F0/F2 cover 268/310 media):
            # absence only disables stratum A / D for this media, never skips it
            def _opt(path):
                try:
                    return load_json(path)
                except Exception:
                    return {}
            asd = _opt(CAT / 'active_speaker' / f'{mid}.json')
            face_assignments = _opt(CAT / 'identity_overlay' / f'{mid}.json')
            identity_map = _opt(CAT / 'identity_map' / f'{mid}.json')
        except FileNotFoundError as e:
            logger.warning('Skipping %s: missing %s', mid, e)
            continue
        except Exception as e:
            logger.error('Failed to load data for %s: %s', mid, e)
            continue

        all_A.extend(generate_A(mid, transcript, adjud_events, asd, face_assignments))
        all_B.extend(generate_B(mid, transcript, render_events))
        all_C.extend(generate_C(mid, adjud_events))
        all_D.extend(generate_D(mid, transcript, identity_map))

    # post‑generation selections
    # A – top 5 per person by (coverage * conf_speaker)
    a_by_person: Dict[str, List[Dict]] = defaultdict(list)
    for c in all_A:
        a_by_person[c['person_id']].append(c)
    selected_A = []
    for pid, clist in a_by_person.items():
        clist.sort(key=lambda x: x['_rank'], reverse=True)
        selected_A.extend(clist[:5])
    logger.info('Stratum A – total generated %d, after top‑5 selection %d',
                len(all_A), len(selected_A))

    # C – top 25 by duration desc (across whole corpus)
    all_C.sort(key=lambda x: x['duration'], reverse=True)
    selected_C = all_C[:25]
    logger.info('Stratum C – total generated %d, top‑25 selected %d',
                len(all_C), len(selected_C))

    # combine all candidates that survived selection
    all_candidates = selected_A + all_B + selected_C + all_D
    logger.info('Candidates before dedup: A=%d B=%d C=%d D=%d  total=%d',
                len(selected_A), len(all_B), len(selected_C), len(all_D),
                len(all_candidates))

    deduped = deduplicate(all_candidates)
    logger.info('After overlap dedup: %d', len(deduped))

    deduped.sort(key=lambda c: (c['media_id'], c['start_ms'], c['stratum']))
    final = assign_golden_ids(deduped)
    # remove internal / selection-only keys before writing
    for c in final:
        c.pop('_rank', None)
        c.pop('person_id', None)
        c.pop('duration', None)

    write_jsonl_atomic(OUT, final)
    logger.info('Wrote %d candidates to %s', len(final), OUT)


if __name__ == '__main__':
    main()

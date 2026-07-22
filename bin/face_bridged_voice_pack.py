#!/usr/bin/env python3
"""
Generate a human-confirmation pack of "probably PERSON (by FACE+ASD)" voice clips.
See requirement in code comments.
"""

import json
import logging
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'catalog' / 'identity' / 'face_bridged_voice_pack'
CLIPS_DIR = OUTPUT_DIR / 'clips'

GALLERY_CLEAN_DIR = ROOT / 'catalog' / 'voiceprint_v2' / 'gallery_clean'
PERSONS_DIR = ROOT / 'catalog' / 'persons'
IDENTITY_OVERLAY_DIR = ROOT / 'catalog' / 'identity_overlay'
ACTIVE_SPEAKER_DIR = ROOT / 'catalog' / 'active_speaker'
ASR_DIR = ROOT / 'catalog' / 'asr'
PROXIES_DIR = ROOT / 'media' / 'proxies'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_gallery_clean():
    """Return dict {person_id: n_clips} from all clean gallery JSON files."""
    persons = {}
    if not GALLERY_CLEAN_DIR.exists():
        logging.warning(f"Gallery clean dir not found: {GALLERY_CLEAN_DIR}")
        return persons
    for fp in GALLERY_CLEAN_DIR.glob('*.json'):
        with open(fp, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            pid = data.get('person_id')
            if pid:
                persons[pid] = data.get('n_clips', 0)
        elif isinstance(data, list):
            for entry in data:
                pid = entry.get('person_id')
                if pid:
                    persons[pid] = entry.get('n_clips', 0)
    return persons


def all_portrait_persons():
    """Return set of all person IDs from p_*.json files, excluding gallery/manual files."""
    ids = set()
    for fp in PERSONS_DIR.glob('p_*.json'):
        if fp.name in ('_gallery.json', '_manual_links.jsonl'):
            continue
        ids.add(fp.stem)  # stem removes .json, e.g. 'p_123'
    return ids


def load_face_match_floor():
    """Read face_match_floor from catalog/persons/_gallery.json, default 0.5."""
    fp = PERSONS_DIR / '_gallery.json'
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        return data.get('face_match_floor', 0.5)
    return 0.5


def discover_media_ids():
    """List media IDs for which all required input files exist."""
    ids = []
    for fp in IDENTITY_OVERLAY_DIR.glob('*.json'):
        mid = fp.stem
        if not (ACTIVE_SPEAKER_DIR / f'{mid}.json').exists():
            logging.warning(f"Missing active_speaker for {mid}, skip")
            continue
        if not (ASR_DIR / f'{mid}.diar.jsonl').exists():
            logging.warning(f"Missing diarization for {mid}, skip")
            continue
        if not (PROXIES_DIR / f'{mid}.mp4').exists():
            logging.warning(f"Missing proxy video for {mid}, skip")
            continue
        ids.append(mid)
    return ids


def collect_candidates(media_id, target_persons, face_match_floor):
    """
    For a media item, yield candidate dicts for any face_assignment
    that matches a target person and passes ASD thresholds.
    """
    overlay_path = IDENTITY_OVERLAY_DIR / f'{media_id}.json'
    with open(overlay_path) as f:
        overlay = json.load(f)
    assignments = overlay.get('face_assignments', [])

    as_path = ACTIVE_SPEAKER_DIR / f'{media_id}.json'
    with open(as_path) as f:
        as_data = json.load(f)
    turn_active_speaker = as_data.get('turn_active_speaker', [])

    # index active speakers by face_track_id for fast lookup
    as_by_face = defaultdict(list)
    for entry in turn_active_speaker:
        as_by_face[entry.get('face_track_id')].append(entry)

    candidates = []
    for fa in assignments:
        pid = fa.get('person_id')
        if pid not in target_persons:
            continue
        score = fa.get('match_score', 0)
        if score < face_match_floor:
            continue
        ft_id = fa.get('face_track_id')
        for as_entry in as_by_face.get(ft_id, []):
            conf = as_entry.get('confidence', 0)
            speaking_ms = as_entry.get('speaking_ms', 0)
            if conf < 0.5 or speaking_ms < 2000:
                continue
            cand_score = score * conf * (min(speaking_ms, 4000) / 4000.0)
            candidates.append({
                'media_id': media_id,
                'speaker_label': as_entry['speaker_label'],
                'person_id': pid,
                'score': cand_score,
                'face_match_score': score,
                'asd_confidence': conf,
                'speaking_ms': speaking_ms,
                'face_track_id': ft_id,  # may not be used later
            })
    return candidates


def dedup_candidates(candidates):
    """Deduplicate by (media_id, speaker_label, person_id) keeping best score."""
    best = {}
    for cand in candidates:
        key = (cand['media_id'], cand['speaker_label'], cand['person_id'])
        if key not in best or cand['score'] > best[key]['score']:
            best[key] = cand
    return list(best.values())


def select_persons_and_candidates(deduped, max_persons=60, max_per_person=6):
    """
    Select up to max_persons persons (ranked by their best candidate score),
    then take up to max_per_person top-scoring candidates for each.
    """
    person_candidates = defaultdict(list)
    for cand in deduped:
        person_candidates[cand['person_id']].append(cand)

    person_max_score = {pid: max(c['score'] for c in cands) for pid, cands in person_candidates.items()}
    sorted_persons = sorted(person_max_score.keys(), key=lambda p: person_max_score[p], reverse=True)
    selected_persons = sorted_persons[:max_persons]

    final = []
    for pid in selected_persons:
        cands = sorted(person_candidates[pid], key=lambda x: x['score'], reverse=True)
        final.extend(cands[:max_per_person])
    return final


def get_media_duration(proxy_path):
    """Return duration in seconds of a video file using ffprobe."""
    if not proxy_path.exists():
        return None
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'csv=p=0', str(proxy_path)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode == 0:
            dur_str = out.stdout.strip()
            return float(dur_str) if dur_str else None
    except Exception as e:
        logging.error(f"ffprobe failed for {proxy_path}: {e}")
    return None


def load_diarization(media_id):
    """Load diarization turns for a media from its JSONL file."""
    diar_path = ASR_DIR / f'{media_id}.diar.jsonl'
    if not diar_path.exists():
        return None
    turns = []
    with open(diar_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns.append(json.loads(line))
    return turns


def get_clip_window(media_id, speaker_label):
    """
    Return (start_sec, dur_sec, end_sec) for the longest diarization turn
    of this speaker, clamped to [3s, 10s] and within media duration.
    Returns (None, None, None) if no valid window.
    """
    diar_turns = load_diarization(media_id)
    if diar_turns is None:
        return None, None, None
    speaker_turns = [t for t in diar_turns if t.get('speaker_label') == speaker_label]
    if not speaker_turns:
        logging.warning(f"No diar turns for speaker {speaker_label} in media {media_id}")
        return None, None, None

    longest = max(speaker_turns, key=lambda t: t['end_ms'] - t['start_ms'])
    duration_ms = longest['end_ms'] - longest['start_ms']
    start_sec = longest['start_ms'] / 1000.0
    dur_sec = max(min(duration_ms / 1000.0, 10.0), 3.0)

    media_dur = get_media_duration(PROXIES_DIR / f'{media_id}.mp4')
    if media_dur is not None:
        if start_sec + dur_sec > media_dur:
            dur_sec = media_dur - start_sec
            if dur_sec < 1.0:
                logging.warning(f"Remaining duration too short for {media_id}/{speaker_label}")
                return None, None, None

    return start_sec, dur_sec, start_sec + dur_sec


def extract_clip(media_id, start_sec, dur_sec, out_path):
    """Run ffmpeg to extract a clip; returns True on success."""
    proxy_in = PROXIES_DIR / f'{media_id}.mp4'
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_sec:.3f}',
        '-i', str(proxy_in),
        '-t', f'{dur_sec:.3f}',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"Created clip {out_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed for {out_path.name}: {e.stderr.decode()}")
        return False


def main():
    # 1. Identify target persons
    gallery_clean = load_gallery_clean()
    logging.info(f"Loaded {len(gallery_clean)} clean-gold persons")

    portrait_persons = all_portrait_persons()
    target_persons = set()
    for pid in portrait_persons:
        if pid not in gallery_clean or gallery_clean[pid] < 5:
            target_persons.add(pid)
    logging.info(f"Target persons (n_clips<5 or absent): {len(target_persons)}")

    face_match_floor = load_face_match_floor()
    logging.info(f"Face match floor: {face_match_floor}")

    # 2. Iterate media and collect candidates
    media_ids = discover_media_ids()
    logging.info(f"Media with required files: {len(media_ids)}")

    all_cands = []
    for mid in media_ids:
        cands = collect_candidates(mid, target_persons, face_match_floor)
        all_cands.extend(cands)
        logging.debug(f"Media {mid}: {len(cands)} raw candidates")

    logging.info(f"Total raw candidates: {len(all_cands)}")
    deduped = dedup_candidates(all_cands)
    logging.info(f"After dedup: {len(deduped)}")

    # 3. Cap persons and select best candidates per person
    selected = select_persons_and_candidates(deduped, max_persons=60, max_per_person=6)
    logging.info(f"Selected candidates after person cap: {len(selected)}")

    # 4. Determine clip windows using diarization
    candidates_with_window = []
    for cand in selected:
        start, dur, end = get_clip_window(cand['media_id'], cand['speaker_label'])
        if start is None:
            continue
        cand['clip_start'] = start
        cand['clip_dur'] = dur
        cand['clip_end'] = end
        candidates_with_window.append(cand)

    # 5. Assign IDs and extract clips
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, cand in enumerate(candidates_with_window, start=1):
        cand['candidate_id'] = f"candidate_{idx:04d}"
        cand['clip_path'] = f"clips/{cand['candidate_id']}.mp4"

    successful = []
    for cand in candidates_with_window:
        out_path = CLIPS_DIR / f"{cand['candidate_id']}.mp4"
        if extract_clip(cand['media_id'], cand['clip_start'], cand['clip_dur'], out_path):
            successful.append(cand)
        else:
            logging.warning(f"Dropped candidate {cand['candidate_id']}")

    logging.info(f"Successfully extracted {len(successful)} clips")

    # 6. Write JSONL
    jsonl_path = OUTPUT_DIR / 'face_bridged_candidates.jsonl'
    tmp_jsonl = jsonl_path.with_suffix('.tmp')
    with open(tmp_jsonl, 'w') as f:
        for entry in successful:
            rec = {
                "candidate_id": entry['candidate_id'],
                "media_id": entry['media_id'],
                "speaker": entry['speaker_label'],
                "proposed_person_id": entry['person_id'],
                "clip_path": entry['clip_path'],
                "start": round(entry['clip_start'], 3),
                "end": round(entry['clip_end'], 3),
                "evidence": {
                    "proposed_by": "face_asd",
                    "face_match_score": entry['face_match_score'],
                    "asd_confidence": entry['asd_confidence'],
                    "speaking_ms": entry['speaking_ms'],
                    "person_target": entry['person_id'],
                },
                "review_status": "pending",
                "review_person_id": None,
                "review_notes": None,
            }
            f.write(json.dumps(rec) + '\n')
    os.replace(tmp_jsonl, jsonl_path)
    logging.info(f"JSONL written to {jsonl_path}")

    # 7. Write review HTML
    html_path = OUTPUT_DIR / 'review.html'
    tmp_html = html_path.with_suffix('.tmp')
    with open(tmp_html, 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Face-Bridged Voice Pack Review</title>
<style>
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 8px; vertical-align: top; }
</style>
</head>
<body>
<h2>Review likely person clips (face + ASD)</h2>
<table>
<tr>
<th>Clip</th>
<th>Proposed Person</th>
<th>Face Score</th>
<th>ASD Conf</th>
<th>Decision</th>
<th>Correct Person</th>
<th>Notes</th>
</tr>
''')
        for entry in successful:
            cid = entry['candidate_id']
            evidence_str = json.dumps(entry['evidence'])
            row = (
                f'<tr data-candidate-row data-candidate-id="{cid}" '
                f'data-media-id="{entry["media_id"]}" data-speaker="{entry["speaker_label"]}" '
                f'data-proposed-person="{entry["person_id"]}" '
                f'data-clip-path="{entry["clip_path"]}" '
                f'data-start="{entry["clip_start"]}" data-end="{entry["clip_end"]}" '
                f'data-evidence=\'{evidence_str}\'>'
                f'<td><video controls width="300" src="{entry["clip_path"]}"></video></td>'
                f'<td>PROBABLY (by face): {entry["person_id"]}</td>'
                f'<td>{entry["face_match_score"]:.3f}</td>'
                f'<td>{entry["asd_confidence"]:.3f}</td>'
                f'<td>'
                f'<label><input type="radio" name="review_{cid}" value="accept"> Accept</label><br/>'
                f'<label><input type="radio" name="review_{cid}" value="reject"> Reject</label>'
                f'</td>'
                f'<td><input type="text" name="rename_{cid}" placeholder="Person ID"></td>'
                f'<td><input type="text" name="notes_{cid}" placeholder="Notes"></td>'
                f'</tr>'
            )
            f.write(row)
        f.write('''</table>
<br/>
<button onclick="downloadDecisions()">Download decisions</button>
<script>
function downloadDecisions() {
    var rows = document.querySelectorAll('[data-candidate-row]');
    var output = [];
    rows.forEach(function(row) {
        var cid = row.getAttribute('data-candidate-id');
        var acceptRadio = row.querySelector('input[name="review_' + cid + '"]:checked');
        var review_status = acceptRadio ? acceptRadio.value : 'pending';
        var renameInput = row.querySelector('input[name="rename_' + cid + '"]');
        var notesInput = row.querySelector('input[name="notes_' + cid + '"]');
        var proposed = row.getAttribute('data-proposed-person');
        var record = {
            candidate_id: cid,
            media_id: row.getAttribute('data-media-id'),
            speaker: row.getAttribute('data-speaker'),
            proposed_person_id: proposed,
            clip_path: row.getAttribute('data-clip-path'),
            start: parseFloat(row.getAttribute('data-start')),
            end: parseFloat(row.getAttribute('data-end')),
            evidence: JSON.parse(row.getAttribute('data-evidence')),
            review_status: review_status,
            review_person_id: review_status === 'accept' ? (renameInput.value || proposed) : null,
            review_notes: notesInput.value
        };
        output.push(record);
    });
    var blob = new Blob([output.map(JSON.stringify).join('\\n')], {type: 'text/plain'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'face_bridged_candidates_reviewed.jsonl';
    a.click();
}
</script>
</body>
</html>''')
    os.replace(tmp_html, html_path)
    logging.info(f"HTML written to {html_path}")

    # 8. Print per-person clip counts
    person_counts = Counter(e['person_id'] for e in successful)
    for pid, cnt in sorted(person_counts.items()):
        logging.info(f"Person {pid}: {cnt} clips")


if __name__ == '__main__':
    main()

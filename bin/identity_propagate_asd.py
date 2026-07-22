#!/usr/bin/env python3
"""
bin/identity_propagate_asd.py

Propagate person_id from ASD face-anchored diarization turns to ALL turns of
the same speaker_label (face-voice hopscotch).

For each media: find ASD-confirmed face turns (confidence >= threshold) → map
face_track_id → person_id (from face_assignments) → majority-vote per
speaker_label → write updated speaker_assignments to identity_overlay.
"""

import json
import os
import sys
import argparse
import logging
import tempfile
from pathlib import Path
from collections import defaultdict, Counter

try:
    from erf.runlog import log_event
except ImportError:
    def log_event(*args, **kwargs):
        pass


ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "catalog"
ACTIVE_SPEAKER_DIR = CAT / "active_speaker"
IDENTITY_OVERLAY_DIR = CAT / "identity_overlay"


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logging.warning(f"Invalid JSON in {path}: {e}")
        return None


def process_media(media_id, args):
    active_path = ACTIVE_SPEAKER_DIR / f"{media_id}.json"
    overlay_path = IDENTITY_OVERLAY_DIR / f"{media_id}.json"

    asd_data = _load_json(active_path)
    if asd_data is None:
        return None

    overlay = _load_json(overlay_path)
    if overlay is None:
        logging.warning(f"{media_id}: no identity_overlay — skipping (run F1 first)")
        return None

    # face_track_id -> person_id lookup from face_assignments list
    fa_list = overlay.get("face_assignments", [])
    face_map = {
        fa["face_track_id"]: fa["person_id"]
        for fa in fa_list
        if isinstance(fa, dict) and fa.get("face_track_id") and fa.get("person_id")
    }

    # Accumulate ASD evidence per speaker_label
    lbl_evidence = defaultdict(list)  # label -> [(person_id, conf, face_track_id)]
    for turn in asd_data.get("turn_active_speaker", []):
        ft = turn.get("face_track_id")
        if not ft:
            continue
        conf = float(turn.get("confidence", 0.0))
        if conf < args.asd_threshold:
            continue
        pid = face_map.get(ft)
        if pid:
            lbl_evidence[turn["speaker_label"]].append((pid, conf, ft))

    if not lbl_evidence:
        return {"fills": 0, "reinforcements": 0, "overrides": 0, "skipped_conflicts": 0}

    sa_list = overlay.get("speaker_assignments", [])
    existing = {a["speaker_label"]: a for a in sa_list if isinstance(a, dict)}

    fills = reinforcements = overrides = skipped = 0
    changes = []

    for label, votes in lbl_evidence.items():
        pid_counter = Counter(pid for pid, _, _ in votes)
        ranked = pid_counter.most_common(2)

        # Tie → skip
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            logging.warning(f"{media_id} {label}: tie {ranked} — skipping")
            skipped += 1
            continue

        winning_pid = ranked[0][0]
        winning_votes = [(pid, conf, ft) for pid, conf, ft in votes if pid == winning_pid]
        mean_conf = sum(c for _, c, _ in winning_votes) / len(winning_votes)
        n_anchors = len(winning_votes)
        face_tracks = sorted({ft for _, _, ft in winning_votes})

        ex = existing.get(label)
        ex_pid = ex.get("person_id") if ex else None
        ex_method = ex.get("method", "unknown") if ex else None

        if not ex_pid:
            action = "FILL"
        elif ex_pid == winning_pid:
            action = "REINFORCE"
        elif mean_conf >= args.override_threshold:
            action = "OVERRIDE"
        else:
            if args.verbose:
                print(f"{media_id} {label}: SKIP_CONFLICT existing={ex_pid} asd={winning_pid} conf={mean_conf:.3f}")
            skipped += 1
            continue

        if action == "REINFORCE":
            new_entry = dict(ex)
            new_entry["method"] = "asd_propagated"
            ev = list(new_entry.get("evidence", []))
            ev.append(f"asd_confirmed: {n_anchors} face-anchored turns agree with {ex_method}")
            new_entry["evidence"] = ev
            reinforcements += 1
            if args.verbose:
                print(f"{media_id} {label}: REINFORCE {winning_pid} (n={n_anchors}, conf={mean_conf:.3f})")
        else:
            evidence = [
                f"asd_anchored: {n_anchors} turns confirmed face",
                f"face_tracks: {','.join(face_tracks)}",
            ]
            if action == "OVERRIDE":
                evidence.append(f"overrode {ex_method}:{ex_pid} (asd_conf={mean_conf:.3f})")
                overrides += 1
                if args.verbose:
                    print(f"{media_id} {label}: OVERRIDE {ex_pid} -> {winning_pid} (n={n_anchors}, conf={mean_conf:.3f})")
            else:
                fills += 1
                if args.verbose:
                    print(f"{media_id} {label}: FILL {winning_pid} (n={n_anchors}, conf={mean_conf:.3f})")
            new_entry = {
                "speaker_label": label,
                "person_id": winning_pid,
                "method": "asd_propagated",
                "confidence": mean_conf,
                "evidence_seconds": 0.0,
                "evidence": evidence,
            }

        changes.append((label, new_entry))

    if changes:
        sa_map = {a["speaker_label"]: a for a in sa_list if isinstance(a, dict)}
        for label, entry in changes:
            sa_map[label] = entry
        overlay["speaker_assignments"] = list(sa_map.values())

        if not args.dry_run:
            fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=IDENTITY_OVERLAY_DIR, text=True)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(overlay, f, indent=2, ensure_ascii=False)
                os.replace(tmp, overlay_path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        elif args.verbose:
            print(f"[DRY RUN] Would write {overlay_path}")

    return {"fills": fills, "reinforcements": reinforcements, "overrides": overrides, "skipped_conflicts": skipped}


def main():
    ap = argparse.ArgumentParser(description="ASD face-anchored identity propagation")
    ap.add_argument("--media-ids", nargs="*", metavar="M", help="Specific media IDs (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--asd-threshold", type=float, default=0.5, metavar="T",
                    help="Min ASD confidence to consider a turn face-anchored (default 0.5)")
    ap.add_argument("--override-threshold", type=float, default=0.75, metavar="T",
                    help="Mean ASD conf required to override existing assignment (default 0.75)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.media_ids:
        media_ids = args.media_ids
    else:
        media_ids = [p.stem for p in sorted(ACTIVE_SPEAKER_DIR.glob("*.json"))]

    total = fills = reinforcements = overrides = skipped = errors = 0
    for mid in media_ids:
        total += 1
        if not args.dry_run:
            log_event("F2:ASD_PROP", mid, "start", {})
        try:
            counts = process_media(mid, args)
            if counts is None:
                if not args.dry_run:
                    log_event("F2:ASD_PROP", mid, "skip", {"reason": "missing file"})
                continue
            fills += counts["fills"]
            reinforcements += counts["reinforcements"]
            overrides += counts["overrides"]
            skipped += counts["skipped_conflicts"]
            if not args.dry_run:
                log_event("F2:ASD_PROP", mid, "ok", counts)
        except Exception as e:
            logging.exception(f"{mid}: {e}")
            errors += 1
            if not args.dry_run:
                log_event("F2:ASD_PROP", mid, "error", {"error": str(e)})

    print(f"Processed {total} media: {fills} fills, {reinforcements} reinforcements, "
          f"{overrides} overrides, {skipped} skipped conflicts, {errors} errors")


if __name__ == "__main__":
    main()

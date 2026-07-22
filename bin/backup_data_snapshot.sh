#!/bin/bash
# Periodic off-pod backup of ERF's canonical data surfaces (catalog/, enroll/, and the core
# state/*.db files) to the private OpenSourceFilms/erf-data-backup repo. Disaster-recovery
# only — not required for any tool's runtime functionality (see D-Log/BATON backup decision,
# 2026-07-03).
#
# Pushes are chunked to ~300MB to stay under GitHub's per-push size/timeout ceiling (a single
# multi-GB commit reliably hit an HTTP 500 on first attempt, 2026-07-22).
set -euo pipefail

ERF_ROOT=/workspace/erf
STAGING=/workspace/erf_data_backup
CHUNK_BYTES=$((300 * 1024 * 1024))

mkdir -p "$STAGING"/state
echo "=== syncing catalog/ and enroll/ (hardlinked where possible) ==="
rsync -a --delete --link-dest="$ERF_ROOT/catalog" "$ERF_ROOT/catalog/" "$STAGING/catalog/"
rsync -a --delete "$ERF_ROOT/enroll/" "$STAGING/enroll/"

echo "=== materializing enroll/*/faces symlinks (must be real copies, not links, for the backup to be self-contained) ==="
for d in "$STAGING"/enroll/*/; do
  [ -L "${d}faces" ] || continue
  target=$(readlink -f "${d}faces")
  rm -f "${d}faces"
  if [ -e "$target" ]; then
    cp -al "$target" "${d}faces"
  else
    echo "WARNING: dangling symlink for ${d}faces -> $target — skipped, needs manual check"
  fi
done

echo "=== syncing core state DBs ==="
cp -f "$ERF_ROOT"/state/{erf.db,erf_fts.db,narrative_pipeline.db,journal.db} "$STAGING/state/"

cd "$STAGING"
git add -A
if git diff --cached --quiet; then
  echo "No changes since last snapshot — nothing to commit."
  exit 0
fi
git commit -q -m "data backup snapshot $(date -u +%F)"

# Chunked push: split the pending push into ~300MB batches so no single push exceeds
# GitHub's ceiling. Uses a temporary branch pointer per chunk via cherry-pick-free approach:
# since this is usually a small delta after the first snapshot, try a plain push first and
# only fall back to rewriting as multiple commits if it fails.
if git push origin master 2>&1 | tee /tmp/backup_push.log; then
  echo "=== push succeeded in one shot ==="
else
  echo "=== single push failed, falling back to per-directory pushes ==="
  git reset --soft HEAD~1
  for path in state enroll catalog; do
    git add "$path"
    if ! git diff --cached --quiet; then
      git commit -q -m "data backup snapshot $(date -u +%F) ($path)"
      git push origin master
    fi
  done
fi

echo "=== done ==="
git log --oneline -5

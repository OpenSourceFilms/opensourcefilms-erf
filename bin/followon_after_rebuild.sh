#!/usr/bin/env bash
# Follow-on after rebuild_all.py completes.
# Re-runs F2→RECONCILE to fix ~119 ASD failures (onnxruntime missing in asd venv
# for media 1-119 of the first rebuild; installed 2026-06-24 mid-F2 run).
# Usage: setsid bash bin/followon_after_rebuild.sh [REBUILD_PID] &

set -o pipefail

LOG=/workspace/erf/logs.all/followon_console.log
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

cd /workspace/erf || exit 1
REBUILD_PID=${1:-1563624}

if [ -z "$DEEPSEEK_API_KEY" ]; then
    log "ERROR: DEEPSEEK_API_KEY not set — aborting"
    exit 1
fi

log "follow-on started: waiting for rebuild PID $REBUILD_PID"
while kill -0 "$REBUILD_PID" 2>/dev/null; do sleep 30; done
log "PID $REBUILD_PID exited"

# Summarise first rebuild
REPORT1=$(ls -t logs.all/rebuild_report_*.json 2>/dev/null | head -1)
if [ -n "$REPORT1" ]; then
    log "first rebuild report: $REPORT1"
    PYTHONPATH=src venvs/embed/bin/python - <<'PY' "$REPORT1" 2>&1 | tee -a "$LOG"
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  finished: {r.get('finished_at','?')}")
for s in r.get('stages', []):
    flag = ' FAIL' if s['rc'] != 0 else ''
    print(f"  {s['name']:10} rc={s['rc']} {s['elapsed_s']}s{flag}")
print(f"  non-fatal failures: {r.get('failures', [])}")
PY
fi

# Pre-check: C/C_vis errors (FileNotFoundError 'No text embeddings' seen in BASE)
log "pre-check: C/C_vis stage errors in outcomes"
PYTHONPATH=src venvs/embed/bin/python - 2>&1 | tee -a "$LOG" <<'PY'
import json, glob
errs = []
for f in sorted(glob.glob("catalog/outcomes/*.json")):
    d = json.load(open(f))
    for stage in ("C", "C_vis"):
        s = d.get("stages", {}).get(stage, {})
        if s.get("status") == "error":
            errs.append((d["media_id"], stage, s.get("reason", "?")))
print(f"  C/C_vis errors: {len(errs)}")
for mid, stage, reason in errs[:15]:
    print(f"    {mid} {stage}: {reason}")
if len(errs) > 15:
    print(f"    ... and {len(errs)-15} more")
PY

# Re-run only F2 (fixes 100 ASD onnxruntime errors; staleness gate skips the ~172 already-ok)
# plus PURGE+RECONCILE to clean journal. V0/E1/CARD/EV/LABEL/VOICE don't read ASD data
# and already ran correctly in the first rebuild — no need to re-run them.
log "starting rebuild --only F2 PURGE RECONCILE (fixes ASD onnxruntime failures, skips already-ok)"
PYTHONPATH=src venvs/embed/bin/python bin/rebuild_all.py --only F2 PURGE RECONCILE >> "$LOG" 2>&1
RC=$?
log "rebuild --from F2 exited rc=$RC"

# Summarise follow-on rebuild
REPORT2=$(ls -t logs.all/rebuild_report_*.json 2>/dev/null | head -1)
if [ -n "$REPORT2" ] && [ "$REPORT2" != "$REPORT1" ]; then
    log "follow-on report: $REPORT2"
    PYTHONPATH=src venvs/embed/bin/python - <<'PY' "$REPORT2" 2>&1 | tee -a "$LOG"
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  finished: {r.get('finished_at','?')}")
for s in r.get('stages', []):
    flag = ' FAIL' if s['rc'] != 0 else ''
    print(f"  {s['name']:10} rc={s['rc']} {s['elapsed_s']}s{flag}")
print(f"  non-fatal failures: {r.get('failures', [])}")
PY
fi

log "follow-on complete. Full log: $LOG"

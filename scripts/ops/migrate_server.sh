#!/usr/bin/env bash
# Non-destructive, resumable compute deployment; results stay on the source.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE=${MIGRATION_REMOTE:?Set MIGRATION_REMOTE=user@host}
DEST=${MIGRATION_DEST:?Set MIGRATION_DEST to the dedicated absolute project directory}
KEY=${MIGRATION_KEY:?Set MIGRATION_KEY to your SSH private key outside this project}
RECORD_REL=${MIGRATION_RECORD:-results/migration/project_copy}
[[ "$REMOTE" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*@[a-zA-Z0-9.-]+$ ]] || exit 2
[[ "$DEST" =~ ^/[-a-zA-Z0-9_./]+$ && "$DEST" != *'/../'* && "$DEST" != */.. ]] || exit 2
case "$DEST" in /|/home|/root|/mnt|/mnt/shared|/data0|/data1) echo 'Refusing broad destination' >&2; exit 2;; esac
[[ "$RECORD_REL" == results/migration/* && "$RECORD_REL" != *..* ]] || exit 2
KEY=$(realpath -e -- "$KEY")
[[ "$KEY" != "$PROJECT_ROOT/"* ]] || { echo 'Private key must remain outside copied project' >&2; exit 2; }
RECORD="$PROJECT_ROOT/$RECORD_REL"
mkdir -p "$RECORD"
exec 9>"$RECORD/transfer.lock"
flock -n 9 || { echo 'A copy with this record directory is already running.' >&2; exit 2; }
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
printf -v RSYNC_RSH 'ssh -i %q -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ProxyCommand=none -o ProxyJump=none' "$KEY"
export RSYNC_RSH
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=30 -o ProxyCommand=none -o ProxyJump=none)
COMMON=(-a --no-owner --no-group --partial --partial-dir=.rsync-partial
        --human-readable --info=progress2,stats2 --exclude="/$RECORD_REL/"
        --exclude=/results/ --exclude=/.archive/)
CODE_ONLY=(--exclude=/checkpoints/ --exclude=/datasets/ --exclude=/environment/
           --exclude=/.cache/ --exclude=/.uv-cache/ --exclude=/results/
           --exclude=/third_party/ --exclude=/.archive/ --exclude=/.tools/)
phase() { printf '%s %s\n' "$(date -Is)" "$1" | tee "$RECORD/status.txt"; }
trap 'rc=$?; if ((rc)); then phase "failed_exit_$rc; outputs retained; rerun to resume"; fi' EXIT

"${SSH[@]}" "$REMOTE" "test -d '$DEST' && test -w '$DEST' && id && df -h '$DEST'"
phase copying_code
rsync "${COMMON[@]}" "${CODE_ONLY[@]}" "$PROJECT_ROOT/" "$REMOTE:$DEST/"
phase copying_bootstrap_tools
"${SSH[@]}" "$REMOTE" "mkdir -p '$DEST/.tools/python'"
UV_BIN=${UV_BIN:-$(command -v uv || true)}
if [[ -n "$UV_BIN" && -x "$UV_BIN" ]]; then
    rsync -a --no-owner --no-group "$UV_BIN" "$REMOTE:$DEST/.tools/uv"
    PYTHON_SOURCE=${MIGRATION_PYTHON_SOURCE:-$("$UV_BIN" python dir)}
    if [[ -d "$PYTHON_SOURCE" ]]; then
        rsync -a --no-owner --no-group "$PYTHON_SOURCE/" "$REMOTE:$DEST/.tools/python/"
    fi
fi
phase copying_all_project_files
for attempt in 1 2 3; do
    rc=0
    rsync "${COMMON[@]}" "$PROJECT_ROOT/" "$REMOTE:$DEST/" || rc=$?
    printf '%s attempt=%s exit=%s\n' "$(date -Is)" "$attempt" "$rc" >> "$RECORD/passes.log"
    if ((rc == 0)); then break; fi
    # Only retry vanished live-source files. Other errors remain failures.
    if ((rc != 24 || attempt == 3)); then exit "$rc"; fi
done
phase verifying_code_checksums_and_remaining_delta
rsync -anc --omit-dir-times --no-owner --no-group --itemize-changes "${CODE_ONLY[@]}" \
    "$PROJECT_ROOT/" "$REMOTE:$DEST/" > "$RECORD/code_checksum_delta.txt"
rsync -an --omit-dir-times --no-owner --no-group --itemize-changes --exclude="/$RECORD_REL/" \
    --exclude=/results/ --exclude=/.archive/ \
    "$PROJECT_ROOT/" "$REMOTE:$DEST/" > "$RECORD/live_source_delta.txt"
if [[ -s "$RECORD/code_checksum_delta.txt" ]]; then phase code_changed_during_copy_needs_resync; exit 3; fi
if [[ -s "$RECORD/live_source_delta.txt" ]]; then
    phase initial_copy_finished_live_source_delta_pending
else
    phase initial_copy_verified_size_mtime_and_code_checksums
fi
echo 'Copy only: results, archives and transfer records remain on the source; no inference or deletion.'

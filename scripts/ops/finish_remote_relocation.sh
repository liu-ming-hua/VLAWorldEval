#!/usr/bin/env bash
# Explicit copy follow-up; all machine-specific values come from the caller.
set -euo pipefail
COPY_PID=${1:?Pass the original copy controller PID}
[[ "$COPY_PID" =~ ^[0-9]+$ ]] || exit 2
REMOTE=${MIGRATION_REMOTE:?Set MIGRATION_REMOTE=user@host}
DEST=${MIGRATION_DEST:?Set MIGRATION_DEST to the dedicated absolute project directory}
KEY=${MIGRATION_KEY:?Set MIGRATION_KEY to your SSH key}
[[ "$REMOTE" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*@[a-zA-Z0-9.-]+$ ]] || exit 2
[[ "$DEST" =~ ^/[-a-zA-Z0-9_./]+$ && "$DEST" != *'/../'* && "$DEST" != */.. ]] || exit 2
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=30 -o ProxyCommand=none -o ProxyJump=none)
COPY_START=""
if [[ -r /proc/$COPY_PID/cmdline ]]; then
    command_line=$(tr '\0' ' ' < "/proc/$COPY_PID/cmdline")
    [[ "$command_line" == *scripts/ops/migrate_server.sh* ]] || {
        echo 'PID does not identify the requested copy controller' >&2; exit 2;
    }
    COPY_START=$(sed 's/^.*) //' "/proc/$COPY_PID/stat" | cut -d' ' -f20)
fi
echo "Waiting for copy controller $COPY_PID; no GPU work."
while [[ -n "$COPY_START" && -r /proc/$COPY_PID/stat ]]; do
    current=$(sed 's/^.*) //' "/proc/$COPY_PID/stat")
    [[ "$(cut -d' ' -f1 <<< "$current")" != Z ]] || break
    [[ "$(cut -d' ' -f20 <<< "$current")" == "$COPY_START" ]] || break
    sleep 30
done
echo 'Controller ended; repairing available runtime files, not declaring copy complete.'
"${SSH[@]}" "$REMOTE" \
    "python3 '$DEST/scripts/ops/relocate_runtime.py' --root '$DEST' --apply"
"${SSH[@]}" "$REMOTE" \
    "cd '$DEST' && bash scripts/ops/remote.sh environment/core/bin/python -m cli --help"

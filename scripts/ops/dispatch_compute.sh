#!/usr/bin/env bash
# Source-side transfer dependencies and remote native-study orchestration.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE=${MIGRATION_REMOTE:?Set MIGRATION_REMOTE}
DEST=${MIGRATION_DEST:?Set MIGRATION_DEST}
KEY=${MIGRATION_KEY:?Set MIGRATION_KEY}
STAGE_PID=${STAGE_PID:?Set explicit staging controller PID}
PREP_PID=${PREP_PID:?Set explicit Cosmos preparation PID}
STAGE_LOG=${STAGE_LOG:?Set source staging log}
HANDOFF=${REMOTE_HANDOFF:?Set project-relative handoff directory}
[[ "$REMOTE" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*@[a-zA-Z0-9.-]+$ ]] || exit 2
[[ "$DEST" =~ ^/[-a-zA-Z0-9_./]+$ && "$DEST" != *'/../'* && "$DEST" != */.. ]] || exit 2
[[ "$STAGE_PID" =~ ^[0-9]+$ && "$PREP_PID" =~ ^[0-9]+$ ]] || exit 2
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ProxyCommand=none -o ProxyJump=none)
if [[ -n "${MIGRATION_HOST_KEY_ALIAS:-}" ]]; then
    SSH+=(-o "HostKeyAlias=$MIGRATION_HOST_KEY_ALIAS" -o StrictHostKeyChecking=yes)
fi
cd "$TASK_ROOT"
live() { [[ -r /proc/$1/stat ]] && [[ "$(sed 's/^.*) //' /proc/"$1"/stat | cut -d' ' -f1)" != Z ]]; }
wait_stage() {
    while ! grep -Fq "$1" "$STAGE_LOG"; do
        live "$STAGE_PID" || { echo 'Staging terminated before required phase' >&2; exit 1; }
        sleep 15
    done
}
if [[ "${CAMPAIGN_MODE:-full}" == budget400 ]]; then
    BUDGET=results/checkpoint_capability_v1/budget400_v1
    PLAN=$BUDGET/cosmos20k/plan.json
    PROGRESS=$BUDGET/cosmos20k/preparation/progress.json
    BATCHES=$BUDGET/cosmos20k/batches
    wait_stage 'Cosmos runtime and checkpoint transferred'
    printf -v RSYNC_RSH '%q ' "${SSH[@]}"
    export RSYNC_RSH
    rsync -aR --no-owner --no-group --partial \
        checkpoints/world_models/cosmos3_nano_droid_fd/vae \
        checkpoints/world_models/cosmos3_nano_droid_fd/text_tokenizer "$REMOTE:$DEST/"
    rsync -aR --no-owner --no-group evaluation/planning/native_batches.py \
        cli/parser.py cli/commands/studies.py "$PLAN" "$REMOTE:$DEST/"
    while true; do
        # A cancelled transfer may have sealed a batch without delivering it.
        # Retry those immutable batches before allocating any new ownership.
        PENDING_MANIFEST=''
        for CANDIDATE in "$BATCHES"/batch*/prepared_manifest.json; do
            [[ -f "$CANDIDATE" ]] || continue
            ACK=${CANDIDATE%/*}/transfer_ack.json
            if [[ ! -f "$ACK" ]]; then PENDING_MANIFEST=$CANDIDATE; break; fi
            environment/core/bin/python -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["remote"]==sys.argv[2] and x["destination"]==sys.argv[3]' "$ACK" "$REMOTE" "$DEST"
        done
        if [[ -n "$PENDING_MANIFEST" ]]; then
            BATCH=${PENDING_MANIFEST%/*}; BATCH=${BATCH##*/}
            BATCH_PLAN=${PENDING_MANIFEST%/*}/plan.json
            MANIFEST=$PENDING_MANIFEST
        else
            BATCH_JSON=$(environment/core/bin/python -m cli native-study batch --plan "$PLAN" \
                --prepared-progress "$PROGRESS" --directory "$BATCHES" --batch-limit 32)
        mapfile -t META < <(environment/core/bin/python -c \
            'import json,sys; x=json.load(sys.stdin); print(int(x["ready"])); print(int(x["all_scheduled"])); print(x.get("batch","")); print(x.get("plan","")); print(x.get("manifest",""))' <<< "$BATCH_JSON")
        if [[ "${META[0]}" != 1 ]]; then
            if [[ "${META[1]}" == 1 ]]; then
                echo "$(date -Is) All400 frozen Cosmos cases scheduled; inference may still be running"
                exit 0
            fi
            live "$PREP_PID" || { echo 'Subset preparation stopped; unresolved cases remain pending, no fallback' >&2; exit 1; }
            sleep 15
            continue
        fi
        BATCH=${META[2]}
        BATCH_PLAN=${META[3]}
        MANIFEST=${META[4]}
        fi
        environment/core/bin/python - "$BATCH_PLAN" <<'PY' | rsync -arR --no-owner --no-group --partial --files-from=- ./ "$REMOTE:$DEST/"
import json
import sys
plan = json.load(open(sys.argv[1]))
for case in plan["cases"]:
    print(case["prepared"] + "/")
PY
        rsync -aR --no-owner --no-group "$BATCH_PLAN" "$MANIFEST" "$REMOTE:$DEST/"
        if [[ "${RESIDENT_STREAM:-0}" == 1 ]]; then
            echo "$(date -Is) Delivered $BATCH for the single resident worker; no new model process launched"
        else
            RUN=$BUDGET/remote_runs/cosmos_runs/$BATCH
            "${SSH[@]}" "$REMOTE" "cd '$DEST' && bash scripts/ops/remote.sh environment/core/bin/python -m cli native-study prepare --directory '$RUN' --plan '$BATCH_PLAN' --prepared-manifest '$MANIFEST' && tmux new-session -d -s 'vwe-budget400-cosmos-$BATCH' -c '$DEST' \"bash scripts/ops/remote.sh environment/core/bin/python -m cli native-study run --directory '$RUN' --gpus 0 1 2 3 --watch --exclusive-gpu > '$BUDGET/remote_runs/cosmos_$BATCH.log' 2>&1\""
            echo "$(date -Is) Queued $BATCH on physical GPUs0–3; original400-case cohort and all chunks unchanged"
        fi
        environment/core/bin/python -c 'import json,sys; f=open(sys.argv[1],"x"); json.dump({"remote":sys.argv[2],"destination":sys.argv[3]},f); f.close()' "${MANIFEST%/*}/transfer_ack.json" "$REMOTE" "$DEST"
    done
fi
wait_stage 'IRASim remote native acceptance completed'
while [[ ! -f "$HANDOFF/handoff.json" ]]; do
    tmux has-session -t '=vwe-forgeai-native-handoff' || { echo 'Handoff failed' >&2; exit 1; }
    sleep 15
done
tar -cf - "$HANDOFF" scripts/ops/launch_remote_native.py | "${SSH[@]}" "$REMOTE" "tar --no-same-owner -xf - -C '$DEST'"
printf -v RSYNC_RSH '%q ' "${SSH[@]}"
export RSYNC_RSH
for gpu in 0 1 2 3; do
    # Start each disjoint queue as soon as its own inputs arrive; do not wait
    # for completed source cases or unrelated caches to finish migrating.
    environment/core/bin/python - "$HANDOFF/gpu${gpu}_plan.json" <<'PY' | rsync -arR --no-owner --no-group --partial --files-from=- ./ "$REMOTE:$DEST/"
import json
import sys
for case in json.load(open(sys.argv[1]))["cases"]:
    print(case["prepared"] + "/")
PY
    "${SSH[@]}" "$REMOTE" "cd '$DEST' && bash scripts/ops/remote.sh environment/core/bin/python scripts/ops/launch_remote_native.py --mode irasim --handoff '$HANDOFF' --origin-root '$TASK_ROOT' --shard gpu${gpu}"
done
echo "$(date -Is) IRASim remaining-case queues launched on physical GPUs 0–3 only"
wait_stage 'Cosmos runtime and checkpoint transferred'
tar -cf - checkpoints/world_models/cosmos3_nano_droid_fd/vae checkpoints/world_models/cosmos3_nano_droid_fd/text_tokenizer \
    | "${SSH[@]}" "$REMOTE" "tar --no-same-owner -xf - -C '$DEST'"
while [[ ! -f datasets/cosmos3_droid_prepared/full_holdout_v2/plan.json ]]; do
    live "$PREP_PID" || { echo 'Cosmos preparation stopped before full plan' >&2; exit 1; }
    sleep 30
done
tar -cf - datasets/cosmos3_droid_prepared/full_holdout_v2 | "${SSH[@]}" "$REMOTE" "tar --no-same-owner -xf - -C '$DEST'"
"${SSH[@]}" "$REMOTE" "cd '$DEST' && bash scripts/ops/remote.sh environment/core/bin/python scripts/ops/launch_remote_native.py --mode cosmos"
echo "$(date -Is) Cosmos full native cohort scheduled on physical GPUs 0–3 only"

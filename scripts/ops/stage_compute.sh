#!/usr/bin/env bash
# Priority native-compute assets; no cache download and no private model loop.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE=${MIGRATION_REMOTE:?Set MIGRATION_REMOTE}
DEST=${MIGRATION_DEST:?Set MIGRATION_DEST}
KEY=${MIGRATION_KEY:?Set MIGRATION_KEY}
ACCEPTANCE_OUTPUT=${ACCEPTANCE_OUTPUT:-results/remote_acceptance/irasim_native_v1}
[[ "$REMOTE" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*@[a-zA-Z0-9.-]+$ ]] || exit 2
[[ "$DEST" =~ ^/[-a-zA-Z0-9_./]+$ && "$DEST" != *'/../'* && "$DEST" != */.. ]] || exit 2
[[ "$ACCEPTANCE_OUTPUT" == results/remote_acceptance/* && "$ACCEPTANCE_OUTPUT" != *..* ]] || exit 2
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ProxyCommand=none -o ProxyJump=none)
if [[ -n "${MIGRATION_HOST_KEY_ALIAS:-}" ]]; then
    SSH+=(-o "HostKeyAlias=$MIGRATION_HOST_KEY_ALIAS" -o StrictHostKeyChecking=yes)
fi
printf -v RSYNC_RSH '%q ' "${SSH[@]}"
export RSYNC_RSH
cd "$TASK_ROOT"
stream() {
    echo "$(date -Is) transferring: $*"
    rsync -aR --no-owner --no-group --partial --info=progress2,stats2 \
        --exclude='__pycache__' "$@" "$REMOTE:$DEST/"
}
# Stage one complete unchanged native case before the entire corpus.
stream .cache/uv/python environment/world_models/irasim third_party/irasim \
    checkpoints/world_models/irasim \
    datasets/irasim/prepared_full_short_v1/cases/irasim_lt_000027_start0000_seed3407
stream .cache/irasim/checkpoint_shards/part_*/.vlaworldeval-download.json
"${SSH[@]}" "$REMOTE" "cd '$DEST' && python3 scripts/ops/relocate_runtime.py --root '$DEST' --apply"
"${SSH[@]}" "$REMOTE" "cd '$DEST' && bash scripts/ops/remote.sh environment/world_models/irasim/bin/python -c 'import torch,torchvision,diffusers; print(torch.__version__, torch.cuda.is_available())'"
# This is explicitly an environment acceptance repeat, not part of full-cohort scoring.
"${SSH[@]}" "$REMOTE" "cd '$DEST' && bash scripts/ops/remote.sh environment/core/bin/python -m cli replay --config datasets/irasim/prepared_full_short_v1/cases/irasim_lt_000027_start0000_seed3407/config.json --input datasets/irasim/prepared_full_short_v1/cases/irasim_lt_000027_start0000_seed3407/input.npz --reference datasets/irasim/prepared_full_short_v1/cases/irasim_lt_000027_start0000_seed3407/reference.npz --output '$ACCEPTANCE_OUTPUT'"
echo "$(date -Is) IRASim remote native acceptance completed"
stream datasets/irasim/prepared_full_short_v1 datasets/irasim/prepared_full_long_v1
echo "$(date -Is) IRASim complete prepared corpora transferred"
stream environment/world_models/cosmos3_nano environment/world_models/cosmos3_nano_droid_fd \
    third_party/cosmos-framework third_party/cosmos \
    checkpoints/world_models/cosmos3_nano_droid_fd/community_jere_mybao_v1_20k_ema
"${SSH[@]}" "$REMOTE" "cd '$DEST' && python3 scripts/ops/relocate_runtime.py --root '$DEST' --apply"
echo "$(date -Is) Cosmos runtime and checkpoint transferred"

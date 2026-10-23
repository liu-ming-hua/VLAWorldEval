#!/usr/bin/env bash
# Run from a relocated project root, only after the old resident worker closes.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com
campaign=results/checkpoint_capability_v1/budget400_v1
environment/core/bin/python - <<'PY'
import json
from pathlib import Path
import time
path = Path('results/checkpoint_capability_v1/budget400_v1/persistent_v1/resident_session/session_close.json')
deadline = time.monotonic()+1800
while not path.is_file():
    if time.monotonic() > deadline:
        raise TimeoutError('Old resident session did not reach a safe close boundary')
    time.sleep(5)
if json.loads(path.read_text())['status'] != 'closed':
    raise RuntimeError('Old session did not certify a clean close; refusing to overlap GPUs')
PY
if environment/core/bin/python -m evaluation.execution.native_batch_benchmark \
  --plan "$campaign/cosmos20k/plan.json" \
  --directory "$campaign/batch_qualification_v1" --sizes 1 2 4 8; then
if environment/core/bin/python -m evaluation.execution.native_batch_stream \
  --plan "$campaign/cosmos20k/plan.json" \
  --directory "$campaign/batched_v1" \
  --previous "$campaign/remote_runs/cosmos_runs" "$campaign/persistent_v1" \
  --benchmark "$campaign/batch_qualification_v1/progress.json"; then
  exit 0
fi
fi
# Never leave four GPUs idle if qualification or the batched runtime fails.
environment/core/bin/python -m cli native-study stream \
    --plan "$campaign/cosmos20k/plan.json" \
    --directory "$campaign/resumed_serial_v2" \
    --batch-directory "$campaign/cosmos20k/batches" \
    --previous-directory "$campaign" --gpus 0 1 2 3

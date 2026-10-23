#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT=https://hf-mirror.com
source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${ROOT_DIR}/third_party/world-model-eval" \
  https://github.com/world-model-eval/world-model-eval.git \
  709665b29f5c95124e7dbc05b0d4d47cd353de97
cd "${ROOT_DIR}"
"${CORE_PYTHON_BIN:-${ROOT_DIR}/environment/core/bin/python}" -c \
  'from pathlib import Path; from vlaworldeval.worldgym_runtime import prepare_runtime, runtime_directory; root=Path.cwd(); print(prepare_runtime(root, root/"third_party/world-model-eval", runtime_directory(root)))'

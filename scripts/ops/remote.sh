#!/usr/bin/env bash
# Source-free remote launcher: no dependency installation or GPU launch by default.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export VLAWORLDEVAL_ROOT="$TASK_ROOT"
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="$TASK_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/.tools/python"
export HF_HOME="$TASK_ROOT/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PATH="$TASK_ROOT/.tools:$TASK_ROOT/environment/core/bin:$PATH"
cd "$TASK_ROOT"
if (($# == 0)); then
    exec "$TASK_ROOT/environment/core/bin/python" -m cli --help
fi
exec "$@"

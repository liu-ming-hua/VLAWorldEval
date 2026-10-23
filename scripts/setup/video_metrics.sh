#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
while IFS= read -r metrics_proxy_name; do
  if [[ "${metrics_proxy_name,,}" == *_proxy ]]; then unset "${metrics_proxy_name}"; fi
done < <(compgen -v)
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_HTTP_TIMEOUT=600
export XDG_CACHE_HOME="${ROOT_DIR}/.cache" HF_ENDPOINT=https://hf-mirror.com
export TORCH_HOME="${ROOT_DIR}/.cache/torch" HF_HOME="${ROOT_DIR}/.cache/huggingface"
UV_BIN="${UV_BIN:-$(command -v uv)}"
METRICS_ENV="${ROOT_DIR}/environment/evaluators/video"
if [[ ! -x "${METRICS_ENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python "${ROOT_DIR}/environment/core/bin/python" "${METRICS_ENV}"
fi
"${UV_BIN}" pip install --python "${METRICS_ENV}/bin/python" -r "${ROOT_DIR}/environment/requirements/video_metrics.txt"

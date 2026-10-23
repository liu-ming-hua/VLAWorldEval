#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for irasim_proxy_name in $(compgen -v); do
  if [[ "${irasim_proxy_name,,}" == *_proxy ]]; then
    unset "${irasim_proxy_name}"
  fi
done
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${ROOT_DIR}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${ROOT_DIR}/.cache/huggingface/datasets"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"
export TORCH_HOME="${ROOT_DIR}/.cache/torch"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export UV_INDEX_URL="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${UV_INDEX_URL}"
export CUDA_VISIBLE_DEVICES=""
UV_BIN="${UV_BIN:-$(command -v uv)}"
SOURCE_DIR="${ROOT_DIR}/third_party/irasim"
VENV_DIR="${ROOT_DIR}/environment/world_models/irasim"
source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/bytedance/IRASim.git c72b6dade6fcd65971e0aa8ab49ea39b15108c90
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.11
fi
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -r "${ROOT_DIR}/environment/requirements/irasim.txt"
cd "${ROOT_DIR}"
"${VENV_DIR}/bin/python" scripts/download/irasim.py "${@}"

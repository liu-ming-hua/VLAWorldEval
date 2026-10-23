#!/usr/bin/env bash
set -euo pipefail

# Install the official NVIDIA Cosmos3 Diffusers runtime and download the
# nvidia/Cosmos3-Nano checkpoint. Network access deliberately bypasses proxies.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/environment/world_models/cosmos3_nano"
SOURCE_DIR="${ROOT_DIR}/third_party/cosmos"
MODEL_DIR="${ROOT_DIR}/checkpoints/world_models/cosmos3_nano/Cosmos3-Nano"
CACHE_DIR="${ROOT_DIR}/.uv-cache"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT_DIR}/.cache/huggingface"
export UV_INDEX_URL="${PYPI_MIRROR}"
export PIP_INDEX_URL="${PYPI_MIRROR}"
export UV_CACHE_DIR="${CACHE_DIR}"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE="copy"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}. Set UV_BIN=/path/to/uv and rerun." >&2
  exit 1
fi
source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/NVIDIA/cosmos.git \
  b0e54e88c322695dab188e6ed160c4d6d071c39d

cd "${ROOT_DIR}"
mkdir -p "${MODEL_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.11
fi

"${UV_BIN}" pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "${PYPI_MIRROR}" \
  --link-mode=copy \
  -r "${ROOT_DIR}/environment/requirements/cosmos3_nano.txt"

"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
  --root "${ROOT_DIR}" --repo nvidia/Cosmos3-Nano --destination "${MODEL_DIR}"

"${VENV_DIR}/bin/python" -c 'from diffusers import Cosmos3OmniPipeline, CosmosActionCondition; print("Cosmos3 Diffusers API: OK")'

echo
echo "Cosmos3-Nano setup finished."
echo "Python:    ${VENV_DIR}/bin/python"
echo "Source:    ${SOURCE_DIR}"
echo "Models:    ${MODEL_DIR}"
echo "HF mirror: ${HF_ENDPOINT}"
echo "PyPI mirror: ${PYPI_MIRROR}"

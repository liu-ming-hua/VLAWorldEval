#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/third_party/lerobot"
ENV_DIR="${ROOT_DIR}/environment/vla/smolvla"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
EXPECTED_REVISION="e595b7902714ba51f91e47523f66f89c5181b649"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export PIP_INDEX_URL="${PYPI_MIRROR}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"

source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/huggingface/lerobot.git "${EXPECTED_REVISION}"

mkdir -p "${ROOT_DIR}/environment/vla" "${UV_CACHE_DIR}" "${HF_HOME}"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${ENV_DIR}" --python 3.12
fi
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" --index-url "${PYPI_MIRROR}" \
  --link-mode=copy "torch==2.7.1" "torchvision==0.22.1"
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" --index-url "${PYPI_MIRROR}" \
  --link-mode=copy -e "${SOURCE_DIR}[smolvla,dataset]"
"${ENV_DIR}/bin/python" -c "import lerobot, torch; print('SmolVLA ready', torch.__version__)"

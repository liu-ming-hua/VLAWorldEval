#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/third_party/isaac-groot-n17"
ENV_DIR="${ROOT_DIR}/environment/vla/groot_n17"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
EXPECTED_REVISION="23ace64f17aa5015259b8609d371eb61a357c776"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export UV_INDEX_URL="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/NVIDIA/Isaac-GR00T.git "${EXPECTED_REVISION}"

mkdir -p "${ROOT_DIR}/environment/vla" "${UV_CACHE_DIR}" "${HF_HOME}"
"${UV_BIN}" sync --project "${SOURCE_DIR}" --python 3.10 --frozen --no-dev
"${ENV_DIR}/bin/python" -c "import gr00t, torch; print('GR00T N1.7 ready', torch.__version__)"

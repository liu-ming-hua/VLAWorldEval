#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/third_party/openpi"
ENV_DIR="${ROOT_DIR}/environment/vla/openpi"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
EXPECTED_REVISION="215abfb217dbac7d5f1273282331b9b1866c0479"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export OPENPI_DATA_HOME="${ROOT_DIR}/.cache/openpi"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export UV_INDEX_URL="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/Physical-Intelligence/openpi.git "${EXPECTED_REVISION}"

mkdir -p "${ROOT_DIR}/environment/vla" "${UV_CACHE_DIR}" "${HF_HOME}" "${OPENPI_DATA_HOME}"
GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" sync --project "${SOURCE_DIR}" --python 3.11 --frozen --no-dev
"${ENV_DIR}/bin/python" -c "import jax, openpi; print('OpenPI ready', jax.__version__)"

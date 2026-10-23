#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/third_party/isaac-groot-n15"
ENV_DIR="${GROOT_N15_ENV_DIR:-${ROOT_DIR}/environment/vla/groot_n15}"
ENV_DIR="$(realpath -m "${ENV_DIR}")"
if [[ "${ENV_DIR}" != "${ROOT_DIR}/environment/"* ]]; then
  echo "GROOT_N15_ENV_DIR must remain under the project's environment/" >&2; exit 1
fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
EXPECTED_REVISION="4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"

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
ensure_source "${SOURCE_DIR}" https://github.com/NVIDIA/Isaac-GR00T.git "${EXPECTED_REVISION}"

mkdir -p "${ROOT_DIR}/environment/vla" "${UV_CACHE_DIR}" "${HF_HOME}"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${ENV_DIR}" --python "${GROOT_N15_PYTHON:-3.10}"
fi
"${ENV_DIR}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,10), "GR00T N1.5 profile requires Python 3.10"'
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" --index-url "${PYPI_MIRROR}" \
  --link-mode=copy -e "${SOURCE_DIR}[base]"

# Runtime build tools are not installed by the isolated package build above.
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" --index-url "${PYPI_MIRROR}" \
  -r "${ROOT_DIR}/environment/locks/build-tools-py311.txt" "packaging==26.3" "ninja==1.13.0"

# Upstream pins flash-attn separately because it must build against installed Torch.
FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE MAX_JOBS="${MAX_JOBS:-8}" \
  "${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" --index-url "${PYPI_MIRROR}" \
  --link-mode=copy --no-build-isolation "flash-attn==2.7.1.post4"

"${ENV_DIR}/bin/python" -c "import gr00t, torch; print('GR00T N1.5 ready', torch.__version__)"

#!/usr/bin/env bash
set -euo pipefail

# Install GE-Sim 2.0 in its own environment and download only the released
# GE-Sim world-model checkpoint. Network access deliberately bypasses proxies.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/environment/world_models/ge_sim_v2"
SOURCE_DIR="${ROOT_DIR}/third_party/ge-sim-v2"
MODEL_DIR="${ROOT_DIR}/checkpoints/world_models/ge_sim_v2"
CACHE_DIR="${ROOT_DIR}/.uv-cache"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT_DIR}/.cache/huggingface"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
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
ensure_source "${SOURCE_DIR}" https://github.com/AgibotTech/GE-Sim-V2.git \
  4a34498e326dfa88d6300b2e476e8a67795c430a

cd "${ROOT_DIR}"
mkdir -p "${MODEL_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.11
fi

"${UV_BIN}" pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "${PYPI_MIRROR}" \
  --link-mode=copy \
  -r "${ROOT_DIR}/environment/requirements/ge_sim_v2.txt"

# Install the official package without allowing the editable install to alter
# the explicitly selected environment profile.
"${UV_BIN}" pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "${PYPI_MIRROR}" \
  --link-mode=copy \
  --no-deps \
  -e "${SOURCE_DIR}"

CHECKPOINT_DIR="${MODEL_DIR}/checkpoints/gesim_community_v2.0.1_g01op_distill_2B"
"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
  --root "${ROOT_DIR}" --repo agibot-world/Genie-Envisioner-Sim-v2.0 \
  --destination "${MODEL_DIR}" --allow 'checkpoints/gesim_community_v2.0.1_g01op_distill_2B/**'

"${VENV_DIR}/bin/python" -c 'import gesim; print("GE-Sim package:", gesim.__version__)'

echo
echo "GE-Sim 2.0 setup finished."
echo "Python:    ${VENV_DIR}/bin/python"
echo "Source:    ${SOURCE_DIR}"
echo "Models:    ${MODEL_DIR}"
echo "HF mirror: ${HF_ENDPOINT}"
echo "PyPI mirror: ${PYPI_MIRROR}"
echo
echo "Start server with: bash scripts/serve/ge_sim_v2.sh"

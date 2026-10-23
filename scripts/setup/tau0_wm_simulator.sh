#!/usr/bin/env bash
set -euo pipefail

# Install the official tau_0-WM Action-Conditioned Video Simulator in an
# isolated, project-local uv environment. Network access deliberately
# bypasses proxies and uses the configured PyPI/Hugging Face mirrors.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/third_party/tau-0-wm"
VENV_DIR="${ROOT_DIR}/environment/world_models/tau0_wm_simulator"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/world_models/tau0_wm_simulator"
TAU0_DIR="${CHECKPOINT_DIR}/tau-0-wm"
WAN_DIR="${CHECKPOINT_DIR}/Wan2.2-TI2V-5B"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${ROOT_DIR}/.cache/huggingface/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_INDEX_URL="${PYPI_MIRROR}"
export PIP_INDEX_URL="${PYPI_MIRROR}"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE="copy"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}; set UV_BIN=/path/to/uv and rerun." >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${CHECKPOINT_DIR}" "${TAU0_DIR}" "${WAN_DIR}"

source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/sii-research/tau-0-wm.git \
  a3e4c0fe58b88974ab6d9b1522af97e5e46be04d

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.11
fi

echo "[tau0-wm] Installing the official pinned dependency set..."
# The released requirements currently contain an upstream resolver conflict:
# cmeel-boost==1.89.0 requires numpy>=2.3,<2.4 while the same file pins
# numpy==2.2.6. The released lerobot pin also conflicts with the pinned
# packaging/opencv versions. Those packages are training/data-collection
# extras and are not imported by the simulator server/client, so remove only
# those extras from the temporary inference manifest. xformers, bitsandbytes,
# DeepSpeed, TorchAO, TensorBoard, W&B, and dataset/Parquet stacks are optional
# training/optimization
# packages; the official attention module falls back to PyTorch SDPA when
# xformers is absent.
mkdir -p "${ROOT_DIR}/.cache/build_tmp"
TAU0_REQUIREMENTS="$(mktemp "${ROOT_DIR}/.cache/build_tmp/tau0_wm_requirements.XXXXXX.txt")"
trap 'rm -f "${TAU0_REQUIREMENTS}"' EXIT
sed -E \
  -e '/^(cmeel-|coal==|eigenpy==|libcoal==|libpinocchio==|pin==|lerobot==|xformers==|bitsandbytes==|deepspeed==|torchao==|wandb==|tensorboard==|tensorboard-data-server==|datasets==|dill==|fastparquet==|multiprocess==|pandas==|polars==|pyarrow==)/d' \
  "${SOURCE_DIR}/requirements.txt" > "${TAU0_REQUIREMENTS}"
"${UV_BIN}" pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "${PYPI_MIRROR}" \
  --link-mode=copy \
  -r "${TAU0_REQUIREMENTS}"

echo "[tau0-wm] Downloading the simulator checkpoint from the HF mirror..."
"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
  --root "${ROOT_DIR}" --repo sii-research/tau-0-wm --destination "${TAU0_DIR}" --allow 'simulator/**'

if [[ "${TAU0_DOWNLOAD_WAN_BASE:-1}" == "1" ]]; then
  echo "[tau0-wm] Downloading Wan2.2-TI2V-5B base assets from the HF mirror..."
  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
    --root "${ROOT_DIR}" --repo Wan-AI/Wan2.2-TI2V-5B --destination "${WAN_DIR}" \
    --allow 'Wan2.2_VAE.pth' --allow 'models_t5_umt5-xxl-enc-bf16.pth' --allow 'google/umt5-xxl/**'
else
  echo "[tau0-wm] TAU0_DOWNLOAD_WAN_BASE=0: skipping Wan2.2 base assets."
fi

echo
echo "tau_0-WM Simulator setup finished."
echo "Python:        ${VENV_DIR}/bin/python"
echo "Official code: ${SOURCE_DIR}"
echo "Simulator:     ${TAU0_DIR}/simulator"
echo "Wan base:      ${WAN_DIR}"
echo "HF mirror:     ${HF_ENDPOINT}"
echo "PyPI mirror:   ${PYPI_MIRROR}"

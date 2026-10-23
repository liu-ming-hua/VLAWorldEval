#!/usr/bin/env bash
set -euo pipefail

# Install the official A2World runtime and download its base tokenizer plus
# released checkpoints. Network access deliberately bypasses proxies and uses
# the configured HF/PyPI mirrors.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/environment/world_models/a2world"
SOURCE_DIR="${ROOT_DIR}/third_party/a2world"
UPSTREAM_DIR="${SOURCE_DIR}/world_model"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/world_models/a2world"
BASE_DIR="${CHECKPOINT_DIR}/base"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TRANSFORMER_ENGINE_WHEEL_URL="${A2WORLD_TRANSFORMER_ENGINE_WHEEL_URL:-https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/transformer_engine-1.13.0%2Bcu126.torch260-cp310-cp310-linux_x86_64.whl}"
FLASH_ATTN_WHEEL_URL="${A2WORLD_FLASH_ATTN_WHEEL_URL:-https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/flash_attn-2.6.3%2Bcu126.torch260-cp310-cp310-linux_x86_64.whl}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${ROOT_DIR}/.cache/huggingface/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_INDEX_URL="${PYPI_MIRROR}"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE="copy"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}; set UV_BIN=/path/to/uv" >&2
  exit 1
fi
source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/LogosRoboticsGroup/A2World.git \
  077e10ad6cee07342b5e779f11fea78247584834

mkdir -p "${CHECKPOINT_DIR}" "${BASE_DIR}"
cd "${ROOT_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.10
fi

# The full upstream lockfile includes CUDA extension indexes that may be
# unreachable on restricted machines. The default path installs the complete
# Python inference dependency set from the configured PyPI mirror, then adds
# the upstream package without re-resolving its metadata. Set
# A2WORLD_FULL_CUDA=1 to use the official CUDA-12.6 lockfile instead.
if [[ "${A2WORLD_FULL_CUDA:-0}" == "1" ]]; then
  UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
    "${UV_BIN}" sync --project "${UPSTREAM_DIR}" --extra cu126 --inexact
else
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    --index-url "${PYPI_MIRROR}" --link-mode=copy \
    -r "${ROOT_DIR}/environment/requirements/a2world.txt"
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    --index-url "${TORCH_INDEX_URL}" --link-mode=copy --reinstall \
    "torch==2.6.0" "torchvision==0.21.0"
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    --index-url "${PYPI_MIRROR}" --link-mode=copy \
    -e "${UPSTREAM_DIR}" --no-deps
fi

# A2World imports Transformer Engine's PyTorch extension at runtime. The
# public PyPI package is only a source/meta package for this pinned release and
# does not provide the ``transformer_engine_torch`` binary expected by the
# Cosmos runtime. Use the exact official CUDA-12.6 / Torch-2.6 wheel instead.
if [[ "${A2WORLD_FULL_CUDA:-0}" != "1" && "${A2WORLD_SKIP_TRANSFORMER_ENGINE:-0}" != "1" ]]; then
  echo "[a2world] Installing the official Transformer Engine runtime wheel..."
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    --no-deps --link-mode=copy "${TRANSFORMER_ENGINE_WHEEL_URL}"
  if ! "${VENV_DIR}/bin/python" -c 'import flash_attn' >/dev/null 2>&1; then
    echo "[a2world] Installing the official FlashAttention runtime wheel..."
    "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
      --no-deps --link-mode=copy "${FLASH_ATTN_WHEEL_URL}"
  fi
fi

"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
  --root "${ROOT_DIR}" --repo Fleurrr/A2World-World-Model --destination "${CHECKPOINT_DIR}" \
  --allow 'a2world-pretrained.pt' --allow 'a2world-libero.pt'

if [[ "${A2WORLD_SKIP_BASE_ASSETS:-0}" == "1" ]]; then
  echo "Skipping gated NVIDIA base tokenizer; set A2WORLD_SKIP_BASE_ASSETS=0 after accepting the model terms."
else
  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
    --root "${ROOT_DIR}" --repo nvidia/Cosmos-Predict2-2B-Video2World \
    --destination "${BASE_DIR}/nvidia/Cosmos-Predict2-2B-Video2World" --allow 'tokenizer/tokenizer.pth'
fi

"${VENV_DIR}/bin/python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "third_party/a2world/world_model")
from a2world.checkpoints import inspect_checkpoint
for path in Path("checkpoints/world_models/a2world").glob("a2world-*.pt"):
    print(inspect_checkpoint(path).to_dict())
PY

echo
echo "A2World setup finished."
echo "Python:       ${VENV_DIR}/bin/python"
echo "Runtime:      ${UPSTREAM_DIR}"
echo "Checkpoints:  ${CHECKPOINT_DIR}"
echo "Base assets:  ${BASE_DIR}"
echo "HF mirror:    ${HF_ENDPOINT}"
echo "PyPI mirror:  ${PYPI_MIRROR}"
echo "Torch index:  ${TORCH_INDEX_URL}"

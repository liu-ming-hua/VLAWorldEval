#!/usr/bin/env bash
set -euo pipefail

# WorldGym project-local environment and checkpoint setup.
# This script intentionally disables proxy variables. The WorldGym checkpoint
# is hosted on Google Drive, not Hugging Face, so HF_ENDPOINT is only relevant
# to optional Hugging Face policy downloads.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${WORLDGYM_ENV_DIR:-${ROOT_DIR}/environment/world_models/worldgym}"
VENV_DIR="$(realpath -m "${VENV_DIR}")"
if [[ "${VENV_DIR}" != "${ROOT_DIR}/environment/"* ]]; then
  echo "WORLDGYM_ENV_DIR must remain inside the project's environment directory" >&2
  exit 1
fi
CACHE_DIR="${ROOT_DIR}/.uv-cache"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/world_models/worldgym/checkpoint"
CHECKPOINT_NAME="mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_INDEX_URL="${PYPI_MIRROR}"
export PIP_INDEX_URL="${PYPI_MIRROR}"
export UV_CACHE_DIR="${CACHE_DIR}"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_LINK_MODE="copy"
# PyTorch wheels are large; the default 30-second HTTP timeout is too short
# on the current network path.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}. Set UV_BIN=/path/to/uv and rerun." >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.11
fi

# Install only the base WorldGym package here. The repository's optional
# OpenVLA and SpatialVLA extras pin incompatible Transformers versions
# (4.49.0 vs 4.48.1). Install one optional runner separately only when it is
# actually needed.
"${UV_BIN}" pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-url "${PYPI_MIRROR}" \
    --constraint "${ROOT_DIR}/environment/locks/worldgym-openvla-py311.txt" \
    --link-mode=copy \
  ".[worldgym]" \
  "gdown==6.2.0"
CORE_PYTHON_BIN="${VENV_DIR}/bin/python" bash "${ROOT_DIR}/scripts/setup/worldgym_runtime.sh"

CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/worldgym_checkpoint.py" \
  --destination "${CHECKPOINT_PATH}"

VAE_DIR="${ROOT_DIR}/checkpoints/world_models/worldgym/vae/sd3_vae"
if [[ ! -s "${VAE_DIR}/config.json" || ! -s "${VAE_DIR}/diffusion_pytorch_model.safetensors" ]]; then
  echo "WorldGym also needs the SD3 VAE. Accept its upstream gated terms before downloading."
  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
    --root "${ROOT_DIR}" --repo stabilityai/stable-diffusion-3-medium-diffusers \
    --destination "${ROOT_DIR}/checkpoints/world_models/worldgym/vae/sd3_snapshot" \
    --allow 'vae/config.json' --allow 'vae/diffusion_pytorch_model.safetensors'
  mkdir -p "${VAE_DIR}"
  # Copy model assets only; never overwrite an existing locally supplied VAE.
  cp -n "${ROOT_DIR}/checkpoints/world_models/worldgym/vae/sd3_snapshot/vae/config.json" "${VAE_DIR}/config.json"
  cp -n "${ROOT_DIR}/checkpoints/world_models/worldgym/vae/sd3_snapshot/vae/diffusion_pytorch_model.safetensors" \
    "${VAE_DIR}/diffusion_pytorch_model.safetensors"
fi

echo
echo "WorldGym setup finished; inference has not been performed by this command."
echo "Python:     ${VENV_DIR}/bin/python"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "HF mirror:  ${HF_ENDPOINT}"
echo "PyPI mirror: ${PYPI_MIRROR}"

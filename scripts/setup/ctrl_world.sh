#!/usr/bin/env bash
set -euo pipefail

# Project-local Ctrl-World setup. Network access intentionally bypasses proxies.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/environment/world_models/ctrl_world"
SOURCE_DIR="${ROOT_DIR}/third_party/ctrl-world"
MODEL_DIR="${ROOT_DIR}/checkpoints/world_models/ctrl_world"
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

cd "${ROOT_DIR}"
mkdir -p "${MODEL_DIR}"

source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/Robert-gyj/Ctrl-World.git \
  99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.11
fi

"${UV_BIN}" pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "${PYPI_MIRROR}" \
  --link-mode=copy \
  -r "${ROOT_DIR}/environment/requirements/ctrl_world.txt"

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --index-url "${PYPI_MIRROR}" --link-mode=copy huggingface_hub

download_model() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
    --root "${ROOT_DIR}" --repo "${repo_id}" --destination "${local_dir}" "$@"
}

# Match the exact runtime manifests. Do not fetch duplicate fp16/flat exports,
# TensorFlow/Flax weights or media that this adapter never loads.
download_model "stabilityai/stable-video-diffusion-img2vid" "${MODEL_DIR}/svd" \
  --allow model_index.json --allow scheduler/scheduler_config.json \
  --allow feature_extractor/preprocessor_config.json \
  --allow unet/config.json --allow unet/diffusion_pytorch_model.safetensors \
  --allow vae/config.json --allow vae/diffusion_pytorch_model.safetensors \
  --allow image_encoder/config.json --allow image_encoder/model.safetensors
download_model "openai/clip-vit-base-patch32" "${MODEL_DIR}/clip" \
  --allow config.json --allow pytorch_model.bin --allow tokenizer_config.json \
  --allow special_tokens_map.json --allow vocab.json --allow merges.txt
download_model "yjguo/Ctrl-World" "${MODEL_DIR}/ctrl_world" --allow checkpoint-10000.pt

echo
echo "Ctrl-World setup finished."
echo "Python:    ${VENV_DIR}/bin/python"
echo "Source:    ${SOURCE_DIR}"
echo "Models:    ${MODEL_DIR}"
echo "HF mirror: ${HF_ENDPOINT}"
echo "PyPI mirror: ${PYPI_MIRROR}"

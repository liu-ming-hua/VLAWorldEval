#!/usr/bin/env bash
set -euo pipefail
# Independent Cosmos FD environment: no base-Cosmos .pth or copied site-packages.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/third_party/cosmos-framework"
ENV_DIR="${COSMOS_FD_ENV_DIR:-${ROOT_DIR}/environment/world_models/cosmos3_nano_droid_fd}"
ENV_DIR="$(realpath -m "${ENV_DIR}")"
if [[ "${ENV_DIR}" != "${ROOT_DIR}/environment/"* ]]; then
  echo "COSMOS_FD_ENV_DIR must remain under the project's environment/" >&2; exit 1
fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv unavailable; set UV_BIN=/path/to/uv" >&2; exit 1
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export UV_INDEX_URL="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"
export TORCH_EXTENSIONS_DIR="${ROOT_DIR}/.cache/torch_extensions"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE_DIR}" https://github.com/NVIDIA/cosmos-framework.git \
  96303bb0bdd14d9efa18f24d8ef98c7a0bfb8412
PATCH="${ROOT_DIR}/scripts/setup/patches/cosmos_framework_py311_torch27.patch"
if git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  echo "Compatibility patch already applied; preserving it."
elif git -C "${SOURCE_DIR}" apply --check "${PATCH}"; then
  git -C "${SOURCE_DIR}" apply "${PATCH}"
else
  echo "Source differs from the compatibility patch; not changing your checkout" >&2; exit 1
fi
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  FD_PYTHON="${COSMOS_FD_PYTHON:-${ROOT_DIR}/environment/core/bin/python}"
  if [[ ! -x "${FD_PYTHON}" ]]; then
    echo "Set COSMOS_FD_PYTHON to an installed Python 3.11, or set up core first" >&2; exit 1
  fi
  "${FD_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3,11), "FD runtime requires Python 3.11"'
  UV_PYTHON_DOWNLOADS=never "${UV_BIN}" venv "${ENV_DIR}" --python "${FD_PYTHON}"
fi
# Do not operate on the legacy .pth-shared environment: use a new destination
# first and select it via VLAWORLDEVAL_ENVIRONMENTS after real inference passes.
if [[ -f "${ENV_DIR}/lib/python3.11/site-packages/tianji_cosmos_inference_env.pth" ]]; then
  echo "Legacy shared environment detected. Set COSMOS_FD_ENV_DIR to a new project-local directory." >&2
  exit 1
fi
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  --index-url "${UV_INDEX_URL}" --override environment/locks/cosmos-fd-overrides.txt \
  -r environment/requirements/cosmos3_nano_droid_fd.txt
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  --no-deps --no-build-isolation --editable "${SOURCE_DIR}"
# The 4-GPU two-way attention path cannot run on plain SDPA. Upstream's newest
# prebuilt NATTEN wheels do not target our Torch 2.7 ABI: build this fixed source
# release against the installed Torch, never upgrade Torch implicitly.
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" --index-url "${UV_INDEX_URL}" \
  -r "${ROOT_DIR}/environment/locks/cosmos-fd-native-build.txt"
if [[ -z "${CUDA_HOME:-}" ]]; then
  NVCC="$(command -v nvcc || true)"
  if [[ -n "$NVCC" ]]; then
    export CUDA_HOME="$(dirname "$(dirname "$(realpath "$NVCC")")")"
  fi
fi
if [[ ! -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]]; then
  echo "NATTEN source build requires CUDA toolkit nvcc; set CUDA_HOME (a driver alone is not enough)" >&2
  exit 1
fi
export NATTEN_CUDA_ARCH="${NATTEN_CUDA_ARCH:-$("${ENV_DIR}/bin/python" -c 'import torch; major,minor=torch.cuda.get_device_capability(); print(f"{major}.{minor}")')}"
"${ENV_DIR}/bin/python" -c 'import os,re; assert re.fullmatch(r"[0-9]+\.[0-9]+",os.environ["NATTEN_CUDA_ARCH"]), "Use a decimal compute capability, e.g. NATTEN_CUDA_ARCH=8.0, not 80"'
export NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-8}"
mkdir -p "${ROOT_DIR}/.cache/build_tmp"
TMPDIR="${ROOT_DIR}/.cache/build_tmp" "${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  --index-url "${UV_INDEX_URL}" --no-deps --no-build-isolation --no-binary natten "natten==0.21.7"
COSMOS_TRAINING=0 COSMOS_INTERNAL=0 CUDA_VISIBLE_DEVICES= \
  "${ENV_DIR}/bin/python" -c 'import torch,cosmos_framework,natten; from cosmos_framework.inference.args import OmniSetupOverrides; from importlib.metadata import version; print("FD imports ready:",torch.__version__,version("transformers"),version("natten"))'
echo "Environment-only setup complete; model inference has not been verified by this command."
echo "Download FD weights/assets separately: bash scripts/download/cosmos3_nano_droid_fd.sh"

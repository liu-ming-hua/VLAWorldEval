#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HELPER_ENV="${ROOT_DIR}/environment/tools/huggingface"
requested=("$@")
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/download/vla_hf.sh openvla|spatialvla|octo|groot_n15|groot_n17|smolvla [...]" >&2
  exit 2
fi
for task_family in "${requested[@]}"; do
  case "${task_family}" in
    openvla|spatialvla|octo|groot_n15|groot_n17|smolvla) ;;
    *) echo "Unknown family ${task_family}; no download performed" >&2; exit 2 ;;
  esac
done
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv unavailable; set UV_BIN=/path/to/uv" >&2; exit 1
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_DISABLE_XET=1

mkdir -p "${ROOT_DIR}/environment/tools" "${HF_HOME}"
if [[ ! -x "${HELPER_ENV}/bin/hf" ]]; then
  "${UV_BIN}" venv "${HELPER_ENV}" --python 3.11
  "${UV_BIN}" pip install --python "${HELPER_ENV}/bin/python" --index-url "${PYPI_MIRROR}" \
    "huggingface_hub==0.34.4"
fi

download_if_missing() {
  local repo="$1"
  local destination="$2"
  local family="$3"
  if [[ " ${requested[*]} " != *" ${family} "* ]]; then return; fi
  "${HELPER_ENV}/bin/python" "${ROOT_DIR}/scripts/download/hf_snapshot.py" \
    --root "${ROOT_DIR}" --repo "${repo}" --destination "${destination}"
}

download_if_missing openvla/openvla-7b "${ROOT_DIR}/checkpoints/vla/openvla-7b" openvla
download_if_missing IPEC-COMMUNITY/spatialvla-4b-224-pt "${ROOT_DIR}/checkpoints/vla/spatialvla-4b-224-pt" spatialvla
download_if_missing nvidia/GR00T-N1.5-3B "${ROOT_DIR}/checkpoints/vla/groot/n1.5-3b" groot_n15
download_if_missing nvidia/GR00T-N1.7-3B "${ROOT_DIR}/checkpoints/vla/groot/n1.7-3b" groot_n17
download_if_missing lerobot/smolvla_base "${ROOT_DIR}/checkpoints/vla/smolvla/base" smolvla
download_if_missing HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  "${ROOT_DIR}/checkpoints/vla/smolvla/smolvlm2-500m-video-instruct" smolvla
download_if_missing rail-berkeley/octo-base-1.5 \
  "${ROOT_DIR}/checkpoints/vla/octo-base-1.5" octo

# Octo's checkpoint config references this encoder by the legacy repository
# id `t5-base`, so it must exist in the project-local HF cache under that exact
# id for fully offline inference.
if [[ " ${requested[*]} " == *" octo "* ]]; then
  "${HELPER_ENV}/bin/hf" download t5-base
fi

if [[ " ${requested[*]} " == *" groot_n17 "* && "${DOWNLOAD_GROOT_N17_BACKBONE:-1}" == "1" ]]; then
  if compgen -G "${HF_HOME}/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/*/model.safetensors" >/dev/null; then
    echo "Reuse cached GR00T N1.7 backbone: nvidia/Cosmos-Reason2-2B"
  else
    echo "Caching gated GR00T N1.7 backbone: nvidia/Cosmos-Reason2-2B"
    echo "This requires prior acceptance of NVIDIA's model terms and 'hf auth login'."
    "${HELPER_ENV}/bin/hf" download nvidia/Cosmos-Reason2-2B
  fi
fi

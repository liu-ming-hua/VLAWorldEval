#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/environment/world_models/playworld"
SOURCE_DIR="${ROOT_DIR}/third_party/playworld"
REVISION=457942696e322cb24b00450d854fbf9875e01888
while IFS= read -r playworld_proxy_name; do
  if [[ "${playworld_proxy_name,,}" == *_proxy ]]; then unset "${playworld_proxy_name}"; fi
done < <(compgen -v)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export PIP_CACHE_DIR="${ROOT_DIR}/.cache/pip"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_INDEX_URL="${UV_INDEX_URL}"
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2
UV_BIN="${UV_BIN:-$(command -v uv)}"
cd "${ROOT_DIR}"
# Intentionally do not initialize policy/WEAVER submodules.
if [[ ! -e "${SOURCE_DIR}" ]]; then
  git -c http.proxy= -c https.proxy= clone --no-checkout https://github.com/irom-princeton/open-world.git "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" checkout --detach "${REVISION}"
fi
[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${REVISION}" ]] || {
  echo "PlayWorld source revision mismatch; preserving existing checkout" >&2; exit 1;
}
[[ -z "$(git -C "${SOURCE_DIR}" status --porcelain)" ]] || {
  echo "PlayWorld source has local edits; preserving them and refusing unverified runtime" >&2; exit 1;
}
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python 3.11
fi
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -r environment/requirements/playworld.txt
"${VENV_DIR}/bin/python" scripts/download/playworld.py "$@"

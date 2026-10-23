#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
while IFS= read -r proxy_name; do
  if [[ "${proxy_name,,}" == *_proxy ]]; then unset "${proxy_name}"; fi
done < <(compgen -e)
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ ! -x "${UV_BIN}" ]]; then
  echo 'uv not found; set UV_BIN to an executable uv path' >&2; exit 1
fi
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${ROOT_DIR}/.cache/dreamtrue/huggingface"
export MODELSCOPE_CACHE="${ROOT_DIR}/.cache/dreamtrue/modelscope"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache/dreamtrue"
export TORCH_HOME="${ROOT_DIR}/.cache/dreamtrue/torch"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_URL="${PIP_INDEX_URL}"
export UV_HTTP_TIMEOUT=600
export UV_LINK_MODE=copy
SOURCE="${ROOT_DIR}/third_party/dreamtrue"
REVISION=991e2963c015be9c5b7f21695bb82bfaaceffed8
source "${ROOT_DIR}/scripts/setup/common.sh"
ensure_source "${SOURCE}" https://github.com/brave-eai/DreamTrue.git "${REVISION}"
# Never checkout over another collaborator's source edits.
[[ "$(git -C "${SOURCE}" rev-parse HEAD)" == "${REVISION}" ]] || {
  echo "DreamTrue source must be at ${REVISION}; refusing to overwrite checkout" >&2; exit 1;
}
[[ -z "$(git -C "${SOURCE}" status --porcelain)" ]] || {
  echo 'DreamTrue checkout is modified; refusing unversioned runtime' >&2; exit 1;
}
VENV="${ROOT_DIR}/environment/world_models/dreamtrue"
if [[ ! -x "${VENV}/bin/python" ]]; then "${UV_BIN}" venv --python 3.11 "${VENV}"; fi
"${UV_BIN}" pip install --python "${VENV}/bin/python" --index-url "${PIP_INDEX_URL}" \
  -r environment/requirements/dreamtrue.txt
# Import the pinned source directly, as the official demo does. No copied venv
# or nonportable entry-point scripts; the adapter inserts this same source path.
PYTHONPATH="${ROOT_DIR}:${SOURCE}/wmvideo" "${VENV}/bin/python" -c 'from Script.Demo.inference_demo_v2v_vace import WanV2VVaceInferenceModule; print("DreamTrue official import OK; no inference performed")'
if [[ "${1:-}" == --download ]]; then
  PYTHONPATH="${ROOT_DIR}" "${VENV}/bin/python" scripts/download/dreamtrue.py --base --base-source modelscope --raw-droid
fi

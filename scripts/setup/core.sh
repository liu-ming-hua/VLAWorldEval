#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv unavailable; set UV_BIN to its executable path" >&2
  exit 1
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.cache/uv/python"
export UV_INDEX_URL="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_PYTHON_DOWNLOADS=never
CORE_PYTHON="${CORE_PYTHON:-3.11}"
for task_flag in CORE_OFFLINE CORE_DROID CORE_DEV; do
  if [[ "${!task_flag:-0}" != "0" && "${!task_flag:-0}" != "1" ]]; then
    echo "${task_flag} must be 0 or 1 (no environment changes made)" >&2
    exit 1
  fi
done
ENV_DIR="${ROOT_DIR}/environment/core"
OFFLINE_ARGS=()
if [[ "${CORE_OFFLINE:-0}" == "1" ]]; then
  OFFLINE_ARGS+=(--offline)
elif [[ "${CORE_OFFLINE:-0}" != "0" ]]; then
  echo "CORE_OFFLINE must be 0 or 1" >&2
  exit 1
fi
cd "${ROOT_DIR}"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${ENV_DIR}" --python "${CORE_PYTHON}" "${OFFLINE_ARGS[@]}"
fi
"${ENV_DIR}/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Core lock currently requires Python 3.11"'
LOCK_FILE=environment/locks/core-py311.txt
if [[ "${CORE_DROID:-0}" == "1" ]]; then
  LOCK_FILE=environment/locks/core-droid-py311.txt
elif [[ "${CORE_DROID:-0}" != "0" ]]; then
  echo "CORE_DROID must be 0 or 1" >&2
  exit 1
fi
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  --index-url "${UV_INDEX_URL}" "${OFFLINE_ARGS[@]}" \
  --requirements "${LOCK_FILE}"
# Build backend tools are needed only for editable installation, not inference.
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  --index-url "${UV_INDEX_URL}" "${OFFLINE_ARGS[@]}" \
  --requirements environment/locks/build-tools-py311.txt
# Do not jointly solve incompatible model extras.
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  --no-deps --no-build-isolation "${OFFLINE_ARGS[@]}" --editable .
if [[ "${CORE_DEV:-0}" == "1" ]]; then
  "${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
    --index-url "${UV_INDEX_URL}" "${OFFLINE_ARGS[@]}" \
    --requirements environment/locks/engineering-tools-py311.txt
elif [[ "${CORE_DEV:-0}" != "0" ]]; then
  echo "CORE_DEV must be 0 or 1" >&2
  exit 1
fi
echo "Core ready: ${ENV_DIR}/bin/python (proxy-free, mirror=${UV_INDEX_URL})"

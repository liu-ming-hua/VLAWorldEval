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
export UV_PYTHON_DOWNLOADS=never
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
ENV_DIR="${ROOT_DIR}/environment/tools/docs"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${ENV_DIR}" --python "${DOCS_PYTHON:-3.11}"
fi
"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" \
  --index-url "${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --requirements "${ROOT_DIR}/environment/locks/docs-py311.txt"
echo "Build: environment/tools/docs/bin/python -m sphinx -W --keep-going -b html docs .cache/site/html"

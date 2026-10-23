#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.cache/uv/python"
export UV_HTTP_TIMEOUT=600
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
source "$ROOT_DIR/scripts/setup/common.sh"
ensure_source "$ROOT_DIR/third_party/dreamzero" https://github.com/dreamzero0/dreamzero.git \
  ab790c198fbce33503358efbbd4187ce9a89adf3
UV_BIN="${UV_BIN:-$(command -v uv)}"
ENV_DIR="$ROOT_DIR/environment/vla/dreamzero"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv "$ENV_DIR" --python "$ROOT_DIR/environment/core/bin/python"
fi
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r "$ROOT_DIR/environment/requirements/dreamzero-inference.txt"
# Keep DreamZero's `groot` isolated from Isaac-GR00T N1.5 and N1.7.
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --no-deps --no-build-isolation \
  -e "$ROOT_DIR/third_party/dreamzero"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT_DIR}/environment/vla/openpi/bin/python"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export OPENPI_DATA_HOME="${ROOT_DIR}/.cache/openpi"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"

if [[ ! -x "${PYTHON}" ]]; then
  echo "OpenPI environment is missing; run bash scripts/setup/openpi.sh first." >&2
  exit 1
fi

# OpenPI publishes these policy checkpoints in Google Cloud Storage, not on
# Hugging Face. This downloads directly from the official anonymous bucket.
"${PYTHON}" "${ROOT_DIR}/scripts/download/openpi_checkpoints.py" \
  --output-root "${ROOT_DIR}/checkpoints/vla/openpi" "$@"

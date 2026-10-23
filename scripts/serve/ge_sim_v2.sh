#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/environment/world_models/ge_sim_v2"
CONFIG_PATH="${GE_SIM_CONFIG:-${ROOT_DIR}/configs/world_models/ge_sim_v2.yaml}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "GE-Sim environment is missing: ${VENV_DIR}" >&2
  echo "Run bash scripts/setup/ge_sim_v2.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec "${VENV_DIR}/bin/python" -m gesim.server \
  --model gesim_v2 \
  --config "${CONFIG_PATH}" \
  --host "${GE_SIM_HOST:-127.0.0.1}" \
  --port "${GE_SIM_PORT:-9000}" \
  "$@"

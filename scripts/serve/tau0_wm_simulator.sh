#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/third_party/tau-0-wm"
VENV_DIR="${ROOT_DIR}/environment/world_models/tau0_wm_simulator"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/world_models/tau0_wm_simulator"
HOST="${1:-127.0.0.1}"
PORT="${2:-8002}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "τ₀-WM environment not found; run scripts/setup/tau0_wm_simulator.sh first." >&2
  exit 1
fi
if [[ ! -f "${SOURCE_DIR}/web_infer_utils/simulator/server_sim.py" ]]; then
  echo "Official τ₀-WM source not found at ${SOURCE_DIR}." >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/tau-0-wm/simulator/diffusion_pytorch_model.bin" ]]; then
  echo "Simulator checkpoint is not downloaded yet: ${CHECKPOINT_DIR}/tau-0-wm/simulator" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ]]; then
  echo "Wan2.2 base assets are not downloaded yet: ${CHECKPOINT_DIR}/Wan2.2-TI2V-5B" >&2
  exit 1
fi

# Upstream resolves its paths from the source checkout, so generate machine-local
# absolute paths at launch, never into the checked-in portable template.
RUNTIME_DIR="${ROOT_DIR}/.cache/runtime/tau0_wm_simulator"
mkdir -p "${RUNTIME_DIR}"
RUNTIME_CONFIG="$(mktemp "${RUNTIME_DIR}/server.XXXXXX.yaml")"
# The current HF mirror snapshot exposes the simulator as one PyTorch
# checkpoint file, while the upstream loader also supports a directory of
# indexed shards.  Pass the file directly when no index exists; this keeps the
# upstream loader unchanged and makes both layouts work.
TAU0_MODEL_PATH="${CHECKPOINT_DIR}/tau-0-wm/simulator"
if [[ -f "${TAU0_MODEL_PATH}/diffusion_pytorch_model.bin" &&
      ! -f "${TAU0_MODEL_PATH}/diffusion_pytorch_model.bin.index.json" &&
      ! -f "${TAU0_MODEL_PATH}/diffusion_pytorch_model.safetensors.index.json" ]]; then
  TAU0_MODEL_PATH="${TAU0_MODEL_PATH}/diffusion_pytorch_model.bin"
fi
sed \
  -e "s|/path/to/pretrained/model|${TAU0_MODEL_PATH}|g" \
  -e "s|/path/to/Wan2.2-TI2V-5B|${CHECKPOINT_DIR}/Wan2.2-TI2V-5B|g" \
  "${SOURCE_DIR}/configs/deployment/tau_simulator.yaml" \
  > "${RUNTIME_CONFIG}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"

cd "${SOURCE_DIR}"
exec "${VENV_DIR}/bin/python" -m web_infer_utils.simulator.server_sim \
  --config "${RUNTIME_CONFIG}" \
  --host "${HOST}" \
  --port "${PORT}"
